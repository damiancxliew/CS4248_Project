"""EvidenceModel (G1): ranker head, L_rank / L_neg in every neg_mode, online mined negatives.

C2 windows and C4 sentence fields are mocked here (synthetic token ids, no tokenizer) until
features.py (X9) and sentence_features.py (J2) land. Offline tiny encoders only.
"""
import random

import pytest
import torch
from datasets import Dataset
from transformers import BertConfig

from qa.models.evidence_model import (
    IGNORE,
    MASKED_LOGIT,
    NEG_MODES,
    EvidenceModel,
    negative_index,
    pool_sentences,
    ranking_losses,
)
from qa.models.span_model import span_loss
from qa.predict import batches, import_object, run_inference

HPARAMS = {"alpha": 0.3, "beta": 1.0, "margin": 0.5}
CLS, SEP, KEY = 2, 3, 7  # KEY marks the answer token of a synthetic window
VOCAB_SIZE, SEQ_LEN, QUESTION_LEN = 40, 24, 3


def make_model(neg_mode="model", **overrides):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=VOCAB_SIZE, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    cfg = {"encoder": "not-downloaded", "neg_mode": neg_mode, **HPARAMS, **overrides}
    return EvidenceModel.build(cfg, encoder_config=config)


def synthetic_windows(n, seed=0):
    """Mock C2 + C4 rows: ``[CLS] q q q [SEP] context [SEP]``, context split into 2-4 sentences.

    The gold sentence holds the KEY token, which is the one-token answer. Each other sentence is
    a negative candidate with probability 0.8 (the rest stand for answer-containing sentences).
    About 15% of windows hide the answer: CLS labels, ``sent_label = -100``.
    """
    rng = random.Random(seed)
    ctx_start, ctx_len = QUESTION_LEN + 2, SEQ_LEN - QUESTION_LEN - 3
    rows = []
    for _ in range(n):
        n_sent = rng.randint(2, 4)
        cuts = sorted(rng.sample(range(1, ctx_len), n_sent - 1))
        bounds = list(zip([0, *cuts], [*cuts, ctx_len]))
        input_ids = [CLS] + [rng.randrange(10, VOCAB_SIZE) for _ in range(SEQ_LEN - 2)] + [SEP]
        input_ids[ctx_start - 1] = SEP
        sent_ids = [-1] * SEQ_LEN
        for j, (s, e) in enumerate(bounds):
            sent_ids[ctx_start + s:ctx_start + e] = [j] * (e - s)
        gold = rng.randrange(n_sent)
        cands = [int(j != gold and rng.random() < 0.8) for j in range(n_sent)]
        if rng.random() < 0.15:
            answer, gold, static = 0, IGNORE, IGNORE
        else:
            s, e = bounds[gold]
            answer = ctx_start + rng.randrange(s, e)
            input_ids[answer] = KEY
            static = rng.choice([j for j, c in enumerate(cands) if c] or [IGNORE])
        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * SEQ_LEN,
            "token_type_ids": [0] * ctx_start + [1] * (SEQ_LEN - ctx_start),
            "sent_ids": sent_ids,
            "start_positions": answer,
            "end_positions": answer,
            "sent_label": gold,
            "neg_cand_mask": cands,
            "static_neg_label": static,
        })
    return Dataset.from_list(rows)


def first_batch(features, model, labels=True, batch_size=8):
    keys = model.input_keys + (model.label_keys if labels else ())
    return next(iter(batches(features, keys, model.pad_values, batch_size)))


def test_c5_outputs_and_mean_pooled_sentence_scores():
    model = make_model().eval()
    features = synthetic_windows(8)
    batch = first_batch(features, model, labels=False)
    out = model(**batch)
    n_slots = max(len(c) for c in features["neg_cand_mask"])
    assert "loss" not in out
    assert out["start_logits"].shape == out["end_logits"].shape == (8, SEQ_LEN)
    assert out["sent_logits"].shape == out["sent_mask"].shape == (8, n_slots)
    for i in range(8):
        for j in range(n_slots):
            tokens = batch["sent_ids"][i] == j
            if not tokens.any():
                assert not out["sent_mask"][i, j] and out["sent_logits"][i, j] == MASKED_LOGIT
                continue
            expected = model.ranker(out["hidden_states"][i, tokens].mean(0))
            assert out["sent_mask"][i, j]
            assert torch.allclose(out["sent_logits"][i, j], expected.squeeze(), atol=1e-6)


def test_pooling_uses_visible_tokens_only():
    hidden = torch.tensor([[[9.0, 9], [1, 2], [3, 4], [5, 6], [9, 9]]])
    sent_ids = torch.tensor([[-1, 0, 0, 2, -1]])  # sentence 1 lies outside the window
    pooled, mask = pool_sentences(hidden, sent_ids, n_slots=4)
    assert mask.tolist() == [[True, False, True, False]]
    assert pooled[0, 0].tolist() == [2.0, 3.0] and pooled[0, 2].tolist() == [5.0, 6.0]
    assert pooled[0, 1].tolist() == pooled[0, 3].tolist() == [0.0, 0.0]


def test_mined_negative_is_the_top_scoring_candidate():
    logits = torch.tensor([[1.0, 3, 2, 9], [9.0, 1, 2, 0], [0.0, 1, 5, 5]])
    mask = torch.tensor([[True] * 4, [True] * 4, [True, True, False, False]])
    label = torch.tensor([0, 0, 1])
    # w0: slot 3 scores highest but is not a candidate (e.g. repeats the answer) -> slot 1.
    # w1: the gold slot is never a negative, even if the mask lists it -> slot 2.
    # w2: the only candidate is not visible in this window -> no negative.
    cand = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]])
    assert negative_index("model", logits, mask, label, cand).tolist() == [1, 2, IGNORE]

    static = torch.tensor([2, IGNORE, 0])
    for mode in ("lexical", "random"):
        assert negative_index(mode, logits, mask, label, cand, static).tolist() == [2, IGNORE, 0]
    assert negative_index("none", logits, mask, label, cand, static).tolist() == [IGNORE] * 3


def test_ranking_losses_skip_windows_without_gold_or_negative():
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0], [5.0, 5.0, 5.0]])
    label = torch.tensor([0, 1, IGNORE])
    neg = torch.tensor([2, IGNORE, 0])
    l_rank, l_neg = ranking_losses(logits, label, neg, margin=1.5)
    expected_rank = (torch.logsumexp(logits[0], 0) - 2.0 + torch.logsumexp(logits[1], 0) - 4.0) / 2
    assert torch.isclose(l_rank, expected_rank)
    assert torch.isclose(l_neg, torch.tensor(0.5))  # only window 0: 1.5 - 2 + 1

    none = torch.full((3,), IGNORE)
    assert [x.item() for x in ranking_losses(logits, none, none, margin=1.0)] == [0.0, 0.0]


@pytest.mark.parametrize("neg_mode", NEG_MODES)
def test_loss_is_span_plus_weighted_ranking_terms(neg_mode):
    model = make_model(neg_mode, beta=2.0).eval()
    batch = first_batch(synthetic_windows(16, seed=1), model, batch_size=16)
    out = model(**batch)
    neg = negative_index(neg_mode, out["sent_logits"], out["sent_mask"], batch["sent_label"],
                         batch["neg_cand_mask"], batch["static_neg_label"])
    l_rank, l_neg = ranking_losses(out["sent_logits"], batch["sent_label"], neg, HPARAMS["margin"])
    span = span_loss(out["start_logits"], out["end_logits"], batch["start_positions"],
                     batch["end_positions"])
    assert torch.isclose(out["loss"], span + HPARAMS["alpha"] * l_rank + 2.0 * l_neg)
    assert l_rank > 0
    assert (l_neg == 0) if neg_mode == "none" else (l_neg > 0)


@pytest.mark.parametrize("neg_mode", NEG_MODES)
def test_gradients_reach_both_heads_and_the_encoder(neg_mode):
    model = make_model(neg_mode).train()
    out = model(**first_batch(synthetic_windows(16, seed=2), model, batch_size=16))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for param in (model.qa_outputs.weight, model.ranker.weight,
                  model.encoder.embeddings.word_embeddings.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0


def test_overfits_200_synthetic_windows():
    torch.manual_seed(0)
    model = make_model("model").train()
    features = synthetic_windows(200, seed=3)
    loader = batches(features, model.input_keys + model.label_keys, model.pad_values, 50,
                     shuffle=True, generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    epoch_losses = []
    for _ in range(15):
        total = 0.0
        for batch in loader:
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_losses.append(total / len(loader))
    assert epoch_losses[-1] < 0.5 * epoch_losses[0]

    model.eval()
    with torch.no_grad():
        batch = first_batch(features, model, batch_size=200)
        top = model(**batch)["sent_logits"].argmax(-1)
    has_gold = batch["sent_label"] != IGNORE
    assert (top[has_gold] == batch["sent_label"][has_gold]).float().mean() > 0.9


def test_runs_through_the_predict_helpers():
    assert import_object("qa.models.evidence_model:EvidenceModel") is EvidenceModel
    model = make_model()
    features = synthetic_windows(20, seed=4)
    outputs = run_inference(model, features, batch_size=8, device=torch.device("cpu"))
    n_slots = max(len(c) for c in features["neg_cand_mask"])
    assert outputs["start_logits"].shape == (20, SEQ_LEN)
    assert outputs["sent_logits"].shape == outputs["sent_mask"].shape == (20, n_slots)


def test_checkpoint_rebuild_keeps_head_and_config():
    trained = make_model("lexical", alpha=1.0)
    with torch.no_grad():
        trained.ranker.weight.add_(1.0)  # differ from a fresh init
    rebuilt = make_model("lexical", alpha=1.0)
    rebuilt.load_state_dict(trained.state_dict())
    assert torch.equal(rebuilt.ranker.weight, trained.ranker.weight)
    assert (rebuilt.neg_mode, rebuilt.alpha, rebuilt.beta, rebuilt.margin) == ("lexical", 1.0, 1.0, 0.5)
    hidden = trained.encoder.config.hidden_size
    assert sum(p.numel() for p in trained.ranker.parameters()) == hidden + 1  # 769 for base


def test_config_and_data_errors_name_the_problem():
    with pytest.raises(ValueError, match="neg_mode"):
        make_model("bm25")
    with pytest.raises(NotImplementedError, match="J2"):
        EvidenceModel.prepare_features(synthetic_windows(2), {})
    model = make_model("random")
    batch = first_batch(synthetic_windows(4), model)
    del batch["static_neg_label"]
    with pytest.raises(ValueError, match="static_neg_label"):
        model(**batch)
