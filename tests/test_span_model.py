"""SpanModel: C5 outputs, loss, gradients, config pass-through. Offline tiny encoders only."""
import pytest
import torch
from transformers import DebertaV2Config

from damian_mocks import make_tiny_encoder
from qa.models.span_model import SpanModel, span_loss


@pytest.fixture(scope="module")
def encoder_dir(tmp_path_factory):
    return make_tiny_encoder(tmp_path_factory.mktemp("tiny_bert"))


def batch(seq_len=12, labels=True):
    torch.manual_seed(1)
    out = {
        "input_ids": torch.randint(5, 30, (3, seq_len)),
        "attention_mask": torch.ones(3, seq_len, dtype=torch.long),
        "token_type_ids": torch.zeros(3, seq_len, dtype=torch.long),
    }
    if labels:
        out["start_positions"] = torch.tensor([2, 0, 5])
        out["end_positions"] = torch.tensor([3, 0, 7])
    return out


def test_c5_outputs_and_loss(encoder_dir):
    model = SpanModel.build({"encoder": str(encoder_dir)})
    out = model(**batch())
    assert out["start_logits"].shape == out["end_logits"].shape == (3, 12)
    assert out["hidden_states"].shape == (3, 12, 32)
    assert torch.isfinite(out["loss"])
    assert "loss" not in model(**batch(labels=False))


def test_gradients_reach_head_and_encoder(encoder_dir):
    model = SpanModel.build({"encoder": str(encoder_dir)})
    model(**batch())["loss"].backward()
    assert model.qa_outputs.weight.grad.abs().sum() > 0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0


def test_model_specific_keys_pass_through(encoder_dir):
    cfg = {"class": "qa.models.span_model:SpanModel", "encoder": str(encoder_dir),
           "tokenizer": None, "neg_mode": "model", "alpha": 0.3}
    model = SpanModel.build(cfg)
    assert model.model_cfg == {"neg_mode": "model", "alpha": 0.3}


def test_build_from_encoder_config_skips_download(encoder_dir):
    trained = SpanModel.build({"encoder": str(encoder_dir)})
    rebuilt = SpanModel.build({"encoder": "not-a-real/model"}, encoder_config=trained.encoder.config)
    assert rebuilt.load_state_dict(trained.state_dict()) is not None
    trained.eval(), rebuilt.eval()
    b = batch(labels=False)
    assert torch.equal(trained(**b)["start_logits"], rebuilt(**b)["start_logits"])


def test_deberta_ignores_token_type_ids():
    config = DebertaV2Config(vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=32, max_position_embeddings=32, relative_attention=True,
                             position_biased_input=False, type_vocab_size=0)
    model = SpanModel.build({"encoder": "microsoft/deberta-v3-base"}, encoder_config=config).eval()
    b = batch(labels=False)
    with_types = model(**b)["start_logits"]
    b["token_type_ids"] = torch.ones_like(b["token_type_ids"])
    assert torch.equal(with_types, model(**b)["start_logits"])


def test_span_loss_ignores_positions_past_the_window():
    logits = torch.zeros(2, 4)
    logits[0, 1] = 10.0
    inside = span_loss(logits[:1], logits[:1], torch.tensor([1]), torch.tensor([1]))
    mixed = span_loss(logits, logits, torch.tensor([1, 99]), torch.tensor([1, 99]))
    assert torch.allclose(inside, mixed)
