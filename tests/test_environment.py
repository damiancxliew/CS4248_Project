"""Offline CPU checks for X1; no pretrained weights or SQuAD data needed."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizerFast,
    DebertaV2Config,
    DebertaV2ForQuestionAnswering,
)


ROOT = Path(__file__).resolve().parents[1]


def test_project_packages_import():
    import qa.baselines
    import qa.data
    import qa.decoding
    import qa.eval
    import qa.models


def test_official_evaluator_imports_and_scores():
    path = ROOT / "src/qa/eval/evaluate-v2.0.py"
    spec = importlib.util.spec_from_file_location("official_squad_eval", path)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    assert evaluator.compute_exact("The Nile", "nile") == 1
    assert evaluator.compute_f1("red blue", "red green") == pytest.approx(0.5)


def test_local_tokenizer_dataset_and_config(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhat\nis\nblue\n?\nthe\nsky\n.\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    data = Dataset.from_dict({"question": ["What is blue?"], "context": ["The sky is blue."]})
    features = data.map(
        lambda batch: tokenizer(
            batch["question"], batch["context"], return_offsets_mapping=True
        ),
        batched=True,
    )
    encoding = tokenizer("What is blue?", "The sky is blue.", return_offsets_mapping=True)
    context_tokens = [i for i, sequence in enumerate(encoding.sequence_ids()) if sequence == 1]
    blue_index = context_tokens[-2]
    start, end = features[0]["offset_mapping"][blue_index]
    assert data[0]["context"][start:end] == "blue"
    assert yaml.safe_load("seed: 2026\n")["seed"] == 2026
    assert torch.from_numpy(np.array([1, 2], dtype=np.int64)).tolist() == [1, 2]


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [(BertConfig, BertForQuestionAnswering), (DebertaV2Config, DebertaV2ForQuestionAnswering)],
    ids=["bert", "deberta"],
)
def test_tiny_qa_forward_backward_on_cpu(config_class, model_class):
    # DeBERTa-v3 checkpoints use the Transformers DebertaV2 architecture.
    torch.manual_seed(42)
    config = config_class(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        relative_attention=True,
        position_biased_input=False,
    )
    model = model_class(config).cpu()
    inputs = torch.tensor([[2, 5, 3, 6, 7, 8, 3, 0]])
    output = model(
        input_ids=inputs,
        attention_mask=inputs.ne(0).long(),
        start_positions=torch.tensor([3]),
        end_positions=torch.tensor([4]),
    )
    assert output.start_logits.shape == output.end_logits.shape == (1, 8)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradient = model.qa_outputs.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
