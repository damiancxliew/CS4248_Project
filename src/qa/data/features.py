"""C2 sliding-window QA features; hook: ``qa.data.features:build_features``."""

from datasets import Dataset, Features, Sequence, Value


def build_features(examples, tokenizer, max_seq_len=384, doc_stride=128):
    """Return fixed-width features for training and decoding with a fast tokenizer.

    Questions are the first sequence and contexts the second, even with left
    padding. Offsets index the original context in Python characters; question,
    special and padding offsets are None. ``sequence_ids`` preserves 0/1/None.
    The first reference supplies labels (SQuAD training has one). Missing or
    partially visible answers target the actual CLS position. Token boundaries
    can enclose more text than a reference. SentencePiece whitespace at token
    edges is trimmed for both training and decoding, independently of answers;
    intra-token reference boundaries are not adjusted to force exact recovery.
    Process in bounded batches and bypass cached results for reproducibility.
    """
    if not tokenizer.is_fast:
        raise ValueError("C2 requires a fast tokenizer for offsets and sequence_ids")
    if tokenizer.cls_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("C2 requires CLS and padding tokens")
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if not isinstance(doc_stride, int) or not 0 <= doc_stride < max_seq_len:
        raise ValueError("doc_stride must satisfy 0 <= doc_stride < max_seq_len")
    required = {"id", "question", "context", "answers"}
    if not required <= set(examples.column_names):
        raise ValueError(f"Missing example columns: {sorted(required - set(examples.column_names))}")
    schema = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "offset_mapping": Sequence(Sequence(Value("int64"), length=2)),
        "sequence_ids": Sequence(Value("int64")),
        "example_id": Value("string"),
        "start_positions": Value("int64"), "end_positions": Value("int64"),
    })
    if "token_type_ids" in tokenizer.model_input_names:
        schema["token_type_ids"] = Sequence(Value("int64"))
    if not len(examples):
        return Dataset.from_dict({name: [] for name in schema}, features=schema)

    def encode(batch):
        questions = [q.lstrip() for q in batch["question"]]
        lengths = tokenizer(questions, add_special_tokens=False)["input_ids"]
        for qid, ids in zip(batch["id"], lengths):
            capacity = max_seq_len - len(ids) - tokenizer.num_special_tokens_to_add(pair=True)
            if capacity <= doc_stride:
                raise ValueError(f"{qid}: question leaves {capacity} context tokens; "
                                 "must exceed doc_stride (shorten question or reduce stride)")
        enc = tokenizer(
            questions, batch["context"], truncation="only_second",
            max_length=max_seq_len, stride=doc_stride, padding="max_length",
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        result = {key: enc[key] for key in schema if key in enc}
        result.update(sequence_ids=[], example_id=[], start_positions=[], end_positions=[])
        for i, sample in enumerate(sample_map):
            ids = enc.sequence_ids(i)
            context = batch["context"][sample]
            offsets = []
            for (a, b), sid in zip(enc["offset_mapping"][i], ids):
                if sid == 1:
                    while a < b and context[a].isspace():
                        a += 1
                    while b > a and context[b - 1].isspace():
                        b -= 1
                offsets.append((a, b))
            cls = next(t for t, token in enumerate(enc["input_ids"][i])
                       if token == tokenizer.cls_token_id and ids[t] is None)
            context_tokens = [t for t, sid in enumerate(ids)
                              if sid == 1 and offsets[t][1] > offsets[t][0]]
            start = end = cls
            answers = batch["answers"][sample]
            texts, positions = answers["text"], answers["answer_start"]
            if len(texts) != len(positions):
                raise ValueError(f"{batch['id'][sample]}: answer lists must be parallel")
            if texts:
                a = positions[0]
                b = a + len(texts[0])
                if a < 0 or b > len(context) or not texts[0] or context[a:b] != texts[0]:
                    raise ValueError(f"{batch['id'][sample]}: invalid answer text/offset")
                if (context_tokens and offsets[context_tokens[0]][0] <= a
                        and offsets[context_tokens[-1]][1] >= b):
                    # Smallest token span enclosing the annotated characters.
                    starts = [t for t in context_tokens if offsets[t][0] <= a]
                    ends = [t for t in context_tokens if offsets[t][1] >= b]
                    if starts and ends and starts[-1] <= ends[0]:
                        start, end = starts[-1], ends[0]
            result["offset_mapping"][i] = [list(o) if sid == 1 and o[1] > o[0] else None
                                            for o, sid in zip(offsets, ids)]
            result["sequence_ids"].append(ids)
            result["example_id"].append(batch["id"][sample])
            result["start_positions"].append(start)
            result["end_positions"].append(end)
        return result

    return examples.map(encode, batched=True, batch_size=256,
                        remove_columns=examples.column_names, features=schema,
                        load_from_cache_file=False, desc="QA sliding windows")
