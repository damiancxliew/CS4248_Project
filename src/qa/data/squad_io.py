"""SQuAD v1.1 loading, deterministic article splits (C1), and descriptive statistics.

Prepare the training split only (official dev remains the final test set)::

    PYTHONPATH=src python -m qa.data.squad_io prepare
    PYTHONPATH=src python -m qa.data.squad_io check

Damian's data.examples_fn hook can use ``qa.data.squad_io:load_examples``.
Statistics use whitespace-separated words, not encoder-specific token counts;
answer lengths use the first reference per question. No model evaluation occurs here.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WH_WORDS = {"what", "who", "whom", "whose", "when", "where", "which", "why", "how"}
# Original CodaLab bundles and SHA-256 values recorded in the HF dataset's
# legacy download metadata: https://huggingface.co/datasets/stanfordnlp/squad_adversarial/commit/878c103ae1e50c1874f51df47321c70624713161
ADVERSARIAL = {
    "addsent": ("0xb765680b60c64d088f5daccac08b3905",
                "40e3602aa5195cdacd03904a9c301ceb17ccf730cc32bd3ab998b66b4401e660"),
    "addonesent": ("0x3ac9349d16ba4e7bb9b5920e3b1af393",
                   "50420ac8d8b7547cd3715347c9a276802bc9466328ba0814adce4c20495e2889"),
}


def load_squad(path):
    """Read SQuAD without rewriting raw files; reject malformed or duplicate questions."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("data"), list):
        raise ValueError("Expected a SQuAD object with a data list")
    qids = set()
    for article in document["data"]:
        if not isinstance(article.get("title"), str):
            raise ValueError("Every article needs a string title")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str):
                raise ValueError("Every context must be a string")
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if not isinstance(qid, str) or qid in qids:
                    raise ValueError(f"Invalid or duplicate question id: {qid!r}")
                qids.add(qid)
                if not isinstance(qa["question"], str) or not qa["answers"]:
                    raise ValueError(f"Question {qid} needs a question and answer references")
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    if (type(start) is not int or not isinstance(text, str) or not text
                            or start < 0 or start + len(text) > len(context)):
                        raise ValueError(f"Invalid answer offset/text for {qid}")
    return document


def load_examples(path):
    """Return an HF Dataset: id, title, context, question, answers (lists of text/offsets)."""
    from datasets import Dataset, Features, Sequence, Value

    rows = []
    for article in load_squad(path)["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"], "title": article["title"],
                    "context": paragraph["context"], "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    },
                })
    schema = Features({
        "id": Value("string"), "title": Value("string"), "context": Value("string"),
        "question": Value("string"),
        "answers": Sequence({"text": Value("string"), "answer_start": Value("int64")}),
    })
    return Dataset.from_dict({key: [row[key] for row in rows] for key in schema}, features=schema)


def split_by_article(document, val_fraction=0.1, seed=2026):
    """Hold out round(fraction * distinct titles); group repeated titles together.

    Titles are sorted before shuffling so membership is independent of input order.
    Original article/paragraph/question order and all answer annotations are retained.
    """
    titles = sorted({article["title"] for article in document["data"]})
    if len(titles) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two article titles and 0 < val_fraction < 1")
    shuffled = titles.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, min(len(titles) - 1, round(len(titles) * val_fraction)))
    held_out = set(shuffled[:n_val])
    train = {**document, "data": [a for a in document["data"] if a["title"] not in held_out]}
    val = {**document, "data": [a for a in document["data"] if a["title"] in held_out]}
    manifest = {
        "seed": seed, "val_fraction": val_fraction,
        "train": [t for t in titles if t not in held_out], "val": sorted(held_out),
    }
    return train, val, manifest


def _distribution(values):
    if not values:
        return {key: None for key in ("min", "mean", "median", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {
        "min": min(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "p90": percentile(0.9),
        "p95": percentile(0.95), "p99": percentile(0.99), "max": max(values),
    }


def dataset_statistics(document):
    """Count stored paragraphs once each; count answer lengths once per question."""
    contexts, answers, question_types = [], [], Counter()
    questions, offset_mismatches = 0, 0
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            contexts.append(len(context.split()))
            for qa in paragraph["qas"]:
                questions += 1
                words = re.findall(r"[a-z]+", qa["question"].lower())
                question_types[next((w for w in words if w in WH_WORDS), "other")] += 1
                answers.append(len(qa["answers"][0]["text"].split()))
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    offset_mismatches += context[start:start + len(text)] != text
    return {
        "articles": len({a["title"] for a in document["data"]}),
        "contexts": len(contexts), "questions": questions,
        "context_words": _distribution(contexts), "answer_words": _distribution(answers),
        "question_types": dict(sorted(question_types.items())),
        "reference_offset_mismatches": offset_mismatches,
    }


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_raw_data(raw_dir=ROOT / "data/raw"):
    """Verify both committed files as bytes, without parsing/evaluating official dev."""
    raw_dir = Path(raw_dir)
    checked = {}
    for line in (raw_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        if name not in {"train-v1.1.json", "dev-v1.1.json"}:
            raise ValueError(f"Unexpected raw-data checksum entry: {name}")
        actual = sha256(raw_dir / name)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
        checked[name] = actual
    if set(checked) != {"train-v1.1.json", "dev-v1.1.json"}:
        raise ValueError("SHA256SUMS must cover both committed SQuAD files")
    return checked


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ASCII escapes preserve Unicode content and support existing Windows readers.
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_statistics_tables(stats, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "data_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "articles", "contexts", "questions", "mean_context_words",
                         "median_context_words", "p95_context_words", "mean_answer_words"])
        for split, row in stats.items():
            writer.writerow([split, row["articles"], row["contexts"], row["questions"],
                             row["context_words"]["mean"], row["context_words"]["median"],
                             row["context_words"]["p95"], row["answer_words"]["mean"]])
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Split & Articles & Contexts & Questions & Context words & Answer words \\",
             r"\hline"]
    for split, row in stats.items():
        lines.append(f"{split} & {row['articles']} & {row['contexts']} & {row['questions']} & "
                     f"{row['context_words']['mean']:.1f} & {row['answer_words']['mean']:.1f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", "% Word lengths are means; first answer per question."])
    (directory / "data_stats.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_splits(input_path, output_dir, tables_dir, seed=2026, val_fraction=0.1):
    """Write C1 and §4.1 statistics; refuses official dev and overwriting input data."""
    input_path, output_dir, tables_dir = Path(input_path), Path(output_dir), Path(tables_dir)
    if input_path.name == "dev-v1.1.json":
        raise ValueError("Official dev is the test set; split train-v1.1.json only")
    raw = (ROOT / "data/raw").resolve()
    for directory in (output_dir.resolve(), tables_dir.resolve()):
        if directory == raw or raw in directory.parents:
            raise ValueError("data/raw is read-only; choose a generated-output directory")
    document = load_squad(input_path)
    train, val, manifest = split_by_article(document, val_fraction, seed)
    manifest["source_sha256"] = sha256(input_path)
    stats = {"train": dataset_statistics(train), "val": dataset_statistics(val)}
    outputs = {"train.json": train, "val.json": val, "split_titles.json": manifest,
               "dataset_stats.json": stats}
    if input_path.resolve() in {(output_dir / name).resolve() for name in outputs}:
        raise ValueError("Generated outputs must not overwrite the source file")
    for name, value in outputs.items():
        _write_json(output_dir / name, value)
    _write_statistics_tables(stats, tables_dir)
    return stats


def _from_flat_examples(examples):
    articles = {}
    for example in examples:
        title = example["title"]
        paragraphs = articles.setdefault(title, {})
        paragraph = paragraphs.setdefault(example["context"], {
            "context": example["context"], "qas": [],
        })
        answers = example["answers"]
        if isinstance(answers, dict):
            answers = [{"text": text, "answer_start": start}
                       for text, start in zip(answers["text"], answers["answer_start"])]
        paragraph["qas"].append({"id": example["id"], "question": example["question"],
                                 "answers": answers})
    return {"version": "1.1", "data": [
        {"title": title, "paragraphs": list(paragraphs.values())}
        for title, paragraphs in articles.items()
    ]}


def download_adversarial(source, output_dir, revision=None):
    """Fetch evaluation-only data; source must be explicitly chosen by the caller.

    CodaLab bytes are checksum-verified. HF requires an explicit Parquet-conversion
    commit SHA and converts flat records back to SQuAD format. Provenance is saved locally.
    Neither source is ever consumed by prepare_splits or used for training here.
    """
    if source not in {"codalab", "hf"}:
        raise ValueError("Choose source codalab or hf explicitly")
    if source == "hf" and (not revision or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("HF downloads require --revision with a full dataset commit SHA")
    output_dir = Path(output_dir)
    raw = (ROOT / "data/raw").resolve()
    if output_dir.resolve() == raw or raw in output_dir.resolve().parents:
        raise ValueError("data/raw is read-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"source": source, "evaluation_only": True, "files": {}}
    if source == "hf":
        provenance.update(repository="stanfordnlp/squad_adversarial", revision=revision)
    # Validate both files before publishing either; leave prior downloads intact on failure.
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        staging = Path(temporary)
        for name, (bundle, expected) in ADVERSARIAL.items():
            destination = staging / f"{name}.json"
            if source == "codalab":
                url = f"https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/"
                with urllib.request.urlopen(url, timeout=60) as response:
                    destination.write_bytes(response.read())
                if sha256(destination) != expected:
                    raise ValueError(f"SHA-256 mismatch for downloaded {name}")
                provenance["files"][name] = {"url": url, "sha256": expected}
            else:
                from datasets import load_dataset

                config = {"addsent": "AddSent", "addonesent": "AddOneSent"}[name]
                # The repository's main revision contains a legacy Python loader
                # that fetches CodaLab. Use the pinned data-only conversion instead.
                url = (f"https://huggingface.co/datasets/stanfordnlp/squad_adversarial/resolve/"
                       f"{revision}/{config}/squad_adversarial-validation.parquet")
                examples = load_dataset("parquet", data_files={"validation": url},
                                        split="validation")
                _write_json(destination, _from_flat_examples(examples))
                provenance["files"][name] = {"config": config, "url": url,
                                              "sha256": sha256(destination)}
            load_squad(destination)
        for name in ADVERSARIAL:
            (staging / f"{name}.json").replace(output_dir / f"{name}.json")
    _write_json(output_dir / "adversarial_sources.json", provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check the committed raw SQuAD checksums")
    check.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    prepare = commands.add_parser("prepare", help="Generate article splits and §4.1 statistics")
    prepare.add_argument("--input", type=Path, default=ROOT / "data/raw/train-v1.1.json")
    prepare.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    prepare.add_argument("--tables-dir", type=Path, default=ROOT / "results/tables")
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    download = commands.add_parser("download", help="Fetch evaluation-only AddSent/AddOneSent")
    download.add_argument("--source", choices=("codalab", "hf"), required=True)
    download.add_argument("--revision", help="Full HF refs/convert/parquet commit SHA (required for hf)")
    download.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(verify_raw_data(args.raw_dir), indent=2))
    elif args.command == "prepare":
        stats = prepare_splits(args.input, args.output_dir, args.tables_dir, args.seed, args.val_fraction)
        print(json.dumps(stats, indent=2))
    else:
        verify_raw_data()
        print(json.dumps(download_adversarial(args.source, args.output_dir, args.revision), indent=2))


if __name__ == "__main__":
    main()
