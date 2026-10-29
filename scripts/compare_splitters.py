"""Compare sentence splitters on SQuAD train contexts.

Usage: python compare_splitters.py <path/to/train-v1.1.json> [n_contexts] [seed]

For each splitter we report:
  - answers cut: % of gold answers whose start and end fall in different sentences
  - offset errors: contexts where the splitter's reported span != the text it returned
  - lost text: contexts where non-whitespace text falls outside every sentence
  - sentences per context, and wall-clock time
Then we print boundaries where pysbd and spaCy (parser) disagree, for reading by eye.
"""
import bisect
import json
import random
import sys
import time

import pysbd
import spacy


def load_contexts(path, n, seed):
    data = json.load(open(path))["data"]
    paras = [p for art in data for p in art["paragraphs"]]
    rng = random.Random(seed)
    sample = rng.sample(paras, min(n, len(paras)))
    return [
        (p["context"], [(a["answer_start"], len(a["text"])) for qa in p["qas"] for a in qa["answers"]])
        for p in sample
    ]


def strip_span(text, s, e):
    """Shrink [s, e) so it has no leading/trailing whitespace."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def split_pysbd(contexts):
    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    out, offset_errs = [], 0
    for ctx in contexts:
        spans, bad = [], False
        for s in seg.segment(ctx):
            if ctx[s.start:s.end] != s.sent:
                bad = True
            spans.append(strip_span(ctx, s.start, s.end))
        offset_errs += bad
        out.append([sp for sp in spans if sp[0] < sp[1]])
    return out, offset_errs


def make_spacy(mode):
    def split(contexts):
        if mode == "parser":
            nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "attribute_ruler"])
        else:  # senter: lightweight sentence tagger, no full parse
            nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer", "attribute_ruler"])
            nlp.enable_pipe("senter")
        out, offset_errs = [], 0
        for ctx, doc in zip(contexts, nlp.pipe(contexts, batch_size=64)):
            spans, bad = [], False
            for sent in doc.sents:
                if ctx[sent.start_char:sent.end_char] != sent.text:
                    bad = True
                spans.append(strip_span(ctx, sent.start_char, sent.end_char))
            offset_errs += bad
            out.append([sp for sp in spans if sp[0] < sp[1]])
        return out, offset_errs
    return split


def lost_text(ctx, spans):
    """True if any non-whitespace char is outside every span (or spans overlap / are unsorted)."""
    pos = 0
    for s, e in spans:
        if s < pos or ctx[pos:s].strip():
            return True
        pos = e
    return bool(ctx[pos:].strip())


def answer_cut(spans, start, length):
    starts = [s for s, _ in spans]
    i = bisect.bisect_right(starts, start) - 1
    j = bisect.bisect_right(starts, start + length - 1) - 1
    return i != j


def show(ctx, pos, width=60):
    return (ctx[max(0, pos - width):pos] + " ‖ " + ctx[pos:pos + width]).replace("\n", " ")


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    items = load_contexts(path, n, seed)
    contexts = [c for c, _ in items]
    n_ans = sum(len(a) for _, a in items)
    print(f"{len(contexts)} contexts, {n_ans} answers (seed {seed}); spaCy {spacy.__version__}, pysbd {pysbd.__version__}\n")

    splitters = {"pysbd": split_pysbd, "spacy-parser": make_spacy("parser"), "spacy-senter": make_spacy("senter")}
    results, cut_examples = {}, {}
    print(f"{'splitter':<14}{'answers cut':>14}{'offset errs':>13}{'lost text':>11}{'sents/ctx':>11}{'time (s)':>10}{'ms/ctx':>8}")
    for name, fn in splitters.items():
        t0 = time.perf_counter()
        spans, offset_errs = fn(contexts)
        dt = time.perf_counter() - t0
        results[name] = spans
        cuts = [(ctx, st, ln) for (ctx, ans), sp in zip(items, spans) for st, ln in ans if answer_cut(sp, st, ln)]
        cut_examples[name] = cuts
        lost = sum(lost_text(c, sp) for c, sp in zip(contexts, spans))
        spc = sum(map(len, spans)) / len(spans)
        print(f"{name:<14}{len(cuts):>7} ({100 * len(cuts) / n_ans:4.2f}%){offset_errs:>13}{lost:>11}{spc:>11.2f}{dt:>10.1f}{1000 * dt / len(contexts):>8.1f}")

    for name, cuts in cut_examples.items():
        print(f"\n=== {name}: sample of cut answers ===")
        for ctx, st, ln in cuts[:5]:
            print(f"  answer={ctx[st:st + ln]!r}\n    ...{ctx[max(0, st - 50):st + ln + 50]!r}...")

    a, b = "pysbd", "spacy-parser"
    n_diff, shown = 0, 0
    print(f"\n=== boundaries where {a} and {b} disagree ('‖' marks the boundary) ===")
    for ctx, sa, sb in zip(contexts, results[a], results[b]):
        ba = {s for s, _ in sa[1:]}
        bb = {s for s, _ in sb[1:]}
        if ba != bb:
            n_diff += 1
        for pos in sorted(ba ^ bb):
            if shown < 25:
                who = f"only {a}" if pos in ba else f"only {b}"
                print(f"  [{who:<17}] ...{show(ctx, pos)}...")
                shown += 1
    print(f"\n{n_diff}/{len(contexts)} contexts ({100 * n_diff / len(contexts):.1f}%) have at least one differing boundary")


if __name__ == "__main__":
    main()
