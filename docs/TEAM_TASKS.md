# G14 Team Plan: Evidence-Guided Extractive QA (SQuAD v1.1)

**Team:** Clemen Ong · Damian Liew · Jun Yip Chan · Ravichandran Gokul · Wong Xin Hui
**Topic:** II. Extractive Question Answering (CS4248 AY2026/27 Sem 1)
**Status:** living document. Change it through a PR so everyone sees the diff.

> **v2 changes (after TA feedback):**
> 1. **Lexical overlap dropped.** The lexical-overlap QA baseline is gone. Our main hard negatives are now **model-mined**: the model picks them during training. Lexical (BM25) negatives stay only as an ablation.
> 2. **Sentence ranking.** The evidence head is now a passage-level sentence ranker, evaluated with Recall@k and MRR.
> 3. **Sequential computation.** A rank-then-read (top-k) decoder is compared against joint decoding.
>
> **v3 changes (Xin Hui ↔ Clemen swap):** Xin Hui is busy in the final weeks, so she takes Clemen's early work and Clemen takes her late work. Moved work packages were renumbered to match the new owner:
> - **To Xin Hui:** C1 → **X8** (data split), C2 → **X9** (features), C6 → **X10** (literature survey). She also writes §2 and §4.1 and presents video segment 2.
> - **To Clemen:** X4 → **C8** (error analysis), X6 → **C9** (reproducibility). Clemen also writes §5.1 and §6 and presents video segment 5.
> - X3 (significance script) is now built early, by M2.

---

## 0. TL;DR

| Member | Owns (one line) | Main code they own | Paper sections | Video segment |
|---|---|---|---|---|
| **Clemen** | BERT baseline, sequential (rank-then-read) decoder, error analysis, reproducibility package, slides/video lead | `src/qa/decoding/sequential_decoder.py`, `src/qa/eval/error_analysis.py`, `scripts/run_all.sh`, `configs/final_checkpoints.yaml` | §3.4 Sequential decoding, §5.1 Main results, §5.3 Joint vs. sequential, §6 Error analysis | Results, joint vs. sequential, errors, robustness, conclusion |
| **Damian** | Training/inference engine, DeBERTa span-only model, hyper-parameter tuning, efficiency | `src/qa/train.py`, `src/qa/predict.py`, `src/qa/models/span_model.py`, `src/qa/decoding/span_decoder.py`, `README.md`, `.gitignore` | §1 Intro, §3.1 Base model, §4.2 Training setup | Motivation + system overview |
| **Jun Yip** | Evidence labels, negative candidates (lexical/random), BM25 ranker baseline, sentence features, ranking metrics, adversarial robustness | `src/qa/data/evidence_labels.py`, `src/qa/data/sentence_features.py`, `src/qa/baselines/bm25_ranker.py`, `src/qa/eval/evidence_metrics.py`, `src/qa/eval/adversarial.py` | §3.2 Evidence labels & negatives, §5.4 Robustness | Evidence labels, negatives, ranking quality |
| **Gokul** | Sentence-ranker head, joint loss with model-mined negatives, passage-level ranking, joint decoding, ablations | `src/qa/models/evidence_model.py`, `src/qa/decoding/sentence_ranking.py`, `src/qa/decoding/evidence_decoder.py`, `configs/ablations/` | §3.3 Ranker, joint objective & decoding, §5.2 Ablations & ranking quality | Model + ablations |
| **Xin Hui** | Data + features pipeline, evaluation harness, significance tests, related-work survey, paper integration (front-loaded: most of her work is due by M2) | `src/qa/data/squad_io.py`, `src/qa/data/features.py`, `src/qa/eval/run_eval.py`, `src/qa/eval/significance.py`, `scripts/download_data.sh`, `scripts/make_tables.py`, `paper/main.tex` | §2 Related Work, §4.1 Data, §7 Conclusion & Limitations | Task data, splits, baselines, prior work |

Everyone: trains 3–6 GPU runs from the run matrix (§6), writes their own sections, reviews 1 PR per day of a teammate, and presents about 3 minutes.

Estimated effort is **about 38–41 h per person** (breakdown in §9). Each person owns separate files, so two people should never need to edit the same file (§3).

---

## 1. What "EXCELLENT" means here (rubric → concrete targets)

The rubric: **Content 60%**, **Presentation 30%**, **Novelty 10%**. Every task in this plan maps to one of the rows below.

| Rubric item | What graders ask | Our concrete bar | Owner(s) |
|---|---|---|---|
| Content: valid & technically sound | Is the setup fair and leak-free? | Validation split held out **by article**, not by question. `dev-v1.1.json` touched **only** for final, pre-chosen checkpoints. All DeBERTa variants share the same preprocessing, hyper-parameters and seeds. Official `evaluate-v2.0.py`, unmodified. | Clemen, Damian, Xin Hui |
| Content: quantity of work | Is there enough work? | 2 QA baselines + full model + 5 ablation variants × 3 seeds. 2 sequential pipelines. Sentence-ranking metrics against 2 reference rankers. Adversarial robustness. Significance tests. Error analysis with a manual audit. Efficiency table. | All |
| Content: performance | How good is the system? | BERT-base reproduced within about 1 point of the published 80.8 EM / 88.5 F1. DeBERTa-v3-base span-only should be at least about 92 F1 (check public reports and cite what we compare to). The full model should beat span-only on mean ± std, and the paper should report a significance test either way. Our ranker should beat BM25 on Recall@1. | Damian, Gokul, Jun Yip |
| Content: literature cited & compared | Do we place results in context? | Table comparing published SQuAD v1.1 dev numbers (logistic regression baseline, BiDAF, BERT, SpanBERT, RoBERTa/XLNet, DeBERTa) against ours. Prior work on sentence selection and rank-then-read pipelines, supporting-fact supervision and hard-negative mining is cited, so our novelty claim is honest. | Xin Hui |
| Presentation: report | Is the paper clear and correct? | ACL template, Times 11pt, at most 4 pages of content plus 1 page of references, checked mechanically. One clear method figure, consistent plots, a **contribution statement** (ours vs. external code), and a limitations paragraph. | Xin Hui (integration), all (sections) |
| Presentation: oral | Clear 15-min video, everyone speaks | 5 × about 3 min, one slide template, a rehearsal before recording, timed to 14:30 or less. | Clemen (slides/video lead), all |
| Novelty | Anything unique? | (a) a **sentence-evidence ranker trained jointly with the reader, using model-mined hard negatives** (compared against lexical and random negatives), (b) **joint vs. sequential (rank-then-read) decoding** on the same model, with the ranker's Recall@k ceiling, (c) a test of whether this improves **robustness to distracting sentences (AddSent)**. This links the method to measurable claims. | Gokul, Jun Yip, Clemen |

**Honesty rule:** if the full model does **not** beat span-only on in-distribution dev, report that, and use the ranking metrics, robustness results and error analysis to explain where it helps or hurts. A well-analysed negative result scores better than an unsupported claim.

---

## 2. System specification (agree on this at kickoff; it is the shared reference)

### 2.1 Data and splits
- `data/raw/train-v1.1.json`, `data/raw/dev-v1.1.json` (from Canvas, byte-identical to the official SQuAD release) **are committed**, with `data/raw/SHA256SUMS` and an attribution note in `data/raw/README.md`. Never edit them. Generated data goes in `data/processed/` (gitignored). `scripts/download_data.sh` fetches the evaluation-only AddSent/AddOneSent data and checks the SQuAD checksums.
- **Train/val split by Wikipedia article (`title`)**: hold out about 10% of the 442 train articles (seed 2026). This gives about 8.7k val questions. Splitting by question would leak passages between train and val.
- Splits: `train` (about 90% of articles) → fit; `val` → all tuning and model selection (including `λ` and `k`); `test` = `dev-v1.1.json` → **final numbers only**.
- Extra **evaluation-only** data: Adversarial SQuAD, AddSent and AddOneSent (Jia & Liang, 2017; HF `squad_adversarial`, or the original CodaLab release). Never used for training. State this in the report.

### 2.2 Shared defaults (unless an ablation changes it)
| Setting | Value |
|---|---|
| Encoders | `bert-base-uncased` (baseline), `microsoft/deberta-v3-base` (ours); stretch: `deberta-v3-large` |
| `max_seq_len` / `doc_stride` | 384 / 128 |
| `max_answer_len` / `n_best` | 30 tokens / 20 |
| Optimiser | AdamW, weight decay 0.01, linear decay, 10% warm-up |
| LR / epochs / batch | BERT 3e-5, 2 ep, effective batch 32. DeBERTa: tuned by Damian on val, then **frozen for all DeBERTa variants** |
| Precision | bf16 on Ampere+ GPUs (A100/H100). If only fp16 is available (T4), watch for DeBERTa-v3 overflow and fall back to fp32 |
| Seeds | 13, 42, 87 |

### 2.3 Proposed model (precise version, v2)
1. **Sentence segmentation** of each context with character offsets. If a split falls inside an answer span (e.g. "St. Louis"), merge the pieces.
2. **Gold evidence sentence** = the sentence containing `answer_start`.
3. **Negative candidates** = non-gold sentences that **do not contain the gold answer string**. A sentence that repeats the answer is not a true negative, so excluding these avoids false negatives.
4. **Sentence-ranker head:** for each sentence visible in a window, mean-pool its token states → linear layer → score `s_j`.
5. **Negatives**, chosen by the config key `neg_mode`:
   - `model` (**ours, default**): mined online. At each training step, the negative is the highest-scoring candidate in the window under the *current* ranker head: `s_neg = max_{j ∈ cand} s_j`. This needs no extra stage and no dependency on another run.
   - `lexical` (ablation; the original proposal): a fixed negative, chosen as the candidate with the highest BM25 score against the question.
   - `random` (ablation): a fixed random candidate.
   - `none` (ablation): no negative term (`β = 0`).
6. **Loss:** `L = (L_start + L_end)/2 + α·L_rank + β·L_neg`
   - `L_rank` = listwise cross-entropy over the window's sentences against the gold sentence (ignored, `-100`, when the gold sentence is not in the window).
   - `L_neg` = margin loss `max(0, m − s_gold + s_neg)` (skipped when no negative is in the window).
7. **Passage-level ranking:** compute each sentence's log-probability per window (log-softmax), then take its max over the windows that contain it. This gives a ranked list of all passage sentences. Evaluate it with Recall@1, Recall@3 and MRR.
8. **Decoding, two modes on the same trained model:**
   - **Joint** (Gokul): `score(span) = start_logit + end_logit + λ·log p(sentence containing span)` over all candidate spans. Tune `λ` on val.
   - **Sequential / rank-then-read** (Clemen): keep only the top-`k` sentences from step 7, then pick the best span (joint score) inside them. Sweep `k ∈ {1, 2, 3}` on val. EM is then capped by the ranker's Recall@k; report that ceiling. Stretch: re-encode only the top-`k` sentences with the reader, which gives shorter inputs and lower cost (Min et al., 2018).
   - `λ = 0` with `k = all` gives exactly the standard decoder.

Gokul tunes `α, β, m, λ` and Clemen tunes `k`, both on val (§6). The values are then frozen.

---

## 3. Repository layout and file ownership (prevents conflicts)

**Rule:** you edit only files you own. To change someone else's file, open an issue or comment on their PR. If you need an urgent one-line fix, get their 👍 in chat first.

```
CS4248_Project/
├── README.md                         Damian    (done: architecture + training pipeline; §6 setup/reproduction filled at M4)
├── docs/TEAM_TASKS.md                Damian    (this file; changes via PR)
├── requirements.txt                  Xin Hui   (others: request additions in your PR description)
├── .gitignore                        Damian    (done: data/, outputs/, checkpoints/, wandb/, model weights, .env, LaTeX build files, videos)
├── data/
│   ├── raw/                          Damian    (committed: SQuAD v1.1 train/dev JSON, SHA256SUMS, README.md; read-only)
│   └── processed/                    gitignored (splits, features, evidence annotations; regenerated by code)
├── configs/
│   ├── base.yaml                     Damian    (schema and shared defaults)
│   ├── baselines/*.yaml              Clemen    (BERT)
│   ├── deberta_span/*.yaml           Damian
│   ├── evidence/*.yaml               Gokul
│   ├── ablations/*.yaml              Gokul     (every member's run config lives here, written by Gokul)
│   └── final_checkpoints.yaml        Clemen    (frozen at M3; the only checkpoints evaluated on test)
├── scripts/
│   ├── download_data.sh              Xin Hui
│   ├── run_all.sh                    Clemen    (reproduce everything)
│   └── make_tables.py / make_figs.py Xin Hui
├── src/qa/
│   ├── data/
│   │   ├── squad_io.py               Xin Hui   (load, article-level split, stats)
│   │   ├── features.py               Xin Hui   (tokenise, sliding window, offsets, start/end labels)
│   │   ├── evidence_labels.py        Jun Yip   (sentence split, gold sentence, negative candidates, lexical/random negatives, BM25 scores)
│   │   └── sentence_features.py      Jun Yip   (per-window sentence ids, labels, candidate masks)
│   ├── models/
│   │   ├── span_model.py             Damian
│   │   └── evidence_model.py         Gokul     (ranker head, listwise + margin loss, online negative mining)
│   ├── decoding/
│   │   ├── span_decoder.py           Damian    (standard n-best post-processing)
│   │   ├── sentence_ranking.py       Gokul     (window → passage-level sentence ranking, C8)
│   │   ├── evidence_decoder.py       Gokul     (joint decoding)
│   │   └── sequential_decoder.py     Clemen    (rank-then-read top-k decoding)
│   ├── baselines/bm25_ranker.py      Jun Yip   (BM25 sentence ranker; ranking baseline and pipeline input)
│   ├── train.py                      Damian
│   ├── predict.py                    Damian
│   └── eval/
│       ├── evaluate-v2.0.py          Xin Hui   (committed; OFFICIAL, UNMODIFIED; credited in README §5)
│       ├── run_eval.py               Xin Hui
│       ├── significance.py           Xin Hui
│       ├── error_analysis.py         Clemen
│       ├── evidence_metrics.py       Jun Yip   (Recall@k, MRR)
│       └── adversarial.py            Jun Yip
├── tests/test_<module>.py            owner of the module under test
├── results/
│   ├── runs/<run_id>.json            whoever ran it (one file per run, so no shared-file conflicts)
│   ├── predictions/<run_id>.<split>.json   final test predictions only (small; needed for significance tests)
│   ├── audits/                       Jun Yip (label/negative audit), Clemen (error audit); separate files
│   └── tables/, figures/             generated by Xin Hui's scripts, never hand-edited
├── paper/
│   ├── main.tex, acl.sty, refs.bib   Xin Hui   (bib entries: add yours in your PR, sorted by key)
│   └── sections/<nn>_<name>.tex      owner per §8 (one file per section, so no conflicts)
└── slides/                           Clemen (template), each member their own slides
```

### 3.1 Interfaces (freeze by Milestone M0; change only by team agreement)
These contracts let people build in parallel against mocks before upstream code exists.

| Contract | Producer → Consumer | Format |
|---|---|---|
| **C1 Splits** | Xin Hui → all | `data/processed/{train,val}.json` in SQuAD v1.1 format plus `data/processed/split_titles.json`. Test = raw `dev-v1.1.json`. |
| **C2 Base features** | Xin Hui → Damian, Gokul, Jun Yip | HF `Dataset` with `input_ids, attention_mask, (token_type_ids), offset_mapping, example_id, start_positions, end_positions`. Context tokens identifiable through `sequence_ids`. |
| **C3 Evidence annotations** | Jun Yip → Gokul, Clemen, Xin Hui | `data/processed/evidence_{split}.jsonl`, one line per question: `{"id", "sentences": [[cs, ce], ...], "gold_sent": int, "neg_candidates": [int], "lex_neg": int\|null, "rand_neg": int\|null, "bm25": [float], "straddle_merged": bool}` (char offsets into the context; `bm25` = one score per sentence against the question). |
| **C4 Sentence features** | Jun Yip → Gokul | `add_sentence_fields(features, evidence, neg_mode) -> features` adds `sent_ids` (per token, `-1` outside context), `sent_label` (window-local index or `-100`), `neg_cand_mask [S_max]` (window-local candidate mask, used by `model` mode), `static_neg_label` (window-local index or `-100`; only for `lexical`/`random`). |
| **C5 Model output** | Damian/Gokul → decoders | `start_logits [B,T]`, `end_logits [B,T]`, optional `sent_logits [B,S_max]` with `sent_mask`. |
| **C6 Predictions** | `predict.py` → eval | Official format `{qid: answer_text}` plus `nbest.json` `{qid: [{text, start_logit, end_logit, sent_idx, sent_logprob, score}]}`. |
| **C7 Run record** | everyone → Xin Hui | `results/runs/<run_id>.json`: `{run_id, system, variant, seed, split, exact, f1, recall_at_1, recall_at_3, mrr, n, params_M, train_gpu_hours, gpu, infer_qps, git_sha, config}` (ranking fields `null` for span-only). |
| **C8 Sentence ranking** | Gokul (model), Jun Yip (BM25) → Clemen, Jun Yip, Xin Hui | `outputs/<run_id>/<split>.sent_rank.json`: `{qid: [[sent_idx, score], ...]}`, sorted best-first, covering every passage sentence. `score` = passage-level log-prob for the model, raw BM25 for the baseline. |

Use a tiny model (e.g. `hf-internal-testing/tiny-random-bert`) and about 200 examples for smoke tests and unit tests so nobody waits for a GPU.

---

## 4. Git workflow
1. `main` is protected: PRs only, at least 1 approving review, squash-merge. Damian (repo owner) sets this up.
2. Branch names: `<name>/<topic>`, e.g. `junyip/negative-candidates`. Keep PRs small (under about 400 lines) and merge often.
3. PR checklist: tests pass (`pytest -q`), no data or checkpoints committed, config committed for any run you report, and the run JSON is in `results/runs/`.
4. Reviewer pairs (rotate weekly): Xin Hui ↔ Damian (data/features ↔ training engine), Jun Yip ↔ Gokul, Clemen reviews eval- and analysis-touching PRs (Xin Hui does so until M2). The reviewer must run the code, not just read it.
5. **Never commit:** anything in `data/` other than the raw SQuAD files already there, model checkpoints, `outputs/`, W&B dirs, `.env` or tokens. Large artefacts go to a shared Google Drive or HF Hub (private), linked in the README.
6. Every reported number must be traceable: `run_id → config → git_sha`.

---

## 5. Work packages per member

Each WP lists **deliverable → done when**. Items marked ⏱ are on the critical path; finish them first.

### Clemen Ong: BERT baseline, sequential decoding, error analysis, reproducibility, video lead
C1, C2 and C6 moved to Xin Hui (now X8, X9, X10). Most of Clemen's work falls in M3–M6.

| # | Work package | Done when |
|---|---|---|
| C3 | `sequential_decoder.py` (rank-then-read): reads C8 (any ranker) + logits, restricts spans to the top-`k` sentences, sweeps `k ∈ {1, 2, 3, all}` on val. Runs **R9a** (BM25 top-k → R3 span-only reader; the classic pipeline) and **R9b** (our ranker top-k → R4). Reports the **Recall@k ceiling** and splits the remaining errors into ranker errors (gold sentence not in top-k) and reader errors (gold sentence kept, wrong span) | Test: `k = all` reproduces the joint decoder's output exactly. `k` frozen on val before test. Joint-vs-sequential table and ceiling figure drafted for §5.3. |
| C4 | BERT-base baseline: config plus 3 seeds (runs R2) | 3 run JSONs. Mean within about 1 point of 80.8 / 88.5. If not, find out why before M3. |
| C5 | Lexical-negative ablation, 3 seeds (runs R6b; config written by Gokul) | 3 run JSONs. |
| C7 | Slides template and video assembly: one deck, consistent style, stitches 5 recordings, checks total under 15:00 | Final MP4 and PDF slides ready by M6. |
| C8 (was X4) | `error_analysis.py`: automatic, mutually exclusive categories, applied in this order: **(1) wrong sentence** (prediction not in the gold sentence; uses contract C3), **(2) boundary** (overlaps gold, EM = 0), **(3) numeric/date** (gold is a number or date by spaCy NER or regex), **(4) entity** (gold is a named entity, prediction is a different span), **(5) other**. Slices: question type (wh-word), context length (single vs. multi-window), answer length. **Evidence-conditioned accuracy:** P(answer correct \| top-1 sentence correct) vs. P(answer correct \| top-1 wrong), for span-only (span-implied ranking) and the full model (contract C8). **Manual audit** of 100 errors of the full model and 100 of span-only. Start the script on seed-42 outputs at M2, so only the final re-run and the audit fall in M4 | Error table and one figure for §6. Examples box with 2–3 illustrative cases. "Wrong sentence" errors compared between span-only and the full model, which directly tests the method. This links to C3: wrong-sentence errors are the joint-decoding counterpart of ranker errors. |
| C9 (was X6) | Reproducibility package: `scripts/run_all.sh` (one command per results table; Damian copies these into README §6), `configs/final_checkpoints.yaml`, the attribution list of external code, and the final submission zip (code + external code + data instructions). **Reproduction check:** a teammate who did not write the code reproduces one table from the README alone | Check passed at M4. Submission zip built and tested at M6. |

### Damian Liew: Training engine, DeBERTa span model, tuning, efficiency
| # | Work package | Done when |
|---|---|---|
| D0 ⏱ | Repo admin: ~~make the repo private~~ (done), ~~README.md and .gitignore~~ (done), ~~add 4 collaborators~~ (done), branch protection | All 5 can push branches; `main` protected. |
| D1 ⏱ | `train.py` + `configs/base.yaml`: YAML config, seeding, bf16/fp16, gradient accumulation, eval on val each epoch, best-checkpoint selection on val F1, writes C7 run JSON, optional W&B logging. Supports a pluggable model class (span-only or Gokul's evidence model), extra feature columns, and passes model-specific config keys (e.g. `neg_mode`) through unchanged | Smoke test with the tiny model passes on CPU in under 2 min. Gokul can plug in the evidence model **without editing `train.py`** (model registry or config key). |
| D2 ⏱ | `span_model.py` (BERT/DeBERTa span head) + `span_decoder.py` (n-best over windows, max answer length, maps back to text through offsets) + `predict.py` (C6 output; also saves raw logits so decoders can be swept without re-running the model) | Standard decoder reproduces HF reference post-processing on a tiny sample. `predict.py` can run any checkpoint on any split. |
| D3 | DeBERTa-v3-base span-only **hyper-parameter search on val**: LR ∈ {1e-5, 2e-5, 3e-5}, epochs ∈ {2, 3}, warm-up ∈ {6%, 10%}, seed 42 | Chosen setting committed to `configs/deberta_span/best.yaml`, and frozen for all DeBERTa variants. |
| D4 | DeBERTa span-only, 3 seeds (runs R3) | 3 run JSONs, mean ± std on val and test. |
| D5 | Efficiency table: parameters, training GPU-hours, inference throughput (questions/s, batch 64, same GPU) for BERT, DeBERTa span-only, full model (joint and sequential decoding) | Table in `results/tables/efficiency.csv`. The ranker head's overhead and the decoding cost are quantified. |
| D6 (stretch) | DeBERTa-v3-large span-only and full model, 1 seed each, if compute allows. Or help Clemen build the re-encode-top-k sequential reader | Reported as "scaling check", not in main claims. |
| D7 | Keep `README.md` current: fill §6 *Setup and reproduction* (environment, data download, one command per results table taken from Clemen's `run_all.sh`, checkpoint links) and the *TBD* results/hyper-parameter fields | Done at M4, before Clemen's reproduction check (C9). |

### Jun Yip Chan: Evidence labels, negatives, ranking quality, robustness
| # | Work package | Done when |
|---|---|---|
| J1 ⏱ | `evidence_labels.py`: sentence splitter with char offsets (spaCy or pysbd; justify the choice), merge splits inside answers, gold sentence, **negative candidates** (exclude sentences containing the answer string), per-sentence **BM25 scores**, **lexical negative** (top-BM25 candidate) and **random negative** → C3 files for train/val/test/AddSent | Statistics: sentences per context, % answers that needed merging, % sentences excluded as answer-containing, % contexts with no candidate (single sentence). Test: the gold sentence span always contains the answer span. |
| J2 ⏱ | `sentence_features.py`: C4. Map tokens to sentences per window, window-local `sent_label`, `neg_cand_mask` and `static_neg_label`, handle partially visible sentences | Test on hand-built examples, including an answer in the overlap region of two windows and a gold sentence cut by the window edge. |
| J3 | **Label and negative audit**: manually inspect 100 random train questions. Is the gold sentence the right evidence? Compare **model-mined** (from Gokul's dump, G1), **lexical** and **random** negatives: is each a plausible distractor, or a false negative (actually answers the question)? | Numbers for §3.2 (e.g. "x/100 model-mined negatives judged plausible distractors vs y/100 lexical vs z/100 random; w false negatives"). Annotation sheet in `results/audits/`. |
| J4 | Random-negative variant runs (R6, 3 seeds) | 3 run JSONs. |
| J5 | `adversarial.py`: run **every final system** (including the sequential pipelines R9a/R9b) on AddSent and AddOneSent, and report EM/F1 and the drop from clean dev. Also measure (a) how often the prediction falls inside the inserted adversarial sentence and (b) how often each ranker puts the adversarial sentence at rank 1 | Robustness table and one figure for §5.4. This is the main test of our novelty claim. |
| J6 | `bm25_ranker.py` (R1; writes C8 from C3 BM25 scores) + `evidence_metrics.py`: **Recall@1, Recall@3, MRR** on val/test/AddSent for: BM25, the span-implied ranking (sentence containing R3's top span), and our ranker (R4, R5, R6, R6b) | Ranking table for §5.2 and §5.4. Our ranker vs. BM25 compared with Xin Hui's significance script. |

### Ravichandran Gokul: Ranker, joint model, joint decoding, ablations
| # | Work package | Done when |
|---|---|---|
| G1 ⏱ | `evidence_model.py`: wraps the span model, adds the sentence-ranker head, listwise `L_rank` and margin `L_neg`, with `neg_mode ∈ {model, lexical, random, none}`. `model` mode mines the top-scoring candidate online from `neg_cand_mask`. Registered so `train.py` can load it from config. At the end of training, dumps the mined negatives for 200 val questions for Jun Yip's audit (J3) | Unit test on synthetic `sent_ids` (can start before J2 is done, using mocks). Each `neg_mode` tested. Loss decreases on a 200-example overfit test. Gradients reach both heads. |
| G2 ⏱ | `sentence_ranking.py` (C8: window log-softmax → passage-level max) + `evidence_decoder.py` (joint decoding) | Test: a sentence's passage score equals its best window score. `λ = 0` gives output identical to Damian's `span_decoder.py`. |
| G3 | Tune on val (seed 42, DeBERTa hyper-parameters frozen from D3, `neg_mode = model`): `α ∈ {0.1, 0.3, 1.0}`, `β ∈ {0.5, 1.0}`, `m ∈ {0.5, 1.0}`. Then a `λ` sweep {0, 0.25, 0.5, 1, 2} **at inference only** | Chosen values in `configs/evidence/best.yaml`. Sweep curve saved (good small figure for §5.2). |
| G4 | Write **all** ablation configs (R4–R7, R6b) so every run uses identical settings, then run the full model with 3 seeds (R4) and compute R7 (no evidence in decoding) from R4 checkpoints | Configs merged before M3. 3 run JSONs for R4 and R7. |
| G5 | Ablation table and analysis: full vs. −ranking objective (= R3) vs. −negatives (R5) vs. random (R6) vs. lexical (R6b) vs. −evidence in decoding (R7). Mean ± std, with Recall@1/MRR from J6 and significance stars from Xin Hui's script | Table drafted for §5.2 with a 3–4 sentence interpretation, including whether model-mined beats lexical. |

### Wong Xin Hui: Data pipeline, evaluation harness, significance, related work, paper integration
Xin Hui is busy in the final weeks, so her work is front-loaded: almost everything is due by M2. X4 and X6 moved to Clemen (now C8, C9).

| # | Work package | Done when |
|---|---|---|
| X1 ⏱ | Repo skeleton: folder layout (§3), `requirements.txt` with pinned versions (include `numpy`, which the official `evaluate-v2.0.py` imports), `pytest` setup, optional GitHub Actions running the CPU smoke tests | Fresh clone + `pip install -r requirements.txt` + `pytest` passes. |
| X2 ⏱ | `run_eval.py`: wraps the **unmodified** official `evaluate-v2.0.py`, writes C7 JSON. `make_tables.py` aggregates `results/runs/*.json` → mean ± std tables (CSV + LaTeX) | A dummy prediction file (e.g. first-sentence answers) evaluates end to end. Script output matches the official script exactly. |
| X8 ⏱ (was C1) | `squad_io.py`: load SQuAD, article-level train/val split (seed 2026), dataset statistics (articles, contexts, questions, context-length distribution, answer length, question-type distribution). Reads the committed `data/raw/` files. Also `scripts/download_data.sh` (fetches AddSent/AddOneSent, runs the SHA-256 check on the SQuAD files) | Split files exist and are reproducible. Stats table drafted for §4.1. Test shows no title is in both train and val. |
| X9 ⏱ (was C2) | `features.py`: tokenisation with sliding windows and offset mapping; start/end labels; out-of-window answers → CLS | Test: decoding gold labels back through offsets recovers the gold answer text for ≥ 99.5% of train questions (report the remainder and why). Works for BERT and DeBERTa tokenisers. |
| X10 (was C6) | Literature survey and comparison table: published SQuAD v1.1 dev EM/F1 for LR baseline (Rajpurkar 2016), BiDAF (Seo 2017), BERT-base/large (Devlin 2019), SpanBERT (Joshi 2020), XLNet / RoBERTa (Yang 2019; Liu 2019), DeBERTa / DeBERTaV3 (He 2021; He 2023). Evidence-related prior work: sentence selection and rank-then-read (Min et al. 2018), supporting-fact supervision (HotpotQA, Yang et al. 2018), hard-negative mining in retrieval (e.g. DPR, Karpukhin et al. 2020), adversarial SQuAD (Jia & Liang 2017) | `refs.bib` entries verified against ACL Anthology / arXiv. Table drafted. Novelty boundaries written down (what is prior work vs. ours). |
| X3 | `significance.py`: paired bootstrap (10k resamples) and approximate randomisation on per-question EM/F1 (and per-question Recall@1 for rankers) between any two systems | Built and unit-tested on synthetic per-question scores **by M2**. The table owners (Gokul, Jun Yip, Clemen) run it on final outputs at M4 through `run_all.sh`. |
| X5 | −Negatives runs (R5, 3 seeds) | 3 run JSONs. |
| X7 | Paper integration: ACL template `main.tex`, section files, `refs.bib`, consistent notation and figure style, **page-limit and font check** (4 pages of content + 1 page of references, Times 11pt) | Template and section files set up by M2. Final compile and page/font check at M5: compiles cleanly, within limits, no overfull boxes. |

---

## 6. Run matrix (who trains what)

Every row uses the frozen hyper-parameters. All models select their checkpoint on **val**. **Test** is evaluated once per final checkpoint.

| ID | System | Encoder | Ranking loss | Negatives (`neg_mode`) | Decoding | Seeds | Runner |
|---|---|---|---|---|---|---|---|
| R1 | BM25 sentence ranker (ranking baseline only, no QA) | n/a | n/a | n/a | n/a | n/a | Jun Yip (CPU) |
| R2 | BERT span-only | bert-base-uncased | ✗ | n/a | standard | 3 | Clemen |
| R3 | DeBERTa span-only (= −ranking objective) | deberta-v3-base | ✗ | n/a | standard | 3 | Damian |
| R4 | **Full model (ours)** | deberta-v3-base | ✓ | **model-mined** | joint (tuned `λ`) | 3 | Gokul |
| R5 | −negatives (`β = 0`) | deberta-v3-base | ✓ | none | joint | 3 | Xin Hui |
| R6 | random negatives | deberta-v3-base | ✓ | random | joint | 3 | Jun Yip |
| R6b | lexical negatives (the original proposal) | deberta-v3-base | ✓ | lexical (BM25) | joint | 3 | Clemen |
| R7 | −evidence in decoding (`λ = 0` on R4 checkpoints) | deberta-v3-base | ✓ | model-mined | standard | (reuses R4) | Gokul |
| R9a | Sequential pipeline: BM25 top-k → span-only reader (R3) | deberta-v3-base | n/a | n/a | sequential (tuned `k`) | (reuses R3) | Clemen |
| R9b | Sequential: our ranker top-k → our reader (R4) | deberta-v3-base | ✓ | model-mined | sequential (tuned `k`) | (reuses R4) | Clemen |
| R8 | stretch: large, span-only and full | deberta-v3-large | ✗ / ✓ | n/a / model | standard / joint | 1 each | Damian |

All final systems (R2–R7, R9a, R9b) are also evaluated on AddSent and AddOneSent by Jun Yip. R7, R9a and R9b need no training: they only re-decode saved logits.

**Compute:** about 18 DeBERTa-base trainings. Each is roughly 1–2 GPU-hours on an A100 and several times longer on a T4. Use the SoC compute cluster where possible. Every member confirms their GPU access at kickoff. If you have no GPU, swap a run with a teammate and take an equivalent analysis task.

---

## 7. Timeline and milestones

The proposal's timeline is not binding. **Dates below assume a submission deadline of Fri 13 Nov 2026 (end of Week 13). Confirm the real deadline on Canvas and shift all dates by the same offset.**

| Milestone | Target date | Exit criteria (all must be true) |
|---|---|---|
| **M0 Kickoff & contracts** | Tue 27 Oct | Repo private, skeleton merged (X1), contracts C1–C8 agreed, GPU access confirmed per person, this plan approved. |
| **M1 End-to-end smoke test** | Sat 31 Oct | C1, C2 merged. `train.py` + `predict.py` + `run_eval.py` run end to end on the tiny model. C3/C4 merged with tests. BM25 ranker Recall@k on val computed (R1). |
| **M2 Baselines + evidence model integrated** | Wed 4 Nov | R2 and R3 (seed 42) done. D3 tuning done, so hyper-parameters are frozen. Evidence model trains end to end in all four `neg_mode`s. C8 merged; sequential decoder tested on R3 logits (R9a). Drafts of §2, §3.1–3.4, §4 written. |
| **M3 Full run matrix** | Sun 8 Nov | G3 tuning done, `λ` and `k` frozen, `final_checkpoints.yaml` frozen. R2–R7 + R6b all seeds done, run JSONs merged. Error-analysis, ranking and adversarial scripts already tested on seed-42 outputs. |
| **M4 Analysis + code freeze** | Tue 10 Nov | All tables and figures generated by script. Significance done. Negative audit and error audit done. README reproduction check passed. **No more training after this.** |
| **M5 Paper + video** | Thu 12 Nov | Full paper draft within page limit. Slides done. Each member records their segment. Video assembled. |
| **M6 Submit** | Fri 13 Nov | Final proofread by 2 people not involved in writing the section. Submission zip built and tested. Submitted. |

**Rhythm:** a 15-min sync twice a week (e.g. Mon/Thu), a short async stand-up in the group chat on other days (done / next / blocked), and a 30-min integration meeting at each milestone. If a ⏱ item will be late, say so **as soon as you know**, so the team can swap tasks.

---

## 8. Report & presentation split

### 8.1 Paper (ACL short paper: ≤ 4 pages of content + ≤ 1 page of references)
| § | Section (file in `paper/sections/`) | Owner | Page budget |
|---|---|---|---|
| — | Abstract | Damian (final pass by all) | 0.15 |
| 1 | Introduction: task, motivation, contributions list | Damian | 0.45 |
| 2 | Related Work: QA models on SQuAD, sentence selection / rank-then-read, evidence supervision, hard-negative mining, adversarial robustness | Xin Hui | 0.35 |
| 3.1 | Base span extractor | Damian | 0.25 |
| 3.2 | Evidence labels, negative candidates, lexical/random negatives (+ audit numbers) | Jun Yip | 0.3 |
| 3.3 | Sentence ranker, model-mined negatives, joint objective & joint decoding (+ method figure) | Gokul | 0.4 |
| 3.4 | Sequential (rank-then-read) decoding | Clemen | 0.15 |
| 4 | Experimental setup: data/splits (Xin Hui), training/hyper-parameters/efficiency (Damian) | Xin Hui + Damian (separate subsection files) | 0.35 |
| 5.1 | Main results vs. literature table (uses Xin Hui's X10 table) | Clemen | 0.3 |
| 5.2 | Ablations + `λ` curve + ranking quality (Recall@k, MRR) | Gokul | 0.35 |
| 5.3 | Joint vs. sequential decoding + Recall@k ceiling | Clemen | 0.2 |
| 5.4 | Robustness (AddSent/AddOneSent) | Jun Yip | 0.25 |
| 6 | Error analysis | Clemen | 0.35 |
| 7 | Conclusion + Limitations | Xin Hui | 0.15 |
| — | **Contribution statement:** what the group implemented vs. external code (HF Transformers/Datasets, official eval script, spaCy, BM25 library, pretrained checkpoints). Required by the brief | Xin Hui (each member confirms their row) | short |

Rules: write in the same tense and notation (`EM`, `F1`, `λ`, `α`, `β`, `k`), use the same table and figure style, and generate every number by script. No hand-copied numbers.

### 8.2 Video (15 min, every member speaks)
| Order | Speaker | Content | Time |
|---|---|---|---|
| 1 | Damian | Task, motivation, system overview, contributions | 2:30 |
| 2 | Xin Hui | Data and splits, baselines (BERT setup, BM25 ranker), prior-work landscape. Content is fixed early, so she can record by M3 | 2:45 |
| 3 | Jun Yip | Evidence labels, negative candidates, negative audit, ranking quality vs. BM25 | 2:45 |
| 4 | Gokul | Ranker head, model-mined negatives, joint loss and decoding, ablations | 3:00 |
| 5 | Clemen | Main results, joint vs. sequential decoding and its Recall@k ceiling, significance, error analysis, robustness, conclusion | 3:00 |
| | | Buffer | 1:00 |

One rehearsal with the full team before recording.

---

## 9. Workload balance (estimated hours)

| Member | Build | Experiments | Analysis | Writing + slides | Review/integration | **Total** |
|---|---|---|---|---|---|---|
| Clemen | 8 (C3, C9) | 6 (R2, R6b) | 11 (C8 error analysis) | 8 (§3.4, §5.1, §5.3, §6, slides/video lead) + 2 talk | 3 | **≈ 38** |
| Damian | 17 (D0–D2) | 9 (D3, R3) | 2 (D5) | 6 (§1, §3.1, §4.2, README D7) + 2 talk | 4 (most integration PRs) | **≈ 40** |
| Jun Yip | 12 (J1–J2) | 3 (R6) | 15 (J3, J5, J6) | 5 (§3.2, §5.4) + 2 talk | 3 | **≈ 40** |
| Gokul | 18 (G1–G2) | 10 (G3, G4) | 3 (G5) | 5 (§3.3, §5.2) + 2 talk | 3 | **≈ 41** |
| Xin Hui | 20 (X1–X2, X8–X9) | 3 (R5) | 10 (X3, X10 survey) | 6 (§2, §4.1, §7, X7) + 2 talk | — (reviews until M2) | **≈ 41** |

**Timing, not just totals:** Xin Hui is front-loaded on purpose (about 20 h of critical-path work due by 31 Oct). Clemen carries much of the final week (about 25 h across M4–M6: C8, C9, C7, §5.1, §6). If Clemen's final week gets too heavy, move C9 (reproducibility) to Damian, who already owns README §6 (D7). Re-check this table at M2. If someone is more than about 5 h ahead or behind, move a WP (good candidates to move: X10 survey, J6 ranking metrics, D5 efficiency, D6 stretch).

---

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Upstream ⏱ items late, which blocks everyone | Contracts C1–C8 plus mocks and the tiny model let downstream work start on Day 1. Owners of ⏱ items get review priority. |
| Not enough GPU time | Confirm access at M0. Start R2/R3 seed 42 as soon as M1 passes. Drop R8 first, then reduce R6 and R6b to 1 seed each. Never drop seeds for R3/R4/R5. |
| DeBERTa-v3 fp16 instability | Use bf16. If on a T4, use fp32 or a lower LR and check for NaNs in the first 200 steps. |
| Model-mined negatives are false negatives (the sentence also answers the question) | Candidates exclude answer-containing sentences (§2.3 step 3). Jun Yip's audit (J3) measures the remaining false-negative rate. If it is high, raise the margin warm-up or switch to mining from epoch 2. |
| Early in training, mined negatives are noise | Option: use `random` negatives for the first 10% of steps, then switch to `model`. Gokul decides on val in G3. |
| Sequential decoding loses answers through ranker errors | Expected: report the Recall@k ceiling and the ranker-error vs. reader-error split (C3). Choose `k` on val. A loss at `k = 1` is itself a finding in favour of joint decoding. |
| Full model gives no gain | Already planned for: ranking metrics, robustness and error analysis explain the result. Report honestly (§1). |
| Test-set leakage or tuning on dev | Only Clemen's `run_all.sh` evaluates `test`, and only for checkpoints listed in `configs/final_checkpoints.yaml`, which is frozen at M3. `λ` and `k` are tuned on val only. |
| Page-limit overrun | Page budgets in §8.1. Xin Hui checks at M5. Move details to fewer, denser tables, not smaller fonts. |
| Merge conflicts | Single-owner files, one JSON per run, one `.tex` per section, small PRs. |

---

## 11. Repository & submission compliance

- **Repo visibility: private (done).** Keep it private until grading is over: other students could copy a public project, and NUS plagiarism rules can hold the source group responsible too. Add the other 4 members as collaborators.
- The brief requires us to submit **source code, all external (supporting) code and data, and instructions on how to run**. This is covered by C9. The SQuAD JSON files and the official eval script are in the repo, so they go into the submission zip automatically. AddSent and checkpoints stay out of git: the submission must say exactly how to obtain them, and include them in the bundle if the size limit allows. SQuAD is CC BY-SA 4.0, so keep the attribution in `data/raw/README.md`.
- **Closed-source LLMs and APIs (ChatGPT, Gemini, …) are not allowed** in the project pipeline. All models here are open checkpoints.
- The report must state clearly **which parts are ours vs. external**. The file-ownership table (§3) and run records make this easy to write.
- The method changed from the proposal: the lexical baseline was dropped, and ranking plus sequential decoding were added, following TA feedback. Say this briefly in the report so graders don't see a mismatch with the proposal.
- Ask the lecturer early if unsure about using **Adversarial SQuAD as evaluation-only data**. The proposal said no additional *training* data. Evaluation-only use is consistent with that, but mention it in the report.
- Do not put student IDs, emails, tokens (HF/W&B) or other personal data in this repo.

---

## 12. Kickoff checklist (Day 1)
- [ ] Everyone reads this file and comments on the PR (approve or request changes)
- [ ] Confirm the deadline from Canvas and shift §7 dates
- [x] Damian: repo private, README.md, .gitignore
- [x] Damian: all 4 collaborators added
- [ ] Damian: branch protection on `main` (D0)
- [ ] Xin Hui: skeleton PR (X1: folders, requirements.txt, pytest)
- [ ] Each member: GPU access confirmed (cluster / Colab Pro / own GPU) and posted in chat
- [ ] Agree on contracts C1–C8 (15-min discussion at the kickoff meeting)
- [ ] Decide on LaTeX in the repo vs. Overleaf (if Overleaf, still keep one file per section to avoid edit clashes)
