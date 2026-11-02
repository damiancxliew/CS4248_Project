# Jun Yip Chan: agent task brief

> **Who this is for:** Jun Yip's AI coding agent. Jun Yip is the human you work for and report to.
> **Source of truth:** [`docs/TEAM_TASKS.md`](../TEAM_TASKS.md) (plan v3). This file is Jun Yip's slice of it. If the two disagree, TEAM_TASKS.md wins: stop and tell Jun Yip, who raises it with Damian (plan owner).
> **Owner of this file:** Jun Yip. You (the agent) may update §9 *Progress*. Everything else here changes only when the plan changes.

## How to work from this file (every session)

1. Read this whole file. Then read TEAM_TASKS.md §2 (system spec), §3 (file ownership), §3.1 (contracts C1–C8), §4 (git workflow), and the repo [README](../../README.md).
2. Sync with `git fetch origin`. Judge what exists by `origin/main`, not by your local checkout.
3. Pick the next work package (WP) from §3: ⏱ critical-path items first, then in the order listed.
4. **Run the blocker check (§5) for that WP before writing any code.** If something it needs is missing, report the blocker in the §5.4 format and stop work on that WP. Do not build around it; the only exception is the mock-only work §5.3 allows.
5. Edit only files you own (§2), on a branch named `junyip/<topic>`.
6. A WP is finished when its "Done when" is met and `pytest -q` passes. Update §9, then tell Jun Yip. Don't push, open a PR, or post anywhere unless Jun Yip says so.

**Start every session** by giving Jun Yip a three-line status: (1) next unblocked WP, (2) blocked WPs and **who** is blocking each, (3) anything waiting on Jun Yip's own decision.

---

## 1. Role
Evidence labels, negative candidates (lexical/random), BM25 ranker baseline, sentence features, ranking metrics, and adversarial robustness.

J1 and J2 are on the critical path: Gokul's evidence model, and every run that uses it, depends on them. J5 is the main test of the project's novelty claim. Reviewer pair: Jun Yip ↔ Gokul.

## 2. Files you own (edit only these)
| Path | For |
|---|---|
| `src/qa/data/evidence_labels.py` | J1 |
| `src/qa/data/sentence_features.py` | J2 |
| `src/qa/baselines/bm25_ranker.py`, `src/qa/eval/evidence_metrics.py` | J6 |
| `src/qa/eval/adversarial.py` | J5 |
| `tests/test_evidence_labels.py`, `tests/test_sentence_features.py`, `tests/test_bm25_ranker.py`, `tests/test_evidence_metrics.py`, `tests/test_adversarial.py` | tests for your modules |
| your label/negative audit files in `results/audits/` (Clemen has separate files there) | J3 |
| `results/runs/<run_id>.json` for R1, R6 and your adversarial evaluations | one file per run |
| generated, gitignored: `data/processed/evidence_{split}.jsonl` (C3), `outputs/<R1 run>/<split>.sent_rank.json` (C8) | J1, J6 |
| `paper/sections/` files for §3.2 and §5.4 | one file per section |
| your own slides in `slides/` | Clemen owns the template |
| `docs/tasks/junyip.md` | this file |

**Not yours:** `src/qa/data/features.py` (Xin Hui; you consume C2), `src/qa/models/evidence_model.py` (Gokul; consumes your C4), the R6 config in `configs/ablations/` (Gokul writes it, you run it).

## 3. Work packages (in order)
| # | Work package | Done when | Due |
|---|---|---|---|
| J1 ⏱ | `evidence_labels.py`: sentence splitter with char offsets (spaCy or pysbd; the choice and its justification are Jun Yip's), merge splits that fall inside answers, gold sentence, **negative candidates** (exclude sentences containing the answer string), per-sentence **BM25 scores**, **lexical negative** (top-BM25 candidate), **random negative** → C3 files for train/val/test/AddSent | Statistics: sentences per context, % answers that needed merging, % sentences excluded as answer-containing, % contexts with no candidate. Test: the gold sentence span always contains the answer span | M1 |
| J2 ⏱ | `sentence_features.py`: C4 `add_sentence_fields(features, evidence, neg_mode)`. Map tokens to sentences per window, window-local `sent_label`, `neg_cand_mask`, `static_neg_label`, partially visible sentences | Tests on hand-built examples, including an answer in the overlap of two windows and a gold sentence cut by the window edge | M1 |
| J6a | `bm25_ranker.py` (R1): writes C8 from the C3 BM25 scores | BM25 Recall@k on val computed | M1 |
| J6b | `evidence_metrics.py`: **Recall@1, Recall@3, MRR** on val/test/AddSent for BM25, the span-implied ranking (sentence containing R3's top span), and our ranker (R4, R5, R6, R6b) | Script tested on seed-42 outputs by M3. Ranking table for §5.2 and §5.4. Our ranker vs. BM25 compared with Xin Hui's significance script | M4 |
| J5 | `adversarial.py`: run **every final system** (R2–R7, R6b, R9a, R9b) on AddSent and AddOneSent. EM/F1 and drop from clean dev. Also (a) how often the prediction falls inside the inserted adversarial sentence, (b) how often each ranker puts the adversarial sentence at rank 1 | Script tested on seed-42 outputs by M3. Robustness table + 1 figure for §5.4 | M4 |
| J4 | Random-negative variant, 3 seeds (R6) | 3 run JSONs | M3 |
| J3 | **Label and negative audit** (the human labels; the agent prepares the sheet and computes the counts): is the gold sentence the right evidence? Are model-mined, lexical and random negatives plausible distractors or false negatives? | Numbers for §3.2. Annotation sheet in `results/audits/` | M4 |

## 4. Dependencies

### Needs (check these before starting the WP)
| Your WP | Needs | From (their WP) | Mock OK? |
|---|---|---|---|
| J1 | Repo skeleton, `requirements.txt` with your splitter + BM25 library | Xin Hui (X1) | Ask for the packages in your PR description |
| J1 | C1 split files (`data/processed/{train,val}.json`) | Xin Hui (X8) | Partly. Develop and test on `data/raw/`, but the final train/val C3 files need C1 |
| J1 | AddSent / AddOneSent data (`scripts/download_data.sh`) | Xin Hui (X8) | Only blocks the AddSent C3 file |
| J2 | C2 base features (`features.py`) | Xin Hui (X9) | Yes. Hand-built examples in your tests are what the plan asks for. Integration needs the real `features.py` |
| J6a | J1 (your own), C1 val split | Xin Hui (X8) | — |
| J6b | C8 model rankings for R4, R5, R6, R6b | Gokul (G2, G4), Xin Hui (X5), Clemen (C5) | Yes for the metric code. Not for the table |
| J6b | R3 predictions (span-implied ranking) | Damian (D4) | Yes for the metric code |
| J6b | `significance.py` | Xin Hui (X3) | No |
| J5 | AddSent / AddOneSent download | Xin Hui (X8) | No |
| J5 | `predict.py`, `run_eval.py` | Damian (D2), Xin Hui (X2) | Yes for script development |
| J5 | Final checkpoints: R2, R6b (Clemen), R3 (Damian), R4, R7 (Gokul), R5 (Xin Hui) | each runner | No |
| J5 | Sequential pipelines R9a/R9b | Clemen (C3) | No |
| J5 | C8 model rankings on AddSent | Gokul (G2) | No |
| J4 | R6 config, `evidence_model.py` with `neg_mode = random` | Gokul (G4, G1) | **No** |
| J4 | `train.py`, frozen DeBERTa hyper-parameters | Damian (D1, D3) | **No** |
| J3 | Dump of model-mined negatives | Gokul (G1) | No. See §8, the plan is inconsistent here |

### Who is waiting on you
- **J1 (C3)** → Gokul (G1 lexical/random negatives, G2 passage sentences), Clemen (C3 Recall@k ceiling, C8 "wrong sentence" errors), Xin Hui (C3 consumer).
- **J2 (C4)** → Gokul G1/G3 and, through the evidence model, every evidence run: R4, R5 (Xin Hui), R6, R6b (Clemen).
- **J6a (R1 BM25 ranking)** → Clemen R9a pipeline.
- **J6b** → Gokul G5 (Recall@1/MRR in the ablation table).
- **J3 false-negative rate** → Gokul (may change margin warm-up or mining start, TEAM_TASKS §10).
- **J5** → §5.4 and Clemen's video segment (robustness).

---

## 5. Blockers: check, stop, report

If a WP needs something another member produces and it isn't ready, that WP is **blocked**. Report it and say **who** is blocking it. Never write their part yourself, never "temporarily" fill in their file, never guess.

### 5.1 Check before starting a WP
For every row of that WP in §4 *Needs*:

```bash
git fetch origin
git ls-tree -r --name-only origin/main -- <path>        # merged on main? (empty output = no)
gh pr list --state open --search "<filename or WP id>"   # still in review?
ls results/runs/                                         # for runs: is the run JSON there?
```

- Being on `main` is not enough. The upstream module's own tests must pass (`pytest -q tests/test_<module>.py`), and its output must match the contract in TEAM_TASKS §3.1.
- Checkpoints, logits and `outputs/` files are **not in git** (gitignored). If you need one and it isn't on this machine or in the team's shared Drive / HF Hub link, the person who ran it is the blocker.
- If it's only in an open PR, it's still a blocker. Say "in review, PR #n" in the report.

### 5.2 What counts as a blocker
- Something in §4 *Needs* (file, contract output, config, run, artefact) is missing, fails its tests, or doesn't match its contract.
- You would have to edit a file another member owns, even a one-line fix.
- A contract (C1–C8) looks wrong or incomplete for your use.
- A value that must be frozen first isn't frozen yet: D3 hyper-parameters, `α/β/m/λ`, `k`, `configs/final_checkpoints.yaml`.
- The plan is ambiguous or contradicts itself (see §8), or the choice belongs to a human: hyper-parameter picks, whether a result is good enough, moving work between people, GPU use.
- You need a new package in `requirements.txt` (Xin Hui owns it).

### 5.3 What you may still do while blocked
- Work on another of your WPs that is not blocked.
- Where §4 says *Mock OK*, build your own module against a **mock of the contract**. The mock lives only in your own test files. Never create or stub a file at another member's path. Mark the WP 🟡 *partial (mocked)* in §9. It isn't done until it passes against the real upstream, and you still report the blocker.
- Never change a contract's shape to make your side easier.

### 5.4 Report format
Stop and give Jun Yip one block per blocker, ready to paste into the group-chat stand-up:

```
🚧 BLOCKED: Jun Yip / <WP id> <short name>
Needs:      <exactly what: contract id, file path, run id, frozen value>
Blocked by: <member name> (<their WP id>, due <milestone>)
Evidence:   <what you checked, e.g. "not on origin/main", "PR #12 open", "tests/test_features.py fails: ...">
Meanwhile:  <what you can do instead, or "nothing on this WP">
Ask:        <one-line request to the blocking person>
```

If the blocker is a decision, `Blocked by:` is whoever owns it: Jun Yip for their own choices, Damian for plan changes, the whole team for contract changes. Record every blocker in §9. To request a change to someone else's file, draft the issue or PR comment for Jun Yip to post. Don't post it yourself.

---

## 6. Repo standards (non-negotiable)

Full text is in TEAM_TASKS.md §2–§4 and §11. Short version:

**Ownership and git**
- Edit only files you own (§2 here, TEAM_TASKS §3). To change anyone else's file, open an issue or comment on their PR. An urgent one-liner needs their 👍 in chat first, and that is Jun Yip's call, not yours.
- `main` is protected: PRs only, ≥ 1 approving review, squash-merge. Branches are named `junyip/<topic>`. Keep PRs under about 400 lines.
- PR checklist: `pytest -q` passes, no data or checkpoints committed, config committed for any run you report, run JSON in `results/runs/`.
- Tests for module `x` go in `tests/test_x.py`, written by the module's owner. Use a tiny model (e.g. `hf-internal-testing/tiny-random-bert`) and about 200 examples, so tests run on CPU.
- Match the style of code already on `main`. Need a new package? Ask for it in your PR description (Xin Hui owns `requirements.txt`).

**Data and evaluation validity**
- `data/raw/` is read-only. Generated data goes in `data/processed/` (gitignored).
- The train/val split is by article (C1). All tuning and model selection happens on **val**.
- `dev-v1.1.json` is the **test set**. It is used only for checkpoints listed in `configs/final_checkpoints.yaml`, and only through Clemen's `scripts/run_all.sh`. Never evaluate on it while developing.
- AddSent and AddOneSent are evaluation-only. Never train on them.
- `src/qa/eval/evaluate-v2.0.py` is the official script. Never modify it.
- All DeBERTa variants share preprocessing, the frozen D3 hyper-parameters, and seeds 13, 42 and 87 (TEAM_TASKS §2.2).

**Results**
- Every reported number traces `run_id → config → git_sha` (C7) and is produced by a script. No hand-copied numbers. Never hand-edit `results/tables/` or `results/figures/`.
- Honesty rule: report negative results as they are. Never tune on test to improve them.

**Compliance**
- No closed-source LLMs or APIs in the project pipeline (TEAM_TASKS §11). All models must be open checkpoints.
- Never commit student IDs, emails, HF/W&B tokens, `.env`, checkpoints, `outputs/`, or W&B dirs.
- Don't start GPU training or other long jobs without Jun Yip's go-ahead.

**Milestones** (these dates assume submission on Fri 13 Nov 2026. Confirm on Canvas; if the deadline differs, every date shifts by the same amount.)

| M0 Kickoff | M1 Smoke test | M2 Baselines + evidence model | M3 Full run matrix | M4 Analysis + code freeze | M5 Paper + video | M6 Submit |
|---|---|---|---|---|---|---|
| Tue 27 Oct | Sat 31 Oct | Wed 4 Nov | Sun 8 Nov | Tue 10 Nov | Thu 12 Nov | Fri 13 Nov |

---

## 7. Runs, paper, video
- **Runs:** R1 (BM25 ranker, CPU), R6 (random negatives, 3 seeds), and AddSent/AddOneSent evaluation of every final system.
- **Paper:** §3.2 Evidence labels, negative candidates, lexical/random negatives + audit numbers (0.3 p); §5.4 Robustness (0.25 p).
- **Video:** segment 3, 2:45. Evidence labels, negative candidates, negative audit, ranking quality vs. BM25.

## 8. Open questions (raise these; don't resolve them yourself)
- **J3 vs. G1 mismatch.** J3 audits "100 random **train** questions", but G1 dumps model-mined negatives for "200 **val** questions". So there are no model-mined negatives for the train questions J3 audits. Don't pick one yourself: ask Damian (plan owner) and Gokul to fix the plan.
- **Sentence splitter.** spaCy vs. pysbd is Jun Yip's call and needs a justification in §3.2. The agent may compare them on data and recommend, but Jun Yip decides.
- **Span-implied ranking.** Clemen's C8 also needs it. Agree with Clemen who computes it, so it isn't built twice.

## 9. Progress (agent keeps this current)
Status key: ☐ not started · 🟡 in progress / partial (mocked) · 🚧 blocked (say by whom) · ✅ merged (PR link)

| WP | Status | Notes / PR / blocker |
|---|---|---|
| J1 | ☐ | |
| J2 | 🟡 | Partial (mocked). PR #47 (in review, Gokul). `sentence_features.py` + 18 tests on branch `junyip/sentence-features`. A sentence counts as visible if any token of it is; tokens go to the sentence they overlap most (DeBERTa offsets include the leading space). 🚧 Real-C2 check blocked by Xin Hui X9 (`features.py`, not started on GitHub). Slot-to-passage column for G2 awaiting agreement with Gokul and Damian |
| J6a (R1) | ☐ | |
| J6b | ☐ | |
| J5 | ☐ | |
| J4 (R6) | ☐ | |
| J3 | ☐ | |
