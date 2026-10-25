# Clemen Ong: agent task brief

> **Who this is for:** Clemen's AI coding agent. Clemen is the human you work for and report to.
> **Source of truth:** [`docs/TEAM_TASKS.md`](../TEAM_TASKS.md) (plan v3). This file is Clemen's slice of it. If the two disagree, TEAM_TASKS.md wins: stop and tell Clemen, who raises it with Damian (plan owner).
> **Owner of this file:** Clemen. You (the agent) may update §9 *Progress*. Everything else here changes only when the plan changes.

## How to work from this file (every session)

1. Read this whole file. Then read TEAM_TASKS.md §2 (system spec), §3 (file ownership), §3.1 (contracts C1–C8), §4 (git workflow), and the repo [README](../../README.md).
2. Sync with `git fetch origin`. Judge what exists by `origin/main`, not by your local checkout.
3. Pick the next work package (WP) from §3: ⏱ critical-path items first, then in the order listed.
4. **Run the blocker check (§5) for that WP before writing any code.** If something it needs is missing, report the blocker in the §5.4 format and stop work on that WP. Do not build around it; the only exception is the mock-only work §5.3 allows.
5. Edit only files you own (§2), on a branch named `clemen/<topic>`.
6. A WP is finished when its "Done when" is met and `pytest -q` passes. Update §9, then tell Clemen. Don't push, open a PR, or post anywhere unless Clemen says so.

**Start every session** by giving Clemen a three-line status: (1) next unblocked WP, (2) blocked WPs and **who** is blocking each, (3) anything waiting on Clemen's own decision.

---

## 1. Role
BERT baseline, sequential (rank-then-read) decoder, error analysis, reproducibility package, and slides/video lead. Most of your work falls in M3–M6.

You are also the **test-set gatekeeper**. Only `scripts/run_all.sh` evaluates `test`, and only for checkpoints listed in `configs/final_checkpoints.yaml`, which is frozen at M3. From M2 on, you review eval- and analysis-touching PRs (Xin Hui covers this until M2).

C1, C2 and C6 moved to Xin Hui (now X8, X9, X10). They are no longer yours.

## 2. Files you own (edit only these)
| Path | For |
|---|---|
| `src/qa/decoding/sequential_decoder.py` | C3 |
| `src/qa/eval/error_analysis.py` | C8 |
| `scripts/run_all.sh`, `configs/final_checkpoints.yaml` (frozen at M3), the external-code attribution list, the submission zip | C9 |
| `configs/baselines/*.yaml` | C4 (BERT) |
| `tests/test_sequential_decoder.py`, `tests/test_error_analysis.py` | tests for your modules |
| your error-audit files in `results/audits/` (Jun Yip has separate files there) | C8 |
| `results/runs/<run_id>.json` for R2, R6b, R9a, R9b | one file per run |
| `paper/sections/` files for §3.4, §5.1, §5.3, §6 | one file per section |
| `slides/` template, your own slides, the final video | C7 |
| `docs/tasks/clemen.md` | this file |

**Not yours,** even though you use them: the R6b config in `configs/ablations/` (Gokul writes it, you run it), `src/qa/decoding/evidence_decoder.py` (Gokul), `README.md` (Damian copies your `run_all.sh` commands into §6).

## 3. Work packages (in order)
| # | Work package | Done when | Due |
|---|---|---|---|
| C7a | Slide template: one deck, consistent style | Template in `slides/` before Xin Hui records her segment | before M3 |
| C4 | BERT-base baseline: config + 3 seeds (R2) | 3 run JSONs. Mean within about 1 point of 80.8 EM / 88.5 F1. If it isn't, find out why before M3. Seed 42 by M2 | M3 |
| C3 | `sequential_decoder.py`: read C8 (any ranker) + logits, restrict spans to the top-`k` sentences, sweep `k ∈ {1, 2, 3, all}` on val. Run **R9a** (BM25 top-k → R3 reader) and **R9b** (our ranker top-k → R4). Report the **Recall@k ceiling** and split errors into ranker errors vs. reader errors | Test: `k = all` reproduces the joint decoder's output exactly. Tested on R3 logits by M2. `k` frozen on val before test (M3). Joint-vs-sequential table and ceiling figure drafted for §5.3 | M2 / M3 |
| C8 | `error_analysis.py`: mutually exclusive categories in this order: (1) wrong sentence, (2) boundary, (3) numeric/date, (4) entity, (5) other. Slices by question type, context length and answer length. Evidence-conditioned accuracy for span-only and the full model. **Manual audit** of 100 full-model errors + 100 span-only errors (the human labels these; the agent only prepares the sheet) | Error table + 1 figure + 2–3 examples for §6. Script started on seed-42 outputs at M2 and tested by M3. Final re-run and audit at M4 | M4 |
| C5 | Lexical-negative ablation, 3 seeds (R6b; config written by Gokul) | 3 run JSONs | M3 |
| C9 | Reproducibility: `run_all.sh` (one command per results table), `final_checkpoints.yaml`, attribution list, submission zip (code + external code + data instructions). **Reproduction check:** a teammate who didn't write the code reproduces one table from the README alone | `final_checkpoints.yaml` frozen at M3. Check passed at M4. Zip built and tested at M6 | M3 / M4 / M6 |
| C7b | Video assembly: stitch the 5 recordings, total under 15:00 (aim ≤ 14:30), one full-team rehearsal before recording | Final MP4 + PDF slides | M5 / M6 |

## 4. Dependencies

### Needs (check these before starting the WP)
| Your WP | Needs | From (their WP) | Mock OK? |
|---|---|---|---|
| C4 | `train.py`, `configs/base.yaml`, `span_model.py`, `predict.py` | Damian (D1, D2) | **No** (real runs) |
| C4 | C1 splits, C2 features, `run_eval.py` | Xin Hui (X8, X9, X2) | **No** |
| C3 | C8 BM25 ranking (`outputs/<run_id>/<split>.sent_rank.json`) | Jun Yip (J6, R1) | Yes in tests. R9a needs the real one |
| C3 | C8 model ranking (`sentence_ranking.py`) + R4 logits | Gokul (G2, G4) | Yes in tests. R9b needs the real ones |
| C3 | R3 saved logits (`predict.py`) | Damian (D2, D4) | Yes in tests |
| C3 | `evidence_decoder.py` (for the `k = all` equality test) and the frozen `λ` | Gokul (G2, G3) | **No** for the equality test or for freezing `k` |
| C3, C8 | C3 evidence annotations (sentence offsets, `gold_sent`) | Jun Yip (J1) | Yes in tests |
| C8 | Predictions + `nbest.json` (C6) for R3 and R4 seed 42 | Damian (D4), Gokul (G4) | Yes for script development |
| C8 | spaCy NER in `requirements.txt` | Xin Hui (X1) | Ask for it in your PR description |
| C5 | R6b config, `evidence_model.py` with `neg_mode = lexical` | Gokul (G4, G1) | **No** |
| C5 | `lex_neg` in C3/C4 | Jun Yip (J1, J2) | **No** |
| C5 | Frozen DeBERTa hyper-parameters | Damian (D3) | **No** |
| C9 | All table scripts: `make_tables.py` / `significance.py` (Xin Hui X2, X3), J5/J6 (Jun Yip), G5 (Gokul), D5 (Damian) | all | No |
| C9 | README §6 | Damian (D7) | No |
| C9 | A teammate to run the reproduction check | team decision (see §8) | — |
| C7b | Every member's slides and recording | everyone (M5) | — |

### Who is waiting on you
- **Slide template (C7a)** → everyone's slides. Xin Hui records by M3.
- **R2 (C4)** → Damian D5 (efficiency), Jun Yip J5 (robustness), your own §5.1.
- **C3 (R9a/R9b)** → Jun Yip J5 (robustness of the sequential pipelines), Damian D5 (sequential throughput).
- **R6b (C5)** → Gokul G5 (ablation table), Jun Yip J5/J6.
- **`final_checkpoints.yaml` and frozen `k` (M3)** → every final test number.
- **`run_all.sh` (C9)** → Damian D7 (README §6), and the significance runs of Gokul and Jun Yip at M4.

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
Stop and give Clemen one block per blocker, ready to paste into the group-chat stand-up:

```
🚧 BLOCKED: Clemen / <WP id> <short name>
Needs:      <exactly what: contract id, file path, run id, frozen value>
Blocked by: <member name> (<their WP id>, due <milestone>)
Evidence:   <what you checked, e.g. "not on origin/main", "PR #12 open", "tests/test_features.py fails: ...">
Meanwhile:  <what you can do instead, or "nothing on this WP">
Ask:        <one-line request to the blocking person>
```

If the blocker is a decision, `Blocked by:` is whoever owns it: Clemen for their own choices, Damian for plan changes, the whole team for contract changes. Record every blocker in §9. To request a change to someone else's file, draft the issue or PR comment for Clemen to post. Don't post it yourself.

---

## 6. Repo standards (non-negotiable)

Full text is in TEAM_TASKS.md §2–§4 and §11. Short version:

**Ownership and git**
- Edit only files you own (§2 here, TEAM_TASKS §3). To change anyone else's file, open an issue or comment on their PR. An urgent one-liner needs their 👍 in chat first, and that is Clemen's call, not yours.
- `main` is protected: PRs only, ≥ 1 approving review, squash-merge. Branches are named `clemen/<topic>`. Keep PRs under about 400 lines.
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
- Don't start GPU training or other long jobs without Clemen's go-ahead.

**Milestones** (these dates assume submission on Fri 13 Nov 2026. Confirm on Canvas; if the deadline differs, every date shifts by the same amount.)

| M0 Kickoff | M1 Smoke test | M2 Baselines + evidence model | M3 Full run matrix | M4 Analysis + code freeze | M5 Paper + video | M6 Submit |
|---|---|---|---|---|---|---|
| Tue 27 Oct | Sat 31 Oct | Wed 4 Nov | Sun 8 Nov | Tue 10 Nov | Thu 12 Nov | Fri 13 Nov |

---

## 7. Runs, paper, video
- **Runs:** R2 (BERT span-only, 3 seeds), R6b (lexical negatives, 3 seeds), R9a and R9b (re-decode saved logits, no training).
- **Paper:** §3.4 Sequential decoding (0.15 p), §5.1 Main results vs. literature (0.3 p; uses Xin Hui's X10 table), §5.3 Joint vs. sequential + Recall@k ceiling (0.2 p), §6 Error analysis (0.35 p).
- **Video:** segment 5, 3:00. Main results, joint vs. sequential and its Recall@k ceiling, significance, error analysis, robustness, conclusion. You're also the slides/video lead (C7).

## 8. Open questions (raise these; don't resolve them yourself)
- **Reproduction checker (C9).** The plan says "a teammate who did not write the code" but doesn't name one. Pick at M3 with the team.
- **Span-implied ranking.** C8 (evidence-conditioned accuracy for span-only) and Jun Yip's J6 both need "the sentence containing R3's top span". Agree with Jun Yip who computes it, so it isn't built twice.
- **Final-week load.** If M4–M6 gets too heavy, C9 can move to Damian (TEAM_TASKS §9). Raise it at the M2 workload check. Don't hand it over on your own.

## 9. Progress (agent keeps this current)
Status key: ☐ not started · 🟡 in progress / partial (mocked) · 🚧 blocked (say by whom) · ✅ merged (PR link)

| WP | Status | Notes / PR / blocker |
|---|---|---|
| C7a slide template | ☐ | |
| C4 (R2) | ☐ | |
| C3 (R9a, R9b) | 🟡 | Mocked: top-k restriction, k-sweep, Recall@k ceiling, ranker-vs-reader error split (branch `clemen/sequential-decoder`). Waiting on: `k = all` equality test needs Gokul's `evidence_decoder.py` (G2); real C6/C8 inputs (Damian D2/D4, Jun Yip J6, Gokul G2/G4); freezing `k` needs G3. |
| C8 | ☐ | |
| C5 (R6b) | ☐ | |
| C9 | ☐ | |
| C7b video | ☐ | |
