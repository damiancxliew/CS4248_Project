# Ravichandran Gokul: agent task brief

> **Who this is for:** Gokul's AI coding agent. Gokul is the human you work for and report to.
> **Source of truth:** [`docs/TEAM_TASKS.md`](../TEAM_TASKS.md) (plan v3). This file is Gokul's slice of it. If the two disagree, TEAM_TASKS.md wins: stop and tell Gokul, who raises it with Damian (plan owner).
> **Owner of this file:** Gokul. You (the agent) may update §9 *Progress*. Everything else here changes only when the plan changes.

## How to work from this file (every session)

1. Read this whole file. Then read TEAM_TASKS.md §2 (system spec), §3 (file ownership), §3.1 (contracts C1–C8), §4 (git workflow), and the repo [README](../../README.md).
2. Sync with `git fetch origin`. Judge what exists by `origin/main`, not by your local checkout.
3. Pick the next work package (WP) from §3: ⏱ critical-path items first, then in the order listed.
4. **Run the blocker check (§5) for that WP before writing any code.** If something it needs is missing, report the blocker in the §5.4 format and stop work on that WP. Do not build around it; the only exception is the mock-only work §5.3 allows.
5. Edit only files you own (§2), on a branch named `gokul/<topic>`.
6. A WP is finished when its "Done when" is met and `pytest -q` passes. Update §9, then tell Gokul. Don't push, open a PR, or post anywhere unless Gokul says so.

**Start every session** by giving Gokul a three-line status: (1) next unblocked WP, (2) blocked WPs and **who** is blocking each, (3) anything waiting on Gokul's own decision.

---

## 1. Role
Sentence-ranker head, joint loss with model-mined negatives, passage-level ranking, joint decoding, and ablations. This is the core of the project's method.

G1 and G2 are on the critical path. You also write the ablation configs for **everyone's** runs (R4–R7, R6b), so all variants use identical settings, and three teammates can't train until those configs are merged. Reviewer pair: Jun Yip ↔ Gokul.

## 2. Files you own (edit only these)
| Path | For |
|---|---|
| `src/qa/models/evidence_model.py` | G1 |
| `src/qa/decoding/sentence_ranking.py`, `src/qa/decoding/evidence_decoder.py` | G2 |
| `configs/evidence/*.yaml` (incl. `best.yaml`) | G3 |
| `configs/ablations/*.yaml` (every member's ablation run config) | G4 |
| `tests/test_evidence_model.py`, `tests/test_sentence_ranking.py`, `tests/test_evidence_decoder.py` | tests for your modules |
| `results/runs/<run_id>.json` for R4, R7 and G3 tuning runs | one file per run |
| generated, gitignored: `outputs/<run_id>/<split>.sent_rank.json` (C8) for your model runs | G2 |
| `paper/sections/` files for §3.3 (with the method figure) and §5.2 | one file per section |
| your own slides in `slides/` | Clemen owns the template |
| `docs/tasks/gokul.md` | this file |

**Not yours,** even though you plug into them: `src/qa/train.py` (Damian; register your model through its registry/config key, don't edit it), `src/qa/models/span_model.py` (Damian; wrap it, don't edit it), `src/qa/data/sentence_features.py` (Jun Yip; you consume C4).

## 3. Work packages (in order)
| # | Work package | Done when | Due |
|---|---|---|---|
| G1 ⏱ | `evidence_model.py`: wraps the span model and adds the sentence-ranker head, listwise `L_rank` and margin `L_neg`, with `neg_mode ∈ {model, lexical, random, none}`. `model` mode mines the top-scoring candidate online from `neg_cand_mask`. Registered so `train.py` loads it from config. Provides the mined-negative export for Jun Yip's audit (J3); the export itself runs once after training (G3/R4) | Unit test on synthetic `sent_ids` (allowed before J2 lands, using mocks). Each `neg_mode` tested. Loss decreases on a 200-example overfit test. Gradients reach both heads. Trains end to end in all four modes by M2 | M2 |
| G2 ⏱ | `sentence_ranking.py` (C8: window log-softmax → passage-level max) + `evidence_decoder.py` (joint decoding) | Test: a sentence's passage score equals its best window score. `λ = 0` gives output identical to Damian's `span_decoder.py`. C8 merged by M2 | M2 |
| G3 | Tune on val (seed 42, D3 hyper-parameters frozen, `neg_mode = model`): `α ∈ {0.1, 0.3, 1.0}`, `β ∈ {0.5, 1.0}`, `m ∈ {0.5, 1.0}`. Then a `λ` sweep {0, 0.25, 0.5, 1, 2} **at inference only**. Final picks are Gokul's decision | Chosen values in `configs/evidence/best.yaml`. Sweep curve saved for §5.2. `λ` frozen | M3 |
| G4 | Write **all** ablation configs (R4–R7, R6b) with identical settings. Run the full model, 3 seeds (R4), and compute R7 (`λ = 0`) from the R4 checkpoints. After training, run the mined-negative export once on the finished model for J3's 100 train questions | Configs merged **before M3** (others are waiting on them). 3 run JSONs each for R4 and R7 | M3 |
| G5 | Ablation table and analysis: full vs. −ranking (R3) vs. −negatives (R5) vs. random (R6) vs. lexical (R6b) vs. −evidence in decoding (R7). Mean ± std, Recall@1/MRR from J6, significance stars from Xin Hui's script | Table drafted for §5.2 with a 3–4 sentence interpretation, including whether model-mined beats lexical | M4 |

## 4. Dependencies

### Needs (check these before starting the WP)
| Your WP | Needs | From (their WP) | Mock OK? |
|---|---|---|---|
| G1 | `span_model.py` to wrap | Damian (D2) | Yes in unit tests. Final integration needs the real one |
| G1 | `train.py` model registry / config key | Damian (D1) | Yes for unit tests. Registration needs the real one. If the registry can't load your model, that's a blocker on Damian. Don't edit `train.py` |
| G1 | C4 sentence fields (`add_sentence_fields`) | Jun Yip (J2) | **Yes**, the plan explicitly allows synthetic `sent_ids` |
| G1 | C2 features (for the 200-example overfit test) | Xin Hui (X9) | Yes until X9 lands, but the overfit test must run on real features before G1 counts as done |
| G2 | `span_decoder.py` (λ = 0 equality test), C5 logits from `predict.py` | Damian (D2) | **No** for the equality test |
| G2 | C3 sentence offsets, C4 | Jun Yip (J1, J2) | Yes in tests |
| G3 | Frozen DeBERTa hyper-parameters | Damian (D3) | **No** |
| G3 | Real C1/C2, C3/C4 merged, `train.py`, GPU | Xin Hui (X8, X9), Jun Yip (J1, J2), Damian (D1) | **No** |
| G4 | G3 values, D3 values | own, Damian (D3) | **No** |
| G5 | R3 runs | Damian (D4) | No |
| G5 | R5 runs | Xin Hui (X5) | No |
| G5 | R6 runs, J6 ranking metrics | Jun Yip (J4, J6) | No |
| G5 | R6b runs | Clemen (C5) | No |
| G5 | `significance.py`, `make_tables.py` | Xin Hui (X3, X2) | No |

### Who is waiting on you
- **G1** → Jun Yip J3 (the export tool; the export itself runs once after G3/R4 training) and every evidence run: R5 (Xin Hui), R6 (Jun Yip), R6b (Clemen).
- **G2 (C8 rankings)** → Clemen C3 (R9b) and C8 (error analysis), Jun Yip J5/J6, Xin Hui (C8 consumer).
- **G2 (`evidence_decoder.py`)** → Clemen C3 (`k = all` must equal joint decoding).
- **G3 (`λ` frozen)** → Clemen C3 (joint score inside the top-k), and your own G4.
- **G4 configs** → Xin Hui (R5), Jun Yip (R6), Clemen (R6b). They can't train until these are merged.
- **R4** → Clemen (R9b, error analysis), Jun Yip (J5, J6), Damian (D5 efficiency).

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
Stop and give Gokul one block per blocker, ready to paste into the group-chat stand-up:

```
🚧 BLOCKED: Gokul / <WP id> <short name>
Needs:      <exactly what: contract id, file path, run id, frozen value>
Blocked by: <member name> (<their WP id>, due <milestone>)
Evidence:   <what you checked, e.g. "not on origin/main", "PR #12 open", "tests/test_features.py fails: ...">
Meanwhile:  <what you can do instead, or "nothing on this WP">
Ask:        <one-line request to the blocking person>
```

If the blocker is a decision, `Blocked by:` is whoever owns it: Gokul for their own choices, Damian for plan changes, the whole team for contract changes. Record every blocker in §9. To request a change to someone else's file, draft the issue or PR comment for Gokul to post. Don't post it yourself.

---

## 6. Repo standards (non-negotiable)

Full text is in TEAM_TASKS.md §2–§4 and §11. Short version:

**Ownership and git**
- Edit only files you own (§2 here, TEAM_TASKS §3). To change anyone else's file, open an issue or comment on their PR. An urgent one-liner needs their 👍 in chat first, and that is Gokul's call, not yours.
- `main` is protected: PRs only, ≥ 1 approving review, squash-merge. Branches are named `gokul/<topic>`. Keep PRs under about 400 lines.
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
- Don't start GPU training or other long jobs without Gokul's go-ahead.

**Milestones** (these dates assume submission on Fri 13 Nov 2026. Confirm on Canvas; if the deadline differs, every date shifts by the same amount.)

| M0 Kickoff | M1 Smoke test | M2 Baselines + evidence model | M3 Full run matrix | M4 Analysis + code freeze | M5 Paper + video | M6 Submit |
|---|---|---|---|---|---|---|
| Tue 27 Oct | Sat 31 Oct | Wed 4 Nov | Sun 8 Nov | Tue 10 Nov | Thu 12 Nov | Fri 13 Nov |

---

## 7. Runs, paper, video
- **Runs:** G3 tuning runs (seed 42), R4 (full model, 3 seeds), R7 (re-decode R4 with `λ = 0`, no training).
- **Paper:** §3.3 Sentence ranker, model-mined negatives, joint objective and decoding, with the method figure (0.4 p); §5.2 Ablations, `λ` curve, ranking quality (0.35 p).
- **Video:** segment 4, 3:00. Ranker head, model-mined negatives, joint loss and decoding, ablations.

## 8. Open questions (raise these; don't resolve them yourself)
- ~~**G1 dump vs. J3 audit mismatch.**~~ Resolved 3 Nov (Gokul with Damian and Jun Yip): the export and the audit use the same 100 train questions. See §9.
- **Negative warm-up.** Using `random` negatives for the first 10% of steps, then switching to `model`, is an option Gokul decides on val in G3 (TEAM_TASKS §10). The agent can run the comparison only with Gokul's go-ahead.
- **High false-negative rate from J3.** If it's high, the fix (margin warm-up, or mining from epoch 2) is a decision for Gokul and the team, not the agent.

## 9. Progress (agent keeps this current)
Status key: ☐ not started · 🟡 in progress / partial (mocked) · 🚧 blocked (say by whom) · ✅ merged (PR link)

| WP | Status | Notes / PR / blocker |
|---|---|---|
| G1 | ✅ merged | [PR #44](https://github.com/damiancxliew/CS4248_Project/pull/44), merged into `main` 3 Nov. Uses #52's `prepare_features(…, split, data_file)` to load `<evidence_dir>/evidence_{split}.jsonl` and apply J2's `add_sentence_fields`. Adds the mined-negative dump (`python -m qa.models.evidence_model dump-negatives`). `pytest -q`: 192 passed. **Real-data check (3 Nov; `python tests/test_evidence_model.py --overfit-real`)**: real C1/C2/C3/C4 for 200 train questions and real `deberta-v3-base`, MPS fp32. `model` mode over 8 epochs: loss 7.22 → 0.50, F1 on the same questions 13.9 → 96.5. `lexical` / `random` / `none` each train 1 epoch without errors. Test export on that checkpoint, 100 train questions: 99 have a mined negative, 0 invalid; it equals the lexical negative on 54 and the random one on 32. G1 is done as is (no 200-question export needed). Left: plan wording with Damian (below) |
| G2 | ✅ merged | [PR #51](https://github.com/damiancxliew/CS4248_Project/pull/51), merged into `main` 3 Nov (closes #24). C8 passage ranking (slots via J2's `token_sentences`) + joint decoder + CLI. λ = 0 equals `span_decoder.decode`. `main` (with G1) merged in: 203 passed. **Real-data check (3 Nov)**: on the G1 overfit model's real logits (200 questions), λ = 0 gives exactly `predict.py`'s predictions and n-best, and C8 is complete and sorted (Recall@1 0.985 on those training questions). On all 9,400 real val windows (deberta-v3-base tokenizer, real C3/C4; 50 questions span 2–4 windows), G2's slots match the model's `sent_mask` and J2's `sent_label` / `static_neg_label` / `neg_cand_mask` with 0 mismatches, and the passage score is the best window log-prob on all 50 split-window questions. |
| G3 | ✅ merged | [PR #54](https://github.com/damiancxliew/CS4248_Project/pull/54), merged into `main` 4 Nov. The λ figure for §5.2 is `g3_grid.sh curve` (G5 row). **Frozen in `configs/evidence/best.yaml` (Gokul confirmed, 4 Nov): α 0.3, β 0.5, m 1.0, λ 0.25 → val F1 88.40 / EM 76.38, sentence Recall@1 0.965** (span-only R3 s42 on the same GPU: 88.32 / 76.26). 12 runs on the SoC cluster (about 25 min each, 4 GPU-h), all on `h100-47` (MIG 3g.47gb); 60 records in `results/runs/g3_grid/`. All 12 lie within 0.34 F1 (about seed/GPU noise); 6 within the 0.2 tie band, where the best-F1 run also has the top Recall@1. λ barely matters: the best λ gains at most 0.04 F1 over λ = 0. A test checks `best.yaml` against the records. Records' `git_sha` is `1e0bad9-dirty`: only `results/tables/data_stats.csv` (rewritten by `squad_io prepare`) differed on the cluster. Branch `gokul/g3-grid`: `configs/evidence/evidence.yaml` + `g3_grid.sh` (train → predict val → joint decode at λ ∈ {0, 0.25, 0.5, 1, 2} → one C7 record per (run, λ) in `results/runs/g3_grid/`) + `g3_slurm.sbatch` (array 0–11, at most 3 at a time, all on `h100-47`). Each run deletes its model once scored, to save cluster disk (`KEEP_CHECKPOINT=1` keeps it); its val logits stay. Pick rule (Gokul, 3 Nov): best joint val F1; runs within 0.2 F1 of it are tied and the highest Recall@1 wins. Negative warm-up skipped. CPU dry run with a tiny model: records valid, λ = 0 F1 equals `run.json` |
| G4 (configs, R4, R7) | ✅ merged | [PR #55](https://github.com/damiancxliew/CS4248_Project/pull/55), merged into `main` 4 Nov. **Handoff (9 Nov):** `r4-outputs.tar` (R4 + R7 outputs without `best/`, 318 MB) and `r4-checkpoints.tar` (R4 `best/`, 2.2 GB) on Damian's Drive folder; both checked byte for byte against the cluster copies. The three `model.pt` hashes equal the R4 entries in `configs/final_checkpoints.yaml`. **R4, 3 seeds (4 Nov): val F1 88.33 ± 0.02 / EM 76.17 ± 0.12. R7 (the same models at λ 0): 88.33 ± 0.01 / 76.17 ± 0.11.** R3 span-only: 88.39 ± 0.08 / 76.27 ± 0.07. Records in `results/runs/r4_deberta_evidence_s{13,42,87}.json` and `r7_deberta_evidence_s{13,42,87}.json`. All are on `MIG 3g.47gb` with one clean `git_sha` (`034f742`), so `make_tables.py` gives one 3-seed row each. Each R7 F1 equals its R4 `run.json` F1. About 25 min per seed (0.33 GPU-h). Same-seed noise: R4 s42 has G3's pick config, seed and GPU type, with no `src/` change, yet scored 88.33 against G3's 88.40, so reruns of one seed vary by about 0.1 F1. Checkpoints are kept on the cluster at `outputs/r4_deberta_evidence_s{13,42,87}/best` (0.74 GB each). **J3 export** (R4 s42, 100 train questions, seed 2026) is in `outputs/r4_deberta_evidence_s42/mined_negatives_train.jsonl`. 97 questions have a mined negative (3 have none), and none is the gold sentence. It equals the lexical negative on 34 and the random one on 31. Branch `gokul/g4-ablations` (4 Nov, from `gokul/g3-grid`). **Configs** in `configs/ablations/`: `r4_full.yaml` (G3 picks, `decode.lam` 0.25), `r5_no_negatives.yaml` (`none`, β 0), `r6_random_negatives.yaml`, `r6b_lexical_negatives.yaml`, `r7_no_evidence_decoding.yaml` (λ 0, `reuses` R4's logits). A test checks each differs from R4 only in what it ablates. **Runner** `run_ablation.sh` (train → predict val → decode at the config's λ → `results/runs/<run_id>_s<seed>.json`; R7 re-decodes R4) + `ablation_slurm.sbatch` (seeds 13/42/87 as array 0–2, `h100-47`, MIG check, checkpoints kept). Not `train.py --record`: that scores at λ = 0, which is R7's number. CPU dry run with a tiny model: records valid, R7 F1 equals R4's `run.json` per seed, `make_tables.py` gives one row per variant |
| G5 | 🟡 table, figure and §5.2/§3.3 drafts done; method figure left | Branch `gokul/g5-ablation-table` (11 Nov). **`run_ablation.sh table`**: R4, R3, R5, R6, R6b, R7 and BM25 (R1); EM/F1 mean ± sample std over 3 seeds from the records (equal to `make_tables.py`), Recall@1/MRR from J6's `ranking_val.csv`, seed-42 tests against R4: per-question official F1 through `qa.eval.significance` (X3), Recall@1 from J6's `ranking_val_significance.json`; star at p < 0.01 (two-sided randomisation). Each seed-42 answers file must re-score exactly to its record's F1, or the table stops. Writes `results/tables/ablations_val.{csv,tex}`, `ablations_val_macros.tex` (the numbers §5.2 quotes) and `ablations_val_significance.json` (file sha256s, paths relative to `OUTPUT_DIR`). **`g3_grid.sh curve`**: `results/figures/lambda_curve.{pdf,png}` and `lambda_curve_macros.tex`. **Result (val):** no ablation moves seed-42 F1 by more than 0.14 (smallest p 0.17; R6b vs R4 −0.02, p 0.85), Recall@1 is flat across negative types (R6b vs R4 +0.01, p 1.0), R4 beats BM25 by 15.86 Recall@1 (p < 0.001) but span-implied R3 ranks as well (+0.11, p 0.47), and λ = 0 (R7) changes 9 of 9,340 answers (best λ gains at most 0.04 F1). So model-mined negatives don't beat lexical ones. Inputs: R4/R7 from Gokul's handoff, R5 (`r5-handoff.tar`), R6 (`r6_checkpoints.zip`), R3 top-1 of Damian's `val.nbest.json` (re-scores exactly). **R6b:** Clemen's handoff has only `best/`, so its seed-42 val answers were re-made on `h100-47` (MIG 3g.47gb) from his checkpoint (hash equal to `final_checkpoints.yaml`) with `run_ablation.sh`'s predict and decode steps (11 Nov). They re-score exactly to his record: F1 88.3081, EM 76.1670. Paper: §5.2 and §3.3 drafted; the method figure is still to do. 3 new tests (table rows/stars/pending, refusal of mismatched answers, curve); the 4 shell-runner tests skip on Windows. `pytest -q`: 289 passed |

**Resolved:**
- `prepare_features` gets the split: Damian's #52, merged 2 Nov.
- Slot-to-passage mapping: no C4 change. G2 reuses `token_sentences`; Jun Yip confirmed window-local slots in #47.
- J3 question set (3 Nov, Gokul with Damian and Jun Yip): use the same 100 train questions for the export and the audit, for a fair comparison. No 200-question val export. During training, `model` mode picks its tricky wrong sentences on its own; nobody picks them by hand. After training (G3/R4), the export runs once on the finished model for 100 train questions, and Jun Yip judges those picks alongside his lexical and random ones (J3). The export saves its ids (seed 2026); Jun Yip can pass his own list with `--qids`.
- Damian's #43 questions: agreed. Epoch selection uses span-only decoding (λ = 0), and `sent_idx` is the sentence holding the span's start.
- Clemen's sequential decoder (#40, #63): merged with the `k = all` equality test against `decode_joint`.
- J6 ranking definition: `evidence_metrics.py` (#65) uses the same rank of `gold_sent` as `g3_grid.sh summary` (checked when reviewing #65, 9 Nov).
- J3 export: audited by Jun Yip (#66, 10 Nov).
- C9 R4 checkpoints: on Drive (9 Nov); Clemen's `verify` passes 18/18 and the R4 hashes match Gokul's.
- λ figure for §5.2: covered by `g3_grid.sh curve` instead of Xin Hui's `make_figs.py`.
- R6b seed-42 val answers (11 Nov): Clemen asked Gokul to re-make them; done on `h100-47`, exact match with his record (G5 row).
- Significance form (Gokul, 11 Nov): keep the original plan (TEAM_TASKS X3): `significance.py`'s paired bootstrap (10k resamples, 95% interval) and approximate randomisation on per-question scores. The G5 stars use the two-sided randomisation p < 0.01, as J5 and J6 do, not the lecture's one-sided bootstrap p.

**To raise (need an answer from the named person):**
- Damian (plan wording, the last G1 item):
  - TEAM_TASKS §5 G1 row: replace "dumps the mined negatives for 200 val questions" with the 3 Nov agreement (export once after training, on J3's 100 train questions).
  - TEAM_TASKS §2.3 step 6: `L_neg` is also skipped when the gold sentence isn't in the window (`sent_label = -100`), since there is no `s_gold`.
- Damian (cluster, 4 Nov): `--gpus=h100-47` on `xgpi15` gave a whole H100 NVL (torch: `NVIDIA H100 NVL`, 94 GB), not R3's `MIG 3g.47gb`, though Slurm lists the node as `h100-47:4`. `g3_slurm.sbatch` now excludes it and stops any task that doesn't see the MIG slice. Worth knowing for R4+ seeds, and maybe worth reporting to SoC.
- Xin Hui (X7, page check):
  - §5.2 is about 0.5 pages against its 0.35 budget (the full-width table plus the λ figure take about 0.3). The options are dropping the figure (the text already quotes its numbers) or another section giving up space.
  - §3.3 is at its 0.40 budget before the method figure. A TikZ figure needs `\usepackage{tikz}` in `main.tex` (Xin Hui's file).
  - Gokul's local TinyTeX lacks `caption`, `microtype` and `grfext`, so the page check needs Overleaf or a full TeX Live.
- Jun Yip (reviewer pair): review the G5 PR (run `bash configs/ablations/run_ablation.sh table` and `bash configs/evidence/g3_grid.sh curve`).
- Damian (D5): R4's val numbers and checkpoints are in the G4 row, for the efficiency table.
- Everyone (before G5): on val, R4 is 0.06 F1 below R3 and R7 equals R4. Neither the ranker nor joint decoding moves clean-SQuAD F1 beyond seed noise (about 0.1 F1). The method's case then rests on AddSent (J5) and ranking quality (J6). Agree how §5 frames this.
- Damian (#51): G2 reuses `span_decoder._window_candidates` so both decoders share one candidate set. Can it be treated as shared (or made public)? The λ = 0 test fails if they drift.
