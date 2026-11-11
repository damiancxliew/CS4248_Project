# C3 / C9: results and decisions (Clemen)

11 Nov 2026. All numbers are on the **val** split (9,340 questions). **Nothing has been scored on test.** Means over seeds are my own arithmetic.

## Decision: `k = 2` for the sequential pipelines (R9a and R9b)

`k = 2` is the frozen sentence budget, set from val only and written in `configs/final_checkpoints.yaml`. It is used for both R9a (BM25 + R3) and R9b (R4's ranker + reader). The full k = 1, 2, 3, all sweep is still reported for both pipelines. Decided by Clemen as the C3 owner; the team can object in the PR review, and Gokul (G3) is informed because R4 is his model.

**Rule, fixed before the R4 sweep:** the smallest `k` whose val F1 is within 0.1 of `k = all`.

**Evidence (R4 ranker + reader, val, one column per seed 13 / 42 / 87; full tables in section 3.1):**

| | k = 1 | k = 2 | k = 3 | all |
|---|---|---|---|---|
| F1 vs `all` | -0.23 / -0.21 / -0.34 | 0.00 / +0.01 / -0.01 | 0.00 / 0.00 / 0.00 | 0 |
| EM | 75.94 / 76.00 / 75.67 | 76.22 / 76.27 / 76.05 | 76.22 / 76.27 / 76.05 | 76.22 / 76.27 / 76.05 |
| Recall@k | 96.43 / 96.47 / 96.23 | 99.31 / 99.37 / 99.29 | 99.72 / 99.81 / 99.72 | 100 |

- **Why not `k = 1`:** it is 0.21 to 0.34 F1 below `all` on every seed, outside the 0.1 tolerance, and it loses 0.27 to 0.38 EM.
- **Why 2 and not larger:** at `k = 2` EM equals `all` on all three seeds and F1 is within 0.01. `k = 3` and `all` add nothing to EM or F1, only about 0.4 points of Recall@k. So 2 is the smallest budget that costs nothing.
- **Why it is not a seed accident:** all three seeds pick 2 independently. The seed means are k=1 EM 75.87 / F1 88.08 and k=2 EM 76.18 / F1 88.34, the same as `all`.

**Caveats:**
- The plan says "tuned `k`" for each pipeline. I read it as one shared `k` so R9a and R9b are compared at the same budget. Under the same rule BM25 alone would need `k = all`, which is plain R3 span-only (section 3.2); at `k = 2` R9a gets 71.67 EM / 83.49 F1 against 76.27 / 88.33 for R9b (both seed 42).
- The rule is my proposal. A different rule could pick 3, but the F1 differences between 2 and 3 are 0.01 or less.
- If the team objects, change `decode.k` in the yaml **before** the test run, not after.

## 1. Frozen checkpoints (`configs/final_checkpoints.yaml`, PR #64)

- 6 systems x 3 seeds (13, 42, 87) = 18 checkpoints: R2, R3, R4, R5, R6, R6b. R7, R9a and R9b re-decode existing outputs, so they have none.
- `bash scripts/run_all.sh verify`: 18/18 `ok` on the cluster.
- Hashes come from the shared Drive copies only; not yet compared with the runners' own (Gokul for R4, Jun Yip for R6).
- lambda = 0.25 (G3, `configs/evidence/best.yaml`); `k = 2` (section 3).

## 2. Trained systems, val EM and F1 (seeds 13 / 42 / 87)

| System | EM (s13 / s42 / s87) | F1 (s13 / s42 / s87) | Mean EM, F1 (± std) |
|---|---|---|---|
| R2 BERT-base span-only | 69.99 / 70.21 / 70.24 | 83.10 / 83.42 / 83.47 | 70.15 ± 0.14, 83.33 ± 0.20 |
| R3 DeBERTa span-only | 76.21 / 76.26 / 76.35 | 88.36 / 88.32 / 88.48 | 76.27 ± 0.07, 88.39 ± 0.08 |
| R4 evidence, model-mined negatives | 76.13 / 76.30 / 76.07 | 88.35 / 88.32 / 88.32 | 76.17 ± 0.12, 88.33 ± 0.02 |
| R5 evidence, no negatives | 76.09 / 76.37 / 76.34 | 88.37 / 88.46 / 88.45 | 76.27 ± 0.15, 88.43 ± 0.05 |
| R6 evidence, random negatives | 76.34 / 76.23 / 76.27 | 88.50 / 88.34 / 88.39 | 76.28 ± 0.05, 88.41 ± 0.08 |
| R6b evidence, BM25 negatives | 76.18 / 76.17 / 76.13 | 88.43 / 88.31 / 88.37 | 76.16 ± 0.02, 88.37 ± 0.06 |
| R7 R4 at lambda = 0 | 76.13 / 76.28 / 76.08 | 88.34 / 88.33 / 88.31 | 76.17 ± 0.11, 88.33 ± 0.01 |

All from the committed val run records in `results/runs/` (sample standard deviation over three seeds).

**Conclusion.** R3 to R7 are within seed noise (about 88.3 to 88.5 F1, about 76.2 EM). Val F1 does not show that evidence-guided training or any negative type helps. Keep section 5.3 claims modest until the ranking metrics (J6) and significance tests are in.

**R2 vs published BERT (80.8 EM / 88.5 F1).** That target is for SQuAD dev. Our val split is harder (R3 gets about 88.3 F1 vs about 93.9 published on dev, about 5 lower), so R2 looks plausible by the same offset. This is an inference; the real check is the test run.

## 3. Sequential decoding sweeps (the `k` decision)

Keep only spans in the top-k ranked sentences, then pick the best by start + end logit.

### 3.1 R9b: R4's ranker and reader, three seeds

| k | Recall@k | EM | F1 |
|---|---|---|---|
| 1 | 96.43 / 96.47 / 96.23 | 75.94 / 76.00 / 75.67 | 88.14 / 88.11 / 87.98 |
| 2 | 99.31 / 99.37 / 99.29 | 76.22 / 76.27 / 76.05 | 88.37 / 88.33 / 88.31 |
| 3 | 99.72 / 99.81 / 99.72 | 76.22 / 76.27 / 76.05 | 88.37 / 88.32 / 88.32 |
| all | 100 / 100 / 100 | 76.22 / 76.27 / 76.05 | 88.37 / 88.32 / 88.32 |

Seed means: k=1 EM 75.87 / F1 88.08; k=2, 3, all EM 76.18 / F1 88.34.

**Rule (set before the sweep):** smallest `k` whose val F1 is within 0.1 of `k = all`. k=1 is 0.21 to 0.34 below on every seed; k=2 is within 0.01. **`k = 2`** on all three seeds.

### 3.2 R9a: BM25 ranking + R3 reader (seed 42, run locally)

| k | Recall@k | EM | F1 |
|---|---|---|---|
| 1 | 80.62 | 63.95 | 74.76 |
| 2 | 92.21 | 71.67 | 83.49 |
| 3 | 96.22 | 73.94 | 85.99 |
| all | 100 | 76.26 | 88.32 |

`k = all` matches R3 seed 42 in `best.yaml`, so the pipeline is wired correctly. Under the same rule BM25 needs `k = all`, i.e. plain R3.

### 3.3 R5 ranker + reader (seed 42)

k=1: 96.46 / 76.06 / 88.23. k=2: 99.31 / 76.38 / 88.48. k=3: 99.79 / 76.39 / 88.49. all: 100 / 76.39 / 88.49 (Recall / EM / F1).

### 3.4 Conclusions

1. **The learned ranker beats BM25.** At `k = 2`, R9b (seed 42) gets 76.27 EM / 88.33 F1 vs 71.67 / 83.49 for R9a, about +4.6 EM and +4.8 F1. Strongest result for section 5.3. Caveat: both are single-seed.
2. **Remaining error is the reader's.** R4's gold sentence is top-1 for about 96.4% of questions, yet EM is about 76%. Cutting to `k = 2` costs nothing.
3. **Model-mined negatives did not help the ranker.** Recall@1 is 96.47 (R4) vs 96.46 (R5); at k=1, F1 is 88.11 vs 88.23. Single seed, so no ranking implied.
4. **Sequential decoding should not beat joint decoding.** The reader already picks inside the right sentence nearly always.

## 4. Error-analysis dry run (not a result)

`python -m qa.eval.error_analysis` on R5 seed 42 val, regex fallback for numbers (no spaCy, so entity = 0 by construction).

- 2,207 errors of 9,340: boundary 77.3%, other 12.1%, wrong_sentence 10.1%, numeric_date 0.5%, entity 0.0%.
- Accuracy when top-1 sentence is right: 77.8% (n = 9,009); wrong: 36.3% (n = 331).

## 5. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Test on **all three seeds** of all six systems (mean ± std). Fallback: seed 42 only. | PR #64 asks team to confirm |
| 2 | One shared `k = 2` for R9a and R9b, from the R4 val sweeps. Full k = 1, 2, 3, all sweep reported for both. | Decided (C3 owner); open to objection in the PR review before the test run |
| 3 | Error analysis, ranking and adversarial scripts use seed 42 (`analysis_seed`). | Proposed |
| 4 | Test records go in `results/runs/test/`, not `results/runs/` (contract C7), so val and test don't mix. The C7 location would also work since `make_tables.py` groups by split. | Needs Xin Hui (C7 owner) |
| 5 | R9a/R9b test records get their own `variant` (`sequential_bm25_k<k>`, `sequential_model_k<k>`); otherwise `make_tables.py` pools them into R3/R4. | Done in `run_all.sh` |

## 6. Open items

- **No test run yet.** `test`, `sequential`, `errors`, `tables` in `run_all.sh` have never run end to end (need GPU and PR #64 approval). Only `verify` and `data` have run.
- **C9:** attribution list, submission zip, reproduction check by someone other than the `run_all.sh` author (none named; M6 is Fri 13 Nov).
- **Not in `run_all.sh`:** test ranking metrics (J6b), adversarial (J5), significance, efficiency (D5).
- **C8:** context-length slice, command-line figures, manual audit of 100 + 100 errors, 2 to 3 example cases.
- **README section 6** (Damian) is TBD; the reproduction check needs it.
- **Unchecked:** R4 and R6 hashes vs the runners'; R2 vs 80.8 / 88.5 (test only); R9a sweep used the local `evidence_val.jsonl` (sentence counts matched the cluster's).

## 7. Where things are

- `configs/final_checkpoints.yaml`, `scripts/run_all.sh`, entry points: PR #64 (`clemen/c9-run-all`).
- Sweep outputs (cluster): `outputs/r4_deberta_evidence_s{13,42,87}/val.sequential_sweep.json`.
- `k = all` vs `decode_joint` equality test: PR #63 (merged).
