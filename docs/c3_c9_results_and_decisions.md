# C3 / C9: results, conclusions and decisions (Clemen)

Written 11 Nov 2026. Everything here is on the **val** split (9,340 questions) unless it says otherwise. **Nothing has been scored on the test set.** Numbers are copied from command output or run records in this session; where a number is my own arithmetic (a mean over seeds) it says so.

## 1. Frozen checkpoints (`configs/final_checkpoints.yaml`, PR #64)

- 6 systems x 3 seeds (13, 42, 87) = **18 checkpoints**: R2, R3, R4, R5, R6, R6b. R7 re-decodes R4 at lambda = 0, and R9a/R9b re-decode R3/R4 outputs, so none of them has a checkpoint.
- Each entry has the SHA256 of `model.pt`. On the cluster `bash scripts/run_all.sh verify` gave **18 of 18 `ok`**.
- The hashes come from the shared Drive copies only. They have not been compared with hashes the runners (Gokul for R4, Jun Yip for R6) computed themselves.
- lambda = 0.25 (G3, `configs/evidence/best.yaml`). `k = 2` (section 3).

## 2. Validation results of the trained systems (context)

Val F1 per seed, as printed in the run records (2 dp); R2 and R6b also have means.

| System | Seeds 13 / 42 / 87 (F1) | Note |
|---|---|---|
| R2 BERT-base span-only | 83.10 / 83.42 / 83.47 | mean EM 70.15, F1 83.33 |
| R3 DeBERTa span-only | 88.36 / 88.32 / 88.48 | |
| R4 evidence, model-mined negatives | 88.35 / 88.32 / 88.32 | |
| R5 evidence, no negatives | 88.37 / 88.46 / 88.45 | |
| R6b evidence, lexical (BM25) negatives | 88.43 / 88.31 / 88.37 | mean EM 76.16 +- 0.02, F1 88.37 +- 0.06 |
| R7 R4 decoded at lambda = 0 | 88.34 / 88.33 / 88.31 | |

R6 (random negatives) is not summarised here.

**Conclusion.** R3, R4, R5, R6b and R7 are within seed noise of each other (about 88.3 to 88.5 F1, about 76.2 EM). Val F1 does not show that evidence-guided training or any negative type helps. Claims in section 5.3 and the discussion must stay modest and wait for the ranking metrics (J6) and significance tests.

**R2 vs the published BERT number.** The brief's target (about 80.8 EM / 88.5 F1) is for the SQuAD *dev* set; our R2 numbers are on our val split, which is harder for every model (R3 gets about 88.3 F1 on val against about 93.9 published for DeBERTa-v3-base on dev, so val sits about 5 F1 lower). R2 is plausible by the same offset, but that is an inference. The "within about 1 point" check can only be made on the test run.

## 3. Sequential decoding sweeps on val (the `k` decision)

Sequential decoding keeps only the spans in the top-k ranked sentences, then picks the best by start + end logit.

### 3.1 R9b candidate: R4's own ranker and reader, all three seeds

| k | Recall@k (s13 / s42 / s87) | EM (s13 / s42 / s87) | F1 (s13 / s42 / s87) |
|---|---|---|---|
| 1 | 96.43 / 96.47 / 96.23 | 75.94 / 76.00 / 75.67 | 88.14 / 88.11 / 87.98 |
| 2 | 99.31 / 99.37 / 99.29 | 76.22 / 76.27 / 76.05 | 88.37 / 88.33 / 88.31 |
| 3 | 99.72 / 99.81 / 99.72 | 76.22 / 76.27 / 76.05 | 88.37 / 88.32 / 88.32 |
| all | 100.0 / 100.0 / 100.0 | 76.22 / 76.27 / 76.05 | 88.37 / 88.32 / 88.32 |

Means over the three seeds (my arithmetic): k=1 EM 75.87 / F1 88.08; k=2 EM 76.18 / F1 88.34; k=3 EM 76.18 / F1 88.34; all EM 76.18 / F1 88.34.

**Rule (proposed before the R4 sweep, applied unchanged):** the smallest `k` whose val F1 is within 0.1 of `k = all`.
- seed 13: k=1 is 0.23 below, k=2 matches -> **2**
- seed 42: k=1 is 0.21 below, k=2 is 0.01 above -> **2**
- seed 87: k=1 is 0.34 below, k=2 is 0.01 below -> **2**

All three seeds agree, so **`k = 2`**.

### 3.2 R9a: BM25 ranking with R3's reader (seed 42 only, run locally)

| k | Recall@k | EM | F1 |
|---|---|---|---|
| 1 | 80.62 | 63.95 | 74.76 |
| 2 | 92.21 | 71.67 | 83.49 |
| 3 | 96.22 | 73.94 | 85.99 |
| all | 100.0 | 76.26 | 88.32 |

`k = all` here (76.26 EM / 88.32 F1) matches the R3 seed-42 val numbers in Gokul's `best.yaml`, which is the check that the pipeline is wired correctly. Under the same rule BM25 would need `k = all`, which is just R3 span-only.

### 3.3 For comparison: R5's ranker with R5's reader (seed 42, `k = 2` looks the same)

k=1: Recall 96.46, EM 76.06, F1 88.23. k=2: 99.31 / 76.38 / 88.48. k=3: 99.79 / 76.39 / 88.49. all: 100.0 / 76.39 / 88.49.

### 3.4 Conclusions

1. **The learned ranker finds the right sentence far more reliably than BM25.** At `k = 2`, R9a (BM25 + R3) gets 71.67 EM / 83.49 F1 and R9b (R4, seed 42) gets 76.27 EM / 88.33 F1: about 4.6 EM and 4.8 F1 higher at the same sentence budget. This is the strongest result for section 5.3. Caveat: R9a was run on seed 42 only, and both sides are single-seed numbers.
2. **Almost all remaining error is the reader's, not the ranker's.** With R4's ranker the gold sentence is in the top-1 for about 96.4% of questions, but EM is only about 76%. Cutting from `all` to `k = 2` costs essentially nothing (0 EM, +0.01 F1 on seed 42).
3. **Model-mined negatives did not improve the ranker on val.** R4's Recall@1 on seed 42 is 96.47 against 96.46 for R5 (no negatives), and at `k = 1` R4 is slightly below R5 (88.11 vs 88.23 F1). Single seed; do not read a ranking into that gap.
4. **Sequential decoding is not expected to beat joint decoding.** The reader already picks inside the right sentence nearly always, so restricting to `k` sentences can only match it.

## 4. Error-analysis dry run (not a result)

First run of `python -m qa.eval.error_analysis` on **R5 seed 42 val** outputs, regex fallback for numbers only (no spaCy, so the entity category reads 0 by construction). R5 is not the paper's full model.

- 2,207 errors out of 9,340. Categories: boundary 77.3%, other 12.1%, wrong_sentence 10.1%, numeric_date 0.5%, entity 0.0%.
- Accuracy when the top-1 sentence is right: 77.8% (n = 9,009). When it is wrong: 36.3% (n = 331).

## 5. Decisions made

| # | Decision | Status |
|---|---|---|
| 1 | Test is evaluated on **all three seeds** of all six systems (mean +- std), not one seed. Fallback if storage or time fails: seed 42 for every system. | Chosen by Clemen with the team's checkpoints on Drive; PR #64 asks the team to confirm |
| 2 | `k = 2`, same value for R9a and R9b, set from the val sweeps above (all three R4 seeds). The plan says "tuned `k`"; I read it as one shared `k` so the two pipelines are compared at the same budget. The full k = 1, 2, 3, all sweep is reported for both pipelines. | Needs Gokul's (G3) confirmation |
| 3 | Error analysis, ranking and adversarial scripts use **seed 42** (`analysis_seed`). | Proposed |
| 4 | Test run records are written to `results/runs/test/`, not `results/runs/` as contract C7 says, so val and test tables do not mix. `make_tables.py` already groups by split, so the contract location would also work. | Needs Xin Hui (C7 owner) |
| 5 | R9a and R9b test records get their own `variant` (`sequential_bm25_k<k>`, `sequential_model_k<k>`). They reuse R3/R4 records as metadata and would otherwise be pooled into R3/R4 by `make_tables.py`. | Done in `run_all.sh` (bug found and fixed) |

## 6. Not done / open

- **No test run yet.** `test`, `sequential`, `errors` and `tables` of `scripts/run_all.sh` have never run end to end (they need a GPU and PR #64's approval). Only `verify` and `data` have run on the cluster.
- **C9 leftovers:** attribution list, submission zip, reproduction check by a teammate who did not write `run_all.sh` (none named; M4 was Tue 10 Nov, M6 is Fri 13 Nov).
- **Not wired into `run_all.sh`:** ranking metrics on test (J6b), adversarial (J5), significance, efficiency (D5). Not "one command per results table" yet.
- **C8 leftovers:** context-length slice, figures from the command line, manual audit of 100 + 100 errors, 2 to 3 example cases.
- **README section 6** (Damian) is still TBD, which the reproduction check needs.
- **Unchecked:** the R4 and R6 hashes against the runners' own; the R2 reproduction against 80.8 / 88.5 (test only).
- **Sweeps use `data/processed/evidence_val.jsonl`** rebuilt on this machine and on the cluster; they agreed on sentence counts, so no mismatch is known, but the R9a sweep ran on the local copy.

## 7. Where things are

- `configs/final_checkpoints.yaml`, `scripts/run_all.sh` and the two command-line entry points: PR #64 (branch `clemen/c9-run-all`).
- Sweep outputs on the cluster: `outputs/r4_deberta_evidence_s{13,42,87}/val.sequential_sweep.json`.
- The `k = all` equality test against `decode_joint`: PR #63 (merged).
