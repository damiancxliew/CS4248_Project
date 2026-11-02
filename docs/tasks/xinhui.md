# Wong Xin Hui: agent task brief

> **Who this is for:** Xin Hui's AI coding agent. Xin Hui is the human you work for and report to.
> **Source of truth:** [`docs/TEAM_TASKS.md`](../TEAM_TASKS.md) (plan v3). This file is Xin Hui's slice of it. If the two disagree, TEAM_TASKS.md wins: stop and tell Xin Hui, who raises it with Damian (plan owner).
> **Owner of this file:** Xin Hui. You (the agent) may update §9 *Progress*. Everything else here changes only when the plan changes.

## How to work from this file (every session)

1. Read this whole file. Then read TEAM_TASKS.md §2 (system spec), §3 (file ownership), §3.1 (contracts C1–C8), §4 (git workflow), and the repo [README](../../README.md).
2. Sync with `git fetch origin`. Judge what exists by `origin/main`, not by your local checkout.
3. Pick the next work package (WP) from §3: ⏱ critical-path items first, then in the order listed.
4. **Run the blocker check (§5) for that WP before writing any code.** If something it needs is missing, report the blocker in the §5.4 format and stop work on that WP. Do not build around it; the only exception is the mock-only work §5.3 allows.
5. Edit only files you own (§2), on a branch named `xinhui/<topic>`.
6. A WP is finished when its "Done when" is met and `pytest -q` passes. Update §9, then tell Xin Hui. Don't push, open a PR, or post anywhere unless Xin Hui says so.

**Start every session** by giving Xin Hui a three-line status: (1) next unblocked WP, (2) blocked WPs and **who** is blocking each, (3) anything waiting on Xin Hui's own decision.

---

## 1. Role
Data and features pipeline, evaluation harness, significance tests, related-work survey, and paper integration.

Xin Hui's work is **front-loaded on purpose**: almost everything is due by M2, because the final weeks are busy. X1, X2, X8 and X9 are the first links of the critical path. Nearly everyone else is blocked until they land, so they come before anything else. X4 and X6 moved to Clemen (now C8, C9) and are no longer Xin Hui's.

Reviewer pair: Xin Hui ↔ Damian (data/features ↔ training engine). Until M2, Xin Hui also reviews eval-touching PRs.

## 2. Files you own (edit only these)
| Path | For |
|---|---|
| `requirements.txt` (others ask for additions in their PR descriptions), pytest setup, optional GitHub Actions for CPU smoke tests, the folder layout | X1 |
| `src/qa/data/squad_io.py`, `scripts/download_data.sh`, generated split files in `data/processed/` (gitignored) | X8 |
| `src/qa/data/features.py` | X9 |
| `src/qa/eval/run_eval.py`, `scripts/make_tables.py`, `scripts/make_figs.py`, everything in `results/tables/` and `results/figures/` (generated only) | X2 |
| `src/qa/eval/significance.py` | X3 |
| `src/qa/eval/evaluate-v2.0.py`: the official script, already committed. **Never modify it** | — |
| `tests/test_squad_io.py`, `tests/test_features.py`, `tests/test_run_eval.py`, `tests/test_significance.py`, `tests/test_make_tables.py` | tests for your modules |
| `results/runs/<run_id>.json` for R5 | X5 |
| `paper/main.tex`, `acl.sty`, `refs.bib` (others add their bib entries in their own PRs, sorted by key) | X7 |
| `paper/sections/` files for §2, §4.1, §7, and the contribution statement | X10, X7 |
| your own slides in `slides/` | Clemen owns the template |
| `docs/tasks/xinhui.md` | this file |

**X1 scope:** create the folder layout from TEAM_TASKS §3 (directories, empty `__init__.py`), but never write content into modules other members own.

## 3. Work packages (in order)
| # | Work package | Done when | Due |
|---|---|---|---|
| X1 ⏱ | Repo skeleton: folder layout, `requirements.txt` with pinned versions (include `numpy`, which `evaluate-v2.0.py` imports), `pytest` setup, optional GitHub Actions for CPU smoke tests | Fresh clone + `pip install -r requirements.txt` + `pytest` passes | M0 |
| X8 ⏱ | `squad_io.py`: load SQuAD from `data/raw/`, article-level train/val split (seed 2026), dataset statistics (articles, contexts, questions, context-length distribution, answer length, question types). `scripts/download_data.sh`: fetch AddSent/AddOneSent and run the SHA-256 check on the SQuAD files | Split files (C1) exist and are reproducible. Stats table drafted for §4.1. Test: no title is in both train and val | M1 |
| X9 ⏱ | `features.py` (C2): tokenisation with sliding windows and offset mapping, start/end labels, out-of-window answers → CLS | Test: decoding gold labels back through offsets recovers the gold answer text for ≥ 99.5% of train questions (report the rest and why). Works for BERT and DeBERTa tokenisers | M1 |
| X2 ⏱ | `run_eval.py`: wraps the **unmodified** official script and writes the C7 JSON. `make_tables.py`: `results/runs/*.json` → mean ± std tables (CSV + LaTeX) | A dummy prediction file (e.g. first-sentence answers) evaluates end to end, and the output matches the official script exactly | M1 |
| X3 | `significance.py`: paired bootstrap (10k resamples) + approximate randomisation on per-question EM/F1 (and per-question Recall@1 for rankers) between any two systems | Built and unit-tested on synthetic per-question scores | M2 |
| X10 | Literature survey + comparison table: published SQuAD v1.1 dev EM/F1 (LR baseline, BiDAF, BERT-base/large, SpanBERT, XLNet/RoBERTa, DeBERTa/DeBERTaV3) and evidence-related prior work (Min et al. 2018, HotpotQA, DPR, Jia & Liang 2017) | `refs.bib` entries verified against ACL Anthology / arXiv (don't make up citations: every entry must be checked at the source). Table drafted. Prior work vs. our novelty written down | M2 |
| X7 | Paper integration: ACL template `main.tex`, section files, `refs.bib`, consistent notation and figure style, page-limit and font check (4 pages of content + 1 page of references, Times 11pt) | Template and section files set up by M2. Final compile and page/font check at M5: compiles cleanly, within limits, no overfull boxes | M2 / M5 |
| X5 | −Negatives runs, 3 seeds (R5) | 3 run JSONs | M3 |

## 4. Dependencies

### Needs (check these before starting the WP)
| Your WP | Needs | From (their WP) | Mock OK? |
|---|---|---|---|
| X1, X3, X10 | nothing | — | — |
| X8 | X1 | own | — |
| X9 | X1, and X8's train split for the ≥ 99.5% check | own | Develop on `data/raw/` |
| X2 | C6 prediction format | Damian (D2) | **Yes**, the plan says to use a dummy prediction file |
| X2 (real tables) | C7 run JSONs from every runner | everyone | Build and test on synthetic run JSONs |
| X7 (final, M5) | Every member's section drafts; each member confirms their row of the contribution statement | everyone | — |
| X5 | R5 config, `evidence_model.py` with `neg_mode = none` | Gokul (G4, G1) | **No** |
| X5 | C3/C4 | Jun Yip (J1, J2) | **No** |
| X5 | `train.py`, frozen DeBERTa hyper-parameters | Damian (D1, D3) | **No** |

### Who is waiting on you
- **X1** → everyone (folders, `requirements.txt`, pytest). Damian D1/D2 can't merge without it.
- **X8 (C1 splits)** → everyone. Directly: Jun Yip J1 (train/val evidence files), Damian D3, Clemen C4. **AddSent download** → Jun Yip J1/J5.
- **X9 (C2 features)** → Damian D1/D2, Jun Yip J2, Gokul G1.
- **X2 (`run_eval.py`)** → Damian D1 (val F1) and every run's C7 record. **`make_tables.py`** → every results table.
- **X3** → Gokul G5, Jun Yip J6, Clemen's `run_all.sh`.
- **X10** → Clemen §5.1 (literature comparison table).
- **X7 template** → everyone's paper sections.
- **R5 (X5)** → Gokul G5, Jun Yip J5/J6.

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
Stop and give Xin Hui one block per blocker, ready to paste into the group-chat stand-up:

```
🚧 BLOCKED: Xin Hui / <WP id> <short name>
Needs:      <exactly what: contract id, file path, run id, frozen value>
Blocked by: <member name> (<their WP id>, due <milestone>)
Evidence:   <what you checked, e.g. "not on origin/main", "PR #12 open", "tests/test_features.py fails: ...">
Meanwhile:  <what you can do instead, or "nothing on this WP">
Ask:        <one-line request to the blocking person>
```

If the blocker is a decision, `Blocked by:` is whoever owns it: Xin Hui for their own choices, Damian for plan changes, the whole team for contract changes. Record every blocker in §9. To request a change to someone else's file, draft the issue or PR comment for Xin Hui to post. Don't post it yourself.

---

## 6. Repo standards (non-negotiable)

Full text is in TEAM_TASKS.md §2–§4 and §11. Short version:

**Ownership and git**
- Edit only files you own (§2 here, TEAM_TASKS §3). To change anyone else's file, open an issue or comment on their PR. An urgent one-liner needs their 👍 in chat first, and that is Xin Hui's call, not yours.
- `main` is protected: PRs only, ≥ 1 approving review, squash-merge. Branches are named `xinhui/<topic>`. Keep PRs under about 400 lines.
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
- Don't start GPU training or other long jobs without Xin Hui's go-ahead.

**Milestones** (these dates assume submission on Fri 13 Nov 2026. Confirm on Canvas; if the deadline differs, every date shifts by the same amount.)

| M0 Kickoff | M1 Smoke test | M2 Baselines + evidence model | M3 Full run matrix | M4 Analysis + code freeze | M5 Paper + video | M6 Submit |
|---|---|---|---|---|---|---|
| Tue 27 Oct | Sat 31 Oct | Wed 4 Nov | Sun 8 Nov | Tue 10 Nov | Thu 12 Nov | Fri 13 Nov |

---

## 7. Runs, paper, video
- **Runs:** R5 (−negatives, 3 seeds).
- **Paper:** §2 Related Work (0.35 p), §4.1 Data and splits (your subsection file within §4), §7 Conclusion + Limitations (0.15 p), the contribution statement (ours vs. external code; each member confirms their row), and integration of the whole paper (X7).
- **Video:** segment 2, 2:45. Data and splits, baselines (BERT setup, BM25 ranker), prior-work landscape. Content is fixed early, so record **by M3**.

## 8. Open questions (raise these; don't resolve them yourself)
- **LaTeX in the repo vs. Overleaf.** This is a kickoff decision (TEAM_TASKS §12) and it changes how X7 is set up. Don't build the template until it's decided.
- **Late-phase load.** R5 (M3) and the final paper check (M5) fall in the busy weeks. If Xin Hui can't cover them, raise it early so the team can swap (TEAM_TASKS §7 rhythm, §9).
- **C7 writer overlap.** X2 (`run_eval.py`) and Damian's D1 (`train.py`) both "write the C7 run JSON". Agree with Damian which one writes the final record.
- **AddSent source.** HF `squad_adversarial` or the original CodaLab release. This is Xin Hui's choice; the agent can recommend one.

## 9. Progress (agent keeps this current)
Status key: ☐ not started · 🟡 in progress / partial (mocked) · 🚧 blocked (say by whom) · ✅ merged (PR link)

| WP | Status | Notes / PR / blocker |
|---|---|---|
| X1 | ✅ | Merged in [PR #41](https://github.com/damiancxliew/CS4248_Project/pull/41). Damian resolved the run-record ignore rule in [PR #42](https://github.com/damiancxliew/CS4248_Project/pull/42), also merged. Optional remote CI was removed at Xin Hui's request; local CPU tests remain. |
| X8 | ✅ | [PR #45](https://github.com/damiancxliew/CS4248_Project/pull/45) merged on 2 Nov 2026 at `7c4b84e`. Fetched and pulled into `xinhui/token-features`; original X8 commit `1aaa891` is in main. Reproducible split: 78,259 train / 9,340 val questions, seed 2026. |
| X9 | 🟡 | Implementation and acceptance verified at main `7c4b84e`: 61 CPU tests passed; full 78,259-question recovery BERT 99.7329%, DeBERTa 99.7240% (both ≥99.5%). Xin Hui authorised committing and pushing on new branch `xinhui/x9-features` for review. Not merged. |
| X2 | ☐ | |
| X3 | ☐ | |
| X10 | ☐ | |
| X7 | ☐ | |
| X5 (R5) | ☐ | |

### X1 verification and blocker (25 Oct 2026)

- Fetched `origin`; branch started from `origin/main` at `62a466953c525376b1dd6bddb9a02f3cdf9f1eb1`.
- Created the layout from TEAM_TASKS §3, with `.gitkeep` files for otherwise-empty tracked directories. `data/processed/` and the local `.venv/` remain ignored.
- Added [requirements.txt](../../requirements.txt), [pytest.ini](../../pytest.ini), and [CPU smoke tests](../../tests/test_environment.py). Tests use synthetic inputs and locally initialised tiny BERT/DeBERTa models; no checkpoint downloads, GPU jobs, or test-set evaluation.
- Installation exposed an Arrow/NumPy runtime incompatibility despite a clean `pip check`: resolved PyArrow 26 required NumPy 2, while NumPy is pinned to 1.26.4. Explicitly pinning PyArrow 19.0.1 fixed it; all 5 tests passed in 52.36 seconds. A separate fresh-clone reproduction check has not been run.
- GitHub Actions could not start because GitHub reported an account billing/spending-limit block. Xin Hui requested removal of the optional workflow; local tests remain available with `python -m pytest -q` in the project environment. Remote CI is not required for X1 and can be added later if the team chooses.
- Raw SQuAD files and the official evaluator match `origin/main`; other members' module contents were not changed.

```text
RESOLVED (26 Oct 2026, merged PR #42): Xin Hui / X1 tracked run-record directory
Needs:      The planned results/runs/ directory must be trackable normally.
Blocked by: Damian (owner of .gitignore; repository setup, M0)
Evidence:   git check-ignore -v results/runs/.gitkeep identifies .gitignore:14:runs/.
            This rule also ignores future results/runs/<run_id>.json records.
Meanwhile:  Local setup is implemented and its offline CPU tests pass.
Ask:        Anchor runs/ to /runs/, or explicitly allow results/runs/ and its files.
```

Draft for Xin Hui to send to Damian: “The `runs/` rule in `.gitignore` also ignores our required `results/runs/` records. Could you anchor it to `/runs/` (or explicitly unignore `results/runs/` and its contents)? `git check-ignore -v results/runs/.gitkeep` reproduces it. I have left your file unchanged.”

### X8 implementation and verification (26 Oct 2026)

- Xin Hui explicitly asked to start X8 before X1 merged. Created `xinhui/data-pipeline` from the tested X1 commit `8adc0db` while PR #41 was in review. After PRs #41/#43/#42 merged, fast-forwarded to `origin/main` at `e8a037c`, preserving all uncommitted X8 changes. The upstream X1 dependency is now satisfied.
- Added [squad_io.py](../../src/qa/data/squad_io.py), [download_data.sh](../../scripts/download_data.sh), and [test_squad_io.py](../../tests/test_squad_io.py). No other member's files changed.
- C1: `data/processed/train.json`, `val.json`, and `split_titles.json` are generated and ignored. Membership uses sorted unique article titles, seed 2026, and a rounded 10% title holdout; repeated titles stay together and original article/question order and all references are preserved. The manifest includes the input SHA-256.
- Real split: train **398 articles / 16,915 contexts / 78,259 questions**; val **44 articles / 1,981 contexts / 9,340 questions**. Zero overlapping titles or question IDs. All 87,599 source questions, contexts, and answer annotations are preserved. Regeneration produces identical bytes for all split/statistics outputs. Both raw-file SHA-256 values match the committed manifest.
- Generated [data_stats.csv](../../results/tables/data_stats.csv), [data_stats.tex](../../results/tables/data_stats.tex), and ignored `data/processed/dataset_stats.json`. The JSON contains context/answer word-length distributions and question-type counts. Word lengths use whitespace splitting; answer statistics use the first reference per question. No answer text/offset mismatches were found. Official dev was only hashed, not parsed for statistics or evaluated.
- Damian's `data.examples_fn` can be set to `qa.data.squad_io:load_examples`. It returns an HF Dataset with `id`, `title`, `context`, `question`, and `answers` containing parallel `text` and `answer_start` lists. A one-epoch, synthetic CPU test through PR #43's real training hook passed; the X9 features hook remains mocked. No training on real SQuAD data occurred.
- Initial `pytest -q`: **23 passed**, with one upstream Datasets deprecation warning. Git Bash syntax check and `bash scripts/download_data.sh --check-only` passed. Downloader tests cover checksum failure, preserving previous files on failure, HF revision pinning, and SQuAD-format conversion using mocked responses. Live adversarial-data verification was completed on 27 Oct (see below).
- Prepare on PowerShell from the repository root: `$env:PYTHONPATH='src'; .venv/Scripts/python.exe -m qa.data.squad_io prepare`. Check raw files with `bash scripts/download_data.sh --check-only` (Git Bash on Windows). The script automatically uses the local virtual environment, or `QA_PYTHON` if supplied.
- The source is never chosen automatically. Xin Hui's final selection is HF. Reproduce with `bash scripts/download_data.sh --source hf --revision 0eb158aad5fd6cf738033016948d2fd7d30bed8a`. Outputs are ignored `data/processed/addsent.json`, `addonesent.json`, and `adversarial_sources.json`. CodaLab support remains available with known upstream SHA-256 checks; HF uses an explicit data-only Parquet revision. These files remain separate from the training split.
- Post-merge compatibility check against main `e8a037c`: **47 tests passed** (3 upstream Datasets deprecation warnings). Added a permanent test in `test_squad_io.py` exercising the real merged `build_split` import/call path with the X8 loader and mock X9 features. No X8 production-code changes were needed. `data/processed/` remains ignored; `results/runs/` is now trackable under Damian's merged `/runs/` fix. The existing skeleton placeholder `results/runs/.gitkeep` is consequently visible as an untracked file. No production files owned by Damian were edited.

```text
RESOLVED (27 Oct 2026): Xin Hui / X8 live CodaLab download verification
Needs:      A working download endpoint for the selected original CodaLab release.
Blocked by: External CodaLab service (expired HTTPS certificate); source choice is resolved.
Evidence:   Live download failed with CERTIFICATE_VERIFY_FAILED: certificate has expired.
            Independent curl check returned SEC_E_CERT_EXPIRED; HTTP redirects to the same HTTPS URL.
Meanwhile:  C1 splits, dataset statistics, loader integration, and checksum verification are complete locally.
Resolution: Xin Hui authorised HF instead; downloaded and validated the pinned data-only mirror.
```

- Xin Hui selected CodaLab on 27 Oct. Ran `bash scripts/download_data.sh --source codalab` with network access; the first bundle request failed during TLS verification, before any adversarial files were published. The existing raw/train/validation data was not changed. No source switch or certificate-verification bypass was made.
- Xin Hui then explicitly authorised HF. Its main revision `878c103ae1e50c1874f51df47321c70624713161` contains a legacy Python loader that delegates back to CodaLab. Updated the HF path to read data-only Parquet URLs at the verified `refs/convert/parquet` commit `0eb158aad5fd6cf738033016948d2fd7d30bed8a`, using the built-in Parquet loader. No remote dataset code is executed.
- Live verification: **3,560 AddSent** and **1,787 AddOneSent** questions; zero answer-text/offset mismatches across all references; both files load with `load_examples`; saved output hashes match provenance. Generated JSON hashes: AddSent `079ad1b19c43b31325b99ddd45d2d6e5b92e4611dbd6a96b008daa4df2992526`, AddOneSent `59663216a9ed6de4d6bb2c9a3a661e34839bb926039d1e637ac9ebcb70f55101`. These are hashes of converted project JSON, not the original CodaLab byte hashes. Raw SQuAD checksums remain unchanged and all downloaded files are ignored by Git. No model evaluation or training on adversarial data occurred.
- After the HF loader change, the full local suite passed again: **47 passed**, with 3 upstream Datasets deprecation warnings. Editor diagnostics and `git diff --check` also passed.

### X9 implementation and dependency check (27 Oct 2026)

- Fetched origin before changes. Main is `e8a037c`, containing merged X1 and Damian's training engine (PR #43). PR #45 is open, unmerged and unapproved. The initial working tree on `xinhui/data-pipeline` was clean; its X8 commit `1aaa891` and ignored processed data are preserved. Created `xinhui/token-features` from `origin/main` so X9 does not duplicate X8 changes.
- Added [features.py](../../src/qa/data/features.py) and [test_features.py](../../tests/test_features.py). Hook: `qa.data.features:build_features(examples, tokenizer, max_seq_len=384, doc_stride=128)` returns an HF Dataset with C2 columns, masked non-context offsets and `sequence_ids`. Questions remain sequence 0 and contexts sequence 1 with either padding side. Each missing/partial-window answer targets that window's actual CLS position. First reference supplies the labels; invalid text/offset annotations fail explicitly.
- SentencePiece can include leading whitespace in token offsets. Context-token edges are trimmed against the original context, independently of gold answers; whitespace-only offsets are masked. True intra-token boundary mismatches remain visible instead of changing offsets to match annotations.
- Offline tests use real `BertTokenizerFast` and `DebertaV2TokenizerFast` classes, with a tiny local WordPiece vocabulary and locally trained SentencePiece model. Cover Unicode, exact decoder recovery, overlap and window-edge cases, partial/absent answers, left padding, empty datasets/contexts, deterministic first-reference selection, invalid inputs, default lengths, and sample mapping across the 256-example batch boundary. The real merged `build_split` uses X9 with Damian's existing test-only C1 loader mock; no other member's file was edited.
- Initial focused run: 16 passed / 2 failed, identifying SentencePiece leading-space recovery; fixed with answer-independent whitespace trimming. Final `.venv/Scripts/python.exe -m pytest -q`: **42 passed in 45.92 seconds**, with 61 upstream Datasets `co_lnotab` deprecation warnings. Editor diagnostics and `git diff --check` pass. No pretrained checkpoint was downloaded, and these offline fixtures do not establish full-data acceptance for either production tokenizer.
- Full ≥99.5% exact recovery across the 78,259 X8 training questions remains pending until PR #45 merges. Then run both production tokenizers (`bert-base-uncased`, `microsoft/deberta-v3-base`) at 384/128 and report unrecovered IDs and causes (including intra-token boundaries, normalization/unknown tokens, whitespace annotations, and spans not fully covered by any window). Do not claim this acceptance check has passed from synthetic tests.
- No raw dev evaluation, GPU training, long data jobs, commits, pushes, PR creation or messages. Only the two X9 files and this brief's §9 changed; processed datasets remain ignored. The earlier 47-test result belongs to X8's branch; this branch starts from main without X8's tests.

```text
RESOLVED (2 Nov 2026, merged PR #45): Xin Hui / X9 X8 merge dependency
Needs:      Merged X8 C1 train split and qa.data.squad_io:load_examples.
Blocked by: X8 review/merge (Xin Hui's PR #45; Damian is the reviewer pair).
Evidence:   Fetched origin/main e8a037c has no squad_io.py; PR #45 is open/unmerged.
Meanwhile:  X9 implementation and offline CPU tests proceed under the raw-data exception.
Ask:        Review and merge PR #45 before the full training-split acceptance check.
```

### X9 post-X8 readiness check (2 Nov 2026)

- GitHub confirms PR #45 merged on 2 Nov. Fetched origin and pulled with `--ff-only --autostash` into the existing `xinhui/token-features` branch; HEAD and origin/main both equal `7c4b84eb5a1a254ff2ee0460900190e34fa69785`. Preserved both untracked X9 files. Resolved the progress-note autostash conflict by keeping all merged X8 history and the previous X9 notes. The autostash and ignored `.venv/x9-progress-before-pull.md` backup remain available. All changes are unstaged.
- Replaced the X9 test's mock C1 loader with the real merged `qa.data.squad_io:load_examples`; the actual `build_split` integration passes for both fast-tokenizer families. Full local CPU suite: **61 passed in 414.65 seconds**, with 61 upstream Datasets deprecation warnings. The earlier 42-test result preceded X8's merge.
- Added an explicit, opt-in audit entry point to [test_features.py](../../tests/test_features.py), separate from ordinary pytest. It reads only the processed training split, verifies seed/title membership and the 78,259-question count, uses 384/128 windows and the first gold reference, and records exact recovery per question across windows. The ignored JSON report includes unrecovered IDs/text/cause categories, counts, tokenizer revision, input/code hashes and base Git SHA. Tokenizer downloads and reports stay in ignored `data/processed/`; no model weights are needed.
- Full-audit command in PowerShell: `$env:PYTHONPATH='src'; .venv/Scripts/python.exe tests/test_features.py --audit`. Add `--limit 200` for a bounded pilot; a pilot does not satisfy the full ≥99.5% criterion. Xin Hui's earlier instruction requires approval for long jobs, so the full audit remains pending that approval. No commit, push, PR, GPU job or raw-dev evaluation was performed.
- Ran the bounded 200-question pilot with the actual production fast tokenizers: **BERT 200/200 exact (100%)**, **DeBERTa 200/200 exact (100%)**, with 200 windows each and no failures. Ignored report: `data/processed/x9_recovery_pilot.json`. Verified cached revisions are BERT `86b5e0934494bd15c9632b12f734a8a67f723594` and DeBERTa `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`; the initial pilot's report revision fields were null because Transformers did not expose `_commit_hash`, so the audit now reads the actual cached refs for future reports. HF emitted its expected DeBERTa fast-tokenizer byte-fallback warning; full-split coverage is still unverified. Editor diagnostics and `git diff --check` pass. Only the X9 files and §9 differ from current main; all generated data and tokenizer files remain ignored.

### X9 full-training acceptance verified (2 Nov 2026)

- Xin Hui ran the full audit in her terminal. Checked the resulting ignored `data/processed/x9_recovery.json`: `full_split=true`, all **78,259 training questions**, first gold reference, **384/128** windows, both production fast tokenizers. No final-test evaluation or model training was involved.
- **BERT:** 78,050/78,259 exact (**99.732938%**), 79,106 windows, **209** failures. Revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
- **DeBERTa:** 78,043/78,259 exact (**99.723993%**), 78,916 windows, **216** failures. Revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- Both exceed the **≥99.5%** acceptance bar. Every failure has a labelled recovered span containing the gold as a strict substring, consistent with annotation boundaries inside tokens (e.g. `six` → `sixth`, `evolution` → `evolutionary`, `7` → `1817`). The audit records all unrecovered IDs, gold text, character start, recovered text and cause category. No failures were classified as missing complete labelled windows or annotation-edge whitespace. Counts and unique failure IDs are consistent for both models; offsets were not adjusted using gold boundaries to force recovery.
- Verified report provenance against current files: base SHA `7c4b84eb5a1a254ff2ee0460900190e34fa69785`; feature-code SHA-256 `8b12bdb15f8248919c71535718d82efefe8510e0f7090fa23b603b5a44df8b06`; processed-train SHA-256 `3e8de5d6b89354de903801548d8310d66684c0b621ee8cd3956a3234d852f42f`. Both hashes match exactly. The existing 61-test CPU suite result still applies; production code has not changed since it passed. Current editor diagnostics and `git diff --check` are clean.
- **Ready to commit for review.** Commit scope is only [features.py](../../src/qa/data/features.py), [test_features.py](../../tests/test_features.py), and this brief's §9. Data, tokenizer cache and detailed audit reports remain ignored. No commit, push or PR creation was performed; X9 stays 🟡 until merged.
- Xin Hui subsequently authorised committing and pushing the verified X9 changes into a new branch, `xinhui/x9-features`, preserving `xinhui/token-features`. This authorisation does not include opening a PR or merging. The commit contains only the three reviewed files above; generated data and local audit reports are excluded.
