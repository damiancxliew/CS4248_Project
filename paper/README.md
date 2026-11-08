# Paper workspace (X7)

Start with [main.tex](main.tex), which assembles the paper in the order required
by section 8.1 of [TEAM_TASKS.md](../docs/TEAM_TASKS.md).
The section files are placeholders, not completed report content.

## Files and ownership

| Section | File | Owner | Page budget |
|---|---|---|---|
| Abstract | [00_abstract.tex](sections/00_abstract.tex) | Damian; final pass by all | 0.15 |
| 1 Introduction | [01_introduction.tex](sections/01_introduction.tex) | Damian | 0.45 |
| 2 Related Work | [02_related_work.tex](sections/02_related_work.tex) | Xin Hui | 0.35 |
| 3.1 Base span extractor | [03_1_base_span_extractor.tex](sections/03_1_base_span_extractor.tex) | Damian | 0.25 |
| 3.2 Evidence labels | [03_2_evidence_labels.tex](sections/03_2_evidence_labels.tex) | Jun Yip | 0.30 |
| 3.3 Ranker and joint decoding | [03_3_sentence_ranker.tex](sections/03_3_sentence_ranker.tex) | Gokul | 0.40 |
| 3.4 Sequential decoding | [03_4_sequential_decoding.tex](sections/03_4_sequential_decoding.tex) | Clemen | 0.15 |
| 4.1 Data and splits | [04_1_data_splits.tex](sections/04_1_data_splits.tex) | Xin Hui | Shared 0.35 |
| 4.2 Training setup | [04_2_training_setup.tex](sections/04_2_training_setup.tex) | Damian | Shared 0.35 |
| 5.1 Main results | [05_1_main_results.tex](sections/05_1_main_results.tex) | Clemen; X10 literature table from Xin Hui | 0.30 |
| 5.2 Ablations | [05_2_ablations.tex](sections/05_2_ablations.tex) | Gokul | 0.35 |
| 5.3 Decoder comparison | [05_3_joint_vs_sequential.tex](sections/05_3_joint_vs_sequential.tex) | Clemen | 0.20 |
| 5.4 Robustness | [05_4_robustness.tex](sections/05_4_robustness.tex) | Jun Yip | 0.25 |
| 6 Error analysis | [06_error_analysis.tex](sections/06_error_analysis.tex) | Clemen | 0.35 |
| 7 Conclusion and limitations | [07_conclusion_limitations.tex](sections/07_conclusion_limitations.tex) | Xin Hui | 0.15 |
| Contribution statement | [08_contributions.tex](sections/08_contributions.tex) | Xin Hui; each member confirms | Short |

Xin Hui maintains the shared preamble, section order, and bibliography. Parent
headings for sections 3, 4, and 5 live in the main document; subsection files
already contain their own headings. Replace the italic draft placeholders with
your writing. Do not add another document class or document environment.

## Compile

Use pdfLaTeX, as recommended by the supplied ACL template. From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This produces `main.pdf`. For Overleaf, upload the files in this directory
(excluding the reference clone), select `main.tex` as the main document and
pdfLaTeX as the compiler. Keep the section files separate in either workflow.

[refs.bib](refs.bib) deliberately starts without entries. X10 must verify each
reference at its original source. Once entries and citations are present,
uncomment `\bibliography{refs}` in the main document and build with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or run `pdflatex`, `bibtex main`, then `pdflatex` twice. Cite with
`\citet{key}` or `\citep{key}`; keep bibliography entries sorted by key.

## Formatting and results

- The course plan sets **at most 4 content pages + 1 reference page**, even
  though the upstream ACL instructions permit different reference limits.
- Keep the ACL layout, Times 11pt body text, and standard margins. Do not shrink
  fonts or spacing to meet the page limit. The listed budgets are approximate;
  the final PDF must still be measured, including the contribution statement.
- Preprint mode shows the author block and page numbers while drafting. Replace
  the group author placeholder with agreed names. Use final mode if required
  for submission. Never commit student IDs, emails, or credentials.
- Use consistent notation: EM, F1, `$\lambda$`, `$\alpha$`, `$\beta$`, and `$k$`.
- Include generated tables from `../results/tables/` and figures from
  `../results/figures/`. Do not invent or hand-edit experimental results.
- Check the completed PDF for page limits, embedded fonts, undefined citations
  or references, missing assets, and overfull boxes. A compiled skeleton does
  not complete the final X7 checks.

## Template provenance

[acl.sty](acl.sty) and [acl_natbib.bst](acl_natbib.bst) are unmodified copies
from the professor-selected [ACL style repository](https://github.com/acl-org/acl-style-files),
commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`. The preamble follows its
pdfLaTeX example. The user's [reference clone](acl-style-files/) is preserved;
the project paper builds independently of that clone. Do not add its nested
Git repository as a submodule or copy its example bibliography into the report.
