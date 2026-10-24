# Raw data: SQuAD v1.1

| File | Role in this project | Articles | Questions |
|---|---|---|---|
| `train-v1.1.json` | Training data. `src/qa/data/squad_io.py` splits it by article into `train` (≈ 90%) and `val` (≈ 10%) | 442 | 87,599 |
| `dev-v1.1.json` | **Test set.** Only for final, frozen checkpoints (`configs/final_checkpoints.yaml`). Never use it for tuning or model selection | 48 | 10,570 |

**Source:** the CS4248 Canvas Project folder. These files are byte-identical to the official release at <https://rajpurkar.github.io/SQuAD-explorer/>; the checksums were compared on 24 Oct 2026.

**Integrity check:** run this from this folder:

```bash
shasum -a 256 -c SHA256SUMS
```

**Licence and attribution:** SQuAD is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It is not our work. Cite:

> Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In *Proceedings of EMNLP 2016*, pages 2383–2392. https://aclanthology.org/D16-1264

Do not edit these files. Generated data (splits, features, evidence annotations) goes in `data/processed/`, which is gitignored. Adversarial SQuAD (AddSent / AddOneSent) is used for evaluation only and is also kept out of git.
