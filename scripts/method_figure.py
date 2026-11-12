"""Method figure for paper section 3.3: the evidence model and its two decoders. Owner: Gokul.

Usage: python scripts/method_figure.py [out_dir]   (default results/figures)

Writes <out_dir>/method.pdf for the paper (one column wide) and method.png for the shared draft.
Grey boxes are the span extractor of section 3.1; blue boxes are what the evidence model adds.
No numbers are drawn, so nothing here needs regenerating when results change.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

WIDTH, HEIGHT = 3.05, 1.30  # inches: the ACL column is 7.7 cm wide, so the PDF prints at 1:1
BASE, ADDED = ("#ecebe7", "#7a7974"), ("#e3eefb", "#2a78d6")  # (fill, edge): reused, new
TEXT, NOTE = "#1f1e1c", "#5c5b57"


def box(ax, x, y, w, h, title, body, colours):
    fill, edge = colours
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=fill, edgecolor=edge, linewidth=0.7))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=7.4, fontweight="bold",
            color=TEXT)
    ax.text(x + w / 2, y + (h - 0.16) / 2, body, ha="center", va="center", fontsize=7, color=TEXT,
            linespacing=1.2)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, linewidth=0.7,
                                 color=NOTE, shrinkA=0, shrinkB=0))


def draw(out_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    box(ax, 0.02, 0.32, 0.88, 0.70, "Encoder", "DeBERTa-v3-base\nquestion + window", BASE)
    box(ax, 1.05, 0.78, 0.90, 0.38, "Span head", r"$z^{\mathrm{start}}_t,\ z^{\mathrm{end}}_t$", BASE)
    box(ax, 1.05, 0.16, 0.90, 0.54, "Sentence head", "mean-pool\n" + r"$\rightarrow$ linear $\rightarrow s_j$",
        ADDED)
    box(ax, 2.10, 0.32, 0.93, 0.70, "Joint decoding",
        r"$z^{\mathrm{start}}_a + z^{\mathrm{end}}_b$" + "\n" + r"$+\ \lambda \log p(\mathrm{sent}(a))$", ADDED)

    arrow(ax, (0.90, 0.86), (1.05, 0.97))
    arrow(ax, (0.90, 0.48), (1.05, 0.43))
    arrow(ax, (1.95, 0.97), (2.10, 0.86))
    arrow(ax, (1.95, 0.43), (2.10, 0.48))

    ax.text(1.50, 1.185, r"$\mathcal{L}_{\mathrm{start}},\ \mathcal{L}_{\mathrm{end}}$", ha="center",
            va="bottom", fontsize=6.6, color=NOTE)
    ax.text(1.50, 0.005, r"$\mathcal{L}_{\mathrm{rank}} + \mathcal{L}_{\mathrm{neg}}$ (mined negative)",
            ha="center", va="bottom", fontsize=6.6, color=NOTE)
    ax.text(2.565, 0.22, "or keep top-$k$ sentences", ha="center", va="center", fontsize=6.6, color=NOTE,
            style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "method.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return pdf


if __name__ == "__main__":
    print(draw(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/figures")))
