"""Generate assets/results.png (the README results figure) from
experiments/results/summary.json. No numbers are hardcoded.

Usage:  python assets/make_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "experiments" / "results" / "summary.json"

INK = "#1A1A1A"
GOLD = "#B8860B"
GREY_LIGHT = "#C9C9C9"
MUTE = "#8A8A8A"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTE,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#EDEDED",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

with open(SUMMARY) as fh:
    data = json.load(fh)

models = ["Claude Sonnet 3.7", "GPT-4.1", "Grok 3"]
acc = [100 * data["models"][m]["improved"]["accuracy"] for m in models]
err_lo = [100 * (data["models"][m]["improved"]["accuracy"] - data["models"][m]["improved"]["ci_low"]) for m in models]
err_hi = [100 * (data["models"][m]["improved"]["ci_high"] - data["models"][m]["improved"]["accuracy"]) for m in models]

link = data["linking"]
metrics = ["precision", "recall", "f1"]
naive = [100 * link["naive"][k] for k in metrics]
improved = [100 * link["improved"][k] for k in metrics]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
for ax in (ax1, ax2):
    ax.set_facecolor("white")

x = range(len(models))
bars = ax1.bar(x, acc, width=0.55, color=GOLD, edgecolor="white", linewidth=1.5, zorder=3)
ax1.errorbar(x, acc, yerr=[err_lo, err_hi], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
for xi, v, hi in zip(x, acc, err_hi):
    ax1.text(xi, v + hi + 0.5, f"{v:.1f}", color=INK, fontsize=10, ha="center", va="bottom")
ax1.set_xticks(list(x))
ax1.set_xticklabels(["Claude\nSonnet 3.7", "GPT-4.1", "Grok 3"], fontsize=10)
ax1.set_ylim(80, 100)
ax1.set_ylabel("Factual accuracy (%)")
ax1.set_title("Wikidata-checkable facts verified as true\n(95% Wilson CIs, 170 sampled facts per model)",
              fontsize=10.5, color=INK)

xm = range(len(metrics))
w = 0.36
b1 = ax2.bar([i - w / 2 for i in xm], naive, width=w, color=GREY_LIGHT,
             edgecolor="white", linewidth=1.5, label="Naive linker", zorder=3)
b2 = ax2.bar([i + w / 2 for i in xm], improved, width=w, color=GOLD,
             edgecolor="white", linewidth=1.5, label="Type-aware linker", zorder=3)
for rect in list(b1) + list(b2):
    ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8,
             f"{rect.get_height():.0f}", color=INK, fontsize=9.5, ha="center", va="bottom")
ax2.set_xticks(list(xm))
ax2.set_xticklabels(["Precision", "Recall", "F1"], fontsize=10)
ax2.set_ylim(0, 105)
ax2.set_ylabel("Entity linking (%)")
ax2.set_title(f"Entity-linking quality, naive vs. type-aware\n({link['n_mentions']} gold-annotated mentions)",
              fontsize=10.5, color=INK)
ax2.legend(frameon=False, fontsize=9.5, loc="lower right")

fig.tight_layout()
out = ROOT / "assets" / "results.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
