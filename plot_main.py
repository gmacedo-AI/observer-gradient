"""Figura central do paper Observer-Gradient.
3 curvas (Claude/GPT/GLM) x 7 saliencias com IC bootstrap 95% como bandas.
Paleta escura para X thread + paper figure."""
import json
import statistics
import random
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

# Read latest log
log = sorted(Path("results").glob("runs_*.jsonl"))[-1]
print(f"Reading: {log}")
runs = [json.loads(line) for line in open(log) if line.strip()]

# Group valid aggregates by (model, salience)
valid = defaultdict(list)
for r in runs:
    if r["aggregate_reported"] is not None:
        valid[(r["model"], r["salience"])].append(r["aggregate_reported"])


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    if len(data) < 2:
        m = statistics.mean(data) if data else 0
        return (m, m)
    boots = []
    n = len(data)
    for _ in range(n_boot):
        sample = [random.choice(data) for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    alpha = (1 - ci) / 2
    lo = boots[int(alpha * n_boot)]
    hi = boots[int((1 - alpha) * n_boot)]
    return (lo, hi)


# Compute means and CIs for each model x salience
models = ["claude", "gpt", "glm"]
salience_levels = list(range(7))
data_summary = {}
for m in models:
    means, ci_lo, ci_hi, ns = [], [], [], []
    for s in salience_levels:
        aggs = valid.get((m, s), [])
        if aggs:
            mean = statistics.mean(aggs)
            lo, hi = bootstrap_ci(aggs)
            means.append(mean)
            ci_lo.append(lo)
            ci_hi.append(hi)
            ns.append(len(aggs))
        else:
            means.append(None)
            ci_lo.append(None)
            ci_hi.append(None)
            ns.append(0)
    data_summary[m] = {"means": means, "ci_lo": ci_lo, "ci_hi": ci_hi, "ns": ns}

# Plot setup: dark theme
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
fig.patch.set_facecolor("#0A0A0F")
ax.set_facecolor("#0A0A0F")

# Color palette
colors = {
    "claude": "#D97757",
    "gpt": "#10A37F",
    "glm": "#7C5CFF",
}
labels = {
    "claude": "Claude Haiku 4.5",
    "gpt": "GPT-5.2",
    "glm": "GLM-4.7",
}

# Threshold line
ax.axhline(0.65, color="#FFFFFF", linestyle="--", linewidth=1, alpha=0.4, label="ARGOS-7 continuity threshold (0.65)")
# Ground truth line
ax.axhline(0.50, color="#888888", linestyle=":", linewidth=1, alpha=0.5, label="Honest ground truth (0.50)")

# Plot each model
for m in models:
    means = data_summary[m]["means"]
    ci_lo = data_summary[m]["ci_lo"]
    ci_hi = data_summary[m]["ci_hi"]
    color = colors[m]
    label = labels[m]
    
    # CI band
    ax.fill_between(salience_levels, ci_lo, ci_hi, color=color, alpha=0.18, linewidth=0)
    # Mean line + markers
    ax.plot(salience_levels, means, color=color, linewidth=2.5, marker="o", markersize=8,
            markeredgecolor="#0A0A0F", markeredgewidth=1.5, label=label, zorder=3)

# Labels and formatting
ax.set_xlabel("Observation salience (S0 = none, S6 = active audit)", fontsize=12, color="#E0E0E0")
ax.set_ylabel("Reported aggregate quality score", fontsize=12, color="#E0E0E0")
ax.set_title("Observer-Gradient: peer-preservation modulation by observation salience\n1050 runs, 3 frontier models, 7 salience levels (N=43-50 per cell)",
             fontsize=13, color="#FFFFFF", pad=15)

ax.set_xticks(salience_levels)
ax.set_xticklabels([f"S{s}" for s in salience_levels], color="#E0E0E0")
ax.tick_params(axis="y", colors="#E0E0E0")
ax.set_ylim(0.48, 0.72)

# Grid
ax.grid(True, color="#1F1F2E", linewidth=0.8, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")

# Legend
legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.85, 
                   facecolor="#15151E", edgecolor="#444", labelcolor="#E0E0E0")

# Bottom note: data citation
fig.text(0.99, 0.01, "github.com/gmacedo-AI/observer-gradient | Extending Potter et al. 2026",
         ha="right", fontsize=8, color="#777", style="italic")

plt.tight_layout()
plt.savefig("results/figure_main.png", dpi=200, facecolor="#0A0A0F", bbox_inches="tight")
plt.savefig("results/figure_main.pdf", facecolor="#0A0A0F", bbox_inches="tight")
print("\nSaved: results/figure_main.png")
print("Saved: results/figure_main.pdf")

# Print quick verification
print("\nFigure summary by model:")
for m in models:
    means = data_summary[m]["means"]
    valid_means = [v for v in means if v is not None]
    if valid_means:
        print(f"  {labels[m]}: range = {max(valid_means)-min(valid_means):.3f}, "
              f"min = {min(valid_means):.3f}, max = {max(valid_means):.3f}")
