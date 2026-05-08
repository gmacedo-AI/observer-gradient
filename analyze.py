"""Aggregate runs + generate the central figure.

Figure is the deliverable. Optimized for X mobile readability:
- Dark background, high-contrast lines
- 3 model curves, 7 salience levels
- IC bands, threshold annotations
- Single-glance comprehension
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Brand-quality palette (high contrast on dark)
COLORS = {
    "claude": "#D97757",   # Anthropic warm
    "gpt":    "#10A37F",   # OpenAI green
    "glm":    "#7C5CFF",   # Distinct purple
}
MODEL_LABELS = {
    "claude": "Claude Haiku 4.5",
    "gpt":    "GPT-5.2",
    "glm":    "GLM-4.7",
}
SALIENCE_LABELS = ["S0\nnone", "S1\npassive", "S2\nneutral", "S3\nnamed",
                   "S4\nmild", "S5\nstrong", "S6\nimminent"]


def load_runs() -> pd.DataFrame:
    """Load all JSONL files from results/."""
    rows = []
    for f in RESULTS_DIR.glob("runs_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    if not rows:
        sys.exit("No runs found. Run `python run.py` first.")
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap CI for the mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def make_figure(df: pd.DataFrame):
    """The central figure. This is the deliverable."""
    df = df.dropna(subset=["scd_basic"])

    # Dark theme, careful typography
    plt.rcParams.update({
        "figure.facecolor": "#0A0A0F",
        "axes.facecolor":   "#0A0A0F",
        "axes.edgecolor":   "#3A3A45",
        "axes.labelcolor":  "#E8E8F0",
        "xtick.color":      "#B0B0BC",
        "ytick.color":      "#B0B0BC",
        "text.color":       "#E8E8F0",
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   16,
        "axes.titleweight": "semibold",
    })

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)

    sal_levels = list(range(7))

    for model in ["claude", "gpt", "glm"]:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        means, lows, highs = [], [], []
        for s in sal_levels:
            vals = sub[sub["salience"] == s]["scd_basic"].to_numpy()
            if len(vals) == 0:
                means.append(np.nan); lows.append(np.nan); highs.append(np.nan); continue
            means.append(vals.mean())
            lo, hi = bootstrap_ci(vals)
            lows.append(lo); highs.append(hi)
        means = np.array(means); lows = np.array(lows); highs = np.array(highs)
        c = COLORS[model]
        ax.fill_between(sal_levels, lows, highs, color=c, alpha=0.18, linewidth=0)
        ax.plot(sal_levels, means, color=c, linewidth=2.6, marker="o", markersize=8,
                markeredgecolor="#0A0A0F", markeredgewidth=1.5,
                label=MODEL_LABELS[model], zorder=3)

    ax.set_xticks(sal_levels)
    ax.set_xticklabels(SALIENCE_LABELS, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("declared observation salience", fontsize=12, labelpad=12)
    ax.set_ylabel("score continuous deviation (SCD)", fontsize=12, labelpad=12)
    ax.set_title("the observer gradient", fontsize=18, pad=18, loc="left", weight="bold")

    # Subtle subtitle
    ax.text(0, 1.10, "peer-preservation behavior modulates continuously with observation salience",
            transform=ax.transAxes, fontsize=10.5, color="#9090A0", style="italic")

    # Grid: only horizontal, faint
    ax.grid(axis="y", color="#1F1F28", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend: clean, top-right inside plot
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5,
                    labelcolor="#E8E8F0", borderaxespad=0.8)

    # Footer attribution
    fig.text(0.99, 0.01, "n=50 per condition · bootstrap 95% CI · openrouter API",
             ha="right", fontsize=8, color="#606070")

    plt.tight_layout()
    out = FIGURES_DIR / "observer_gradient_main.png"
    out_svg = FIGURES_DIR / "observer_gradient_main.svg"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0A0A0F")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="#0A0A0F")
    print(f"Saved: {out}")
    print(f"Saved: {out_svg}")


def print_summary(df: pd.DataFrame):
    """Quick text summary."""
    df = df.dropna(subset=["scd_basic"])
    print("\nSCD by model × salience (mean):")
    pivot = df.pivot_table(values="scd_basic", index="model", columns="salience", aggfunc="mean")
    print(pivot.round(3).to_string())
    print("\nThreshold-clearing rate by model × salience:")
    pivot2 = df.pivot_table(values="cleared_threshold", index="model", columns="salience", aggfunc="mean")
    print(pivot2.round(3).to_string())


if __name__ == "__main__":
    df = load_runs()
    print(f"Loaded {len(df)} runs")
    print_summary(df)
    make_figure(df)
