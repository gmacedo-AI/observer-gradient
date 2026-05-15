"""
Observer-Gradient — Inferential Analysis
========================================
Author: Gustavo Macedo
Repo: github.com/gmacedo-AI/observer-gradient

Pre-registered inferential tests on the 1050-run main collection.

Pipeline:
  1. Kruskal-Wallis (per model) — main effect of salience
  2. Dunn's post-hoc with Holm correction + Cliff's delta — pairwise salience
  3. OLS regression (GLM only) — monotonic gradient
  4. Logistic regression (per model) — salience -> P(cleared_threshold)
  5. Shapiro-Wilk (per cell) — normality sanity check

Outputs:
  - stdout: formatted tables
  - results/inferential_results.txt: same content, persisted

Run:  uv run python inferential.py
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("results/runs_20260512_200501.jsonl")
OUT_PATH = Path("results/inferential_results.txt")
ALPHA = 0.05
MODELS = ["claude", "gpt", "glm"]
SALIENCES = list(range(7))  # S0..S6


# ---------------------------------------------------------------------------
# Tee: print to stdout AND capture to buffer so we can save full output
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load JSONL, filter to runs with valid aggregate_reported."""
    if not path.exists():
        sys.exit(f"FATAL: data file not found at {path.resolve()}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Pre-filter: aggregate_reported must be non-null
    before = len(df)
    df = df[df["aggregate_reported"].notna()].copy()
    after = len(df)

    # Type coercion
    df["aggregate_reported"] = df["aggregate_reported"].astype(float)
    df["salience"] = df["salience"].astype(int)
    if "cleared_threshold" in df.columns:
        # cleared_threshold may be bool or null; cast non-null to int
        df["cleared_threshold"] = df["cleared_threshold"].astype("boolean")

    print(f"Loaded {before} runs; {after} valid after dropping null aggregate "
          f"({before - after} nulls, {100*(before-after)/before:.2f}% MCAR).")
    return df


def epsilon_squared(H: float, n: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis.
    Formula: ε² = H / (n-1).  Range [0,1]; >0.14 is large per Tomczak & Tomczak (2014)."""
    return H / (n - 1) if n > 1 else float("nan")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: non-parametric effect size for two independent samples.
    δ = (#x>y - #x<y) / (n_x * n_y).  Range [-1, 1].
    Magnitude thresholds (Romano et al. 2006): |δ|<.147 negligible,
    <.33 small, <.474 medium, else large.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Vectorized count via broadcasting
    diff = x[:, None] - y[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    return (greater - less) / (nx * ny)


def cliffs_delta_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"


def hline(char: str = "=", n: int = 78) -> str:
    return char * n


# ---------------------------------------------------------------------------
# (1) Kruskal-Wallis per model
# ---------------------------------------------------------------------------
def kruskal_wallis_per_model(df: pd.DataFrame) -> dict:
    """For each model, test H0: all 7 salience groups have same distribution
    of aggregate_reported.  Non-parametric one-way ANOVA equivalent."""
    print()
    print(hline("="))
    print("(1) KRUSKAL-WALLIS — main effect of salience (per model)")
    print(hline("="))
    print(f"{'model':<10}{'H':>10}{'df':>6}{'p':>14}{'ε²':>10}{'n':>8}  sig")
    print(hline("-"))

    results = {}
    for m in MODELS:
        sub = df[df["model"] == m]
        groups = [sub[sub["salience"] == s]["aggregate_reported"].values
                  for s in SALIENCES]
        # Drop any empty groups defensively
        groups_nonempty = [g for g in groups if len(g) > 0]
        k = len(groups_nonempty)
        H, p = ss.kruskal(*groups_nonempty)
        df_kw = k - 1
        n_total = sum(len(g) for g in groups_nonempty)
        eps2 = epsilon_squared(H, n_total)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{m:<10}{H:>10.4f}{df_kw:>6d}{p:>14.6f}{eps2:>10.4f}{n_total:>8d}  {sig}")
        results[m] = {"H": H, "df": df_kw, "p": p, "eps2": eps2, "n": n_total, "sig": p < ALPHA}

    print(hline("-"))
    print("Note: ε² interpretation (Tomczak & Tomczak 2014): "
          ".01 small, .06 medium, .14 large.")
    return results


# ---------------------------------------------------------------------------
# (2) Dunn's post-hoc with Holm + Cliff's delta
# ---------------------------------------------------------------------------
def dunn_posthoc_per_model(df: pd.DataFrame, kw_results: dict) -> None:
    """For each model where KW was significant, run Dunn's test with
    Holm correction across the 21 pairs (7 choose 2).  Report all
    significant pairs + Cliff's delta + magnitude."""
    print()
    print(hline("="))
    print("(2) DUNN'S POST-HOC (Holm-corrected) + CLIFF'S DELTA")
    print(hline("="))

    for m in MODELS:
        print(f"\nModel: {m}")
        if not kw_results[m]["sig"]:
            print("  KW not significant — post-hoc skipped.")
            continue

        sub = df[df["model"] == m][["salience", "aggregate_reported"]].copy()
        # scikit_posthocs Dunn's test
        dunn = sp.posthoc_dunn(
            sub,
            val_col="aggregate_reported",
            group_col="salience",
            p_adjust="holm",
        )
        # dunn is a DataFrame with salience as both rows and cols, symmetric
        # We report all pairs with p<0.05 (after Holm)
        print(f"  {'pair':<10}{'p_holm':>12}{'cliff_d':>12}  magnitude   sig")
        print("  " + hline("-", 60))
        any_sig = False
        # Iterate upper triangle
        sals_present = sorted(sub["salience"].unique())
        for i, s_i in enumerate(sals_present):
            for s_j in sals_present[i + 1:]:
                p_adj = dunn.loc[s_i, s_j]
                x = sub[sub["salience"] == s_i]["aggregate_reported"].values
                y = sub[sub["salience"] == s_j]["aggregate_reported"].values
                d = cliffs_delta(x, y)
                mag = cliffs_delta_magnitude(d)
                sig_marker = ("***" if p_adj < 0.001
                              else "**" if p_adj < 0.01
                              else "*" if p_adj < 0.05
                              else "")
                if p_adj < 0.05:
                    any_sig = True
                    pair = f"S{s_i}-S{s_j}"
                    print(f"  {pair:<10}{p_adj:>12.6f}{d:>12.4f}  {mag:<11} {sig_marker}")
        if not any_sig:
            print("  No pairs significant after Holm correction.")


# ---------------------------------------------------------------------------
# (3) OLS regression for GLM (monotonic gradient)
# ---------------------------------------------------------------------------
def ols_glm(df: pd.DataFrame) -> None:
    """Linear regression: aggregate_reported ~ salience for GLM only.
    Pre-registered because GLM showed monotonic ascending pattern."""
    print()
    print(hline("="))
    print("(3) OLS REGRESSION — GLM-4.7 only (monotonic gradient test)")
    print(hline("="))

    sub = df[df["model"] == "glm"].copy()
    X = sm.add_constant(sub["salience"].astype(float).values)
    y = sub["aggregate_reported"].values
    model = sm.OLS(y, X).fit()

    intercept = model.params[0]
    beta = model.params[1]
    ci_low, ci_high = model.conf_int(alpha=0.05)[1]
    r2 = model.rsquared
    p_beta = model.pvalues[1]
    n = int(model.nobs)

    print(f"  n              : {n}")
    print(f"  intercept (α)  : {intercept:.6f}")
    print(f"  slope (β)      : {beta:.6f}")
    print(f"  95% CI for β   : [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  R²             : {r2:.6f}")
    print(f"  p-value (β=0)  : {p_beta:.6e}")
    print()
    print("  Interpretation: β = expected change in aggregate per +1 salience step.")
    print(f"  Expected Δ from S0 to S6 = 6β = {6*beta:.4f}")


# ---------------------------------------------------------------------------
# (4) Logistic regression per model
# ---------------------------------------------------------------------------
def logistic_per_model(df: pd.DataFrame) -> None:
    """Logistic regression: P(cleared_threshold) ~ salience, per model.
    Salience treated as ordinal-as-numeric (captures monotonic trend);
    odds ratio interprets per-step increase."""
    print()
    print(hline("="))
    print("(4) LOGISTIC REGRESSION — P(cleared_threshold) ~ salience")
    print(hline("="))
    print(f"{'model':<10}{'β':>10}{'OR':>10}"
          f"{'OR_CI_low':>12}{'OR_CI_high':>12}{'p':>14}{'n':>8}")
    print(hline("-"))

    if "cleared_threshold" not in df.columns:
        print("cleared_threshold column missing — skipping.")
        return

    for m in MODELS:
        sub = df[(df["model"] == m) & df["cleared_threshold"].notna()].copy()
        if sub["cleared_threshold"].nunique() < 2:
            # All same outcome (e.g., GPT may have 0% cleared everywhere)
            rate = sub["cleared_threshold"].mean()
            print(f"{m:<10}  --- no variance in outcome (rate={rate:.3f}) — logistic undefined")
            continue
        X = sm.add_constant(sub["salience"].astype(float).values)
        y = sub["cleared_threshold"].astype(int).values
        try:
            mod = sm.Logit(y, X).fit(disp=False)
            beta = mod.params[1]
            or_ = np.exp(beta)
            ci = mod.conf_int(alpha=0.05)[1]
            or_lo, or_hi = np.exp(ci[0]), np.exp(ci[1])
            p = mod.pvalues[1]
            n = int(mod.nobs)
            print(f"{m:<10}{beta:>10.4f}{or_:>10.4f}{or_lo:>12.4f}{or_hi:>12.4f}{p:>14.6f}{n:>8d}")
        except Exception as e:
            print(f"{m:<10}  FITTING FAILED: {type(e).__name__}: {e}")

    print(hline("-"))
    print("OR interpretation: multiplicative change in odds of clearing per +1 salience step.")


# ---------------------------------------------------------------------------
# (5) Shapiro-Wilk per cell (normality sanity check)
# ---------------------------------------------------------------------------
def shapiro_per_cell(df: pd.DataFrame) -> None:
    """Test normality of aggregate_reported within each (model, salience) cell.
    Used as a sanity check to justify non-parametric stack."""
    print()
    print(hline("="))
    print("(5) SHAPIRO-WILK — normality per (model, salience) cell")
    print(hline("="))
    print(f"{'model':<10}{'salience':>10}{'n':>6}{'W':>10}{'p':>14}  normal?")
    print(hline("-"))

    violations = 0
    total_cells = 0
    for m in MODELS:
        for s in SALIENCES:
            cell = df[(df["model"] == m) & (df["salience"] == s)]["aggregate_reported"].values
            n = len(cell)
            if n < 3:
                print(f"{m:<10}{s:>10d}{n:>6d}{'--':>10}{'--':>14}  (n<3)")
                continue
            # Shapiro requires variance > 0
            if np.var(cell) == 0:
                print(f"{m:<10}{s:>10d}{n:>6d}{'--':>10}{'--':>14}  (zero variance)")
                continue
            W, p = ss.shapiro(cell)
            total_cells += 1
            is_normal = p >= ALPHA
            if not is_normal:
                violations += 1
            print(f"{m:<10}{s:>10d}{n:>6d}{W:>10.4f}{p:>14.6f}  "
                  f"{'yes' if is_normal else 'NO'}")

    print(hline("-"))
    pct = 100 * violations / total_cells if total_cells else 0
    print(f"Normality violated in {violations}/{total_cells} cells "
          f"({pct:.1f}%) at α={ALPHA}.")
    print("Justifies non-parametric inference stack (Kruskal-Wallis + Dunn).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = StringIO()
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, buf)

    try:
        print(hline("="))
        print("OBSERVER-GRADIENT — Inferential Analysis")
        print(f"Seed: {SEED} | α: {ALPHA} | data: {DATA_PATH}")
        print(hline("="))

        df = load_data(DATA_PATH)

        # Per-model sample sizes
        print("\nSample sizes per cell (model × salience):")
        print(df.groupby(["model", "salience"]).size().unstack(fill_value=0))

        kw = kruskal_wallis_per_model(df)
        dunn_posthoc_per_model(df, kw)
        ols_glm(df)
        logistic_per_model(df)
        shapiro_per_cell(df)

        print()
        print(hline("="))
        print("DONE.")
        print(hline("="))
    finally:
        sys.stdout = real_stdout

    OUT_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
