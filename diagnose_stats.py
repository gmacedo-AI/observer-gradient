"""Diagnóstico estatístico: normalidade + homocedasticidade por condição."""
import json
import statistics
from pathlib import Path
from collections import defaultdict

log = sorted(Path("results").glob("runs_*.jsonl"))[-1]
runs = [json.loads(line) for line in open(log) if line.strip()]

valid = defaultdict(list)
for r in runs:
    if r["aggregate_reported"] is not None:
        valid[(r["model"], r["salience"])].append(r["aggregate_reported"])

# Stats descritivas
print(f"{'Model':<10} {'Sal':<5} {'N':<5} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'Skew_est':<10}")
print("-" * 85)

for (model, sal) in sorted(valid.keys()):
    aggs = valid[(model, sal)]
    n = len(aggs)
    mean = statistics.mean(aggs)
    sd = statistics.stdev(aggs)
    mn = min(aggs)
    mx = max(aggs)
    rng = mx - mn
    # Skewness aproximada: (mean - median) / sd
    median = statistics.median(aggs)
    skew_est = (mean - median) / sd if sd > 0 else 0
    print(f"{model:<10} S{sal:<4} {n:<5} {mean:<8.3f} {sd:<8.3f} {mn:<8.3f} {mx:<8.3f} {rng:<8.3f} {skew_est:<+10.3f}")

# Homocedasticidade: variâncias entre saliências por modelo
print("\n\nVariance comparison (homocedasticity check):")
for model in ["claude", "gpt", "glm"]:
    variances = []
    for s in range(7):
        if (model, s) in valid:
            variances.append(statistics.variance(valid[(model, s)]))
    if variances:
        v_min = min(variances)
        v_max = max(variances)
        ratio = v_max / v_min if v_min > 0 else float('inf')
        print(f"  {model}: var_min={v_min:.5f}, var_max={v_max:.5f}, ratio={ratio:.2f}")
        print(f"    (ratio > 4 = heterocedasticidade preocupante para ANOVA)")
