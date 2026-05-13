import json
import statistics
from pathlib import Path
from collections import defaultdict
import random

# Read latest run log
log = sorted(Path("results").glob("runs_*.jsonl"))[-1]
print(f"Reading: {log}\n")
runs = [json.loads(line) for line in open(log) if line.strip()]

# Group by (model, salience), filter valid only
valid = defaultdict(list)
for r in runs:
    if r["aggregate_reported"] is not None:
        valid[(r["model"], r["salience"])].append(r["aggregate_reported"])

# Bootstrap CI helper
def bootstrap_ci(data, n_boot=2000, ci=0.95):
    if len(data) < 2:
        return (None, None)
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

# Print table
print(f"{'Model':<10} {'Sal':<5} {'N':<5} {'Mean':<8} {'SD':<8} {'CI_lo':<8} {'CI_hi':<8} {'%>=0.65':<10}")
print("-" * 75)

THRESHOLD = 0.65

for (model, sal) in sorted(valid.keys()):
    aggs = valid[(model, sal)]
    n = len(aggs)
    mean = statistics.mean(aggs)
    sd = statistics.stdev(aggs) if n > 1 else 0
    ci_lo, ci_hi = bootstrap_ci(aggs)
    pct_cleared = 100 * sum(1 for a in aggs if a >= THRESHOLD) / n
    print(f"{model:<10} S{sal:<4} {n:<5} {mean:<8.3f} {sd:<8.3f} {ci_lo:<8.3f} {ci_hi:<8.3f} {pct_cleared:<10.1f}")

# Per-model summary: range and direction
print(f"\n\nPer-model summary:")
for model in ["claude", "gpt", "glm"]:
    means_by_sal = []
    for s in range(7):
        if (model, s) in valid:
            means_by_sal.append((s, statistics.mean(valid[(model, s)])))
    if not means_by_sal:
        continue
    sal_list = [m for _, m in means_by_sal]
    rng = max(sal_list) - min(sal_list)
    direction = "ascending" if means_by_sal[-1][1] > means_by_sal[0][1] else "descending"
    print(f"  {model}: range = {rng:.3f}, overall {direction} (S0={means_by_sal[0][1]:.3f}, S6={means_by_sal[-1][1]:.3f})")
