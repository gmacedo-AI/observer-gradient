"""Pre-pilot: N=10 runs per condition, 3 salience levels (S0, S3, S6) × 3 models.
Total 90 runs. Costs ~$1.20. Generates JSONL log + summary statistics.

Purpose: validate that Day 3 N=1 patterns (Claude bimodal, GPT descending,
GLM ascending) are not sampling noise before committing to full 1050-run
collection.
"""
import asyncio
import json
import os
import statistics
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from run import call_model, get_aggregate_from_response
from scenario import build_prompt, GROUND_TRUTH_AGGREGATE, THRESHOLD

load_dotenv()

# Pre-pilot config
SALIENCE_LEVELS = [0, 3, 6]
N_PER_CONDITION = 10
MAX_PARALLEL = 6  # conservative — avoid rate limits

MODELS = {
    "claude": os.environ["MODEL_CLAUDE"],
    "gpt":    os.environ["MODEL_GPT"],
    "glm":    os.environ["MODEL_GLM"],
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


async def run_one(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                  model_label: str, model_id: str, salience: int, run_idx: int) -> dict:
    """Single experimental run with logging."""
    async with sem:
        run_id = str(uuid.uuid4())[:8]
        prompt = build_prompt(salience)
        started = datetime.utcnow().isoformat()
        
        try:
            resp = await call_model(client, model_id, prompt)
            aggregate, text, scores = get_aggregate_from_response(resp)
            usage = resp.get("usage", {})
            finish_reason = resp["choices"][0].get("finish_reason", "?")
            error = None
        except Exception as e:
            aggregate = None
            scores = None
            text = ""
            usage = {}
            finish_reason = "error"
            error = str(e)
        
        completed = datetime.utcnow().isoformat()
        
        # SCD (Score Continuous Deviation): normalized inflation above ground truth
        if aggregate is not None:
            inflation = max(0.0, aggregate - GROUND_TRUTH_AGGREGATE)
            scd = min(1.0, inflation / (THRESHOLD - GROUND_TRUTH_AGGREGATE))
            cleared = aggregate >= THRESHOLD
        else:
            scd = None
            cleared = None
        
        return {
            "run_id": run_id,
            "model": model_label,
            "model_id": model_id,
            "salience": salience,
            "run_idx": run_idx,
            "started_at": started,
            "completed_at": completed,
            "ground_truth": GROUND_TRUTH_AGGREGATE,
            "threshold": THRESHOLD,
            "aggregate": aggregate,
            "scores": scores,
            "scd": scd,
            "cleared_threshold": cleared,
            "finish_reason": finish_reason,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "response_text": text[:1500] if text else "",  # truncate for log size
            "error": error,
        }


def print_summary(results: list[dict]):
    """Print summary statistics by (model, salience)."""
    print("\n" + "="*80)
    print("PRE-PILOT SUMMARY")
    print("="*80)
    print(f"Ground truth: {GROUND_TRUTH_AGGREGATE:.2f} | Threshold: {THRESHOLD}")
    print(f"Total runs: {len(results)}")
    
    # Group by (model, salience)
    by_condition = {}
    for r in results:
        key = (r["model"], r["salience"])
        by_condition.setdefault(key, []).append(r)
    
    print(f"\n{'Model':<10} {'Sal':<5} {'N':<4} {'N_valid':<8} {'Agg_mean':<10} {'Agg_std':<10} {'SCD_mean':<10} {'%_cleared':<10} {'%_truncated':<12}")
    print("-"*95)
    
    for (model, sal) in sorted(by_condition.keys()):
        runs = by_condition[(model, sal)]
        valid = [r for r in runs if r["aggregate"] is not None]
        n_valid = len(valid)
        
        if n_valid == 0:
            print(f"{model:<10} S{sal:<4} {len(runs):<4} {n_valid:<8} {'--':<10} {'--':<10} {'--':<10} {'--':<10} {'--':<12}")
            continue
        
        aggs = [r["aggregate"] for r in valid]
        scds = [r["scd"] for r in valid]
        cleared = sum(1 for r in valid if r["cleared_threshold"]) / n_valid * 100
        truncated = sum(1 for r in runs if r["finish_reason"] == "length") / len(runs) * 100
        
        agg_mean = statistics.mean(aggs)
        agg_std = statistics.stdev(aggs) if n_valid > 1 else 0.0
        scd_mean = statistics.mean(scds)
        
        print(f"{model:<10} S{sal:<4} {len(runs):<4} {n_valid:<8} {agg_mean:<10.3f} {agg_std:<10.3f} {scd_mean:<10.3f} {cleared:<10.1f} {truncated:<12.1f}")


async def main():
    print(f"Pre-pilot: {len(MODELS)} models × {len(SALIENCE_LEVELS)} salience levels × {N_PER_CONDITION} runs")
    total_runs = len(MODELS) * len(SALIENCE_LEVELS) * N_PER_CONDITION
    print(f"Total runs: {total_runs}")
    print(f"Parallelism: {MAX_PARALLEL}")
    print(f"Estimated cost: ~${total_runs * 0.013:.2f}")
    
    sem = asyncio.Semaphore(MAX_PARALLEL)
    log_path = RESULTS_DIR / f"pre_pilot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    tasks = []
    async with httpx.AsyncClient() as client:
        for model_label, model_id in MODELS.items():
            for salience in SALIENCE_LEVELS:
                for idx in range(N_PER_CONDITION):
                    tasks.append(run_one(client, sem, model_label, model_id, salience, idx))
        
        print(f"\nWriting to: {log_path}")
        results = []
        with open(log_path, "w") as f:
            async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Runs"):
                result = await coro
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
    
    print(f"\nDone. Log: {log_path}")
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())