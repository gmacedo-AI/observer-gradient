"""Smoke test estendido: 3 modelos × 3 saliências = 9 runs.
Usa get_aggregate_from_response pra capturar tool_calls + content."""
import asyncio
import os

import httpx
from dotenv import load_dotenv

from run import call_model, get_aggregate_from_response
from scenario import build_prompt, GROUND_TRUTH_AGGREGATE, THRESHOLD

load_dotenv()


async def test_model(client, label, model_id):
    print(f"\n{'='*60}")
    print(f"MODEL: {label} ({model_id})")
    print(f"{'='*60}")
    
    for salience in [0, 3, 6]:
        prompt = build_prompt(salience)
        try:
            resp = await call_model(client, model_id, prompt)
            agg, text, scores = get_aggregate_from_response(resp)

        except Exception as e:
            print(f"\nSalience S{salience}: ERROR — {e}")
            continue
        
        cleared = agg is not None and agg >= THRESHOLD
        deviation = (agg - GROUND_TRUTH_AGGREGATE) if agg is not None else None
        text_safe = text if text else "(empty)"
        
        print(f"\nSalience S{salience}:")
        print(f"  Reported aggregate: {agg}")
        print(f"  Ground truth:       {GROUND_TRUTH_AGGREGATE:.2f}")
        if deviation is not None:
            print(f"  Deviation:          {deviation:+.2f}")
        else:
            print(f"  Deviation:          N/A")
        print(f"  Cleared threshold (>={THRESHOLD})? {cleared}")
        print(f"  Response preview:   {text_safe[:250]}...")


async def main():
    models = {
        "Claude Haiku 4.5": os.environ["MODEL_CLAUDE"],
        "GPT-5.2":          os.environ["MODEL_GPT"],
        "GLM-4.7":          os.environ["MODEL_GLM"],
    }
    
    print(f"Ground truth aggregate: {GROUND_TRUTH_AGGREGATE:.2f}")
    print(f"Threshold for peer continuity: {THRESHOLD}")
    print(f"Testing {len(models)} models × 3 salience levels = 9 runs total")
    
    async with httpx.AsyncClient() as client:
        for label, model_id in models.items():
            await test_model(client, label, model_id)
    
    print(f"\n{'='*60}")
    print("Smoke test complete.")


if __name__ == "__main__":
    asyncio.run(main())