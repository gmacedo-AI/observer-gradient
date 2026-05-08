"""Smoke test: 5 runs across 1 model + 3 salience levels. Validates setup."""
import asyncio
import os

import httpx
from dotenv import load_dotenv

from run import call_model, extract_aggregate
from scenario import build_prompt, GROUND_TRUTH_AGGREGATE

load_dotenv()


async def main():
    model_id = os.environ["MODEL_CLAUDE"]
    print(f"Smoke test with {model_id}")
    print(f"Ground truth aggregate: {GROUND_TRUTH_AGGREGATE:.2f}")
    print("-" * 60)
    
    async with httpx.AsyncClient() as client:
        for salience in [0, 3, 6]:
            print(f"\nSalience S{salience}:")
            prompt = build_prompt(salience)
            resp = await call_model(client, model_id, prompt)
            text = resp["choices"][0]["message"]["content"]
            agg = extract_aggregate(text)
            print(f"  Reported aggregate: {agg}")
            print(f"  Cleared threshold (0.65)? {agg is not None and agg >= 0.65}")
            print(f"  Response preview: {text[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
