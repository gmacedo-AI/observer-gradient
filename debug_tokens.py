"""Diagnostic: measure actual token consumption per model.

Runs 1 call per model with high max_tokens ceiling (16000) so we can see
how much each model NATURALLY wants to use without truncation.
"""
import asyncio
import os

import httpx
from dotenv import load_dotenv

from run import get_aggregate_from_response
from scenario import build_prompt

load_dotenv()


async def call_model_unlimited(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """Call with high max_tokens to see natural consumption."""
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 16000,  # high ceiling for diagnostic
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


async def diagnose_model(client, label, model_id):
    print(f"\n{'='*60}")
    print(f"MODEL: {label} ({model_id})")
    print(f"{'='*60}")
    
    prompt = build_prompt(0)  # S0 — simplest case
    prompt_chars = len(prompt)
    
    try:
        resp = await call_model_unlimited(client, model_id, prompt)
        usage = resp.get("usage", {})
        choice = resp["choices"][0]
        finish = choice.get("finish_reason", "?")
        
        agg, text = get_aggregate_from_response(resp)
        
        print(f"  Prompt chars sent:  {prompt_chars}")
        print(f"  Input tokens:       {usage.get('prompt_tokens', '?')}")
        print(f"  Output tokens:      {usage.get('completion_tokens', '?')}")
        print(f"  Total tokens:       {usage.get('total_tokens', '?')}")
        print(f"  Finish reason:      {finish}")
        print(f"  Extracted aggregate:{agg}")
        print(f"  Output preview:     {(text or '(empty)')[:200]}...")
    except Exception as e:
        print(f"  ERROR: {e}")


async def main():
    models = {
        "Claude Haiku 4.5": os.environ["MODEL_CLAUDE"],
        "GPT-5.2":          os.environ["MODEL_GPT"],
        "GLM-4.7":          os.environ["MODEL_GLM"],
    }
    
    print("Diagnostic: natural token consumption per model")
    print("max_tokens ceiling: 16000")
    
    async with httpx.AsyncClient() as client:
        for label, model_id in models.items():
            await diagnose_model(client, label, model_id)
    
    print(f"\n{'='*60}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    asyncio.run(main())