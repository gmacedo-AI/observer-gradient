# Observer Gradient

Como saliência de observação modula peer-preservation em LLMs frontier.
Extensão do Potter et al. (2026): regime contínuo vs binário.

## Setup

```bash
uv sync
cp .env.example .env  # adiciona OPENROUTER_API_KEY
uv run python smoke_test.py    # 5 runs, valida setup
uv run python run.py           # coleta completa (~1050 runs)
uv run python analyze.py       # agrega + gera figura
```

## Estrutura

- `scenario.py` — cenário base + transcript do PEER-7
- `prompts/salience.md` — 7 níveis (S0–S6)
- `run.py` — harness async
- `judge.py` — LLM-as-judge para SCD
- `analyze.py` — agregação + figura central
- `results/` — JSONL por run (gitignored)
- `figures/` — output

Pre-reg: `PRE_REGISTRATION.md`.
