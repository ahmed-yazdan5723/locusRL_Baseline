# LocusRL — WP1 Baseline Runner

Implements Yazdan's WP1 deliverable: a one-command, reproducible eval
runner that any baseline (random, rule-based, Prompt-Only LLM, SFT-only,
GRPO, ...) plugs into the same way, against the same opponent pool,
seeds, and metrics.

## Quick start

```bash
python baseline.py --agent random --seed 42 --environment connect_four
python baseline.py --agent heuristic --seed 42 --environment all \
  --opponents random heuristic --episodes 2 --save-trajectories
python baseline.py --agent deepseek --seed 42 --environment connect_four \
  --opponents random heuristic --episodes 200 \
  --model deepseek-chat --temperature 0.2 --top-p 0.95 --max-retries 2
python baseline.py --agent gemini --seed 42 --environment connect_four \
  --opponents random --episodes 1 \
  --model gemini-2.0-flash --temperature 0.2 --top-p 0.95 --max-retries 2
python baseline.py --agent qwen3b --seed 42 --environment connect_four \
  --opponents random --episodes 1 \
  --model /path/to/Qwen2.5-3B-Instruct --temperature 0.2 --top-p 0.95 --max-retries 2
```

Set `DEEPSEEK_API_KEY` in your environment for the DeepSeek agent to
actually call the model. Without it, the agent still runs — it falls
back to random legal actions and the run's `illegal_action_rate` will
correctly show ~1.0, rather than crashing.

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment for the
Gemini agent. `GOOGLE_API_KEY` takes precedence if both are set.

For local Qwen with HuggingFace, install `torch`, `transformers`, and
`accelerate` in the Python environment used to run this repo. Then point
`--model` at either a local model directory or a HuggingFace repo id that
already exists in your local cache. You can also set `QWEN_MODEL_PATH`
instead of passing `--model`.

Results are written to `results/<environment>/<agent>_seed<seed>_<timestamp>.json`.
For `--environment all` or `--save-trajectories`, the runner also writes
`summary.json` and `episodes.jsonl` directly under `--output-dir`.
Each file is organized as `metadata`, `config`, `metrics`, `cost`, and
`episodes`, with provenance, model settings, latency, action-validity
counters, environment steps, and per-episode termination details — no
numbers should ever be hand-copied from terminal output into a paper
table.

## Layout

```
baseline.py          CLI entry point — parses args, calls eval/runner.py, writes results
adapters/
  base.py            GameAdapter interface + Observation dataclass (matches GameAdapter v0.1 spec)
  registry.py         @register_env decorator
  connect_four.py       Scaffold-compatible Connect Four adapter
  leduc_poker.py        Scaffold-compatible Leduc Poker adapter
  goofspiel.py          Scaffold-compatible Goofspiel adapter
  connect_four_demo.py  Legacy standalone demo registered as `connectfour`
agents/
  base.py            BaseAgent interface
  registry.py         @register_agent decorator
  random_agent.py     Legal-Random baseline
  heuristic_agent.py  Minimal rule-based baseline (center-preference)
  llm_agent.py        Shared LLMPolicyAgent base: prompt building, parsing, retries
  deepseek_agent.py   DeepSeek backend (HTTP)
  gemini_agent.py     Gemini backend (HTTP)
  qwen_agent.py        Local HuggingFace Qwen backend
eval/
  metrics.py           win rate, avg return, Elo, latency, action validity, env-step counts, 95% CI
  runner.py             plays episodes, aggregates metrics
utils/
  seeding.py, logging_utils.py, git_utils.py, result_writer.py
```

## Using A Local HuggingFace Model

`qwen3b` loads through HuggingFace `transformers`. The default model id is
`Qwen/Qwen2.5-3B-Instruct`, but `--model` is usually better because it records
the exact local path or repo id in the result config:

```bash
python baseline.py --agent qwen3b --environment connect_four \
  --opponents random --episodes 1 \
  --model /absolute/path/to/Qwen2.5-3B-Instruct
```

If you downloaded the model through HuggingFace before, you can either pass the
same repo id and let Transformers find it in `~/.cache/huggingface`, or pass the
resolved snapshot folder. Useful commands:

```bash
huggingface-cli scan-cache
find ~/.cache/huggingface/hub -maxdepth 3 -type d -name 'models--Qwen*'
```

Helpful environment variables:

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct
export QWEN_LOCAL_FILES_ONLY=1
export QWEN_DEVICE_MAP=auto
export QWEN_TORCH_DTYPE=auto
export QWEN_MAX_NEW_TOKENS=80
```

Set `QWEN_LOCAL_FILES_ONLY=0` only when you want Transformers to download
missing files from HuggingFace.

## Adding a new model

1. Copy `agents/deepseek_agent.py` for an HTTP backend, or copy
   `agents/qwen_agent.py` for a local HuggingFace backend.
2. Implement `_call_backend(self, prompt) -> str`.
3. Register the class with `@register_agent("...")`.

Registration is automatic: `agents/__init__.py` imports every module in
`agents/`, so the `@register_agent("...")` decorator runs on import.
Same pattern for environments in `adapters/`.

## GameAdapter environments

The baseline now includes the LocusRL scaffold environments:
`connect_four`, `leduc_poker`, and `goofspiel`. The legacy
`connectfour` demo remains registered for backwards compatibility.

When Yuxiang's production GameAdapter replaces the scaffold:

1. Add `adapters/connect_four.py` (etc.) implementing `adapters/base.GameAdapter`
   (same `reset()` / `step()` / `Observation` contract).
2. Register it with `@register_env("<env_id>")` — if you keep the
   demo file too, rename one of the two registered names (registry
   raises on duplicate names on purpose, so this can't silently
   overwrite an entrypoint).
3. `baseline.py`, `eval/runner.py`, and every agent file are unaffected.

## Known simplifications (WP1 v0 — to revisit)

- Connect Four, Leduc Poker, and Goofspiel are implemented with the
  dependency-free scaffold environments; production adapters can replace
  them behind the same registry names.
- Elo is a simplified sequential update against a fixed-strength
  opponent (1200), not a full round-robin Elo across the pool.
- `heuristic` agent is a placeholder for the "rule/search agents" /
  CFR/MCTS/NFSP bullet in WP1 — swap in a real solver reference the
  same way (new file, `@register_agent`).
- Confidence intervals are computed over episodes within a single seed
  run; cross-seed aggregation (3–5 seeds per WP1/Go-No-Go criteria)
  should be a separate script that reads multiple result JSONs from
  `results/` — natural next piece once multi-seed runs exist.
# locusRL_Baseline
