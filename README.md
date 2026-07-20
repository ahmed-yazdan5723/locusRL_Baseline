# LocusRL — WP1 Baseline Runner

Implements Yazdan's WP1 deliverable: a one-command, reproducible eval
runner that any baseline (random, rule-based, Prompt-Only LLM, SFT-only,
GRPO, ...) plugs into the same way, against the same opponent pool,
seeds, and metrics.

## Quick start

```bash
python baseline.py --agent random --seed 42 --environment connectfour
python baseline.py --agent deepseek --seed 42 --environment connectfour --opponents random heuristic --episodes 200
```

Set `DEEPSEEK_API_KEY` in your environment for the DeepSeek agent to
actually call the model. Without it, the agent still runs — it falls
back to random legal actions and the run's `illegal_action_rate` will
correctly show ~1.0, rather than crashing.

Results are written to `results/<environment>/<agent>_seed<seed>_<timestamp>.json`.
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
  connect_four_demo.py  Standalone demo Connect Four — placeholder until Yuxiang's real adapter lands
agents/
  base.py            BaseAgent interface
  registry.py         @register_agent decorator
  random_agent.py     Legal-Random baseline
  heuristic_agent.py  Minimal rule-based baseline (center-preference)
  llm_agent.py        Shared LLMPolicyAgent base: prompt building, parsing, retries
  deepseek_agent.py   DeepSeek backend (HTTP)
  qwen_agent.py        Qwen backend template — NotImplementedError stub to fill in
eval/
  metrics.py           win rate, avg return, Elo, latency, action validity, env-step counts, 95% CI
  runner.py             plays episodes, aggregates metrics
utils/
  seeding.py, logging_utils.py, git_utils.py, result_writer.py
```

## Adding a new model (e.g. Qwen2.5-3B)

1. Copy `agents/deepseek_agent.py` (HTTP backend) or fill in
   `agents/qwen_agent.py` (already stubbed and registered as `qwen3b`).
2. Implement `_call_backend(self, prompt) -> str`.
3. Run `python baseline.py --agent qwen3b ...` — nothing else changes.

Registration is automatic: `agents/__init__.py` imports every module in
`agents/`, so the `@register_agent("...")` decorator runs on import.
Same pattern for environments in `adapters/`.

## Swapping in the real GameAdapter

`adapters/connect_four_demo.py` is a stand-in. Once Yuxiang delivers
GameAdapter v0.1 for Connect Four / Leduc Poker / Goofspiel:

1. Add `adapters/connect_four.py` (etc.) implementing `adapters/base.GameAdapter`
   (same `reset()` / `step()` / `Observation` contract).
2. Register it with `@register_env("connectfour")` — if you keep the
   demo file too, rename one of the two registered names (registry
   raises on duplicate names on purpose, so this can't silently
   overwrite an entrypoint).
3. `baseline.py`, `eval/runner.py`, and every agent file are unaffected.

## Known simplifications (WP1 v0 — to revisit)

- Only Connect Four (demo) is implemented; Leduc Poker / Goofspiel wait
  on the real GameAdapter.
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
