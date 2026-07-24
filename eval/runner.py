"""Core evaluation loop: agent vs. each opponent in the pool, for N
episodes, on a given environment/seed. This is the one place that
actually plays games — baseline.py just parses args and calls this.
"""
from typing import Dict, List
import time

from adapters.registry import get_env_class
from agents.base import BaseAgent
from agents.registry import get_agent_class
from eval.metrics import EpisodeResult, aggregate_across_opponents, compute_metrics_for_opponent
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "max_ms": 0.0,
            "count": 0,
        }
    mean = sum(latencies_ms) / len(latencies_ms)
    if len(latencies_ms) < 2:
        std = 0.0
    else:
        std = (sum((value - mean) ** 2 for value in latencies_ms) / (len(latencies_ms) - 1)) ** 0.5
    return {
        "mean_ms": mean,
        "std_ms": std,
        "max_ms": max(latencies_ms),
        "count": len(latencies_ms),
    }


def _termination_reason(obs, agent_player_idx: int, agent_return: float) -> str:
    reason = obs.info.get("termination_reason") if getattr(obs, "info", None) else None
    if reason:
        return str(reason).upper()
    if agent_return > 0:
        return "WIN"
    if agent_return < 0:
        return "LOSS"
    return "DRAW"


def play_episode(env, agent: BaseAgent, opponent: BaseAgent, agent_player_idx: int) -> EpisodeResult:
    """Plays one full episode with `agent` seated at `agent_player_idx`
    and `opponent` in the other seat(s). Returns per-episode stats.
    """
    obs = env.reset()
    agent.reset()
    opponent.reset()

    illegal_actions = 0
    total_actions = 0
    valid_actions = 0
    retry_count = 0
    parser_failures = 0
    invalid_json = 0
    backend_failures = 0
    api_calls = 0
    environment_steps = 0
    action_latencies_ms = []

    max_steps = getattr(env, "max_steps", 200)
    while not obs.done:
        if environment_steps >= max_steps:
            raise RuntimeError(f"{getattr(env, 'name', 'environment')} did not terminate within {max_steps} steps.")

        current = agent if obs.current_player == agent_player_idx else opponent
        started_at = time.perf_counter()
        action = current.act(obs)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        if action not in obs.legal_actions:
            if current is agent:
                illegal_actions += 1
                total_actions += 1
                action_latencies_ms.append(latency_ms)
                diagnostics = current.consume_action_diagnostics()
                retry_count += diagnostics.get("retry_count", 0)
                parser_failures += diagnostics.get("parser_failures", 0)
                invalid_json += diagnostics.get("invalid_json", 0)
                backend_failures += diagnostics.get("backend_failures", 0)
                api_calls += diagnostics.get("api_calls", 0)
            action = obs.legal_actions[0]  # deterministic safe fallback so the game can continue
        elif current is agent:
            total_actions += 1
            valid_actions += 1
            action_latencies_ms.append(latency_ms)
            diagnostics = current.consume_action_diagnostics()
            retry_count += diagnostics.get("retry_count", 0)
            parser_failures += diagnostics.get("parser_failures", 0)
            invalid_json += diagnostics.get("invalid_json", 0)
            backend_failures += diagnostics.get("backend_failures", 0)
            api_calls += diagnostics.get("api_calls", 0)

        obs = env.step(action)
        environment_steps += 1

    agent_return = obs.outcome.get(agent_player_idx, 0.0) if obs.outcome else 0.0
    return EpisodeResult(
        opponent=opponent.name,
        agent_return=agent_return,
        illegal_actions=illegal_actions,
        total_actions=total_actions,
        valid_actions=valid_actions,
        retry_count=retry_count,
        parser_failures=parser_failures,
        invalid_json=invalid_json,
        backend_failures=backend_failures,
        api_calls=api_calls,
        episode_length=environment_steps,
        environment_steps=environment_steps,
        termination_reason=_termination_reason(obs, agent_player_idx, agent_return),
        action_latency_ms=_latency_stats(action_latencies_ms),
    )


def run_evaluation(
    environment: str,
    agent_name: str,
    opponents: List[str],
    seed: int,
    episodes: int,
    checkpoint_path: str = None,
    agent_kwargs: Dict = None,
    max_steps: int = 200,
) -> Dict:
    """Runs `episodes` games of `agent_name` against each opponent in
    `opponents`, alternating who moves first, and returns a metrics dict
    with per-opponent breakdowns plus an aggregate summary.
    """
    agent_kwargs = agent_kwargs or {}

    EnvClass = get_env_class(environment)
    AgentClass = get_agent_class(agent_name)

    agent = AgentClass(seed=seed, checkpoint_path=checkpoint_path, **agent_kwargs)

    per_opponent_metrics = {}
    raw_episode_log = []

    for opp_name in opponents:
        OpponentClass = get_agent_class(opp_name)
        opponent = OpponentClass(seed=seed + 1)  # decorrelate opponent RNG from agent RNG

        env = EnvClass(seed=seed)
        env.max_steps = max_steps
        episode_results = []
        for ep in range(episodes):
            agent_player_idx = ep % env.num_players  # alternate seat to avoid first-move bias
            result = play_episode(env, agent, opponent, agent_player_idx)
            episode_results.append(result)
            raw_episode_log.append({
                "opponent": opp_name,
                "episode": ep,
                "agent_seat": agent_player_idx,
                "agent_return": result.agent_return,
                "termination_reason": result.termination_reason,
                "episode_length": result.episode_length,
                "environment_steps": result.environment_steps,
                "valid_actions": result.valid_actions,
                "illegal_actions": result.illegal_actions,
                "total_actions": result.total_actions,
                "retry_count": result.retry_count,
                "parser_failures": result.parser_failures,
                "invalid_json": result.invalid_json,
                "backend_failures": result.backend_failures,
                "api_calls": result.api_calls,
                "latency": result.action_latency_ms,
            })
        env.close()

        metrics = compute_metrics_for_opponent(episode_results)
        per_opponent_metrics[opp_name] = metrics
        logger.info("%s vs %s: win_rate=%.3f illegal_rate=%.3f (n=%d)",
                     agent_name, opp_name, metrics["win_rate"],
                     metrics["illegal_action_rate"], metrics["num_episodes"])

    summary = aggregate_across_opponents(per_opponent_metrics)

    return {
        "per_opponent": per_opponent_metrics,
        "summary": summary,
        "raw_episodes": raw_episode_log,
        "agent_config": agent.get_config(),
    }
