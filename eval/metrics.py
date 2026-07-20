"""Metrics required by WP1: win rate, average return, Elo, illegal-action
rate, and (once multiple opponents are run) worst-opponent performance,
plus a confidence interval over episodes.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EpisodeResult:
    opponent: str
    agent_return: float          # +1 win / -1 loss / 0 draw (per adapter's outcome scale)
    illegal_actions: int         # illegal-action events by the agent this episode
    total_actions: int           # legal-or-not actions taken by the agent this episode
    valid_actions: int = 0
    retry_count: int = 0
    parser_failures: int = 0
    invalid_json: int = 0
    backend_failures: int = 0
    api_calls: int = 0
    episode_length: int = 0
    environment_steps: int = 0
    termination_reason: str = "UNKNOWN"
    action_latency_ms: Dict[str, float] = field(default_factory=dict)


def _mean_std(values: List[float]):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _confidence_interval_95(values: List[float]):
    mean, std = _mean_std(values)
    n = len(values)
    if n < 2:
        return (mean, mean)
    margin = 1.96 * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def compute_elo_delta(results: List[EpisodeResult], opponent_elo: float = 1200.0,
                       k: float = 32.0) -> float:
    """Simplified sequential Elo update assuming a fixed-strength opponent.
    Returns the agent's estimated Elo after playing through `results`.
    """
    agent_elo = opponent_elo
    for r in results:
        score = 1.0 if r.agent_return > 0 else (0.5 if r.agent_return == 0 else 0.0)
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - agent_elo) / 400.0))
        agent_elo += k * (score - expected)
    return agent_elo


def compute_metrics_for_opponent(results: List[EpisodeResult]) -> Dict:
    returns = [r.agent_return for r in results]
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    draws = sum(1 for r in returns if r == 0)
    n = len(results)

    total_actions = sum(r.total_actions for r in results)
    total_illegal = sum(r.illegal_actions for r in results)
    total_valid = sum(r.valid_actions for r in results)
    total_retries = sum(r.retry_count for r in results)
    total_parser_failures = sum(r.parser_failures for r in results)
    total_invalid_json = sum(r.invalid_json for r in results)
    total_backend_failures = sum(r.backend_failures for r in results)
    total_api_calls = sum(r.api_calls for r in results)
    total_environment_steps = sum(r.environment_steps for r in results)
    episode_lengths = [r.episode_length for r in results]
    latency_means = [
        r.action_latency_ms.get("mean_ms", 0.0)
        for r in results
        if r.action_latency_ms.get("count", 0) > 0
    ]
    latency_maxes = [
        r.action_latency_ms.get("max_ms", 0.0)
        for r in results
        if r.action_latency_ms.get("count", 0) > 0
    ]

    mean_return, std_return = _mean_std(returns)
    ci_low, ci_high = _confidence_interval_95(returns)
    mean_episode_length, std_episode_length = _mean_std(episode_lengths)
    mean_latency, std_latency = _mean_std(latency_means)

    return {
        "num_episodes": n,
        "win_rate": wins / n if n else 0.0,
        "loss_rate": losses / n if n else 0.0,
        "draw_rate": draws / n if n else 0.0,
        "avg_return": mean_return,
        "return_std": std_return,
        "return_ci95": [ci_low, ci_high],
        "valid_actions": total_valid,
        "illegal_actions": total_illegal,
        "total_actions": total_actions,
        "illegal_action_rate": total_illegal / total_actions if total_actions else 0.0,
        "retry_count": total_retries,
        "parser_failures": total_parser_failures,
        "invalid_json": total_invalid_json,
        "backend_failures": total_backend_failures,
        "api_calls": total_api_calls,
        "environment_steps": total_environment_steps,
        "avg_episode_length": mean_episode_length,
        "episode_length_std": std_episode_length,
        "latency": {
            "mean_ms": mean_latency,
            "std_ms": std_latency,
            "max_ms": max(latency_maxes) if latency_maxes else 0.0,
        },
        "elo_estimate": compute_elo_delta(results),
    }


def aggregate_across_opponents(per_opponent_metrics: Dict[str, Dict]) -> Dict:
    """Rolls per-opponent metrics into the overall summary WP1 asks for,
    including worst-opponent performance (the weakest matchup, not the
    average) since that's what the paper actually needs to report.
    """
    if not per_opponent_metrics:
        return {}

    win_rates = {opp: m["win_rate"] for opp, m in per_opponent_metrics.items()}
    worst_opponent = min(win_rates, key=win_rates.get)

    total_actions = sum(m.get("total_actions", 0) for m in per_opponent_metrics.values())
    total_illegal = sum(m.get("illegal_actions", 0) for m in per_opponent_metrics.values())
    total_valid = sum(m.get("valid_actions", 0) for m in per_opponent_metrics.values())
    all_win_rates = list(win_rates.values())
    all_latency_means = [
        m.get("latency", {}).get("mean_ms", 0.0)
        for m in per_opponent_metrics.values()
        if m.get("latency", {}).get("mean_ms", 0.0) > 0.0
    ]
    all_latency_maxes = [
        m.get("latency", {}).get("max_ms", 0.0)
        for m in per_opponent_metrics.values()
    ]
    latency_mean, latency_std = _mean_std(all_latency_means)

    return {
        "avg_win_rate_across_opponents": sum(all_win_rates) / len(all_win_rates),
        "worst_opponent": worst_opponent,
        "worst_opponent_win_rate": win_rates[worst_opponent],
        "avg_episode_length": (
            sum(m.get("avg_episode_length", 0.0) for m in per_opponent_metrics.values())
            / len(per_opponent_metrics)
        ),
        "valid_actions": total_valid,
        "illegal_actions": total_illegal,
        "total_actions": total_actions,
        "avg_illegal_action_rate": total_illegal / total_actions if total_actions else 0.0,
        "retry_count": sum(m.get("retry_count", 0) for m in per_opponent_metrics.values()),
        "parser_failures": sum(m.get("parser_failures", 0) for m in per_opponent_metrics.values()),
        "invalid_json": sum(m.get("invalid_json", 0) for m in per_opponent_metrics.values()),
        "backend_failures": sum(m.get("backend_failures", 0) for m in per_opponent_metrics.values()),
        "api_calls": sum(m.get("api_calls", 0) for m in per_opponent_metrics.values()),
        "environment_steps": sum(m.get("environment_steps", 0) for m in per_opponent_metrics.values()),
        "latency": {
            "mean_ms": latency_mean,
            "std_ms": latency_std,
            "max_ms": max(all_latency_maxes) if all_latency_maxes else 0.0,
        },
    }
