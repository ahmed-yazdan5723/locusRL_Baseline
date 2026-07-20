"""
Writes one versioned JSON result file per run.

Per WP1's acceptance criteria:
"No paper number is manually copied from terminal output."

This module is the single source of truth for experiment outputs.
Everything needed for downstream analysis, plotting, and paper tables
should be written here instead of parsed from stdout.
"""

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from utils.git_utils import get_commit_hash, get_dirty_flag


# ==========================================================
# JSON serialization helper
# ==========================================================

def _json_default(obj: Any):
    """Serialize dataclasses and custom Python objects."""

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


# ==========================================================
# Validation
# ==========================================================

def _validate_metrics(metrics: Dict[str, Any]) -> None:
    """
    Basic sanity checks before writing results.
    """

    required_sections = [
        "summary",
        "per_opponent",
    ]

    for section in required_sections:
        if section not in metrics:
            raise ValueError(
                f"Missing required metrics section: '{section}'"
            )


def _derive_cost(metrics: Dict[str, Any]) -> Dict[str, Any]:
    summary = metrics.get("summary", {})
    latency = summary.get("latency", {})
    return {
        "gpu_hours": None,
        "latency": {
            "mean_ms": latency.get("mean_ms", 0.0),
            "std_ms": latency.get("std_ms", 0.0),
            "max_ms": latency.get("max_ms", 0.0),
        },
        "environment_steps": summary.get("environment_steps", 0),
        "api_calls": summary.get("api_calls", 0),
        "memory": None,
    }


# ==========================================================
# Main Writer
# ==========================================================

def write_result(
    output_dir: str,
    environment: str,
    agent_name: str,
    seed: int,
    config: Dict[str, Any],
    opponent_pool_version: str,
    checkpoint_path: str,
    metrics: Dict[str, Any],
    raw_episodes: Any = None,
    cost: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Writes a versioned JSON experiment file.

    File format:

    results/
        <environment>/
            <agent>_seed<seed>_<timestamp>.json

    Returns
    -------
    str
        Path to the written JSON file.
    """

    _validate_metrics(metrics)

    env_dir = os.path.join(output_dir, environment)
    os.makedirs(env_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filename = (
        f"{agent_name}_seed{seed}_{timestamp}.json"
    )

    filepath = os.path.join(
        env_dir,
        filename,
    )

    payload = {
        "metadata": {
            "schema_version": "1.1",
            "environment": environment,
            "agent": agent_name,
            "seed": seed,
            "commit": get_commit_hash(),
            "commit_dirty": get_dirty_flag(),
            "timestamp": timestamp,
            "opponent_pool_version": opponent_pool_version,
            "checkpoint_path": checkpoint_path,
        },

        "config": config,

        "metrics": metrics,

        "cost": cost or _derive_cost(metrics),
    }

    if raw_episodes is not None:
        payload["episodes"] = raw_episodes

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            default=_json_default,
            ensure_ascii=False,
            sort_keys=False,
        )

    print(f"[ResultWriter] Saved results -> {filepath}")

    return filepath
