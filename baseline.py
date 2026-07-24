#!/usr/bin/env python3
"""LocusRL WP1 baseline runner.

Usage:
    python baseline.py --agent deepseek --seed 42 --environment connectfour
    python baseline.py --agent random --seed 1 --environment connectfour \
        --opponents random heuristic --episodes 200

Adding a new model or environment does NOT require editing this file —
see agents/<name>.py and adapters/<name>.py for the registration pattern.
"""
import argparse
import json
import os
import sys
import time

from adapters import ENV_REGISTRY
from agents import AGENT_REGISTRY
from eval.runner import run_evaluation
from utils.logging_utils import get_logger
from utils.result_writer import write_result
from utils.seeding import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="LocusRL WP1: reproducible baseline evaluation runner."
    )
    parser.add_argument("--agent", required=True,
                         help=f"Agent to evaluate. Available: {sorted(AGENT_REGISTRY.keys())}")
    parser.add_argument("--environment", "--env", required=True,
                         help=f"Environment to run in. Use 'all' for every P0 env. Available: {['all', *sorted(ENV_REGISTRY.keys())]}")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed controlling env, agent RNG, and episode ordering.")
    parser.add_argument("--episodes", type=int, default=100,
                         help="Number of episodes per opponent.")
    parser.add_argument("--opponents", nargs="+", default=["random"],
                         help="One or more registered agent names to evaluate against.")
    parser.add_argument("--opponent-pool-version", default="v0-demo",
                         help="Tag recorded in the result file for provenance; bump this "
                              "when Yuxiang's real opponent pool replaces the demo one.")
    parser.add_argument("--checkpoint-path", default=None,
                         help="Path to a trained checkpoint, for SFT-only / GRPO agents.")
    parser.add_argument("--model", default=None,
                         help="Optional backend model identifier to pass to model-based agents.")
    parser.add_argument("--temperature", type=float, default=0.7,
                         help="Sampling temperature for model-based agents.")
    parser.add_argument("--top-p", type=float, default=0.95,
                         help="Nucleus sampling top-p value for model-based agents.")
    parser.add_argument("--ensemble", type=int, default=1,
                         help="Number of model samples/votes for ensemble-style agents.")
    parser.add_argument("--mask-actions", action=argparse.BooleanOptionalAction, default=True,
                         help="Whether model-based agents should use legal-action masking when supported.")
    parser.add_argument("--max-retries", type=int, default=2,
                         help="Maximum model retry attempts after invalid or unparseable actions.")
    parser.add_argument("--max-steps", type=int, default=200,
                         help="Safety cap for one episode, to catch non-terminating environment bugs.")
    parser.add_argument("--save-trajectories", action="store_true",
                         help="Also write output-dir/summary.json and output-dir/episodes.jsonl for scaffold-compatible inspection.")
    parser.add_argument("--output-dir", default="results",
                         help="Directory results are written under (per-environment subfolders).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _requested_environments(environment):
    if environment == "all":
        p0_envs = ["connect_four", "goofspiel", "leduc_poker"]
        return [env for env in p0_envs if env in ENV_REGISTRY]
    return [environment]


def _write_suite_files(output_dir, suite_payload, episode_rows):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    episodes_path = os.path.join(output_dir, "episodes.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_payload, f, indent=2, ensure_ascii=False)

    with open(episodes_path, "w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary_path, episodes_path


def main():
    args = parse_args()
    logger = get_logger("baseline", verbose=args.verbose)

    if args.agent not in AGENT_REGISTRY:
        logger.error("Unknown agent '%s'. Available: %s", args.agent, sorted(AGENT_REGISTRY.keys()))
        sys.exit(1)
    if args.environment != "all" and args.environment not in ENV_REGISTRY:
        logger.error("Unknown environment '%s'. Available: %s",
                      args.environment, ["all", *sorted(ENV_REGISTRY.keys())])
        sys.exit(1)
    unknown_opponents = [o for o in args.opponents if o not in AGENT_REGISTRY]
    if unknown_opponents:
        logger.error("Unknown opponent(s) %s. Available: %s",
                      unknown_opponents, sorted(AGENT_REGISTRY.keys()))
        sys.exit(1)

    set_global_seed(args.seed)

    environments = _requested_environments(args.environment)
    logger.info("Running agent=%s env=%s seed=%d episodes=%d opponents=%s",
                args.agent, environments, args.seed, args.episodes, args.opponents)

    agent_kwargs = {
        "model_name": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ensemble": args.ensemble,
        "mask_actions": args.mask_actions,
        "max_retries": args.max_retries,
    }

    suite_summaries = []
    suite_episodes = []
    result_paths = []

    for environment in environments:
        start_time = time.perf_counter()
        eval_results = run_evaluation(
            environment=environment,
            agent_name=args.agent,
            opponents=args.opponents,
            seed=args.seed,
            episodes=args.episodes,
            checkpoint_path=args.checkpoint_path,
            agent_kwargs=agent_kwargs,
            max_steps=args.max_steps,
        )
        duration_hours = (time.perf_counter() - start_time) / 3600.0

        gpu_hours = 0.0
        memory_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_hours = duration_hours
                memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except ImportError:
            pass

        summary = eval_results["summary"]
        latency = summary.get("latency", {})
        cost = {
            "gpu_hours": gpu_hours,
            "latency": {
                "mean_ms": latency.get("mean_ms", 0.0),
                "std_ms": latency.get("std_ms", 0.0),
                "max_ms": latency.get("max_ms", 0.0),
            },
            "environment_steps": summary.get("environment_steps", 0),
            "api_calls": summary.get("api_calls", 0),
            "memory": memory_gb,
        }

        config = {
            "agent": args.agent,
            "environment": environment,
            "seed": args.seed,
            "episodes": args.episodes,
            "opponents": args.opponents,
            "opponent_pool_version": args.opponent_pool_version,
            "checkpoint_path": args.checkpoint_path,
            "model": eval_results["agent_config"].get("model"),
            "temperature": eval_results["agent_config"].get("temperature"),
            "top_p": eval_results["agent_config"].get("top_p"),
            "ensemble": eval_results["agent_config"].get("ensemble"),
            "mask_actions": eval_results["agent_config"].get("mask_actions"),
            "max_retries": eval_results["agent_config"].get("max_retries"),
            "max_steps": args.max_steps,
        }

        filepath = write_result(
            output_dir=args.output_dir,
            environment=environment,
            agent_name=args.agent,
            seed=args.seed,
            config=config,
            opponent_pool_version=args.opponent_pool_version,
            checkpoint_path=args.checkpoint_path,
            metrics={
                "per_opponent": eval_results["per_opponent"],
                "summary": eval_results["summary"],
            },
            raw_episodes=eval_results["raw_episodes"],
            cost=cost,
        )
        result_paths.append(filepath)
        suite_summaries.append({
            "environment": environment,
            "summary": eval_results["summary"],
            "per_opponent": eval_results["per_opponent"],
            "result_path": filepath,
        })
        for row in eval_results["raw_episodes"]:
            suite_episodes.append({"environment": environment, **row})
        logger.info("Result written to %s", filepath)

    suite_payload = {
        "status": "ok",
        "agent": args.agent,
        "environments": environments,
        "opponents": args.opponents,
        "seed": args.seed,
        "episodes_per_opponent": args.episodes,
        "opponent_pool_version": args.opponent_pool_version,
        "checkpoint_path": args.checkpoint_path,
        "result_paths": result_paths,
        "summary": suite_summaries,
    }
    if args.environment == "all" or args.save_trajectories:
        summary_path, episodes_path = _write_suite_files(args.output_dir, suite_payload, suite_episodes)
        logger.info("Suite files written to %s and %s", summary_path, episodes_path)

    print(json.dumps(suite_payload["summary"], indent=2))


if __name__ == "__main__":
    main()
