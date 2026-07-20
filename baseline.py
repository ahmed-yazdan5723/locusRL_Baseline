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
import sys

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
    parser.add_argument("--environment", required=True,
                         help=f"Environment to run in. Available: {sorted(ENV_REGISTRY.keys())}")
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
    parser.add_argument("--output-dir", default="results",
                         help="Directory results are written under (per-environment subfolders).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("baseline", verbose=args.verbose)

    if args.agent not in AGENT_REGISTRY:
        logger.error("Unknown agent '%s'. Available: %s", args.agent, sorted(AGENT_REGISTRY.keys()))
        sys.exit(1)
    if args.environment not in ENV_REGISTRY:
        logger.error("Unknown environment '%s'. Available: %s",
                      args.environment, sorted(ENV_REGISTRY.keys()))
        sys.exit(1)
    unknown_opponents = [o for o in args.opponents if o not in AGENT_REGISTRY]
    if unknown_opponents:
        logger.error("Unknown opponent(s) %s. Available: %s",
                      unknown_opponents, sorted(AGENT_REGISTRY.keys()))
        sys.exit(1)

    set_global_seed(args.seed)

    logger.info("Running agent=%s env=%s seed=%d episodes=%d opponents=%s",
                args.agent, args.environment, args.seed, args.episodes, args.opponents)

    eval_results = run_evaluation(
        environment=args.environment,
        agent_name=args.agent,
        opponents=args.opponents,
        seed=args.seed,
        episodes=args.episodes,
        checkpoint_path=args.checkpoint_path,
    )

    config = {
        "agent": args.agent,
        "environment": args.environment,
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
    }

    filepath = write_result(
        output_dir=args.output_dir,
        environment=args.environment,
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
    )

    logger.info("Result written to %s", filepath)
    print(json.dumps(eval_results["summary"], indent=2))


if __name__ == "__main__":
    main()
