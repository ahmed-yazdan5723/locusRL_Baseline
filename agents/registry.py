"""Agent registry.

Adding a new model (e.g. qwen3b) = add one file to agents/ that calls
@register_agent("qwen3b") on its BaseAgent subclass. baseline.py and
eval/runner.py never change — agents/__init__.py auto-imports every
module in this package so the decorator runs.
"""
from typing import Dict, Type

AGENT_REGISTRY: Dict[str, Type] = {}


def register_agent(name: str):
    def decorator(cls):
        if name in AGENT_REGISTRY:
            raise ValueError(f"Agent '{name}' already registered "
                              f"(by {AGENT_REGISTRY[name].__name__})")
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def get_agent_class(name: str):
    if name not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY.keys())) or "(none registered)"
        raise KeyError(f"Unknown agent '{name}'. Available: {available}")
    return AGENT_REGISTRY[name]
