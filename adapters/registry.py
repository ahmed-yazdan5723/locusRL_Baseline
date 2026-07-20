"""Environment registry.

Adding a new environment = add one file to adapters/ that calls
@register_env("name") on its GameAdapter subclass. Nothing else in the
codebase (including baseline.py) needs to change — adapters/__init__.py
auto-imports every module in this package so the decorator runs.
"""
from typing import Dict, Type

ENV_REGISTRY: Dict[str, Type] = {}


def register_env(name: str):
    def decorator(cls):
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment '{name}' already registered "
                              f"(by {ENV_REGISTRY[name].__name__})")
        ENV_REGISTRY[name] = cls
        return cls
    return decorator


def get_env_class(name: str):
    if name not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys())) or "(none registered)"
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    return ENV_REGISTRY[name]
