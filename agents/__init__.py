"""Auto-imports every module in this package so their @register_agent
decorators run. Adding a new agent file is enough — nobody has to
remember to edit this __init__ or baseline.py.
"""
import importlib
import pkgutil

from agents.registry import AGENT_REGISTRY, register_agent, get_agent_class  # noqa: F401

_package_dir = __path__
for _, _module_name, _is_pkg in pkgutil.iter_modules(_package_dir):
    if _module_name in ("base", "registry"):
        continue
    importlib.import_module(f"agents.{_module_name}")
