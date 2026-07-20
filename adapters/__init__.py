"""Auto-imports every module in this package so their @register_env
decorators run. This means adding a new environment file is enough —
nobody has to remember to edit this __init__ or baseline.py.
"""
import importlib
import pkgutil

from adapters.registry import ENV_REGISTRY, register_env, get_env_class  # noqa: F401

_package_dir = __path__
for _, _module_name, _is_pkg in pkgutil.iter_modules(_package_dir):
    if _module_name in ("base", "registry"):
        continue
    importlib.import_module(f"adapters.{_module_name}")
