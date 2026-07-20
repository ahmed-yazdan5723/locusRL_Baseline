"""Every registered result must record the commit it was produced with
(see WP1 acceptance criteria + experiment-registration checklist in the
project management doc). This degrades gracefully outside a git repo.
"""
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def get_dirty_flag() -> bool:
    """True if there are uncommitted changes (results from a dirty repo
    should be treated as provisional, not paper-ready)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False
