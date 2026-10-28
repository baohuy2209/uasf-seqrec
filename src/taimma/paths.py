"""Repository-relative paths for the standalone source checkout."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    """Resolve a checkout-relative path, preserving an explicit absolute path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
