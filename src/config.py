"""Project configuration loading and path resolution.

All paths inside ``configs/config.yaml`` are relative to the project root
(the directory that contains ``configs/``) unless they are absolute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Path to a config file. Defaults to ``configs/config.yaml``
            inside the project root.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or not a YAML mapping.
    """
    config_path = (
        Path(path) if path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Run commands from the project root, or pass --config explicitly."
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration file is empty or invalid: {config_path}")
    return config


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root if root is not None else PROJECT_ROOT) / path
