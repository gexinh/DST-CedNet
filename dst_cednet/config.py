"""Configuration paths are relative to root_dir, never the caller's cwd."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    root = Path(os.path.expandvars(os.path.expanduser(str(config.get("root_dir", "..")))))
    if not root.is_absolute():
        root = path.parent / root
    config["_root"] = str(root.resolve())
    config["_config_path"] = str(path)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    if not raw or "$" in raw:
        raise ValueError(f"Missing or unresolved configuration path: {value!r}")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(config.get("_root", Path.cwd())) / path
    return path.resolve()
