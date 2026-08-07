"""Load the one YAML file decomb runs on.

Every stage reads its settings from the same file. There is no second configuration to
keep in step, and no key is required: the packaged :file:`defaults.yaml` supplies every
value, and a user's file is merged over it, so a config that changes one number needs to
contain only that number.

Resolution order, highest priority first:

1. a command-line flag
2. ``--config PATH``, or the ``DECOMB_CONFIG`` environment variable
3. ``decomb.yaml`` in the working directory
4. the packaged defaults

Path values may refer to another path with a placeholder -- ``"<bids_root>_clean"`` --
so a derived root can be named without repeating the drive. Relative paths stay relative
to the working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"
ENV_VAR = "DECOMB_CONFIG"
LOCAL_CONFIG_NAME = "decomb.yaml"

#: Paths another path may refer to with a ``<name>`` placeholder.
REFERENCEABLE = ("bids_root", "output_root", "notched_root")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings.

    A scalar or list in ``override`` replaces its counterpart outright. Lists are not
    concatenated: ``notch_bands`` given in a user's file means those bands and no others,
    which is the only reading that lets a user turn a default off.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config_path(config_path: str | Path | None = None) -> Path | None:
    """Locate the user's config: explicit path, then the env var, then the local file."""
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"No config file at {resolved}")
        return resolved

    from_env = os.getenv(ENV_VAR)
    if from_env:
        resolved = Path(from_env).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{ENV_VAR} points at {resolved}, which does not exist")
        return resolved

    local = Path.cwd() / LOCAL_CONFIG_NAME
    return local.resolve() if local.is_file() else None


@dataclass(frozen=True)
class DecombConfig:
    """The settings one run works from."""

    source: Path | None
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a dotted key, e.g. ``removal.estimation_window_s``."""
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def path(self, name: str, *, override: str | Path | None = None) -> Path:
        """Resolve a named path, expanding any ``<other_path>`` placeholders in it."""
        if override is not None:
            return Path(override).expanduser()
        raw = (self.data.get("paths") or {}).get(name)
        if raw is None:
            where = self.source or "the packaged defaults"
            raise KeyError(f"paths.{name} is not set in {where}")
        text = str(raw)
        for token in REFERENCEABLE:
            placeholder = f"<{token}>"
            if placeholder in text:
                if token == name:
                    raise ValueError(f"paths.{name} refers to itself")
                text = text.replace(placeholder, str(self.path(token)))
        return Path(text).expanduser()


def load_config(config_path: str | Path | None = None) -> DecombConfig:
    """Load the packaged defaults and merge the user's file over them."""
    data = _read_yaml(DEFAULTS_PATH)
    source = resolve_config_path(config_path)
    if source is not None:
        data = _deep_merge(data, _read_yaml(source))
    return DecombConfig(source=source, data=data)
