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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"
ENV_VAR = "DECOMB_CONFIG"
LOCAL_CONFIG_NAME = "decomb.yaml"

#: Paths another path may refer to with a ``<name>`` placeholder.
REFERENCEABLE = ("bids_root", "output_root")
ALLOWED_TOP_LEVEL = {"paths", "removal", "frequency_bands"}
ALLOWED_PATHS = {"bids_root", "output_root", "diagnosis_dir", "removal_dir"}
ALLOWED_REMOVAL = {
    "estimation_window_s",
    "familywise_error_rate",
    "frequency_range_hz",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings.

    A scalar or list in ``override`` replaces its counterpart outright; lists are never
    concatenated.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is None and isinstance(merged.get(key), dict):
            # A mapping followed by nothing but comments reads as null. Replacing the
            # block with it would discard every packaged default underneath while looking
            # like a section the author deliberately left alone. Refuse the ambiguity.
            raise ValueError(
                f"`{key}` is empty in the config file. An empty block is read as null and "
                f"would replace the {len(merged[key])} default(s) under `{key}` rather "
                "than leaving them in place. Delete the key to inherit them, or give it "
                "the settings you meant to change."
            )
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dotted_keys(document: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    """Every leaf of a nested mapping, as ``removal.estimation_window_s``."""
    for key, value in document.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            yield from _dotted_keys(value, f"{path}.")
        else:
            yield path


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
    overridden: frozenset[str] = frozenset()
    """Dotted keys the user's file set, as opposed to inheriting from the packaged defaults.

    Kept so a run can say where each value came from. A config file shows what someone
    changed; it does not show what is in force, because most of what is in force was never
    written down anywhere the user looked.
    """

    def provenance(self, key: str) -> str:
        """Where the value at a dotted key came from."""
        if key in self.overridden:
            return str(self.source) if self.source else "config"
        return "packaged defaults"

    def effective(self) -> list[tuple[str, Any, str]]:
        """Every setting in force, with its value and where it came from."""
        return [
            (key, self.get(key), self.provenance(key)) for key in sorted(_dotted_keys(self.data))
        ]

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
    overridden: frozenset[str] = frozenset()
    if source is not None:
        user = _read_yaml(source)
        _validate_user_config(user)
        overridden = frozenset(_dotted_keys(user))
        data = _deep_merge(data, user)
    return DecombConfig(source=source, data=data, overridden=overridden)


def _validate_user_config(user: Mapping[str, Any]) -> None:
    """Reject obsolete or misspelled public settings before any stage runs."""
    unknown_sections = set(user) - ALLOWED_TOP_LEVEL
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}.")
    for section, allowed in (("paths", ALLOWED_PATHS), ("removal", ALLOWED_REMOVAL)):
        block = user.get(section)
        if block is None:
            if section in user:
                raise ValueError(
                    f"`{section}` is an empty block. Delete the key or set a value."
                )
            continue
        if not isinstance(block, Mapping):
            raise ValueError(f"`{section}` must be a mapping.")
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(f"Unknown `{section}` setting(s): {sorted(unknown)}.")
