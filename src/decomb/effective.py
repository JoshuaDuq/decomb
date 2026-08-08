"""Write down every setting a run actually used, and where each one came from.

A config file records what someone changed. It does not record what was in force, and the
difference is most of the file: a user who sets three values inherits eighty, and nothing
in their checkout says what those eighty were. Values the workflow computes are worse
again -- they appear in no file at all, so a reader cannot tell whether a width was chosen
or derived, or from what.

This module produces the missing statement: one row per setting, its value, and its
origin. Written beside a stage's outputs, it makes a run readable a year later without
the reader having to reconstruct the merge order or know which properties are computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from decomb.config import DecombConfig
from decomb.remove import RemovalSettings

#: Settings the workflow computes rather than reads, with the expression that produces
#: each. These appear in no config file, so without this they are invisible to a reader.
DERIVED: tuple[tuple[str, str], ...] = (
    ("removal.spectral_resolution_hz", "1.4382 / estimation_window_s"),
    (
        "removal.max_harmonic_residual_hz",
        "max_harmonic_residual_resolutions * spectral_resolution_hz",
    ),
    (
        "removal.max_fit_residual_rms_hz",
        "max_fit_residual_rms_resolutions * spectral_resolution_hz",
    ),
    ("removal.max_line_width_hz", "max_line_width_resolutions * spectral_resolution_hz"),
    ("removal.protected_bands_hz", "notch_bands + mains_notch_hz when exclude_mains"),
)


def _value_of(settings: RemovalSettings, dotted: str) -> Any:
    return getattr(settings, dotted.split(".", 1)[1])


def rows(config: DecombConfig, settings: RemovalSettings) -> list[tuple[str, str, str]]:
    """Every setting in force: name, value, origin."""
    table = [
        (key, _format(value), origin)
        for key, value, origin in config.effective()
    ]
    table.extend(
        (key, _format(_value_of(settings, key)), f"derived: {expression}")
        for key, expression in DERIVED
    )
    return sorted(table)


def _format(value: Any) -> str:
    if isinstance(value, float):
        # Enough digits to reproduce the value, without padding every integer-valued
        # setting with a tail of zeros.
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return str(value)


def write(
    config: DecombConfig,
    settings: RemovalSettings,
    destination: Path,
    *,
    stage: str,
) -> Path:
    """Write the effective configuration for one stage, and return the path."""
    table = rows(config, settings)
    width = max(len(key) for key, _, _ in table)
    value_width = min(max(len(value) for _, value, _ in table), 40)

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# decomb {stage}: every setting in force, and where it came from.",
        f"# config file: {config.source or '(none -- packaged defaults only)'}",
        "#",
        "# 'packaged defaults' means the value was inherited from src/decomb/defaults.yaml",
        "# and does not appear in the config file. 'derived' means the workflow computed it",
        "# from the settings shown beside it, so it appears in no file at all.",
        "",
    ]
    lines.extend(
        f"{key:<{width}}  {value:<{value_width}}  {origin}" for key, value, origin in table
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def summarise(config: DecombConfig, settings: RemovalSettings) -> str:
    """A one-line count for a stage's console output."""
    table = rows(config, settings)
    changed = sum(1 for _, _, origin in table if origin not in {"packaged defaults"})
    derived = sum(1 for _, _, origin in table if origin.startswith("derived"))
    return (
        f"{len(table)} setting(s) in force: {len(table) - changed} from the packaged "
        f"defaults, {changed - derived} from {config.source.name if config.source else 'none'}, "
        f"{derived} derived"
    )
