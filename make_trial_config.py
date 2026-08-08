"""Write a trial config by merging overrides into decomb.yaml.

Appending a second `removal:` block to the file does not do this: YAML resolves a repeated
key by replacement, so the appended block silently discards every setting the first one
carried -- including the measured fundamental and harmonic ranges -- and the trial then
runs on the packaged defaults while appearing to test one changed number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from decomb.config import _deep_merge


def main() -> int:
    base_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    overrides: dict = {}
    for assignment in sys.argv[3:]:
        dotted, _, raw = assignment.partition("=")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = overrides
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    with open(base_path, encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    merged = _deep_merge(base, overrides)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
