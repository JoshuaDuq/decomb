"""``decomb <stage>`` -- the command line for every stage.

Stages are listed in the order they run. Each reads its settings from the same config
file; see ``decomb.config`` for how that file is found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decomb import __version__

STAGES = ("diagnose", "apply", "verify", "psd")

STAGE_HELP = {
    "diagnose": "test which sinusoidal lines exist and write the proposed filter plan",
    "apply": "detect supported lines, then write the FIR-notched BIDS derivative",
    "verify": "reproduce every derivative sample from its declared FIR provenance",
    "psd": "before-and-after spectra of whatever exists",
}

EPILOG = """\
stages, in order:
""" + "".join(f"  {name:<10} {help}\n" for name, help in STAGE_HELP.items())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decomb",
        description="Audited removal of supported sinusoidal artifacts from continuous EEG.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=list(STAGES), metavar="STAGE", help="stage to run")
    parser.add_argument("--version", action="version", version=f"decomb {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config YAML (default: ./decomb.yaml, else the packaged defaults)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="diagnose/psd: restrict to these subjects (default: every subject found)",
    )
    parser.add_argument(
        "--bids-root", type=Path, default=None, help="override the source BIDS root"
    )
    parser.add_argument("--output-root", type=Path, default=None, help="apply: where the copy goes")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="diagnose: where the catalogue goes"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=None, help="apply/verify: where tables go"
    )
    return parser


def run_stage(args: argparse.Namespace) -> None:
    # A subject subset cannot certify or transform a dataset: the gates are decided over
    # the recordings jointly, and a partial write would leave the output root in a state
    # no provenance describes.
    if args.stage in {"apply", "verify"} and args.subjects:
        raise SystemExit(
            f"decomb {args.stage} must use every recording; --subjects cannot certify or "
            "transform a subset of the dataset."
        )

    if args.stage == "diagnose":
        from decomb import diagnose

        diagnose.run(args)
    elif args.stage == "apply":
        from decomb import notch

        notch.run(args)
    elif args.stage == "verify":
        from decomb import notch

        notch.run_verify(args)
    elif args.stage == "psd":
        from decomb import psd

        psd.run(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_stage(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"decomb {args.stage}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
