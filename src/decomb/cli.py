"""``decomb <stage>`` -- the command line for every stage.

Stages are listed in the order they run. Each reads its settings from the same config
file; see ``decomb.config`` for how that file is found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decomb import __version__

STAGES = ("diagnose", "benchmark", "apply", "verify", "report", "notch", "psd")

STAGE_HELP = {
    "diagnose": "measure which lines exist, whether they form a comb, and what they cost",
    "benchmark": "check the removal against its preservation criteria -- run this first",
    "apply": "write the cleaned BIDS copy (refuses without a passing benchmark)",
    "verify": "re-measure what was written, sweeping the band with no knowledge of the targets",
    "report": "band-by-band outcome tables and figure",
    "notch": "optional: wide FIR notch over configured cluster bands",
    "psd": "before-and-after spectra of whatever exists",
}

EPILOG = """\
stages, in order:
""" + "".join(f"  {name:<10} {help}\n" for name, help in STAGE_HELP.items())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decomb",
        description="Audited removal of narrowband line and comb artifacts from continuous EEG.",
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
        "--report-dir", type=Path, default=None, help="benchmark/verify/report: where tables go"
    )
    parser.add_argument("--filter-length", default=None, help="override the removal filter length")
    parser.add_argument(
        "--mt-bandwidth", type=float, default=None, help="override the multitaper bandwidth"
    )
    return parser


def run_stage(args: argparse.Namespace) -> None:
    """Dispatch one stage to the module that implements it."""
    # A subject subset cannot certify or transform a dataset: the gates are decided over
    # the recordings jointly, and a partial write would leave the output root in a state
    # no provenance describes.
    if args.stage in {"benchmark", "apply", "verify", "notch"} and args.subjects:
        raise SystemExit(
            f"decomb {args.stage} must use every recording; --subjects cannot certify or "
            "transform a subset of the dataset."
        )

    if args.stage == "diagnose":
        from decomb import diagnose

        diagnose.run(args)
    elif args.stage == "report":
        from decomb import report

        args.removal_dir = args.report_dir
        report.run(args)
    elif args.stage == "notch":
        from decomb import notch

        notch.run(args)
    elif args.stage == "psd":
        from decomb import psd

        psd.run(args)
    else:
        from decomb import remove

        remove.run(args)


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
