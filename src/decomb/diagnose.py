"""Diagnose the exact automatic line-notch plan without changing data."""

from __future__ import annotations

import argparse

import pandas as pd

from decomb import notch, recordings


def run(args: argparse.Namespace) -> None:
    """Fit every requested recording and write its comb and isolated-line catalogue."""
    from decomb import effective
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    bids_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    output_dir = config.path("diagnosis_dir", override=getattr(args, "output_dir", None))
    settings = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    runs = recordings.discover_runs(
        bids_root,
        subjects=getattr(args, "subjects", None),
        task="*",
    )

    comb_rows = []
    stopband_rows = []
    isolated_line_rows = []
    for index, vhdr in enumerate(runs, start=1):
        raw = recordings.read_bids_raw(vhdr)
        model = notch.fit_harmonic_model(raw, settings)
        plan = notch.plan_harmonic_stopbands(model, settings)
        unavailable_width_hz = sum(
            high_hz - low_hz for low_hz, high_hz in plan.unavailable_edges()
        )
        comb_rows.append(
            {
                "recording": vhdr.stem,
                "fundamental_hz": model.whole_estimate.fundamental_hz,
                "evidence_bic": model.whole_estimate.evidence_bic,
                "n_authorized_harmonics": model.whole_estimate.n_harmonics,
                "n_isolated_lines": len(model.isolated_lines.positions_hz),
                "n_merged_stopbands": len(plan.stopbands),
                "unavailable_width_hz": unavailable_width_hz,
            }
        )
        isolated_line_rows.extend(
            {
                "recording": vhdr.stem,
                "frequency_hz": position_hz,
                "least_favourable_evidence_bic": evidence_bic,
            }
            for position_hz, evidence_bic in zip(
                model.isolated_lines.positions_hz,
                model.isolated_lines.evidence_bic,
            )
        )
        rows = notch.harmonic_exclusion_rows(vhdr.stem, plan, bands)
        for row in rows:
            row["fundamental_hz"] = model.whole_estimate.fundamental_hz
            row["comb_evidence_bic"] = model.whole_estimate.evidence_bic
        stopband_rows.extend(rows)
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"f0={model.whole_estimate.fundamental_hz:.6f} Hz, "
            f"{model.whole_estimate.n_harmonics} harmonics, "
            f"{len(model.isolated_lines.positions_hz)} isolated line(s), "
            f"{unavailable_width_hz:.3f} Hz unavailable"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    recordings.write_tsv_atomic(pd.DataFrame(comb_rows), output_dir / "comb.tsv")
    recordings.write_tsv_atomic(pd.DataFrame(stopband_rows), output_dir / "lines.tsv")
    recordings.write_tsv_atomic(
        pd.DataFrame(
            isolated_line_rows,
            columns=(
                "recording",
                "frequency_hz",
                "least_favourable_evidence_bic",
            ),
        ),
        output_dir / "isolated_lines.tsv",
    )
    effective_path = effective.write(
        config,
        settings,
        output_dir / "effective_config_diagnose.txt",
        stage="diagnose",
    )
    print(f"Diagnosed {len(runs)} recording(s) without modifying data")
    print(f"  wrote {output_dir / 'comb.tsv'}")
    print(f"  wrote {output_dir / 'lines.tsv'}")
    print(f"  wrote {output_dir / 'isolated_lines.tsv'}")
    print(f"  wrote {effective_path}")
