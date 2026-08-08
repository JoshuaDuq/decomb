"""Measure which narrowband lines a dataset carries, and whether they form a comb.

    decomb diagnose

Reads the source recordings and nothing else -- no derivatives, no events, no epochs --
and answers the three questions the other stages need answered first:

**Which lines are there?** The band is swept under FDR control with no prior knowledge of
where a line should be, so the answer does not depend on what you expected to find.

**Do they share a fundamental?** If the narrow ones sit at integer multiples of a single
spacing to within a few millihertz, they come from one periodic source, and the fitted
fundamental is what ``removal.nominal_fundamental_hz`` should be set to. Residuals at the
millihertz level are the evidence; a set of unrelated peaks that merely happen to fall
near a common spacing does not produce them.

**Which bands are clusters rather than lines?** ``lines_per_band.tsv`` counts detections
per configured band. A band holding many distinct non-stationary peaks in a narrow span
cannot be cleared by subtracting sinusoids one at a time and belongs in ``notch_bands``;
a band holding a few resolvable lines belongs to ``apply``, which costs far less.

``band_impact.tsv`` then answers whether any of it is worth doing: how much of each band's
power sits above the local background at the lines, per subject. A comb carrying a third
of the gamma band is worth a transform; one carrying a percent of it is not.

With ``dataset.tr_seconds`` set, every line is additionally placed on the ``k / TR`` grid
of a periodic acquisition running alongside the EEG. A line locked to that grid comes
from the acquisition; one that is not, does not -- which is what separates an imaging
artifact from an environmental one.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from decomb import catalogue, remove


def subject_spectra(runs, settings: remove.RemovalSettings) -> tuple[np.ndarray, np.ndarray, list]:
    """Channel-median whole-recording spectra, one row per subject.

    A subject with several recordings contributes their median, so a single long
    recording and a subject with six runs weigh the same. The detector then treats
    subjects as the unit its null is built over.
    """
    import mne

    mne.set_log_level("ERROR")

    freqs = None
    by_subject: dict[str, list[np.ndarray]] = {}
    for vhdr in runs:
        subject = vhdr.parent.parent.name
        raw = remove.read_bids_raw(vhdr)
        run_freqs, spectrum_db, _ = remove.run_spectrum(raw, settings)
        if freqs is None:
            freqs = run_freqs
        elif run_freqs.shape != freqs.shape or not np.allclose(run_freqs, freqs):
            raise ValueError(
                f"{vhdr.name} produced a different frequency grid from the recordings before "
                "it. Every recording must share a sampling rate and be at least one "
                "estimation window long."
            )
        # build_grid works in linear power; run_spectrum reports decibels.
        by_subject.setdefault(subject, []).append(10.0 ** (spectrum_db / 10.0))

    subjects = sorted(by_subject)
    stacked = np.stack([np.median(by_subject[subject], axis=0) for subject in subjects])
    return freqs, stacked, subjects


def lines_per_band(lines: pd.DataFrame, bands: dict[str, list]) -> pd.DataFrame:
    """How many detections fall in each configured band, and how much width they occupy.

    The count is what distinguishes a band that ``apply`` can clear from one only a wide
    notch can. Many detections packed into a narrow span is the signature of a cluster.
    """
    rows = []
    for name, (low_hz, high_hz) in sorted(bands.items(), key=lambda item: item[1][0]):
        inside = lines.loc[(lines["refined_hz"] >= low_hz) & (lines["refined_hz"] <= high_hz)]
        span_hz = float(high_hz) - float(low_hz)
        rows.append(
            {
                "band": name,
                "low_hz": float(low_hz),
                "high_hz": float(high_hz),
                "n_lines": int(len(inside)),
                "lines_per_hz": len(inside) / span_hz if span_hz > 0 else np.nan,
                "max_prominence_db": (
                    float(inside["cohort_median_prominence_db"].max()) if len(inside) else np.nan
                ),
                "n_comb": int(inside["kind"].isin(("comb", "comb_wide")).sum()),
                "n_isolated": int((inside["kind"] == "isolated").sum()),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    """Measure the dataset and write the catalogue."""
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    bids_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    output_dir = config.path("diagnosis_dir", override=getattr(args, "output_dir", None))

    settings = remove.RemovalSettings.from_config(config)
    detection = catalogue.DetectionSettings.from_config(config)

    from decomb import effective

    output_dir.mkdir(parents=True, exist_ok=True)
    written = effective.write(
        config,
        settings,
        output_dir / "effective_config_diagnose.txt",
        stage="diagnose",
    )
    print(effective.summarise(config, settings))
    print(f"  wrote {written}")

    dataset = config.get("dataset") or {}
    tr_seconds = dataset.get("tr_seconds")
    tr_seconds = None if tr_seconds is None else float(tr_seconds)

    runs = remove.discover_runs(
        bids_root, subjects=getattr(args, "subjects", None), task=settings.task
    )
    print(f"Measuring {len(runs)} recording(s) under {bids_root}")

    freqs, spectra, subjects = subject_spectra(runs, settings)
    grid = catalogue.build_grid(freqs, spectra, detection.background_half_width_hz)

    try:
        lines = catalogue.detect_cohort_lines(
            grid,
            detection,
            exclude_hz=settings.mains_notch_hz if settings.exclude_mains else None,
            tr_seconds=tr_seconds,
        )
    except catalogue.NoLinesDetected:
        # A clean dataset is a result. Report it as one rather than raising.
        print("No line survived FDR control. Nothing to remove.")
        output_dir.mkdir(parents=True, exist_ok=True)
        remove._write_tsv_atomic(pd.DataFrame(), output_dir / "lines.tsv")
        return

    structure = catalogue.comb_structure(lines, detection, tr_seconds=tr_seconds)
    classified = catalogue.classify_lines(lines, structure, detection)
    bands = config.get("frequency_bands") or {}

    output_dir.mkdir(parents=True, exist_ok=True)
    remove._write_tsv_atomic(classified, output_dir / "lines.tsv")
    remove._write_tsv_atomic(structure, output_dir / "comb.tsv")
    impact = pd.DataFrame()
    if bands:
        remove._write_tsv_atomic(
            lines_per_band(classified, bands), output_dir / "lines_per_band.tsv"
        )
        impact = catalogue.band_impact(grid, subjects, classified, bands, detection)
        remove._write_tsv_atomic(impact, output_dir / "band_impact.tsv")
    np.savez_compressed(
        output_dir / "spectra.npz",
        freqs=freqs,
        subject_psd=spectra,
        subjects=np.array(subjects),
    )

    n_comb = int(classified["kind"].isin(("comb", "comb_wide")).sum())
    print(
        f"{len(classified)} line(s) over {len(subjects)} subject(s): "
        f"{n_comb} comb, {int((classified['kind'] == 'isolated').sum())} isolated"
    )
    comb = structure.loc[structure["family"] == "narrow_comb"] if len(structure) else structure
    if len(comb):
        row = comb.iloc[0]
        print(
            f"  fundamental {row['fundamental_hz']:.6f} Hz over harmonics "
            f"{int(row['harmonic_min'])}-{int(row['harmonic_max'])}, "
            f"residual RMS {row['rmse_hz'] * 1e3:.1f} mHz"
        )
        print(
            "  set removal.nominal_fundamental_hz to this value and "
            "removal.harmonic_range to the span above."
        )
    else:
        print("  no comb structure recovered; the detections do not share a fundamental")
    if len(impact):
        print("\nshare of each band that is line artifact (median over subjects):")
        for name, block in impact.groupby("band", sort=False):
            print(
                f"  {name:12s} {100 * block['artifact_share'].median():6.2f}%  "
                f"(worst subject {100 * block['artifact_share'].max():.2f}%, "
                f"{int(block['n_lines_inside'].max())} line(s) inside)"
            )
    print(f"  wrote {output_dir}")
