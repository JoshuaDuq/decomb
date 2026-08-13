"""Behavioral tests for the site-local overnight runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def pipeline(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repository_root = Path(__file__).resolve().parents[1]
    runner = tmp_path / "run_pipeline.sh"
    shutil.copy2(repository_root / "run_pipeline.sh", runner)

    source_root = tmp_path / "source"
    output_root = tmp_path / "derivative"
    report_root = tmp_path / "reports"
    for subject_index in range(15):
        subject = f"sub-{subject_index:04d}"
        eeg_dir = source_root / subject / "eeg"
        eeg_dir.mkdir(parents=True)
        for run_index in range(1, 7):
            (eeg_dir / f"{subject}_task-test_run-{run_index}_eeg.vhdr").touch()

    binary_dir = tmp_path / ".venv" / "bin"
    binary_dir.mkdir(parents=True)
    python_wrapper = binary_dir / "python"
    python_wrapper.write_text(
        f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    (tmp_path / "decomb.yaml").write_text(
        "\n".join(
            (
                "paths:",
                f'  bids_root: "{source_root}"',
                f'  output_root: "{output_root}"',
                f'  diagnosis_dir: "{report_root / "diagnosis"}"',
                f'  removal_dir: "{report_root / "removal"}"',
                "",
            )
        ),
        encoding="utf-8",
    )

    fake_decomb = binary_dir / "decomb"
    fake_decomb.write_text(
        f"""#!{sys.executable}
import os
import sys
from pathlib import Path

stage = sys.argv[1]
with Path(os.environ["STAGE_RECORD"]).open("a", encoding="utf-8") as handle:
    handle.write(stage + "\\n")
if stage == os.environ.get("FAIL_STAGE"):
    raise SystemExit(23)
if stage == "diagnose":
    Path(os.environ["DIAGNOSIS_DIR"]).mkdir(parents=True, exist_ok=True)
elif stage == "apply":
    output_root = Path(os.environ["OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)
    recordings = sorted(Path(os.environ["SOURCE_ROOT"]).glob("sub-*/eeg/*_eeg.vhdr"))
    rows = ["recording", *(recording.stem for recording in recordings)]
    (output_root / "line_notch_manifest.tsv").write_text("\\n".join(rows) + "\\n")
elif stage == "verify":
    removal_dir = Path(os.environ["REMOVAL_DIR"])
    removal_dir.mkdir(parents=True, exist_ok=True)
    recordings = sorted(Path(os.environ["SOURCE_ROOT"]).glob("sub-*/eeg/*_eeg.vhdr"))
    if os.environ.get("INCOMPLETE_VERIFICATION"):
        recordings = recordings[:-1]
    rows = ["recording", *(recording.stem for recording in recordings)]
    (removal_dir / "line_notch_verification.tsv").write_text("\\n".join(rows) + "\\n")
""",
        encoding="utf-8",
    )
    fake_decomb.chmod(0o755)

    environment = {
        **os.environ,
        "SOURCE_ROOT": str(source_root),
        "OUTPUT_ROOT": str(output_root),
        "REPORT_ROOT": str(report_root),
        "DIAGNOSIS_DIR": str(report_root / "diagnosis"),
        "REMOVAL_DIR": str(report_root / "removal"),
        "STAGE_RECORD": str(tmp_path / "stages.txt"),
        "PYTHONPATH": str(repository_root / "src"),
    }
    return runner, environment


def _run(runner: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(runner)],
        cwd=runner.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _stages(environment: dict[str, str]) -> list[str]:
    stage_record = Path(environment["STAGE_RECORD"])
    return stage_record.read_text(encoding="utf-8").splitlines()


def test_runs_current_stages_in_strict_order(pipeline):
    runner, environment = pipeline

    result = _run(runner, environment)

    assert result.returncode == 0, result.stderr
    assert _stages(environment) == ["diagnose", "apply", "verify", "psd"]


def test_stage_failure_stops_later_stages(pipeline):
    runner, environment = pipeline
    environment["FAIL_STAGE"] = "apply"

    result = _run(runner, environment)

    assert result.returncode == 23
    assert _stages(environment) == ["diagnose", "apply"]


def test_invalid_cohort_stops_before_replacing_outputs(pipeline):
    runner, environment = pipeline
    next(Path(environment["SOURCE_ROOT"]).glob("sub-*/eeg/*_eeg.vhdr")).unlink()
    output_root = Path(environment["OUTPUT_ROOT"])
    output_root.mkdir()
    sentinel = output_root / "old-result.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = _run(runner, environment)

    assert result.returncode != 0
    assert "Expected exactly 15 participants and 90 recordings" in result.stderr
    assert sentinel.is_file()
    assert not Path(environment["STAGE_RECORD"]).exists()


def test_archives_existing_derivative_and_reports_before_replacement(pipeline):
    runner, environment = pipeline
    output_root = Path(environment["OUTPUT_ROOT"])
    report_root = Path(environment["REPORT_ROOT"])
    output_root.mkdir()
    report_root.mkdir()
    (output_root / "old-derivative.txt").write_text("old", encoding="utf-8")
    (report_root / "old-report.txt").write_text("old", encoding="utf-8")

    result = _run(runner, environment)

    assert result.returncode == 0, result.stderr
    derivative_archives = list(output_root.parent.glob("derivative.archive-*"))
    report_archives = list(report_root.parent.glob("reports.archive-*"))
    assert len(derivative_archives) == 1
    assert len(report_archives) == 1
    assert derivative_archives[0].name.removeprefix("derivative.archive-") == (
        report_archives[0].name.removeprefix("reports.archive-")
    )
    assert (derivative_archives[0] / "old-derivative.txt").is_file()
    assert (report_archives[0] / "old-report.txt").is_file()
    assert output_root.is_dir()
    assert Path(environment["DIAGNOSIS_DIR"]).is_dir()
    assert Path(environment["REMOVAL_DIR"]).is_dir()


def test_incomplete_verification_coverage_stops_before_psd(pipeline):
    runner, environment = pipeline
    environment["INCOMPLETE_VERIFICATION"] = "1"

    result = _run(runner, environment)

    assert result.returncode != 0
    assert "verification coverage mismatch" in result.stderr
    assert _stages(environment) == ["diagnose", "apply", "verify"]
