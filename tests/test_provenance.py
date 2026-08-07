"""A cleaned dataset must say what cleaned it.

mirror_sidecars copies every sidecar byte-for-byte, dataset_description.json included, so
without this the cleaned root would declare DatasetType "raw" and credit only MNE-BIDS --
no removal settings, no code revision, no link back to the root it came from. BIDS asks
derivatives to carry GeneratedBy for exactly this reason: without it the delivered data
cannot be traced to the transformation that produced it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from decomb import remove


def test_the_cleaned_dataset_declares_what_made_it(tmp_path):
    source = tmp_path / "raw"
    (source / "sub-0001" / "eeg").mkdir(parents=True)
    (source / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "study",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "mne-bids"}],
            }
        )
    )
    output = tmp_path / "cleaned"
    output.mkdir()
    remove.mirror_sidecars(source, output)

    settings = remove.RemovalSettings()
    remove.write_derivative_description(output, source, settings, source_version="source-digest")

    described = json.loads((output / "dataset_description.json").read_text())
    assert described["DatasetType"] == "derivative"
    names = [entry.get("Name") for entry in described["GeneratedBy"]]
    assert any("decomb" in str(name) for name in names), names

    generated = next(e for e in described["GeneratedBy"] if "decomb" in str(e.get("Name")))
    assert generated.get("Version"), "no code revision recorded"
    assert generated["Parameters"]["settings_fingerprint"] == remove.settings_fingerprint(settings)
    assert described["SourceDatasets"], "no link back to the root this was made from"
    assert described["SourceDatasets"][0]["Version"] == "source-digest"


def test_the_raw_description_is_not_left_in_place(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "dataset_description.json").write_text(json.dumps({"DatasetType": "raw"}))
    output = tmp_path / "cleaned"
    output.mkdir()
    remove.mirror_sidecars(source, output)
    before = (output / "dataset_description.json").read_text()

    remove.write_derivative_description(
        output,
        source,
        remove.RemovalSettings(),
        source_version="source-digest",
    )
    assert (output / "dataset_description.json").read_text() != before


def test_malformed_source_description_surfaces_instead_of_being_replaced(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "dataset_description.json").write_text("not-json", encoding="utf-8")
    output = tmp_path / "cleaned"
    output.mkdir()
    remove.mirror_sidecars(source, output)

    with pytest.raises(json.JSONDecodeError):
        remove.write_derivative_description(
            output,
            source,
            remove.RemovalSettings(),
            source_version="source-digest",
        )


def test_each_manifest_row_identifies_its_input_plan_and_settings():
    row = remove.record_manifest_provenance(
        {"recording": "sub-0001_run-1"},
        input_digest="input-digest",
        plan_digest="plan-digest",
        fingerprint="settings-fingerprint",
    )

    assert row["input_digest"] == "input-digest"
    assert row["plan_digest"] == "plan-digest"
    assert row["settings_fingerprint"] == "settings-fingerprint"


class TestMeasuredBandCostTravelsWithTheData:
    """The delivered dataset must carry what the removal actually cost.

    The cost is a benchmark quantity -- it needs the broadband probe, which only the
    benchmark injects -- so `apply` has to read it from the benchmark it accepted, not
    from its own manifest, where the column never appears.
    """

    def test_the_cost_is_taken_from_the_benchmark(self):
        benchmark = pd.DataFrame(
            {"measured_band_attenuated_1db": [0.12, 0.16, 0.14]},
        )

        assert remove.measured_band_cost(benchmark) == {
            "measured_band_attenuated_1db_median": 0.14,
            "measured_band_attenuated_1db_worst": 0.16,
        }

    def test_a_benchmark_without_the_column_still_applies(self):
        """An older benchmark is missing the measurement, not disqualified by it."""
        assert remove.measured_band_cost(pd.DataFrame({"gate_passed": [True]})) is None

    def test_the_cost_reaches_the_written_description(self, tmp_path):
        source = tmp_path / "raw"
        source.mkdir()
        (source / "dataset_description.json").write_text(json.dumps({"DatasetType": "raw"}))
        output = tmp_path / "cleaned"
        output.mkdir()
        remove.mirror_sidecars(source, output)

        path = remove.write_derivative_description(
            output,
            source,
            remove.RemovalSettings(),
            source_version="digest",
            band_cost=remove.measured_band_cost(
                pd.DataFrame({"measured_band_attenuated_1db": [0.13, 0.15]})
            ),
        )

        described = json.loads(path.read_text(encoding="utf-8"))
        entry = next(e for e in described["GeneratedBy"] if "decomb" in str(e.get("Name")))
        assert entry["Parameters"]["band_cost"]["measured_band_attenuated_1db_worst"] == 0.15
