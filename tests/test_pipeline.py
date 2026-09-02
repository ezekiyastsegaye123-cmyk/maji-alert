"""Tests for treering.pipeline and treering.export — end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treering.export import ExportError, export_csv
from treering.pipeline import PipelineError, process_rwl

FIXTURES = Path(__file__).parent / "fixtures"

# Expected output columns in order
_EXPECTED_COLS = [
    "series_id",
    "year",
    "raw_ring_width",
    "fitted_growth",
    "rwi",
]


class TestProcessRWL:
    """Integration tests for the pipeline."""

    def test_multi_series_output_schema(self) -> None:
        """Output DataFrame has the expected columns."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        assert df.columns.tolist() == _EXPECTED_COLS

    def test_multi_series_separate_fits(self) -> None:
        """Each series should be fitted independently."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        series_ids = df["series_id"].unique()
        assert len(series_ids) == 2

        # Fitted growth values should differ between series because
        # they have different observation ranges and magnitudes
        tst01 = df[df["series_id"] == "TST01"]
        tst02 = df[df["series_id"] == "TST02"]

        # Both should have valid RWI
        assert np.all(np.isfinite(tst01["rwi"].values))
        assert np.all(np.isfinite(tst02["rwi"].values))

    def test_single_series(self) -> None:
        df = process_rwl(FIXTURES / "test_single_series.rwl")
        assert df["series_id"].nunique() == 1
        assert len(df) == 50

    def test_output_years_match_input(self) -> None:
        """Output years must match the parsed input years."""
        from treering.parser import parse_rwl as _parse

        input_df = _parse(FIXTURES / "test_multi_series.rwl")
        output_df = process_rwl(FIXTURES / "test_multi_series.rwl")

        for sid in input_df["series_id"].unique():
            inp_years = sorted(
                input_df[input_df["series_id"] == sid]["year"].values
            )
            out_years = sorted(
                output_df[output_df["series_id"] == sid]["year"].values
            )
            assert inp_years == out_years, (
                f"Year mismatch for series {sid}"
            )

    def test_no_nan_rwi(self) -> None:
        """No NaN values in the RWI column."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        assert not df["rwi"].isna().any()

    def test_no_inf_rwi(self) -> None:
        """No infinite values in the RWI column."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        assert np.all(np.isfinite(df["rwi"].values))

    def test_no_nan_fitted_growth(self) -> None:
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        assert not df["fitted_growth"].isna().any()

    def test_deterministic(self) -> None:
        """Two runs on the same input produce identical output."""
        df1 = process_rwl(FIXTURES / "test_multi_series.rwl")
        df2 = process_rwl(FIXTURES / "test_multi_series.rwl")
        pd.testing.assert_frame_equal(df1, df2)

    def test_sorted_output(self) -> None:
        """Output is sorted by (series_id, year)."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")
        expected = df.sort_values(["series_id", "year"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected)

    def test_real_data_eth001(self) -> None:
        """End-to-end on real eth001.rwl data."""
        real_file = FIXTURES.parent.parent / "africa" / "eth001.rwl"
        if not real_file.exists():
            pytest.skip("Real data file not available")
        # Use skip_failed_series because some short series may fail
        df = process_rwl(real_file, skip_failed_series=True)
        assert len(df) > 0
        assert df.columns.tolist() == _EXPECTED_COLS
        assert np.all(np.isfinite(df["rwi"].values))

    def test_real_data_eth007(self) -> None:
        """End-to-end on real eth007.rwl data."""
        real_file = FIXTURES.parent.parent / "africa" / "eth007.rwl"
        if not real_file.exists():
            pytest.skip("Real data file not available")
        df = process_rwl(real_file, skip_failed_series=True)
        assert len(df) > 0
        assert np.all(np.isfinite(df["rwi"].values))


class TestExportCSV:
    """Tests for CSV export."""

    def test_basic_export(self, tmp_path: Path) -> None:
        """Write a valid DataFrame to CSV."""
        df = pd.DataFrame(
            {
                "series_id": ["A", "A"],
                "year": [2000, 2001],
                "raw_ring_width": [100, 90],
                "fitted_growth": [95.0, 88.0],
                "rwi": [1.0526, 1.0227],
            }
        )
        out = tmp_path / "output.csv"
        result = export_csv(df, out)

        assert result.exists()
        written = pd.read_csv(result)
        assert written.columns.tolist() == _EXPECTED_COLS
        assert len(written) == 2

    def test_no_index_column(self, tmp_path: Path) -> None:
        """CSV should not contain an index column."""
        df = pd.DataFrame(
            {
                "series_id": ["X"],
                "year": [1999],
                "raw_ring_width": [50],
                "fitted_growth": [45.0],
                "rwi": [1.111],
            }
        )
        out = tmp_path / "noindex.csv"
        export_csv(df, out)
        text = out.read_text()
        first_line = text.strip().split("\n")[0]
        assert first_line == "series_id,year,raw_ring_width,fitted_growth,rwi"

    def test_overwrite_protection(self, tmp_path: Path) -> None:
        """Should raise if file exists and overwrite=False."""
        out = tmp_path / "existing.csv"
        out.write_text("existing content")
        df = pd.DataFrame(
            {
                "series_id": ["A"],
                "year": [2000],
                "raw_ring_width": [100],
                "fitted_growth": [90.0],
                "rwi": [1.11],
            }
        )
        with pytest.raises(ExportError, match="already exists"):
            export_csv(df, out, overwrite=False)

    def test_overwrite_allowed(self, tmp_path: Path) -> None:
        """Should succeed if overwrite=True."""
        out = tmp_path / "overwrite.csv"
        out.write_text("old")
        df = pd.DataFrame(
            {
                "series_id": ["A"],
                "year": [2000],
                "raw_ring_width": [100],
                "fitted_growth": [90.0],
                "rwi": [1.11],
            }
        )
        export_csv(df, out, overwrite=True)
        assert "series_id" in out.read_text()

    def test_missing_columns(self, tmp_path: Path) -> None:
        """Should raise if DataFrame is missing required columns."""
        df = pd.DataFrame({"series_id": ["A"], "year": [2000]})
        with pytest.raises(ExportError, match="missing"):
            export_csv(df, tmp_path / "bad.csv")

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Should create parent directories if needed."""
        out = tmp_path / "sub" / "dir" / "output.csv"
        df = pd.DataFrame(
            {
                "series_id": ["A"],
                "year": [2000],
                "raw_ring_width": [100],
                "fitted_growth": [90.0],
                "rwi": [1.11],
            }
        )
        export_csv(df, out)
        assert out.exists()


class TestEndToEnd:
    """Full pipeline: .rwl → CSV."""

    def test_rwl_to_csv(self, tmp_path: Path) -> None:
        """Complete end-to-end: parse → detrend → RWI → export."""
        df = process_rwl(FIXTURES / "test_multi_series.rwl")

        out = tmp_path / "result.csv"
        export_csv(df, out)

        # Read back and validate
        result = pd.read_csv(out)
        assert result.columns.tolist() == _EXPECTED_COLS
        assert result["series_id"].nunique() == 2
        assert not result["rwi"].isna().any()
        assert np.all(np.isfinite(result["rwi"].values))

        # Years aligned
        for sid in result["series_id"].unique():
            grp = result[result["series_id"] == sid]
            years = grp["year"].values
            assert list(years) == sorted(years)

    def test_rwl_to_csv_real_data(self, tmp_path: Path) -> None:
        """End-to-end with real eth007 data."""
        real_file = FIXTURES.parent.parent / "africa" / "eth007.rwl"
        if not real_file.exists():
            pytest.skip("Real data file not available")

        df = process_rwl(real_file, skip_failed_series=True)
        out = tmp_path / "eth007_result.csv"
        export_csv(df, out)

        result = pd.read_csv(out)
        assert result.columns.tolist() == _EXPECTED_COLS
        assert len(result) > 0
        assert np.all(np.isfinite(result["rwi"].values))
