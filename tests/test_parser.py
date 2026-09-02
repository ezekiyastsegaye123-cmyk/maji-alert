"""Tests for treering.parser — Tucson .rwl file parsing."""

from __future__ import annotations

from pathlib import Path
import pytest

from treering.parser import parse_rwl, RWLParseError

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseRWLValid:
    """Tests for valid .rwl files."""

    def test_single_series(self) -> None:
        """Parse a file with one series and verify shape."""
        df = parse_rwl(FIXTURES / "test_single_series.rwl")
        assert set(df.columns) == {"series_id", "year", "ring_width"}
        assert df["series_id"].nunique() == 1
        assert df["series_id"].iloc[0] == "TST01"
        # 50 years: 1900–1949
        assert len(df) == 50
        assert df["year"].min() == 1900
        assert df["year"].max() == 1949

    def test_multi_series(self) -> None:
        """Parse a file with two series (and header lines)."""
        df = parse_rwl(FIXTURES / "test_multi_series.rwl")
        assert df["series_id"].nunique() == 2
        assert set(df["series_id"].unique()) == {"TST01", "TST02"}

    def test_multi_series_counts(self) -> None:
        """Verify correct observation counts per series."""
        df = parse_rwl(FIXTURES / "test_multi_series.rwl")
        counts = df.groupby("series_id").size()
        assert counts["TST01"] == 50  # 1900-1949
        assert counts["TST02"] == 30  # 1920-1949

    def test_years_are_integers(self) -> None:
        df = parse_rwl(FIXTURES / "test_single_series.rwl")
        assert df["year"].dtype in ("int64", "int32", int)

    def test_ring_widths_are_integers(self) -> None:
        df = parse_rwl(FIXTURES / "test_single_series.rwl")
        assert df["ring_width"].dtype in ("int64", "int32", int)

    def test_stop_marker_excluded(self) -> None:
        """The 999 stop marker must not appear as a measurement."""
        df = parse_rwl(FIXTURES / "test_single_series.rwl")
        assert 999 not in df["ring_width"].values

    def test_chronological_order(self) -> None:
        """Years within each series should be sorted ascending."""
        df = parse_rwl(FIXTURES / "test_multi_series.rwl")
        for _, grp in df.groupby("series_id"):
            years = grp["year"].values
            assert list(years) == sorted(years)

    def test_first_measurement_value(self) -> None:
        """Verify first value of TST01 is 500 at year 1900."""
        df = parse_rwl(FIXTURES / "test_single_series.rwl")
        row = df[df["year"] == 1900].iloc[0]
        assert row["ring_width"] == 500

    def test_real_rwl_eth001(self) -> None:
        """Parse the real eth001.rwl file from the africa directory."""
        real_file = FIXTURES.parent.parent / "africa" / "eth001.rwl"
        if not real_file.exists():
            pytest.skip("Real data file not available")
        df = parse_rwl(real_file)
        assert len(df) > 0
        assert df["series_id"].nunique() > 1

    def test_real_rwl_eth007(self) -> None:
        """Parse the real eth007.rwl file."""
        real_file = FIXTURES.parent.parent / "africa" / "eth007.rwl"
        if not real_file.exists():
            pytest.skip("Real data file not available")
        df = parse_rwl(real_file)
        assert len(df) > 0
        assert df["series_id"].nunique() > 1


class TestParseRWLInvalid:
    """Tests for error handling."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_rwl(tmp_path / "nonexistent.rwl")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.rwl"
        empty.write_text("")
        with pytest.raises(RWLParseError, match="empty"):
            parse_rwl(empty)

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        ws = tmp_path / "whitespace.rwl"
        ws.write_text("   \n\n   \n")
        with pytest.raises(RWLParseError, match="empty"):
            parse_rwl(ws)

    def test_no_data_lines(self, tmp_path: Path) -> None:
        """A file with only header lines should raise."""
        hdr = tmp_path / "header_only.rwl"
        hdr.write_text(
            "TESTRE 1 Test Site Name\n"
            "TESTRE 2 TestLand\n"
            "TESTRE 3 Test Author\n"
        )
        with pytest.raises(RWLParseError, match="No valid"):
            parse_rwl(hdr)

    def test_non_numeric_measurement(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_value.rwl"
        bad.write_text("TST01   1900   500   abc   460\n")
        with pytest.raises(RWLParseError, match="non-numeric"):
            parse_rwl(bad)
