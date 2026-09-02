"""Tests for treering.solar_lag — RWI / Sunspot solar-cycle lag analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treering.solar_lag import (
    DataValidationError,
    LagAnalysisError,
    OptimalLagResult,
    SolarLagAnalysisResult,
    apply_centered_smoothing,
    build_final_aligned_dataframe,
    calculate_lag_correlations,
    export_results,
    load_rwi_data,
    load_sunspot_data,
    merge_datasets,
    run_solar_lag_analysis,
    select_optimal_lag,
    standardize_series,
    validate_years,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================================
# 1. Data Ingestion & Validation Tests
# ============================================================================


class TestDataLoading:
    """Tests for loading and validating RWI and Sunspot data."""

    def test_load_rwi_valid_df(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901, 1902], "rwi": [1.05, 0.95, 1.10]})
        loaded = load_rwi_data(df)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["year", "rwi"]
        assert loaded["year"].dtype in (int, np.int64, np.int32)

    def test_load_rwi_missing_columns(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901]})
        with pytest.raises(DataValidationError, match="missing required"):
            load_rwi_data(df)

    def test_load_rwi_negative_values(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901], "rwi": [1.0, -0.5]})
        with pytest.raises(DataValidationError, match="negative"):
            load_rwi_data(df)

    def test_load_rwi_non_finite(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901], "rwi": [1.0, np.nan]})
        with pytest.raises(DataValidationError, match="non-finite"):
            load_rwi_data(df)

    def test_load_rwi_duplicate_years(self) -> None:
        df = pd.DataFrame({"year": [1900, 1900], "rwi": [1.0, 1.1]})
        with pytest.raises(DataValidationError, match="Duplicate"):
            load_rwi_data(df)

    def test_load_sunspot_silso_format(self, tmp_path: Path) -> None:
        """Test loading SILSO semicolon-delimited fractional year format."""
        content = (
            "1700.5;   8.3; -1.0;    -1;1\n"
            "1701.5;  18.3; -1.0;    -1;1\n"
            "1702.5;  26.7; -1.0;    -1;1\n"
        )
        file_path = tmp_path / "sunspot_silso.csv"
        file_path.write_text(content)

        loaded = load_sunspot_data(file_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["year", "sunspot"]
        assert loaded["year"].tolist() == [1700, 1701, 1702]
        np.testing.assert_allclose(loaded["sunspot"].values, [8.3, 18.3, 26.7])

    def test_load_sunspot_header_format(self, tmp_path: Path) -> None:
        content = "year,Sunspot Number\n1950,150.2\n1951,120.5\n"
        file_path = tmp_path / "sunspot_header.csv"
        file_path.write_text(content)

        loaded = load_sunspot_data(file_path)
        assert len(loaded) == 2
        assert loaded["year"].tolist() == [1950, 1951]
        assert loaded["sunspot"].tolist() == [150.2, 120.5]

    def test_load_sunspot_negative_cleaning(self) -> None:
        """Ensure negative missing markers (e.g. -1.0) are cleaned."""
        df = pd.DataFrame({"year": [1800, 1801, 1802], "sunspot": [50.0, -1.0, 60.0]})
        loaded = load_sunspot_data(df)
        assert len(loaded) == 2
        assert 1801 not in loaded["year"].values


# ============================================================================
# 2. Merge & Year Validation Tests
# ============================================================================


class TestMergeAndYearValidation:
    """Tests for dataset merging and temporal validation."""

    def test_merge_matching_years(self) -> None:
        rwi = pd.DataFrame({"year": [1950, 1951, 1952], "rwi": [1.0, 1.1, 0.9]})
        sn = pd.DataFrame({"year": [1950, 1951, 1952, 1953], "sunspot": [100.0, 120.0, 80.0, 50.0]})
        merged = merge_datasets(rwi, sn)

        assert len(merged) == 3
        assert list(merged.columns) == ["year", "rwi", "sunspot"]
        assert merged["year"].tolist() == [1950, 1951, 1952]

    def test_merge_no_overlap(self) -> None:
        rwi = pd.DataFrame({"year": [1800, 1801], "rwi": [1.0, 1.1]})
        sn = pd.DataFrame({"year": [1900, 1901], "sunspot": [100.0, 120.0]})
        with pytest.raises(DataValidationError, match="No overlapping years"):
            merge_datasets(rwi, sn)

    def test_validate_years_gap_detection(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901, 1902, 1905]})
        missing = validate_years(df)
        assert missing == [1903, 1904]

    def test_validate_years_continuous(self) -> None:
        df = pd.DataFrame({"year": [1900, 1901, 1902, 1903]})
        missing = validate_years(df)
        assert missing == []


# ============================================================================
# 3. 11-Year Centered Moving Average Smoothing Tests
# ============================================================================


class TestCenteredSmoothing:
    """Tests for 11-year centered moving average."""

    def test_exact_11_point_smoothing(self) -> None:
        """11 constant values of 10.0 should have middle value 10.0 and boundaries NaN."""
        arr = np.full(11, 10.0)
        smoothed = apply_centered_smoothing(arr, window=11)

        assert len(smoothed) == 11
        # First 5 are NaN
        assert np.all(np.isnan(smoothed[:5]))
        # Last 5 are NaN
        assert np.all(np.isnan(smoothed[6:]))
        # Center (index 5) is exactly 10.0
        assert smoothed[5] == 10.0

    def test_linear_sequence_smoothing(self) -> None:
        """Centered average of arithmetic progression 1..11 is 6.0 at center."""
        arr = np.arange(1.0, 12.0)
        smoothed = apply_centered_smoothing(arr, window=11)
        assert smoothed[5] == 6.0

    def test_short_series_returns_nan(self) -> None:
        arr = np.ones(8)
        smoothed = apply_centered_smoothing(arr, window=11)
        assert np.all(np.isnan(smoothed))

    def test_smoothing_with_year_gaps(self) -> None:
        """Verify that missing years prevent invalid moving average across gaps."""
        # 11 points total, but with a gap: 1900..1904 (5 points), gap at 1905..1909, 1910..1915 (6 points)
        years = np.array([1900, 1901, 1902, 1903, 1904, 1910, 1911, 1912, 1913, 1914, 1915])
        vals = np.ones(len(years)) * 5.0
        smoothed = apply_centered_smoothing(vals, window=11, years=years)
        # Because there are no 11 consecutive calendar years, all should be NaN
        assert np.all(np.isnan(smoothed))


# ============================================================================
# 4. Standardization Tests
# ============================================================================


class TestStandardization:
    """Tests for z-score standardization."""

    def test_standardize_properties(self) -> None:
        np.random.seed(42)
        vals = np.random.normal(loc=50.0, scale=10.0, size=100)
        z = standardize_series(vals, ddof=1)

        np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(z, ddof=1), 1.0, atol=1e-12)

    def test_standardize_with_nans(self) -> None:
        vals = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
        z = standardize_series(vals, ddof=1)

        assert np.isnan(z[0]) and np.isnan(z[4])
        valid_z = z[1:4]
        np.testing.assert_allclose(np.mean(valid_z), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(valid_z, ddof=1), 1.0, atol=1e-12)

    def test_constant_series_raises(self) -> None:
        vals = np.array([5.0, 5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="zero-variance"):
            standardize_series(vals)


# ============================================================================
# 5. Lag Correlation & Alignment Tests
# ============================================================================


class TestLagAnalysis:
    """Tests for lag correlation calculation and optimal lag selection."""

    def test_known_lag_recovery(self) -> None:
        """Construct synthetic data with known lag tau=2 and verify recovery."""
        years = np.arange(1800, 1950)
        # 11-year solar cycle
        sn = 100.0 + 80.0 * np.sin(2 * np.pi * years / 11.0)

        # RWI driven by SN with a 2-year lag: RWI(t) = SN(t - 2)
        sn_series = pd.Series(sn, index=years)
        rwi = sn_series.shift(2).bfill().values

        df = pd.DataFrame({"year": years, "rwi": rwi, "sunspot": sn})
        df["rwi_smoothed"] = apply_centered_smoothing(df["rwi"].values, window=11, years=years)
        df["sunspot_smoothed"] = apply_centered_smoothing(df["sunspot"].values, window=11, years=years)
        df["rwi_z"] = standardize_series(df["rwi_smoothed"], ddof=1)
        df["sunspot_z"] = standardize_series(df["sunspot_smoothed"], ddof=1)

        lag_results = calculate_lag_correlations(df, max_lag=5)
        opt = select_optimal_lag(lag_results)

        assert opt.optimal_lag == 2
        assert opt.optimal_correlation > 0.95
        assert opt.correlation_direction == "positive"

    def test_negative_correlation_optimal_selection(self) -> None:
        """Verify that a strong negative correlation is chosen over a weak positive correlation."""
        lag_df = pd.DataFrame(
            {
                "lag": [0, 1, 2, 3, 4, 5],
                "correlation": [0.10, 0.20, -0.65, 0.30, -0.10, 0.05],
                "p_value": [0.5, 0.2, 0.001, 0.1, 0.5, 0.8],
                "n_observations": [100, 100, 100, 100, 100, 100],
            }
        )
        opt = select_optimal_lag(lag_df)
        assert opt.optimal_lag == 2
        assert opt.optimal_correlation == -0.65
        assert opt.correlation_direction == "negative"

    def test_insufficient_pairs(self) -> None:
        df = pd.DataFrame(
            {
                "year": [1900, 1901],
                "rwi_z": [0.5, -0.5],
                "sunspot_z": [1.0, -1.0],
            }
        )
        lag_results = calculate_lag_correlations(df, max_lag=2)
        assert np.isnan(lag_results.loc[lag_results["lag"] == 0, "correlation"].iloc[0])


# ============================================================================
# 6. End-to-End Pipeline & Export Tests
# ============================================================================


class TestEndToEndSolarLag:
    """Integration test running full pipeline and file export."""

    def test_full_pipeline_with_export(self, tmp_path: Path) -> None:
        # Create synthetic RWI and Sunspot CSVs
        years = np.arange(1900, 1960)
        t = (years - 1900).astype(float)
        sn_vals = 80.0 + 50.0 * np.cos(2 * np.pi * t / 11.0)
        rwi_vals = 1.0 + 0.2 * np.cos(2 * np.pi * (t - 3) / 11.0)

        rwi_file = tmp_path / "rwi.csv"
        pd.DataFrame({"year": years, "rwi": rwi_vals}).to_csv(rwi_file, index=False)

        sn_file = tmp_path / "sunspot.csv"
        pd.DataFrame({"year": years, "sunspot": sn_vals}).to_csv(sn_file, index=False)

        out_dir = tmp_path / "results"
        res = run_solar_lag_analysis(
            rwi_input=rwi_file,
            sunspot_input=sn_file,
            max_lag=5,
            output_dir=out_dir,
            overwrite=True,
        )

        assert isinstance(res, SolarLagAnalysisResult)
        assert isinstance(res.optimal_lag, OptimalLagResult)

        # Check export files exist
        aligned_csv = out_dir / "processed_lagged_data.csv"
        lag_csv = out_dir / "lag_correlation_results.csv"
        summary_json = out_dir / "analysis_summary.json"

        assert aligned_csv.exists()
        assert lag_csv.exists()
        assert summary_json.exists()

        # Validate aligned CSV content
        df_aligned = pd.read_csv(aligned_csv)
        expected_cols = [
            "year",
            "rwi",
            "sunspot",
            "rwi_smoothed",
            "sunspot_smoothed",
            "rwi_z",
            "sunspot_z",
            "sunspot_z_lagged",
            "optimal_lag",
        ]
        for col in expected_cols:
            assert col in df_aligned.columns

        # Validate summary JSON
        with open(summary_json, "r") as f:
            summary = json.load(f)
        assert summary["smoothing_window_years"] == 11
        assert "optimal_lag_years" in summary

    def test_real_sunspot_data_with_generated_rwi(self, tmp_path: Path) -> None:
        """Integration test with project's real SN_y_tot_V2.0.csv and real rwl."""
        from treering.pipeline import process_rwl

        rwl_path = Path(__file__).parent.parent / "africa" / "eth007.rwl"
        sunspot_path = Path(__file__).parent.parent / "SN_y_tot_V2.0.csv"

        if not rwl_path.exists() or not sunspot_path.exists():
            pytest.skip("Real data files missing")

        # 1. Process tree-ring RWL to RWI DataFrame
        rwi_df = process_rwl(rwl_path, skip_failed_series=True)
        assert len(rwi_df) > 0

        # 2. Run solar lag analysis
        res = run_solar_lag_analysis(
            rwi_input=rwi_df,
            sunspot_input=sunspot_path,
            max_lag=5,
            output_dir=tmp_path / "real_test_out",
            overwrite=True,
        )

        assert len(res.lag_correlations) == 6  # lags 0 to 5
        assert res.optimal_lag.optimal_lag in range(6)
        assert np.isfinite(res.optimal_lag.optimal_correlation)
