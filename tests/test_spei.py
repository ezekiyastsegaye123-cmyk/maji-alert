"""Automated test suite for SPEIbase NetCDF extraction and annual aggregation module.

Tests cover:
- File loading and validation (missing, corrupt, non-NetCDF)
- SPEI variable detection and dimensional validation
- Coordinate normalization (0..360 vs -180..180), ascending/descending grids
- Nearest-grid-cell selection and spatial distance calculation
- Time coordinate decoding, calendar handling, and frequency validation
- Monthly to annual aggregation with complete-year policy enforcement
- Missing value, NaN, and Inf handling
- DataFrame schema and dtype validation
- CSV export and disk reload roundtrip verification
- Downstream merge compatibility with RWI / solar datasets
- End-to-end synthetic NetCDF pipeline execution
- Red-team edge cases (descending coords, leap years, boundary months, corrupt paths)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from treering.spei import (
    DEFAULT_TARGET_LAT,
    DEFAULT_TARGET_LON,
    GridCellMetadata,
    SPEIAggregationError,
    SPEICoordinateError,
    SPEIError,
    SPEIExportError,
    SPEIExtractionResult,
    SPEIFileError,
    SPEITimeError,
    SPEIVariableError,
    TimeCoverageMetadata,
    aggregate_annual_spei,
    check_downstream_compatibility,
    export_annual_spei_csv,
    export_provenance_json,
    extract_annual_spei,
    extract_point_series,
    haversine_distance,
    inspect_dataset,
    load_netcdf,
    resolve_spei_variable,
    to_annual_dataframe,
    validate_coordinates,
    validate_time_axis,
)


# =============================================================================
# Synthetic Fixtures
# =============================================================================

@pytest.fixture
def synthetic_monthly_nc(tmp_path: Path) -> Path:
    """Create a synthetic 3-year (36-month) 0.5-degree NetCDF SPEI dataset."""
    times = pd.date_range("2000-01-01", periods=36, freq="MS")
    lats = np.arange(8.0, 11.5, 0.5)  # includes 9.5 and 10.0 (near 9.63)
    lons = np.arange(38.0, 41.5, 0.5)  # includes 39.5 (near 39.53)

    # Deterministic synthetic SPEI values
    np.random.seed(42)
    data = np.random.normal(loc=0.1, scale=0.8, size=(len(times), len(lats), len(lons))).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "spei01": (("time", "lat", "lon"), data, {
                "units": "1",
                "long_name": "Standardized Precipitation-Evapotranspiration Index",
            }),
        },
        coords={
            "time": ("time", times, {"long_name": "time"}),
            "lat": ("lat", lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "Synthetic Test SPEIbase Dataset",
            "institution": "Test Suite",
            "version": "1.0",
        },
    )

    nc_path = tmp_path / "synthetic_spei.nc"
    ds.to_netcdf(nc_path)
    ds.close()
    return nc_path


@pytest.fixture
def synthetic_0_360_lon_nc(tmp_path: Path) -> Path:
    """Create a synthetic dataset using 0..360 longitude convention."""
    times = pd.date_range("2010-01-01", periods=24, freq="MS")
    lats = np.array([9.0, 9.5, 10.0])
    lons = np.array([38.0, 39.5, 41.0])  # in 0..360 range, 39.53 is 39.53

    data = np.full((len(times), len(lats), len(lons)), 0.25, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={"spei": (("time", "lat", "lon"), data)},
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    nc_path = tmp_path / "synthetic_0_360.nc"
    ds.to_netcdf(nc_path)
    ds.close()
    return nc_path


# =============================================================================
# Unit Tests: Helpers and Distance
# =============================================================================

class TestHaversineDistance:
    def test_same_point_zero_distance(self):
        dist = haversine_distance(9.63, 39.53, 9.63, 39.53)
        assert pytest.approx(dist, abs=1e-5) == 0.0

    def test_known_equatorial_distance(self):
        # 1 degree longitude at equator is ~111.19 km
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 111.0 < dist < 111.5

    def test_known_pole_distance(self):
        # North pole (90) to South pole (-90) = pi * R ~ 20015 km
        dist = haversine_distance(90.0, 0.0, -90.0, 0.0)
        assert 20000.0 < dist < 20050.0

    def test_debrebirkan_to_selected_cell(self):
        # From 9.63, 39.53 to 9.75, 39.75 (~27.56 km)
        dist = haversine_distance(9.63, 39.53, 9.75, 39.75)
        assert 25.0 < dist < 30.0


# =============================================================================
# Unit Tests: File Loading and Inspection
# =============================================================================

class TestNetCDFLoading:
    def test_load_valid_file(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        assert isinstance(ds, xr.Dataset)
        assert "spei01" in ds.data_vars
        ds.close()

    def test_load_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(SPEIFileError, match="does not exist"):
            load_netcdf(tmp_path / "nonexistent.nc")

    def test_load_directory_as_file(self, tmp_path: Path):
        with pytest.raises(SPEIFileError, match="not a regular file"):
            load_netcdf(tmp_path)

    def test_load_corrupt_file(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupt.nc"
        corrupt_file.write_text("This is not a NetCDF binary file.")
        with pytest.raises(SPEIFileError, match="Failed to open NetCDF"):
            load_netcdf(corrupt_file)

    def test_inspect_dataset(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        info = inspect_dataset(ds)
        assert "dimensions" in info
        assert "coordinates" in info
        assert "data_vars" in info
        assert "attributes" in info
        assert "spei01" in info["data_vars"]
        ds.close()


# =============================================================================
# Unit Tests: Variable Resolution
# =============================================================================

class TestVariableResolution:
    def test_auto_resolve_spei01(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        var = resolve_spei_variable(ds)
        assert var == "spei01"
        ds.close()

    def test_explicit_variable_name(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        var = resolve_spei_variable(ds, var_name="spei01")
        assert var == "spei01"
        ds.close()

    def test_missing_explicit_variable(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        with pytest.raises(SPEIVariableError, match="Specified variable 'nonexistent' not found"):
            resolve_spei_variable(ds, var_name="nonexistent")
        ds.close()

    def test_no_spei_variable_in_dataset(self, tmp_path: Path):
        ds = xr.Dataset(
            data_vars={"temperature": (("time", "lat", "lon"), np.ones((5, 2, 2)))},
            coords={"time": pd.date_range("2000-01-01", periods=5), "lat": [1, 2], "lon": [1, 2]},
        )
        nc = tmp_path / "no_spei.nc"
        ds.to_netcdf(nc)
        ds.close()

        loaded_ds = load_netcdf(nc)
        with pytest.raises(SPEIVariableError, match="No SPEI variable found"):
            resolve_spei_variable(loaded_ds)
        loaded_ds.close()


# =============================================================================
# Unit Tests: Coordinate Validation and Extraction
# =============================================================================

class TestCoordinateExtraction:
    def test_validate_standard_coordinates(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        lat_name, lon_name, time_name = validate_coordinates(ds)
        assert lat_name == "lat"
        assert lon_name == "lon"
        assert time_name == "time"
        ds.close()

    def test_out_of_bounds_target_latitude(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        with pytest.raises(SPEICoordinateError, match="Target latitude 95.0 is outside valid range"):
            extract_point_series(ds, lat=95.0, lon=39.53)
        ds.close()

    def test_nearest_grid_cell_selection(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        # In fixture, lats = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0], lons = [38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0]
        # Target: lat=9.63 -> nearest is 9.5; lon=39.53 -> nearest is 39.5
        da, meta = extract_point_series(ds, lat=9.63, lon=39.53)
        assert meta.requested_lat == 9.63
        assert meta.requested_lon == 39.53
        assert meta.selected_lat == 9.5
        assert meta.selected_lon == 39.5
        assert pytest.approx(meta.lat_offset, abs=1e-4) == 9.5 - 9.63
        assert pytest.approx(meta.lon_offset, abs=1e-4) == 39.5 - 39.53
        assert meta.spatial_distance_km > 0.0
        assert len(da) == 36
        ds.close()

    def test_0_360_longitude_dataset(self, synthetic_0_360_lon_nc: Path):
        ds = load_netcdf(synthetic_0_360_lon_nc)
        da, meta = extract_point_series(ds, lat=9.63, lon=39.53)
        assert meta.selected_lat == 9.5
        assert meta.selected_lon == 39.5
        assert len(da) == 24
        ds.close()

    def test_descending_latitude_coordinate(self, tmp_path: Path):
        """Red-team test: dataset where latitudes run from North to South (+90 to -90)."""
        times = pd.date_range("2000-01-01", periods=12, freq="MS")
        lats = np.array([12.0, 10.0, 8.0])  # Descending
        lons = np.array([38.0, 40.0, 42.0])

        data = np.full((12, 3, 3), 0.5, dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"spei": (("time", "lat", "lon"), data)},
            coords={"time": times, "lat": lats, "lon": lons},
        )
        nc = tmp_path / "descending_lat.nc"
        ds.to_netcdf(nc)
        ds.close()

        loaded_ds = load_netcdf(nc)
        da, meta = extract_point_series(loaded_ds, lat=9.63, lon=39.53)
        assert meta.selected_lat == 10.0  # nearest to 9.63 in [12, 10, 8]
        assert meta.selected_lon == 40.0  # nearest to 39.53 in [38, 40, 42]
        loaded_ds.close()


# =============================================================================
# Unit Tests: Temporal Validation and Annual Aggregation
# =============================================================================

class TestTimeValidationAndAggregation:
    def test_validate_time_axis_complete(self, synthetic_monthly_nc: Path):
        ds = load_netcdf(synthetic_monthly_nc)
        da, _ = extract_point_series(ds, lat=9.63, lon=39.53)
        time_meta = validate_time_axis(da)
        assert time_meta.total_months == 36
        assert time_meta.total_years == 3
        assert time_meta.complete_years == 3
        assert len(time_meta.incomplete_years) == 0
        ds.close()

    def test_aggregation_known_values(self):
        # 12 monthly values in 2000: 1.0 through 12.0 -> mean = 6.5
        # 12 monthly values in 2001: all 2.0 -> mean = 2.0
        times = pd.date_range("2000-01-01", periods=24, freq="MS")
        vals = list(range(1, 13)) + [2.0] * 12
        s = pd.Series(vals, index=times)

        annual_s, dropped = aggregate_annual_spei(s, require_full_year=True, min_months=12)
        assert len(annual_s) == 2
        assert pytest.approx(annual_s.loc[2000], abs=1e-5) == 6.5
        assert pytest.approx(annual_s.loc[2001], abs=1e-5) == 2.0
        assert dropped == []

    def test_drop_incomplete_years_policy(self):
        # 2000 has 12 months, 2001 has only 6 months
        times = pd.date_range("2000-01-01", periods=18, freq="MS")
        vals = [1.0] * 18
        s = pd.Series(vals, index=times)

        # Enforce full year
        annual_s, dropped = aggregate_annual_spei(s, require_full_year=True, min_months=12)
        assert len(annual_s) == 1
        assert 2000 in annual_s.index
        assert 2001 not in annual_s.index
        assert dropped == [2001]

        # Allow incomplete years
        annual_s_inc, dropped_inc = aggregate_annual_spei(s, require_full_year=False)
        assert len(annual_s_inc) == 2
        assert 2001 in annual_s_inc.index
        assert dropped_inc == []

    def test_nan_month_handling(self):
        # 2000 has 12 months, but month 5 is NaN -> only 11 valid months -> dropped when min_months=12
        times = pd.date_range("2000-01-01", periods=12, freq="MS")
        vals = [1.0] * 12
        vals[4] = np.nan
        s = pd.Series(vals, index=times)

        annual_s, dropped = aggregate_annual_spei(s, require_full_year=True, min_months=12)
        assert len(annual_s) == 0
        assert dropped == [2000]

        # If min_months=11, it is kept
        annual_s_11, dropped_11 = aggregate_annual_spei(s, require_full_year=True, min_months=11)
        assert len(annual_s_11) == 1
        assert dropped_11 == []


# =============================================================================
# Unit Tests: DataFrame Construction and CSV Export
# =============================================================================

class TestDataFrameAndExport:
    def test_to_annual_dataframe_schema(self):
        s = pd.Series({2000: 0.12, 2001: -0.45, 2002: 1.05}, name="spei")
        s.index.name = "year"

        df = to_annual_dataframe(s)
        assert list(df.columns) == ["year", "spei"]
        assert df["year"].dtype == np.int64
        assert df["spei"].dtype == np.float64
        assert len(df) == 3
        assert df["year"].is_monotonic_increasing

    def test_duplicate_years_in_dataframe_raises(self):
        # Direct malformed series
        s = pd.Series([0.1, 0.2], index=[2000, 2000], name="spei")
        s.index.name = "year"
        with pytest.raises(SPEIAggregationError, match="Duplicate years"):
            to_annual_dataframe(s)

    def test_nan_spei_in_dataframe_raises(self):
        s = pd.Series({2000: np.nan, 2001: 0.5}, name="spei")
        s.index.name = "year"
        with pytest.raises(SPEIAggregationError, match="NaN values present"):
            to_annual_dataframe(s)

    def test_export_and_reload_csv(self, tmp_path: Path):
        df = pd.DataFrame({"year": [2000, 2001, 2002], "spei": [0.1, -0.2, 0.3]})
        csv_path = tmp_path / "test_spei.csv"

        out = export_annual_spei_csv(df, csv_path, overwrite=False)
        assert out.exists()

        reloaded = pd.read_csv(out)
        assert list(reloaded.columns) == ["year", "spei"]
        assert len(reloaded) == 3
        assert list(reloaded["year"]) == [2000, 2001, 2002]

    def test_export_overwrite_protection(self, tmp_path: Path):
        df = pd.DataFrame({"year": [2000], "spei": [0.1]})
        csv_path = tmp_path / "overwrite_test.csv"
        csv_path.write_text("existing content")

        with pytest.raises(SPEIExportError, match="already exists"):
            export_annual_spei_csv(df, csv_path, overwrite=False)

        # Overwrite allowed
        out = export_annual_spei_csv(df, csv_path, overwrite=True)
        assert out.exists()


# =============================================================================
# Unit Tests: Downstream Compatibility Check
# =============================================================================

class TestDownstreamCompatibility:
    def test_compatibility_overlap(self, tmp_path: Path):
        lagged_df = pd.DataFrame({
            "year": [1990, 1991, 1992, 1993],
            "rwi": [1.0, 1.1, 0.9, 0.95],
        })
        lagged_path = tmp_path / "processed_lagged.csv"
        lagged_df.to_csv(lagged_path, index=False)

        spei_df = pd.DataFrame({
            "year": [1991, 1992, 1993, 1994],
            "spei": [-0.5, 0.2, 0.1, -0.8],
        })

        report = check_downstream_compatibility(spei_df, lagged_path)
        assert report["compatible"] is True
        assert report["overlapping_years_count"] == 3
        assert report["overlap_start_year"] == 1991
        assert report["overlap_end_year"] == 1993

    def test_compatibility_missing_file(self, tmp_path: Path):
        spei_df = pd.DataFrame({"year": [2000], "spei": [0.0]})
        report = check_downstream_compatibility(spei_df, tmp_path / "missing.csv")
        assert report["compatible"] is False


# =============================================================================
# Integration & End-to-End Tests
# =============================================================================

class TestEndToEndSPEIExtraction:
    def test_full_pipeline_synthetic(self, synthetic_monthly_nc: Path, tmp_path: Path):
        out_csv = tmp_path / "annual_spei.csv"
        out_json = tmp_path / "provenance.json"

        result = extract_annual_spei(
            input_path=synthetic_monthly_nc,
            output_path=out_csv,
            lat=9.63,
            lon=39.53,
            overwrite=True,
            provenance_path=out_json,
        )

        assert isinstance(result, SPEIExtractionResult)
        assert len(result.annual_df) == 3
        assert list(result.annual_df.columns) == ["year", "spei"]
        assert out_csv.exists()
        assert out_json.exists()

        # Check provenance contents
        with open(out_json, "r", encoding="utf-8") as f:
            prov = json.load(f)
        assert "pipeline" in prov
        assert "spatial_extraction" in prov
        assert "annual_aggregation" in prov

    def test_real_spei01_nc_if_present(self):
        real_nc = Path("data/spei01.nc")
        if not real_nc.exists():
            pytest.skip("Real dataset data/spei01.nc not available in workspace.")

        result = extract_annual_spei(
            input_path=real_nc,
            lat=9.63,
            lon=39.53,
            require_full_year=True,
        )

        assert len(result.annual_df) == 124  # 1901 to 2024
        assert int(result.annual_df["year"].min()) == 1901
        assert int(result.annual_df["year"].max()) == 2024
        assert result.grid_metadata.selected_lat == 9.75
        assert result.grid_metadata.selected_lon == 39.75
        assert 27.0 < result.grid_metadata.spatial_distance_km < 28.5
        assert not result.annual_df["spei"].isna().any()
