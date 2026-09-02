"""SPEIbase NetCDF extraction and annual ground-truth dataset engineering module.

Extracts point-location SPEI time series (e.g., Debrebirkan Selassie, Ethiopia),
validates coordinates and time axes, converts monthly SPEI-1 to calendar-year
annual mean ground-truth series, validates data integrity, and exports clean CSVs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Standard target coordinates for Debrebirkan Selassie, Ethiopia
DEFAULT_TARGET_LAT = 9.63
DEFAULT_TARGET_LON = 39.53
EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class SPEIError(Exception):
    """Base exception for all SPEI extraction errors."""


class SPEIFileError(SPEIError):
    """Raised when an input NetCDF file cannot be found, opened, or parsed."""


class SPEIVariableError(SPEIError):
    """Raised when the SPEI variable cannot be found or has invalid dimensions."""


class SPEICoordinateError(SPEIError):
    """Raised when dataset coordinates are invalid, missing, or out of bounds."""


class SPEITimeError(SPEIError):
    """Raised when time coordinates are malformed, ambiguous, or non-chronological."""


class SPEIAggregationError(SPEIError):
    """Raised when annual aggregation encounters invalid or unrecoverable data."""


class SPEIExportError(SPEIError):
    """Raised when output CSV export or verification fails."""


# =============================================================================
# Data Containers
# =============================================================================

@dataclass(frozen=True)
class GridCellMetadata:
    """Metadata for the nearest grid cell selected during spatial extraction."""

    requested_lat: float
    requested_lon: float
    selected_lat: float
    selected_lon: float
    lat_offset: float
    lon_offset: float
    spatial_distance_km: float
    selection_method: str = "nearest"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class TimeCoverageMetadata:
    """Metadata for temporal validation and coverage."""

    start_date: str
    end_date: str
    total_months: int
    total_years: int
    complete_years: int
    incomplete_years: list[int]
    calendar: str
    time_frequency: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class SPEIExtractionResult:
    """Complete container for SPEI extraction results and provenance."""

    annual_df: pd.DataFrame
    monthly_series: pd.Series
    grid_metadata: GridCellMetadata
    time_metadata: TimeCoverageMetadata
    provenance: dict[str, Any]
    output_path: Optional[Path] = None

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary dictionary."""
        return {
            "requested_location": {
                "latitude": self.grid_metadata.requested_lat,
                "longitude": self.grid_metadata.requested_lon,
            },
            "selected_grid_cell": {
                "latitude": self.grid_metadata.selected_lat,
                "longitude": self.grid_metadata.selected_lon,
                "distance_km": round(self.grid_metadata.spatial_distance_km, 2),
            },
            "temporal_range": {
                "start_year": int(self.annual_df["year"].min()) if not self.annual_df.empty else None,
                "end_year": int(self.annual_df["year"].max()) if not self.annual_df.empty else None,
                "complete_annual_observations": len(self.annual_df),
                "dropped_incomplete_years": self.time_metadata.incomplete_years,
            },
            "statistics": {
                "mean_spei": float(self.annual_df["spei"].mean()) if not self.annual_df.empty else None,
                "std_spei": float(self.annual_df["spei"].std()) if not self.annual_df.empty else None,
                "min_spei": float(self.annual_df["spei"].min()) if not self.annual_df.empty else None,
                "max_spei": float(self.annual_df["spei"].max()) if not self.annual_df.empty else None,
            },
            "output_path": str(self.output_path) if self.output_path else None,
        }


# =============================================================================
# Helper Utilities
# =============================================================================

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of point 2 in decimal degrees.

    Returns
    -------
    float
        Great-circle distance in kilometers.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_KM * c)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file.

    Parameters
    ----------
    file_path : str or Path
        Target file path.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# Scientific Data Loading & Inspection
# =============================================================================

def load_netcdf(input_path: Union[str, Path]) -> xr.Dataset:
    """Open a NetCDF dataset using xarray with validation.

    Parameters
    ----------
    input_path : str or Path
        Path to the NetCDF (.nc) file.

    Returns
    -------
    xr.Dataset
        Opened xarray Dataset.

    Raises
    ------
    SPEIFileError
        If file does not exist, is not readable, or is not a valid NetCDF.
    """
    path = Path(input_path)
    if not path.exists():
        raise SPEIFileError(f"Input NetCDF file does not exist: {path.resolve()}")
    if not path.is_file():
        raise SPEIFileError(f"Input path is not a regular file: {path.resolve()}")

    try:
        ds = xr.open_dataset(path, use_cftime=None)
        logger.info("Successfully opened NetCDF dataset: %s", path.name)
        return ds
    except Exception as exc:
        raise SPEIFileError(f"Failed to open NetCDF dataset '{path}': {exc}") from exc


def inspect_dataset(ds: xr.Dataset) -> dict[str, Any]:
    """Inspect dataset metadata, dimensions, coordinates, variables, and attributes.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset to inspect.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with dimensions, coordinates, variables, and global attributes.
    """
    return {
        "dimensions": {dim: int(size) for dim, size in ds.sizes.items()},
        "coordinates": list(ds.coords.keys()),
        "data_vars": list(ds.data_vars.keys()),
        "attributes": dict(ds.attrs),
    }


def resolve_spei_variable(
    ds: xr.Dataset, var_name: Optional[str] = None
) -> str:
    """Identify and validate the SPEI variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset.
    var_name : str, optional
        Explicit variable name to look for. If None, scans common SPEI names.

    Returns
    -------
    str
        Verified variable name.

    Raises
    ------
    SPEIVariableError
        If no suitable SPEI variable is found or variable is invalid.
    """
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise SPEIVariableError(
                f"Specified variable '{var_name}' not found in dataset. "
                f"Available variables: {list(ds.data_vars.keys())}"
            )
        candidate = var_name
    else:
        # Standard candidate priority list for SPEI products
        candidates = ["spei01", "spei", "spei_01", "SPEI01", "SPEI"]
        found = [c for c in candidates if c in ds.data_vars]
        if not found:
            # Fallback: scan variables with 'spei' in name
            matching = [v for v in ds.data_vars if "spei" in v.lower()]
            if matching:
                found = matching
            else:
                raise SPEIVariableError(
                    f"No SPEI variable found in dataset. Available variables: {list(ds.data_vars.keys())}"
                )
        candidate = found[0]

    da = ds[candidate]
    # Check dimensionality: should have at least 1 temporal and spatial dimensions
    if len(da.dims) < 2:
        raise SPEIVariableError(
            f"Variable '{candidate}' has unexpected dimensions {da.dims}. "
            "Expected at least temporal and spatial coordinates."
        )

    logger.info("Resolved SPEI variable: '%s' (dims=%s, shape=%s)", candidate, da.dims, da.shape)
    return candidate


# =============================================================================
# Spatial Coordinate Extraction & Normalization
# =============================================================================

def validate_coordinates(ds: xr.Dataset) -> tuple[str, str, str]:
    """Validate latitude, longitude, and time coordinates and return their names.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset.

    Returns
    -------
    tuple[str, str, str]
        Tuple of (lat_name, lon_name, time_name).

    Raises
    ------
    SPEICoordinateError
        If required spatial/temporal coordinates cannot be identified or validated.
    """
    coords = list(ds.coords.keys()) + list(ds.sizes.keys())

    # Resolve latitude
    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lat_name = next((c for c in lat_candidates if c in coords), None)
    if not lat_name:
        raise SPEICoordinateError(f"Latitude coordinate not found in dataset. Available: {coords}")

    # Resolve longitude
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]
    lon_name = next((c for c in lon_candidates if c in coords), None)
    if not lon_name:
        raise SPEICoordinateError(f"Longitude coordinate not found in dataset. Available: {coords}")

    # Resolve time
    time_candidates = ["time", "TIME", "Time", "date"]
    time_name = next((c for c in time_candidates if c in coords), None)
    if not time_name:
        raise SPEICoordinateError(f"Time coordinate not found in dataset. Available: {coords}")

    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    if len(lat_vals) == 0 or len(lon_vals) == 0:
        raise SPEICoordinateError("Latitude or longitude coordinate array is empty.")

    logger.debug(
        "Validated coordinates: lat='%s' [%.2f, %.2f], lon='%s' [%.2f, %.2f], time='%s'",
        lat_name, float(np.min(lat_vals)), float(np.max(lat_vals)),
        lon_name, float(np.min(lon_vals)), float(np.max(lon_vals)),
        time_name,
    )
    return lat_name, lon_name, time_name


def extract_point_series(
    ds: xr.Dataset,
    lat: float = DEFAULT_TARGET_LAT,
    lon: float = DEFAULT_TARGET_LON,
    var_name: Optional[str] = None,
) -> tuple[xr.DataArray, GridCellMetadata]:
    """Extract 1D monthly SPEI time series for the nearest grid cell to target coordinates.

    Handles -180..180 and 0..360 longitude conventions, ascending or descending grids.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing SPEI data.
    lat : float, default 9.63
        Target latitude in decimal degrees.
    lon : float, default 39.53
        Target longitude in decimal degrees.
    var_name : str, optional
        SPEI variable name. If None, auto-resolved.

    Returns
    -------
    tuple[xr.DataArray, GridCellMetadata]
        Extracted 1D DataArray across time and metadata for the selected grid cell.

    Raises
    ------
    SPEICoordinateError
        If coordinates are out of valid bounds or point extraction fails.
    """
    if not (-90.0 <= lat <= 90.0):
        raise SPEICoordinateError(f"Target latitude {lat} is outside valid range [-90, +90].")

    lat_name, lon_name, time_name = validate_coordinates(ds)
    resolved_var = resolve_spei_variable(ds, var_name)

    lon_vals = ds[lon_name].values
    dataset_lon_min = float(np.min(lon_vals))
    dataset_lon_max = float(np.max(lon_vals))

    # Normalize query longitude to dataset coordinate convention
    target_lon_query = lon
    if dataset_lon_min >= 0.0 and dataset_lon_max > 180.0:
        # Dataset uses 0..360 convention
        if target_lon_query < 0.0:
            target_lon_query = (target_lon_query + 360.0) % 360.0
            logger.info("Converted target longitude %.4f to 0..360 convention: %.4f", lon, target_lon_query)
    else:
        # Dataset uses -180..180 convention
        if target_lon_query > 180.0:
            target_lon_query = ((target_lon_query + 180.0) % 360.0) - 180.0
            logger.info("Converted target longitude %.4f to -180..180 convention: %.4f", lon, target_lon_query)

    sel_dict = {lat_name: lat, lon_name: target_lon_query}

    try:
        point_da = ds[resolved_var].sel(sel_dict, method="nearest")
    except Exception as exc:
        raise SPEICoordinateError(f"Spatial nearest-neighbor selection failed for {sel_dict}: {exc}") from exc

    selected_lat = float(point_da[lat_name].values)
    selected_lon_raw = float(point_da[lon_name].values)

    # Convert selected longitude back to standard -180..180 for uniform reporting
    selected_lon_std = selected_lon_raw
    if selected_lon_std > 180.0:
        selected_lon_std = ((selected_lon_std + 180.0) % 360.0) - 180.0

    lat_offset = selected_lat - lat
    lon_offset = selected_lon_std - lon
    dist_km = haversine_distance(lat, lon, selected_lat, selected_lon_std)

    grid_meta = GridCellMetadata(
        requested_lat=lat,
        requested_lon=lon,
        selected_lat=selected_lat,
        selected_lon=selected_lon_std,
        lat_offset=lat_offset,
        lon_offset=lon_offset,
        spatial_distance_km=dist_km,
        selection_method="nearest",
    )

    logger.info(
        "Extracted nearest grid cell: selected=(%.4f, %.4f), requested=(%.4f, %.4f), "
        "distance=%.2f km, offset=(d_lat=%+.4f, d_lon=%+.4f)",
        selected_lat, selected_lon_std, lat, lon, dist_km, lat_offset, lon_offset,
    )

    return point_da, grid_meta


# =============================================================================
# Temporal Validation & Annual Aggregation
# =============================================================================

def validate_time_axis(da: xr.DataArray) -> TimeCoverageMetadata:
    """Inspect and validate the time coordinate of an extracted 1D monthly DataArray.

    Parameters
    ----------
    da : xr.DataArray
        1D time series DataArray.

    Returns
    -------
    TimeCoverageMetadata
        Metadata regarding temporal span, frequency, and complete/incomplete years.

    Raises
    ------
    SPEITimeError
        If time coordinate is missing, empty, or has non-chronological dates.
    """
    time_coord = da.coords.get("time")
    if time_coord is None:
        raise SPEITimeError("Extracted DataArray does not have a 'time' coordinate.")

    time_vals = time_coord.values
    if len(time_vals) == 0:
        raise SPEITimeError("Extracted DataArray time coordinate is empty.")

    # Convert to pandas DatetimeIndex or cftime representation
    try:
        dt_index = pd.to_datetime(time_vals)
    except Exception:
        # Handle cftime dates if standard datetime conversion fails
        try:
            dt_index = pd.DatetimeIndex([pd.Timestamp(str(t)) for t in time_vals])
        except Exception as exc:
            raise SPEITimeError(f"Failed to decode time coordinate to datetime index: {exc}") from exc

    if not dt_index.is_monotonic_increasing:
        raise SPEITimeError("Time coordinate is not strictly monotonically increasing.")

    start_date = str(dt_index[0])
    end_date = str(dt_index[-1])
    total_months = len(dt_index)

    # Check monthly counts per calendar year
    years = dt_index.year
    months_per_year = pd.Series(1, index=years).groupby(level=0).sum()
    complete_years_count = int((months_per_year == 12).sum())
    incomplete_years = [int(yr) for yr in months_per_year[months_per_year != 12].index]

    calendar = getattr(time_coord, "attrs", {}).get("calendar", "standard")

    meta = TimeCoverageMetadata(
        start_date=start_date,
        end_date=end_date,
        total_months=total_months,
        total_years=len(months_per_year),
        complete_years=complete_years_count,
        incomplete_years=incomplete_years,
        calendar=calendar,
        time_frequency="monthly",
    )

    if incomplete_years:
        logger.warning(
            "Found %d incomplete year(s) with <12 monthly observations: %s",
            len(incomplete_years), incomplete_years,
        )

    return meta


def aggregate_annual_spei(
    da_or_series: Union[xr.DataArray, pd.Series],
    require_full_year: bool = True,
    min_months: int = 12,
) -> tuple[pd.Series, list[int]]:
    """Convert a monthly SPEI time series into an annual calendar-year mean time series.

    Enforces the complete-year ground-truth policy (12 valid months per year required
    by default) to prevent seasonal bias in annual climate signal representation.

    Parameters
    ----------
    da_or_series : xr.DataArray or pd.Series
        Monthly SPEI time series with datetime index or coordinate.
    require_full_year : bool, default True
        If True, only years with at least `min_months` valid (non-NaN, finite)
        monthly values are included in the output. Incomplete years are dropped.
    min_months : int, default 12
        Minimum number of valid monthly observations required to form an annual value.

    Returns
    -------
    tuple[pd.Series, list[int]]
        Annual mean SPEI Series indexed by integer calendar year, and list of dropped years.

    Raises
    ------
    SPEIAggregationError
        If aggregation fails or no valid complete years are available.
    """
    if isinstance(da_or_series, xr.DataArray):
        # Convert DataArray to pandas Series
        series = da_or_series.to_series()
    elif isinstance(da_or_series, pd.Series):
        series = da_or_series.copy()
    else:
        raise SPEIAggregationError(f"Expected DataArray or Series, got {type(da_or_series)}")

    if series.empty:
        raise SPEIAggregationError("Cannot aggregate empty SPEI series.")

    # Ensure index is datetime-like
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as exc:
            raise SPEIAggregationError(f"Series index cannot be converted to DatetimeIndex: {exc}") from exc

    # Filter non-finite monthly values
    finite_mask = np.isfinite(series.values)
    if not np.all(finite_mask):
        nan_count = int(np.isnan(series.values).sum())
        inf_count = int(np.isinf(series.values).sum())
        logger.warning(
            "Monthly series contains %d NaN and %d Inf values. These months will be excluded from annual means.",
            nan_count, inf_count,
        )

    # Group by calendar year
    years = series.index.year
    valid_series = series[finite_mask]
    valid_years = valid_series.index.year

    # Count valid months per year
    month_counts = valid_series.groupby(valid_years).count()
    all_years = np.unique(years)

    annual_means = {}
    dropped_years = []

    for yr in all_years:
        count = int(month_counts.get(yr, 0))
        if require_full_year:
            if count >= min_months:
                yr_vals = valid_series[valid_years == yr].values
                annual_means[int(yr)] = float(np.mean(yr_vals))
            else:
                dropped_years.append(int(yr))
                logger.info(
                    "Year %d has only %d/%d valid monthly observations; dropping per complete-year policy.",
                    yr, count, min_months,
                )
        else:
            if count > 0:
                yr_vals = valid_series[valid_years == yr].values
                annual_means[int(yr)] = float(np.mean(yr_vals))
            else:
                dropped_years.append(int(yr))

    annual_s = pd.Series(annual_means, name="spei", dtype=np.float64).sort_index()
    annual_s.index.name = "year"

    logger.info(
        "Successfully aggregated annual SPEI: %d complete years [%s–%s], dropped %d incomplete year(s)",
        len(annual_s),
        int(annual_s.index.min()) if not annual_s.empty else "N/A",
        int(annual_s.index.max()) if not annual_s.empty else "N/A",
        len(dropped_years),
    )
    return annual_s, dropped_years


# =============================================================================
# DataFrame Construction & Validation
# =============================================================================

def to_annual_dataframe(annual_series: pd.Series) -> pd.DataFrame:
    """Convert an annual SPEI Series to a validated production DataFrame.

    Ensures exact schema: ``['year', 'spei']``, integer years, float SPEI,
    chronological sorting, no NaN, no Inf, and no duplicate years.

    Parameters
    ----------
    annual_series : pd.Series
        Annual SPEI series indexed by integer year.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns ``['year', 'spei']``.

    Raises
    ------
    SPEIAggregationError
        If DataFrame validation fails.
    """
    if annual_series.empty:
        raise SPEIAggregationError("Cannot construct DataFrame from empty annual SPEI series.")

    df = annual_series.reset_index()
    df.columns = ["year", "spei"]

    # Enforce exact dtypes
    try:
        df["year"] = df["year"].astype(np.int64)
        df["spei"] = df["spei"].astype(np.float64)
    except Exception as exc:
        raise SPEIAggregationError(f"Failed to cast DataFrame columns to standard types: {exc}") from exc

    # Sort chronologically
    df = df.sort_values("year").reset_index(drop=True)

    # Validation checks
    if df["year"].duplicated().any():
        dupes = df["year"][df["year"].duplicated()].tolist()
        raise SPEIAggregationError(f"Duplicate years found in annual DataFrame: {dupes}")

    if not df["year"].is_monotonic_increasing:
        raise SPEIAggregationError("Years are not strictly monotonically increasing.")

    if df["spei"].isna().any():
        nan_years = df.loc[df["spei"].isna(), "year"].tolist()
        raise SPEIAggregationError(f"NaN values present in annual SPEI for years: {nan_years}")

    if np.isinf(df["spei"].values).any():
        inf_years = df.loc[np.isinf(df["spei"].values), "year"].tolist()
        raise SPEIAggregationError(f"Infinite values present in annual SPEI for years: {inf_years}")

    # Check for missing calendar year gaps (informational check)
    full_range = np.arange(df["year"].min(), df["year"].max() + 1)
    missing_gaps = sorted(set(full_range) - set(df["year"]))
    if missing_gaps:
        logger.warning(
            "Annual SPEI series has %d temporal gap year(s) between %d and %d: %s",
            len(missing_gaps), df["year"].min(), df["year"].max(), missing_gaps,
        )

    return df


# =============================================================================
# Export & Verification
# =============================================================================

def export_annual_spei_csv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export annual SPEI DataFrame to CSV and verify by re-reading from disk.

    Parameters
    ----------
    df : pd.DataFrame
        Annual SPEI DataFrame with columns ``['year', 'spei']``.
    output_path : str or Path
        Destination CSV file path.
    overwrite : bool, default False
        Whether to overwrite existing file.

    Returns
    -------
    Path
        Resolved output file path.

    Raises
    ------
    SPEIExportError
        If output exists and overwrite is False, or if roundtrip verification fails.
    """
    out_p = Path(output_path)
    if out_p.exists() and not overwrite:
        raise SPEIExportError(
            f"Output file '{out_p.resolve()}' already exists. Pass overwrite=True to replace."
        )

    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write CSV without pandas index, UTF-8 encoded
        df.to_csv(out_p, index=False, encoding="utf-8")
    except Exception as exc:
        raise SPEIExportError(f"Failed to write CSV to '{out_p}': {exc}") from exc

    # Verification: Read back from disk
    try:
        reloaded = pd.read_csv(out_p, encoding="utf-8")
    except Exception as exc:
        raise SPEIExportError(f"Verification failed: cannot reload written CSV '{out_p}': {exc}") from exc

    if list(reloaded.columns) != ["year", "spei"]:
        raise SPEIExportError(f"Reloaded CSV columns {list(reloaded.columns)} != ['year', 'spei']")

    if len(reloaded) != len(df):
        raise SPEIExportError(f"Reloaded CSV row count {len(reloaded)} != original {len(df)}")

    if reloaded["spei"].isna().any():
        raise SPEIExportError("Reloaded CSV contains NaN values.")

    logger.info("Successfully exported and verified SPEI CSV at: %s (%d rows)", out_p.resolve(), len(df))
    return out_p.resolve()


def export_provenance_json(
    provenance: dict[str, Any],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export extraction provenance metadata to JSON.

    Parameters
    ----------
    provenance : dict[str, Any]
        Provenance dictionary.
    output_path : str or Path
        Destination JSON path.
    overwrite : bool, default False
        Whether to overwrite existing file.

    Returns
    -------
    Path
        Resolved JSON path.
    """
    out_p = Path(output_path)
    if out_p.exists() and not overwrite:
        raise SPEIExportError(f"Provenance file '{out_p.resolve()}' already exists.")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    return out_p.resolve()


# =============================================================================
# Downstream ML Compatibility Check
# =============================================================================

def check_downstream_compatibility(
    spei_df: pd.DataFrame,
    lagged_csv_path: Union[str, Path],
) -> dict[str, Any]:
    """Check merge compatibility between extracted annual SPEI and downstream RWI/solar dataset.

    Parameters
    ----------
    spei_df : pd.DataFrame
        Extracted annual SPEI DataFrame (columns: ``year``, ``spei``).
    lagged_csv_path : str or Path
        Path to ``processed_lagged_data.csv``.

    Returns
    -------
    dict[str, Any]
        Compatibility report dictionary with overlap statistics.
    """
    p = Path(lagged_csv_path)
    if not p.exists():
        logger.warning("Downstream dataset '%s' not found for compatibility check.", p)
        return {"compatible": False, "error": f"File not found: {p}"}

    rwi_df = pd.read_csv(p)
    if "year" not in rwi_df.columns:
        return {"compatible": False, "error": "Column 'year' not found in downstream dataset."}

    merged = rwi_df.merge(spei_df, on="year", how="inner")
    rwi_years = set(rwi_df["year"].unique())
    spei_years = set(spei_df["year"].unique())
    overlap_years = sorted(rwi_years.intersection(spei_years))

    rwi_only_years = sorted(rwi_years - spei_years)
    spei_only_years = sorted(spei_years - rwi_years)

    report = {
        "compatible": len(overlap_years) > 0,
        "rwi_total_rows": len(rwi_df),
        "rwi_unique_years": len(rwi_years),
        "rwi_year_range": [int(min(rwi_years)), int(max(rwi_years))] if rwi_years else None,
        "spei_total_rows": len(spei_df),
        "spei_unique_years": len(spei_years),
        "spei_year_range": [int(min(spei_years)), int(max(spei_years))] if spei_years else None,
        "overlapping_rows": len(merged),
        "overlapping_years_count": len(overlap_years),
        "overlap_start_year": int(overlap_years[0]) if overlap_years else None,
        "overlap_end_year": int(overlap_years[-1]) if overlap_years else None,
        "rwi_years_missing_spei_count": len(rwi_only_years),
        "rwi_years_missing_spei": rwi_only_years[:10] + (["..."] if len(rwi_only_years) > 10 else []),
        "spei_years_missing_rwi_count": len(spei_only_years),
        "spei_years_missing_rwi": spei_only_years[:10] + (["..."] if len(spei_only_years) > 10 else []),
    }

    logger.info(
        "Downstream compatibility check: %d overlapping years [%s–%s] across %d merged observations.",
        report["overlapping_years_count"],
        report["overlap_start_year"],
        report["overlap_end_year"],
        report["overlapping_rows"],
    )
    return report


# =============================================================================
# High-Level Extraction Pipeline
# =============================================================================

def extract_annual_spei(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    lat: float = DEFAULT_TARGET_LAT,
    lon: float = DEFAULT_TARGET_LON,
    var_name: Optional[str] = None,
    require_full_year: bool = True,
    overwrite: bool = False,
    provenance_path: Optional[Union[str, Path]] = None,
) -> SPEIExtractionResult:
    """Run end-to-end extraction from NetCDF to validated annual ground-truth SPEI.

    Parameters
    ----------
    input_path : str or Path
        Path to the SPEIbase NetCDF dataset.
    output_path : str or Path, optional
        Path for exporting the annual CSV (e.g. ``spei_debrebirkan.csv``).
    lat : float, default 9.63
        Target latitude in decimal degrees.
    lon : float, default 39.53
        Target longitude in decimal degrees.
    var_name : str, optional
        SPEI variable name. If None, auto-resolved.
    require_full_year : bool, default True
        Whether to enforce complete 12 months for every annual observation.
    overwrite : bool, default False
        Whether to overwrite existing output files.
    provenance_path : str or Path, optional
        Optional path to save provenance metadata JSON.

    Returns
    -------
    SPEIExtractionResult
        Result dataclass containing annual DataFrame, metadata, and provenance.
    """
    in_path = Path(input_path)
    logger.info("Starting SPEI annual extraction pipeline for: %s", in_path.resolve())

    # 1. Load dataset
    ds = load_netcdf(in_path)

    # 2. Extract point series with nearest-grid-cell validation
    point_da, grid_meta = extract_point_series(ds, lat=lat, lon=lon, var_name=var_name)

    # 3. Validate temporal structure
    time_meta = validate_time_axis(point_da)

    # 4. Aggregate monthly to annual calendar-year mean
    annual_s, dropped_years = aggregate_annual_spei(
        point_da, require_full_year=require_full_year, min_months=12
    )

    # 5. Convert to validated production DataFrame
    annual_df = to_annual_dataframe(annual_s)

    # 6. Record full provenance
    provenance = {
        "pipeline": "SPEIbase NetCDF -> Annual Ground-Truth Extractor",
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "xarray_version": xr.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "input_dataset": {
            "path": str(in_path.resolve()),
            "filename": in_path.name,
            "sha256": compute_file_sha256(in_path),
            "global_attributes": dict(ds.attrs),
            "variable_name": point_da.name,
            "variable_attributes": dict(point_da.attrs),
        },
        "spatial_extraction": grid_meta.to_dict(),
        "temporal_validation": time_meta.to_dict(),
        "annual_aggregation": {
            "rule": "calendar-year arithmetic mean of monthly values",
            "require_full_year": require_full_year,
            "min_months_per_year": 12,
            "total_complete_years": len(annual_df),
            "dropped_years": dropped_years,
            "annual_year_range": [int(annual_df["year"].min()), int(annual_df["year"].max())],
            "statistics": {
                "mean": float(annual_df["spei"].mean()),
                "std": float(annual_df["spei"].std()),
                "min": float(annual_df["spei"].min()),
                "max": float(annual_df["spei"].max()),
            },
        },
    }

    # Close dataset
    ds.close()

    # 7. Export CSV if requested
    saved_csv_path = None
    if output_path is not None:
        saved_csv_path = export_annual_spei_csv(annual_df, output_path=output_path, overwrite=overwrite)
        provenance["output_csv"] = str(saved_csv_path)

    # 8. Export provenance JSON if requested
    if provenance_path is not None:
        export_provenance_json(provenance, output_path=provenance_path, overwrite=overwrite)

    result = SPEIExtractionResult(
        annual_df=annual_df,
        monthly_series=point_da.to_series(),
        grid_metadata=grid_meta,
        time_metadata=time_meta,
        provenance=provenance,
        output_path=saved_csv_path,
    )

    logger.info("SPEI extraction pipeline completed successfully.")
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def _build_spei_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="treering.spei",
        description=(
            "SPEIbase NetCDF Annual Ground-Truth Extractor.\n\n"
            "Extracts monthly SPEI time series for a specified target location\n"
            "(e.g., Debrebirkan Selassie at lat 9.63, lon 39.53), performs\n"
            "coordinate validation and nearest grid cell selection, enforces\n"
            "complete 12-month calendar-year policy, and exports clean CSVs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input SPEIbase NetCDF (.nc) file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="spei_debrebirkan.csv",
        help="Path for destination CSV output (default: spei_debrebirkan.csv)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=DEFAULT_TARGET_LAT,
        help=f"Target latitude in decimal degrees (default: {DEFAULT_TARGET_LAT})",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=DEFAULT_TARGET_LON,
        help=f"Target longitude in decimal degrees (default: {DEFAULT_TARGET_LON})",
    )
    parser.add_argument(
        "--var",
        type=str,
        default=None,
        help="Name of SPEI variable in NetCDF (default: auto-detect 'spei01' / 'spei')",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        default=False,
        help="Allow incomplete years (<12 months) instead of dropping them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output CSV if present",
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default=None,
        help="Optional path to export full provenance JSON metadata",
    )
    parser.add_argument(
        "--check-downstream",
        type=str,
        default=None,
        help="Optional path to processed_lagged_data.csv to check merge compatibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SPEI extraction."""
    parser = _build_spei_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        res = extract_annual_spei(
            input_path=args.input,
            output_path=args.output,
            lat=args.lat,
            lon=args.lon,
            var_name=args.var,
            require_full_year=not args.allow_incomplete,
            overwrite=args.overwrite,
            provenance_path=args.provenance,
        )

        print("\n=== SPEI Extraction Summary ===")
        print(f"Output file: {res.output_path}")
        print(f"Rows (complete years): {len(res.annual_df)}")
        print(f"Year range: {res.annual_df['year'].min()}–{res.annual_df['year'].max()}")
        print(
            f"Selected grid cell: ({res.grid_metadata.selected_lat:.4f}, "
            f"{res.grid_metadata.selected_lon:.4f}) "
            f"[distance: {res.grid_metadata.spatial_distance_km:.2f} km]"
        )

        # Verification step: read first 5 rows directly from written CSV
        if res.output_path:
            reloaded = pd.read_csv(res.output_path)
            print("\nFirst 5 rows from generated CSV:")
            print(reloaded.head().to_string(index=False))

        # Check downstream compatibility if requested
        if args.check_downstream:
            compat = check_downstream_compatibility(res.annual_df, args.check_downstream)
            print("\n=== Downstream Compatibility Check ===")
            print(f"Compatible: {compat.get('compatible')}")
            print(f"Overlapping years count: {compat.get('overlapping_years_count')}")
            print(f"Overlap span: {compat.get('overlap_start_year')}–{compat.get('overlap_end_year')}")
            print(f"Overlapping observations: {compat.get('overlapping_rows')}")

        return 0

    except Exception as exc:
        print(f"Error during SPEI extraction: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

