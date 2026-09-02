"""RWI and Sunspot Number Solar-Cycle Lag Analysis.

This module provides a complete, scientifically rigorous pipeline to:
1. Ingest and validate Tree-Ring Width Index (RWI) and Sunspot Number (SN) datasets.
2. Align and merge datasets on chronological calendar years.
3. Apply an 11-year centered moving average to isolate the Schwabe solar cycle.
4. Standardize smoothed series using z-scores (with explicit sample standard deviation ddof=1).
5. Perform lag correlation analysis: R(tau) = corr(RWI(t), SN(t - tau)) for tau in [0, 5].
6. Identify the optimal lag based on maximum absolute correlation |R(tau)|.
7. Build an aligned, standardized dataset and export results.

Scientific Note on Lag Direction:
---------------------------------
R(tau) = corr(RWI(t), SN(t - tau))
A positive lag tau means solar activity (sunspot number) at year (t - tau) is
correlated against tree growth (RWI) at year t. For example, lag tau=2 evaluates
whether solar activity 2 years prior influences current-year tree-ring growth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Schwabe cycle window length in years (must be odd for centered moving average)
SCHWABE_WINDOW: int = 11
HALF_WINDOW: int = SCHWABE_WINDOW // 2  # 5 years on each side


class DataValidationError(Exception):
    """Raised when input data fails schema or numerical validation."""


class TemporalContinuityError(Exception):
    """Raised when temporal continuity checks fail."""


class LagAnalysisError(Exception):
    """Raised when lag analysis cannot be computed."""


@dataclass(frozen=True)
class OptimalLagResult:
    """Container for optimal lag selection results."""

    optimal_lag: int
    optimal_correlation: float
    correlation_direction: str
    p_value: float
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class SolarLagAnalysisResult:
    """Container for the complete solar lag analysis pipeline output."""

    aligned_data: pd.DataFrame
    lag_correlations: pd.DataFrame
    optimal_lag: OptimalLagResult
    summary: dict[str, Any]


def load_rwi_data(
    filepath_or_df: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load and validate RWI dataset.

    Parameters
    ----------
    filepath_or_df : str, Path, or pd.DataFrame
        Path to RWI CSV file or an existing DataFrame. Expected columns
        must include ``year`` and ``rwi``. May also include ``series_id``,
        ``raw_ring_width``, ``fitted_growth``.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with standardized column types and sorted by
        ``(series_id, year)`` or ``year``.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    DataValidationError
        If required columns are missing, data is empty, or values are invalid.
    """
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df.copy()
    else:
        path = Path(filepath_or_df)
        if not path.is_file():
            raise FileNotFoundError(f"RWI file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataValidationError(f"Failed to read RWI CSV from {path}: {exc}") from exc

    if df.empty:
        raise DataValidationError("RWI dataset is empty.")

    # Canonicalize column names (lowercase, strip whitespace)
    col_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Validate required columns
    required_cols = {"year", "rwi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(
            f"RWI dataset missing required column(s): {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    # Validate year column
    try:
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    except Exception as exc:
        raise DataValidationError(f"RWI 'year' column contains non-integer/invalid values: {exc}") from exc

    # Validate rwi column
    try:
        df["rwi"] = pd.to_numeric(df["rwi"], errors="raise").astype(float)
    except Exception as exc:
        raise DataValidationError(f"RWI 'rwi' column contains non-numeric values: {exc}") from exc

    # Check for non-finite RWI values
    non_finite = ~np.isfinite(df["rwi"].values)
    if non_finite.any():
        bad_count = int(non_finite.sum())
        raise DataValidationError(f"RWI dataset contains {bad_count} non-finite/NaN value(s).")

    # Check for negative RWI values (RWI = raw/growth >= 0)
    if (df["rwi"] < 0).any():
        bad_count = int((df["rwi"] < 0).sum())
        raise DataValidationError(f"RWI dataset contains {bad_count} negative value(s).")

    # Sort data
    if "series_id" in df.columns:
        df["series_id"] = df["series_id"].astype(str)
        # Check duplicate (series_id, year)
        dupes = df.duplicated(subset=["series_id", "year"])
        if dupes.any():
            first_dupe = df.loc[dupes].iloc[0]
            raise DataValidationError(
                f"Duplicate (series_id, year) record found in RWI: "
                f"series={first_dupe['series_id']}, year={first_dupe['year']}"
            )
        df = df.sort_values(["series_id", "year"]).reset_index(drop=True)
    else:
        dupes = df.duplicated(subset=["year"])
        if dupes.any():
            first_dupe = df.loc[dupes].iloc[0]
            raise DataValidationError(
                f"Duplicate year record found in RWI dataset: year={first_dupe['year']}"
            )
        df = df.sort_values("year").reset_index(drop=True)

    logger.info(
        "Loaded RWI dataset: %d rows, year range [%d, %d], series count: %d",
        len(df),
        df["year"].min(),
        df["year"].max(),
        df["series_id"].nunique() if "series_id" in df.columns else 1,
    )
    return df


def load_sunspot_data(
    filepath_or_df: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load and normalize historical Sunspot Number (SN) dataset.

    Supports both standard SILSO files (semicolon-delimited, e.g. ``SN_y_tot_V2.0.csv``
    where column 0 is fractional year e.g. 1700.5 and column 1 is Sunspot Number)
    and standard CSV files with header containing ``year`` and sunspot columns
    (``sunspot``, ``sunspot_number``, ``sn``, ``Sunspot Number``).

    Parameters
    ----------
    filepath_or_df : str, Path, or pd.DataFrame
        Path to Sunspot CSV file or DataFrame.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with columns ``['year', 'sunspot']``, sorted
        chronologically with unique integer calendar years.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    DataValidationError
        If file is empty or cannot be parsed into valid (year, sunspot) observations.
    """
    if isinstance(filepath_or_df, pd.DataFrame):
        raw_df = filepath_or_df.copy()
    else:
        path = Path(filepath_or_df)
        if not path.is_file():
            raise FileNotFoundError(f"Sunspot file not found: {path}")

        # Read sample lines to detect delimiter and header
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise DataValidationError(f"Sunspot file is empty: {path}")

        first_line = lines[0]
        delimiter = ";" if ";" in first_line else ","

        # Check if first line is a header
        tokens = [t.strip().lower() for t in first_line.split(delimiter)]
        has_header = any("year" in t or "sn" in t or "sunspot" in t for t in tokens)

        if has_header:
            raw_df = pd.read_csv(path, sep=delimiter)
        else:
            # Headerless SILSO format: col0=fractional_year, col1=SN, col2=std_dev, col3=n_obs, col4=flag
            raw_df = pd.read_csv(path, sep=delimiter, header=None)

    if raw_df.empty:
        raise DataValidationError("Sunspot dataset is empty.")

    # Identify year and sunspot columns
    if isinstance(raw_df.columns[0], int) or all(isinstance(c, int) for c in raw_df.columns):
        # Numeric positional columns (SILSO format)
        if raw_df.shape[1] < 2:
            raise DataValidationError("Sunspot dataset has fewer than 2 columns.")
        # Col 0: fractional year (e.g. 1700.5 -> 1700)
        years_raw = raw_df.iloc[:, 0].astype(float)
        years = np.floor(years_raw).astype(int)
        sunspots = pd.to_numeric(raw_df.iloc[:, 1], errors="coerce")
    else:
        # Header-based DataFrame
        col_map = {c: str(c).strip().lower() for c in raw_df.columns}
        df_named = raw_df.rename(columns=col_map)

        # Locate year column
        year_cand = [c for c in df_named.columns if "year" in c]
        if not year_cand:
            # Fall back to first column
            year_col = df_named.columns[0]
        else:
            year_col = year_cand[0]

        # Locate sunspot column
        sn_cand = [
            c for c in df_named.columns
            if c in ("sunspot", "sunspot_number", "sn", "sunspots", "yearly_mean_total")
            or "sunspot" in c
            or c == "sn"
        ]
        if not sn_cand:
            # If 2 columns, take the other one
            non_year = [c for c in df_named.columns if c != year_col]
            if non_year:
                sn_col = non_year[0]
            else:
                raise DataValidationError(f"Could not identify Sunspot column in: {list(df_named.columns)}")
        else:
            sn_col = sn_cand[0]

        years_raw = pd.to_numeric(df_named[year_col], errors="coerce")
        # Handle fractional years (e.g. 1700.5 -> 1700)
        years = np.floor(years_raw).astype(int)
        sunspots = pd.to_numeric(df_named[sn_col], errors="coerce")

    # Assemble normalized DataFrame
    df = pd.DataFrame({"year": years, "sunspot": sunspots})

    # Validate numeric and finite values
    invalid_mask = df["year"].isna() | df["sunspot"].isna() | ~np.isfinite(df["sunspot"].values)
    if invalid_mask.any():
        bad_count = int(invalid_mask.sum())
        logger.warning("Dropping %d invalid/non-finite Sunspot rows.", bad_count)
        df = df[~invalid_mask].copy()

    # In SILSO data, missing values are denoted as -1. Handle/reject:
    negative_mask = df["sunspot"] < 0
    if negative_mask.any():
        bad_count = int(negative_mask.sum())
        logger.warning(
            "Dropping %d Sunspot rows with negative/missing values (< 0).", bad_count
        )
        df = df[~negative_mask].copy()

    if df.empty:
        raise DataValidationError("No valid Sunspot observations remaining after cleaning.")

    # Deduplicate years
    dupes = df.duplicated(subset=["year"])
    if dupes.any():
        first_dupe = df.loc[dupes].iloc[0]
        raise DataValidationError(
            f"Duplicate year record found in Sunspot dataset: year={int(first_dupe['year'])}"
        )

    df = df.sort_values("year").reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["sunspot"] = df["sunspot"].astype(float)

    logger.info(
        "Loaded Sunspot dataset: %d rows, year range [%d, %d], sunspot range [%.1f, %.1f]",
        len(df),
        df["year"].min(),
        df["year"].max(),
        df["sunspot"].min(),
        df["sunspot"].max(),
    )
    return df


def validate_years(
    df: pd.DataFrame,
    year_col: str = "year",
) -> list[int]:
    """Validate temporal continuity of a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing calendar years.
    year_col : str, default 'year'
        Name of the year column.

    Returns
    -------
    list[int]
        List of missing calendar years within the span [min_year, max_year].

    Raises
    ------
    DataValidationError
        If year column is missing or non-integer.
    """
    if year_col not in df.columns:
        raise DataValidationError(f"DataFrame missing year column '{year_col}'.")

    years = df[year_col].values
    if len(years) == 0:
        return []

    min_year = int(np.min(years))
    max_year = int(np.max(years))
    full_span = set(range(min_year, max_year + 1))
    present_years = set(years)
    missing_years = sorted(full_span - present_years)

    if missing_years:
        logger.warning(
            "Temporal gap detected: %d missing year(s) between %d and %d: %s",
            len(missing_years),
            min_year,
            max_year,
            missing_years[:10] if len(missing_years) > 10 else missing_years,
        )

    return missing_years


def merge_datasets(
    rwi_df: pd.DataFrame,
    sunspot_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge RWI and Sunspot datasets on chronological calendar year (inner join).

    Preserves individual series identifiers if present in RWI dataset.

    Parameters
    ----------
    rwi_df : pd.DataFrame
        Validated RWI dataset (contains ``year``, ``rwi``, optional ``series_id``).
    sunspot_df : pd.DataFrame
        Validated Sunspot dataset (contains ``year``, ``sunspot``).

    Returns
    -------
    pd.DataFrame
        Merged dataset with columns ``[series_id], year, rwi, sunspot, ...``,
        sorted chronologically.

    Raises
    ------
    DataValidationError
        If no overlapping years exist between the two datasets.
    """
    rwi_years = set(rwi_df["year"].unique())
    sun_years = set(sunspot_df["year"].unique())
    overlap_years = sorted(rwi_years & sun_years)

    lost_rwi_years = sorted(rwi_years - sun_years)
    lost_sun_years = sorted(sun_years - rwi_years)

    logger.info(
        "Merge summary: RWI years=%d [%d, %d], Sunspot years=%d [%d, %d], "
        "Overlapping years=%d [%d, %d]. Lost RWI years=%d, Lost Sunspot years=%d",
        len(rwi_years),
        min(rwi_years),
        max(rwi_years),
        len(sun_years),
        min(sun_years),
        max(sun_years),
        len(overlap_years),
        min(overlap_years) if overlap_years else -1,
        max(overlap_years) if overlap_years else -1,
        len(lost_rwi_years),
        len(lost_sun_years),
    )

    if not overlap_years:
        raise DataValidationError(
            f"No overlapping years between RWI [{min(rwi_years)}, {max(rwi_years)}] "
            f"and Sunspot [{min(sun_years)}, {max(sun_years)}]."
        )

    merged = pd.merge(
        rwi_df,
        sunspot_df[["year", "sunspot"]],
        on="year",
        how="inner",
    )

    if "series_id" in merged.columns:
        merged = merged.sort_values(["series_id", "year"]).reset_index(drop=True)
    else:
        merged = merged.sort_values("year").reset_index(drop=True)

    return merged


def apply_centered_smoothing(
    series: Union[pd.Series, np.ndarray],
    window: int = SCHWABE_WINDOW,
    years: Union[pd.Series, np.ndarray, None] = None,
) -> np.ndarray:
    """Apply an 11-year centered moving average with strict Schwabe-cycle boundaries.

    Mathematical Definition:
        x_smoothed(t) = (1 / 11) * sum_{k=-5}^{5} x(t + k)

    Boundary and Gap Requirements:
    -------------------------------
    1. Complete 11-observation window only: The first 5 and last 5 observations
       cannot receive a complete centered 11-year window and MUST be NaN.
    2. Temporal continuity: If ``years`` are provided, gaps in calendar years are
       detected so that observations are strictly averaged across 11 consecutive
       calendar years (t-5 to t+5).

    Parameters
    ----------
    series : array_like
        Input values to smooth.
    window : int, default 11
        Centered moving average window size (must be odd, default 11).
    years : array_like, optional
        Corresponding calendar years. Used to guarantee calendar-year continuity.

    Returns
    -------
    np.ndarray
        Smoothed array of same length as input, with NaN where a full centered
        11-consecutive-year window is not available.

    Raises
    ------
    ValueError
        If window is not a positive odd integer or inputs have mismatched lengths.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"Window size must be a positive odd integer, got {window}")

    values = np.asarray(series, dtype=np.float64)
    n = len(values)

    if n < window:
        # Not enough data for even one full centered window
        return np.full(n, np.nan, dtype=np.float64)

    half = window // 2

    if years is not None:
        years_arr = np.asarray(years, dtype=int)
        if len(years_arr) != n:
            raise ValueError(
                f"Length mismatch: series has {n} items, years has {len(years_arr)}"
            )

        # Place values on a continuous integer year grid to ensure calendar-year continuity
        min_yr = int(years_arr.min())
        max_yr = int(years_arr.max())
        grid_years = np.arange(min_yr, max_yr + 1)
        grid_df = pd.DataFrame({"year": grid_years})
        input_df = pd.DataFrame({"year": years_arr, "val": values})
        merged_grid = pd.merge(grid_df, input_df, on="year", how="left")

        # Rolling centered 11-year mean on complete grid (min_periods=window requires all 11 non-NaN)
        rolled = (
            merged_grid["val"]
            .rolling(window=window, center=True, min_periods=window)
            .mean()
        )
        merged_grid["smoothed"] = rolled

        # Map back to original observations
        res_df = pd.merge(input_df[["year"]], merged_grid[["year", "smoothed"]], on="year", how="left")
        smoothed = res_df["smoothed"].values.astype(np.float64)
    else:
        # Standard contiguous rolling centered mean
        s = pd.Series(values)
        smoothed = s.rolling(window=window, center=True, min_periods=window).mean().to_numpy()

    return smoothed


def standardize_series(
    series: Union[pd.Series, np.ndarray],
    ddof: int = 1,
) -> np.ndarray:
    """Standardize a series using z-scores: z_t = (x_t - mu) / sigma.

    Parameters
    ----------
    series : array_like
        Input series (may contain NaN).
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculation is N - ddof.
        Default 1 represents the sample standard deviation (unbiased estimator).

    Returns
    -------
    np.ndarray
        Standardized array where valid observations have mean ~ 0 and std ~ 1.
        NaN positions in the input remain NaN.

    Raises
    ------
    ValueError
        If the series has zero variance (constant values) or fewer than 2 valid points.
    """
    arr = np.asarray(series, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    n_valid = int(valid_mask.sum())

    if n_valid < 2:
        raise ValueError(
            f"Cannot standardize series with only {n_valid} valid observation(s)."
        )

    valid_vals = arr[valid_mask]
    mu = float(np.mean(valid_vals))
    sigma = float(np.std(valid_vals, ddof=ddof))

    if sigma == 0.0 or not np.isfinite(sigma):
        raise ValueError(
            f"Cannot standardize constant or zero-variance series (mu={mu:.4f}, sigma={sigma:.4f})."
        )

    result = np.full_like(arr, np.nan, dtype=np.float64)
    result[valid_mask] = (valid_vals - mu) / sigma
    return result


def calculate_lag_correlations(
    df: pd.DataFrame,
    rwi_col: str = "rwi_z",
    sunspot_col: str = "sunspot_z",
    max_lag: int = 5,
    series_id: Union[str, None] = None,
) -> pd.DataFrame:
    """Calculate Pearson correlation for lags tau in [0, max_lag].

    Scientific Relationship:
        R(tau) = corr(RWI(t), SN(t - tau))

    A positive lag tau means Sunspot Number at year (t - tau) is correlated
    with RWI at year t.

    Parameters
    ----------
    df : pd.DataFrame
        Merged and standardized dataset containing ``year``, ``rwi_col``,
        and ``sunspot_col``.
    rwi_col : str, default 'rwi_z'
        Name of standardized RWI column.
    sunspot_col : str, default 'sunspot_z'
        Name of standardized Sunspot column.
    max_lag : int, default 5
        Maximum positive lag in years (evaluates 0, 1, ..., max_lag).
    series_id : str, optional
        Series identifier for logging and filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['lag', 'correlation', 'p_value', 'n_observations']``.

    Raises
    ------
    LagAnalysisError
        If fewer than 3 paired observations exist for correlation calculation.
    """
    if series_id is not None and "series_id" in df.columns:
        sub_df = df[df["series_id"] == series_id].copy()
    else:
        sub_df = df.copy()

    # Ensure required columns exist
    for col in ("year", rwi_col, sunspot_col):
        if col not in sub_df.columns:
            raise LagAnalysisError(f"Missing required column '{col}' for lag analysis.")

    sub_df = sub_df.sort_values("year").reset_index(drop=True)

    # Build year-indexed series for precise temporal alignment
    rwi_series = pd.Series(
        data=sub_df[rwi_col].values,
        index=sub_df["year"].values,
        dtype=float,
    )
    sn_series = pd.Series(
        data=sub_df[sunspot_col].values,
        index=sub_df["year"].values,
        dtype=float,
    )

    results: list[dict[str, Any]] = []

    for tau in range(0, max_lag + 1):
        # RWI(t) compared with SN(t - tau)
        # For each year t in rwi_series, find SN at year (t - tau)
        target_years = rwi_series.index
        lagged_sn_years = target_years - tau

        # Retrieve matching sunspot values where year (t - tau) exists
        matching_sn = sn_series.reindex(lagged_sn_years).values
        rwi_vals = rwi_series.values

        # Pairwise finite filter
        valid = np.isfinite(rwi_vals) & np.isfinite(matching_sn)
        n_pairs = int(valid.sum())

        if n_pairs < 3:
            logger.warning(
                "Lag tau=%d: only %d paired valid observations (need >= 3).",
                tau,
                n_pairs,
            )
            results.append(
                {
                    "lag": tau,
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "n_observations": n_pairs,
                }
            )
            continue

        x_pair = rwi_vals[valid]
        y_pair = matching_sn[valid]

        res = stats.pearsonr(x_pair, y_pair)
        r_val = float(res.statistic)
        p_val = float(res.pvalue)

        results.append(
            {
                "lag": tau,
                "correlation": r_val,
                "p_value": p_val,
                "n_observations": n_pairs,
            }
        )
        logger.debug(
            "Lag tau=%d: Pearson R=%.4f, p=%.4e, N=%d",
            tau,
            r_val,
            p_val,
            n_pairs,
        )

    res_df = pd.DataFrame(results)
    return res_df


def select_optimal_lag(lag_results: pd.DataFrame) -> OptimalLagResult:
    """Identify the optimal lag maximizing absolute Pearson correlation |R(tau)|.

    Considers both strongest positive and strongest negative correlations:
        tau* = argmax_{tau} |R(tau)|

    Parameters
    ----------
    lag_results : pd.DataFrame
        DataFrame from ``calculate_lag_correlations()`` with columns
        ``['lag', 'correlation', 'p_value', 'n_observations']``.

    Returns
    -------
    OptimalLagResult
        Named container with ``optimal_lag``, ``optimal_correlation``,
        ``correlation_direction`` ('positive' or 'negative'),
        ``p_value``, and ``n_observations``.

    Raises
    ------
    LagAnalysisError
        If no valid correlation values are found.
    """
    valid_df = lag_results.dropna(subset=["correlation"]).copy()
    if valid_df.empty:
        raise LagAnalysisError("No valid correlation values available to select optimal lag.")

    valid_df["abs_r"] = valid_df["correlation"].abs()
    best_row = valid_df.sort_values(by=["abs_r", "lag"], ascending=[False, True]).iloc[0]

    opt_lag = int(best_row["lag"])
    opt_r = float(best_row["correlation"])
    opt_p = float(best_row["p_value"])
    opt_n = int(best_row["n_observations"])
    direction = "positive" if opt_r >= 0 else "negative"

    logger.info(
        "Optimal lag: tau=%d years, Pearson R=%.4f (%s), p=%.4e, N=%d",
        opt_lag,
        opt_r,
        direction,
        opt_p,
        opt_n,
    )

    return OptimalLagResult(
        optimal_lag=opt_lag,
        optimal_correlation=opt_r,
        correlation_direction=direction,
        p_value=opt_p,
        n_observations=opt_n,
    )


def build_final_aligned_dataframe(
    merged_df: pd.DataFrame,
    optimal_lag: int,
) -> pd.DataFrame:
    """Build final aligned DataFrame containing original, smoothed, standardized, and lagged variables.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame containing ``year``, ``rwi``, ``sunspot``,
        ``rwi_smoothed``, ``sunspot_smoothed``, ``rwi_z``, ``sunspot_z``.
    optimal_lag : int
        The selected optimal lag tau*.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with columns:
        ``[series_id], year, rwi, sunspot, rwi_smoothed, sunspot_smoothed,``
        ``rwi_z, sunspot_z, sunspot_z_lagged, optimal_lag``.
    """
    df = merged_df.copy()

    # Align lagged sunspot_z: for each observation at year t, sunspot_z_lagged is sunspot_z at (t - optimal_lag)
    if "series_id" in df.columns:
        # Group by series_id so lagging respects series boundaries
        lagged_frames = []
        for sid, grp in df.groupby("series_id"):
            grp = grp.sort_values("year").copy()
            sn_map = pd.Series(grp["sunspot_z"].values, index=grp["year"].values)
            grp["sunspot_z_lagged"] = sn_map.reindex(grp["year"] - optimal_lag).values
            grp["optimal_lag"] = optimal_lag
            lagged_frames.append(grp)
        final_df = pd.concat(lagged_frames, ignore_index=True)
        final_df = final_df.sort_values(["series_id", "year"]).reset_index(drop=True)
    else:
        df = df.sort_values("year").reset_index(drop=True)
        sn_map = pd.Series(df["sunspot_z"].values, index=df["year"].values)
        df["sunspot_z_lagged"] = sn_map.reindex(df["year"] - optimal_lag).values
        df["optimal_lag"] = optimal_lag
        final_df = df

    return final_df


def run_solar_lag_analysis(
    rwi_input: Union[str, Path, pd.DataFrame],
    sunspot_input: Union[str, Path, pd.DataFrame],
    max_lag: int = 5,
    series_id: Union[str, None] = None,
    output_dir: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> SolarLagAnalysisResult:
    """Run the complete end-to-end RWI / Sunspot solar-cycle lag analysis pipeline.

    Parameters
    ----------
    rwi_input : str, Path, or pd.DataFrame
        RWI data path or DataFrame.
    sunspot_input : str, Path, or pd.DataFrame
        Sunspot data path or DataFrame.
    max_lag : int, default 5
        Maximum lag in years to evaluate.
    series_id : str, optional
        If specified, filters multi-series RWI data to this specific series_id.
        If None and multiple series exist, a site-level mean chronology is
        analyzed while preserving all individual series in the merged dataset.
    output_dir : str or Path, optional
        Directory to export output CSV files.
    overwrite : bool, default False
        Whether to overwrite existing output files.

    Returns
    -------
    SolarLagAnalysisResult
        Analysis result object containing aligned data, lag correlation table,
        optimal lag, and execution summary.
    """
    # 1. Load and validate
    rwi_df = load_rwi_data(rwi_input)
    sunspot_df = load_sunspot_data(sunspot_input)

    # 2. Merge on calendar year
    merged = merge_datasets(rwi_df, sunspot_df)

    # 2b. Validate temporal continuity before smoothing (spec §9)
    if "series_id" in merged.columns:
        for sid, grp in merged.groupby("series_id"):
            gaps = validate_years(grp, year_col="year")
            if gaps:
                logger.warning(
                    "Series '%s': %d missing year(s) in [%d, %d]. "
                    "Smoothing will produce NaN for windows spanning gaps.",
                    sid,
                    len(gaps),
                    int(grp["year"].min()),
                    int(grp["year"].max()),
                )
    else:
        gaps = validate_years(merged, year_col="year")
        if gaps:
            logger.warning(
                "%d missing year(s) in merged data [%d, %d]. "
                "Smoothing will produce NaN for windows spanning gaps.",
                len(gaps),
                int(merged["year"].min()),
                int(merged["year"].max()),
            )
            if len(gaps) > merged["year"].nunique() * 0.5:
                raise TemporalContinuityError(
                    f"Severe temporal discontinuity: {len(gaps)} missing years "
                    f"exceed 50% of the {merged['year'].nunique()} present years. "
                    f"The 11-year centered moving average cannot produce meaningful results."
                )

    # 3. Apply 11-year centered Schwabe smoothing and standardization
    if "series_id" in merged.columns and series_id is not None:
        merged = merged[merged["series_id"] == series_id].copy()

    if "series_id" in merged.columns:
        # Sunspot Number is a single global time series — smooth it once
        unique_sn = (
            merged[["year", "sunspot"]]
            .drop_duplicates("year")
            .sort_values("year")
            .reset_index(drop=True)
        )
        sn_smoothed_arr = apply_centered_smoothing(
            unique_sn["sunspot"].values,
            window=SCHWABE_WINDOW,
            years=unique_sn["year"].values,
        )
        sn_z_arr = standardize_series(sn_smoothed_arr, ddof=1)
        global_sn_df = pd.DataFrame({
            "year": unique_sn["year"].values,
            "sunspot_smoothed": sn_smoothed_arr,
            "sunspot_z": sn_z_arr,
        })

        # RWI is series-specific — smooth and standardize per series
        smoothed_frames = []
        for sid, grp in merged.groupby("series_id"):
            grp = grp.sort_values("year").copy()
            grp["rwi_smoothed"] = apply_centered_smoothing(
                grp["rwi"].values, window=SCHWABE_WINDOW, years=grp["year"].values
            )
            try:
                grp["rwi_z"] = standardize_series(grp["rwi_smoothed"], ddof=1)
            except ValueError as exc:
                logger.warning("Series '%s': RWI standardization warning: %s", sid, exc)
                grp["rwi_z"] = np.nan
            # Attach globally-smoothed sunspot values (consistent across all series)
            grp = grp.merge(global_sn_df, on="year", how="left")
            smoothed_frames.append(grp)
        processed_df = pd.concat(smoothed_frames, ignore_index=True)
        processed_df = processed_df.sort_values(["series_id", "year"]).reset_index(drop=True)

        # For lag analysis: if series_id was not singled out, build site mean chronology
        if series_id is None:
            logger.info(
                "Computing site-mean RWI chronology across %d series for lag analysis.",
                processed_df["series_id"].nunique(),
            )
            chrono = (
                processed_df.groupby("year")
                .agg(rwi=("rwi", "mean"))
                .reset_index()
                .sort_values("year")
            )
            chrono["rwi_smoothed"] = apply_centered_smoothing(
                chrono["rwi"].values, window=SCHWABE_WINDOW, years=chrono["year"].values
            )
            chrono["rwi_z"] = standardize_series(chrono["rwi_smoothed"], ddof=1)
            # Reuse globally-smoothed sunspot (no redundant recomputation)
            chrono = chrono.merge(unique_sn, on="year", how="left")
            chrono = chrono.merge(global_sn_df, on="year", how="left")
            analysis_df = chrono
        else:
            analysis_df = processed_df
    else:
        # Single series
        merged = merged.sort_values("year").reset_index(drop=True)
        merged["rwi_smoothed"] = apply_centered_smoothing(
            merged["rwi"].values, window=SCHWABE_WINDOW, years=merged["year"].values
        )
        merged["sunspot_smoothed"] = apply_centered_smoothing(
            merged["sunspot"].values, window=SCHWABE_WINDOW, years=merged["year"].values
        )
        merged["rwi_z"] = standardize_series(merged["rwi_smoothed"], ddof=1)
        merged["sunspot_z"] = standardize_series(merged["sunspot_smoothed"], ddof=1)
        processed_df = merged
        analysis_df = merged

    # 4. Calculate lag correlations
    lag_df = calculate_lag_correlations(
        analysis_df,
        rwi_col="rwi_z",
        sunspot_col="sunspot_z",
        max_lag=max_lag,
    )

    # 5. Select optimal lag
    optimal = select_optimal_lag(lag_df)

    # 6. Build final aligned DataFrame
    aligned_df = build_final_aligned_dataframe(processed_df, optimal_lag=optimal.optimal_lag)

    # Summary dictionary
    summary: dict[str, Any] = {
        "analysis": "RWI / Sunspot Solar-Cycle Lag Analysis",
        "smoothing_window_years": SCHWABE_WINDOW,
        "smoothing_type": "11-year centered moving average",
        "standardization": "z-score with sample standard deviation (ddof=1)",
        "lag_formula": "R(tau) = corr(RWI(t), SN(t - tau))",
        "evaluated_lags": list(range(max_lag + 1)),
        "optimal_lag_years": optimal.optimal_lag,
        "optimal_pearson_r": optimal.optimal_correlation,
        "optimal_direction": optimal.correlation_direction,
        "optimal_p_value": optimal.p_value,
        "optimal_n_observations": optimal.n_observations,
        "n_total_merged_rows": len(aligned_df),
        "year_min": int(aligned_df["year"].min()),
        "year_max": int(aligned_df["year"].max()),
    }

    # 7. Export if requested
    if output_dir is not None:
        export_results(aligned_df, lag_df, summary, output_dir=output_dir, overwrite=overwrite)

    return SolarLagAnalysisResult(
        aligned_data=aligned_df,
        lag_correlations=lag_df,
        optimal_lag=optimal,
        summary=summary,
    )


def export_results(
    aligned_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Union[str, Path],
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export analysis results to CSV and JSON files.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned and lagged dataset.
    lag_df : pd.DataFrame
        Lag correlation table.
    summary : dict
        Analysis summary metadata.
    output_dir : str or Path
        Destination directory.
    overwrite : bool, default False
        Whether to overwrite existing files.

    Returns
    -------
    dict[str, Path]
        Paths of the exported files:
        ``'aligned_data'``, ``'lag_correlations'``, ``'summary'``.

    Raises
    ------
    FileExistsError
        If output files already exist and overwrite is False.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = out_dir / "processed_lagged_data.csv"
    lag_path = out_dir / "lag_correlation_results.csv"
    summary_path = out_dir / "analysis_summary.json"

    for p in (aligned_path, lag_path, summary_path):
        if p.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {p} already exists. Set overwrite=True to replace."
            )

    # Export aligned data (no pandas index, UTF-8)
    aligned_df.to_csv(aligned_path, index=False, encoding="utf-8")

    # Export lag correlations (no pandas index, UTF-8)
    lag_df.to_csv(lag_path, index=False, encoding="utf-8")

    # Export summary metadata JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Exported results to directory: %s", out_dir.resolve())
    return {
        "aligned_data": aligned_path.resolve(),
        "lag_correlations": lag_path.resolve(),
        "summary": summary_path.resolve(),
    }
