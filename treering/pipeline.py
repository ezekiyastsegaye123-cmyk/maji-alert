"""End-to-end detrending pipeline.

Orchestrates: parse → fit → RWI → output DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from treering.model import fit_growth_curve, FittingError
from treering.parser import parse_rwl
from treering.rwi import calculate_rwi, RWIError

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the detrending pipeline encounters a fatal error."""


def process_rwl(
    filepath: Union[str, Path],
    *,
    skip_failed_series: bool = False,
) -> pd.DataFrame:
    """Run the full detrending pipeline on a Tucson ``.rwl`` file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.rwl`` file.
    skip_failed_series : bool, default False
        If ``True``, series that fail curve fitting are logged as
        warnings and omitted from the result instead of raising.
        If ``False`` (default), any fitting failure raises immediately.

    Returns
    -------
    pd.DataFrame
        Columns: ``series_id``, ``year``, ``raw_ring_width``,
        ``fitted_growth``, ``rwi``.  Sorted by ``(series_id, year)``.

    Raises
    ------
    PipelineError
        If no series can be successfully processed.
    FittingError
        If a series' curve fit fails and *skip_failed_series* is False.
    RWIError
        If RWI calculation fails.
    """
    raw_df = parse_rwl(filepath)

    series_ids = raw_df["series_id"].unique()
    logger.info("Processing %d series", len(series_ids))

    result_frames: list[pd.DataFrame] = []
    failed_series: list[str] = []

    for sid in series_ids:
        series_df = raw_df[raw_df["series_id"] == sid].copy()
        series_df = series_df.sort_values("year").reset_index(drop=True)

        years = series_df["year"].values
        ring_widths = series_df["ring_width"].values.astype(np.float64)

        # --- Fit growth curve -----------------------------------------------
        try:
            fit = fit_growth_curve(
                years, ring_widths, series_id=sid
            )
        except FittingError as exc:
            if skip_failed_series:
                logger.warning(
                    "Skipping series '%s': %s", sid, exc
                )
                failed_series.append(sid)
                continue
            raise

        # --- Calculate RWI --------------------------------------------------
        try:
            rwi = calculate_rwi(
                ring_widths, fit.fitted_values, series_id=sid
            )
        except RWIError as exc:
            if skip_failed_series:
                logger.warning(
                    "Skipping series '%s' (RWI): %s", sid, exc
                )
                failed_series.append(sid)
                continue
            raise

        # --- Assemble output ------------------------------------------------
        series_df = series_df[["series_id", "year", "ring_width"]].copy()
        series_df = series_df.rename(columns={"ring_width": "raw_ring_width"})
        series_df["fitted_growth"] = fit.fitted_values
        series_df["rwi"] = rwi

        result_frames.append(series_df)

    if not result_frames:
        raise PipelineError(
            f"No series could be successfully processed from {filepath}. "
            f"Failed series: {failed_series}"
        )

    if failed_series:
        logger.warning(
            "%d series failed processing: %s",
            len(failed_series),
            ", ".join(failed_series),
        )

    result = pd.concat(result_frames, ignore_index=True)
    result = result.sort_values(["series_id", "year"]).reset_index(drop=True)

    # Final validation
    assert result.columns.tolist() == [
        "series_id",
        "year",
        "raw_ring_width",
        "fitted_growth",
        "rwi",
    ], "Unexpected output schema"

    _validate_output(result)

    return result


def biweight_robust_mean(
    values: Union[np.ndarray, Sequence[float]],
    c: float = 9.0,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> float:
    """Calculate Tukey's biweight robust mean for an array of observations.

    Parameters
    ----------
    values : array-like
        Numeric array of measurements (e.g. annual RWI across multiple cores).
    c : float, default 9.0
        Tuning constant (typically 6.0 to 9.0 in dendrochronology).
    max_iter : int, default 10
        Maximum iterations for M-estimator convergence.
    tol : float, default 1e-4
        Convergence tolerance on location change.

    Returns
    -------
    float
        The biweight robust mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float(np.nan)
    if n <= 2:
        return float(np.mean(arr))

    m = float(np.median(arr))
    for _ in range(max_iter):
        dev = arr - m
        mad = float(np.median(np.abs(dev)))
        if mad < 1e-8:
            return m

        u = dev / (c * mad)
        mask = np.abs(u) < 1.0
        if not np.any(mask):
            return m

        w = np.zeros(n, dtype=np.float64)
        w[mask] = (1.0 - u[mask] ** 2) ** 2
        sum_w = float(np.sum(w))
        if sum_w < 1e-12:
            return m

        new_m = float(np.sum(w * arr) / sum_w)
        if abs(new_m - m) < tol:
            return new_m
        m = new_m

    return m


def process_multiple_rwl(
    filepaths: Sequence[Union[str, Path]],
    *,
    skip_failed_series: bool = True,
    exclude_holdout: Sequence[str] = ("eth001", "eth001.rwl"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest multiple .rwl files across national sites and compute a unified master chronology.

    Applies negative exponential biological growth detrending to all cores across
    the provided sites and computes an annual biweight robust mean to form the
    unified Ethiopian Master Chronology.

    Parameters
    ----------
    filepaths : sequence of str or Path
        List of paths to .rwl files (e.g. eth002.rwl through eth007.rwl).
    skip_failed_series : bool, default True
        If True, skip series that fail curve fitting instead of raising.
    exclude_holdout : sequence of str, default ("eth001", "eth001.rwl")
        Site identifiers strictly quarantined from the training pipeline.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        1. all_cores_df: Combined detrended series DataFrame across all sites with columns
           ['site', 'series_id', 'year', 'raw_ring_width', 'fitted_growth', 'rwi'].
        2. master_chronology_df: Unified national chronology with columns
           ['year', 'rwi', 'core_count'].

    Raises
    ------
    ValueError
        If any excluded holdout file (such as eth001) is detected in filepaths.
    PipelineError
        If no files or series can be processed successfully.
    """
    if not filepaths:
        raise PipelineError("No .rwl filepaths provided for multi-site processing.")

    # Strict holdout isolation enforcement
    for p in filepaths:
        p_str = str(p)
        p_name = Path(p).name
        for holdout in exclude_holdout:
            if holdout in p_str or holdout == p_name:
                raise ValueError(
                    f"Quarantined geographic holdout '{holdout}' detected in multi-site training list ({p_str})! "
                    f"Holdout dataset (eth001) must remain strictly isolated."
                )

    all_frames: List[pd.DataFrame] = []

    for path_item in filepaths:
        path_obj = Path(path_item)
        site_name = path_obj.stem
        logger.info("Ingesting multi-site RWL: %s (site: %s)", path_obj, site_name)
        try:
            df_site = process_rwl(path_obj, skip_failed_series=skip_failed_series)
            df_site["site"] = site_name
            # Ensure series IDs are globally unique across sites
            df_site["series_id"] = site_name + "_" + df_site["series_id"].astype(str)
            all_frames.append(df_site)
        except Exception as exc:
            if skip_failed_series:
                logger.warning("Failed processing file %s: %s. Skipping.", path_obj, exc)
                continue
            raise

    if not all_frames:
        raise PipelineError("No valid series could be processed across the provided multi-site RWL files.")

    all_cores_df = pd.concat(all_frames, ignore_index=True)
    all_cores_df = all_cores_df.sort_values(["site", "series_id", "year"]).reset_index(drop=True)

    # Calculate unified annual biweight robust mean
    chron_records = []
    for year_val, group in all_cores_df.groupby("year"):
        rwi_vals = group["rwi"].values
        bw_val = biweight_robust_mean(rwi_vals)
        chron_records.append({
            "year": int(year_val),
            "rwi": float(bw_val),
            "core_count": int(len(group)),
        })

    master_chronology_df = pd.DataFrame(chron_records).sort_values("year").reset_index(drop=True)
    logger.info(
        "Built Ethiopian Master Chronology: %d years (%d–%d) from %d series across %d sites.",
        len(master_chronology_df),
        master_chronology_df["year"].min(),
        master_chronology_df["year"].max(),
        all_cores_df["series_id"].nunique(),
        all_cores_df["site"].nunique(),
    )

    return all_cores_df, master_chronology_df


def _validate_output(df: pd.DataFrame) -> None:
    """Sanity-check the final output DataFrame.

    Raises
    ------
    PipelineError
        If any RWI or fitted_growth values are non-finite.
    """
    for col in ("fitted_growth", "rwi"):
        non_finite = ~np.isfinite(df[col].values)
        if non_finite.any():
            bad_rows = df[non_finite][["series_id", "year", col]]
            raise PipelineError(
                f"Output contains non-finite '{col}' values:\n{bad_rows}"
            )
