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
