"""CSV export for detrended tree-ring data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# Expected column order for the output CSV.
_OUTPUT_COLUMNS = [
    "series_id",
    "year",
    "raw_ring_width",
    "fitted_growth",
    "rwi",
]


class ExportError(Exception):
    """Raised when CSV export fails."""


def export_csv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """Export a detrended DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``series_id``, ``year``,
        ``raw_ring_width``, ``fitted_growth``, ``rwi``.
    output_path : str or Path
        Destination CSV file path.
    overwrite : bool, default False
        If ``False``, raise if *output_path* already exists.

    Returns
    -------
    Path
        The absolute path of the written CSV file.

    Raises
    ------
    ExportError
        If the DataFrame is missing required columns, the file
        already exists (and *overwrite* is False), or writing fails.
    """
    path = Path(output_path)

    # Validate columns
    missing = set(_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ExportError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    if not overwrite and path.exists():
        raise ExportError(
            f"Output file already exists: {path}. "
            f"Use overwrite=True or --overwrite to replace."
        )

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df[_OUTPUT_COLUMNS].to_csv(
            path,
            index=False,
            encoding="utf-8",
        )
    except OSError as exc:
        raise ExportError(
            f"Failed to write CSV to {path}: {exc}"
        ) from exc

    logger.info("Exported %d rows to %s", len(df), path)
    return path.resolve()
