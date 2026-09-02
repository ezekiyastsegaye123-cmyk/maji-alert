"""Ring Width Index (RWI) calculation.

RWI Calculation
---------------
For each observation:

    RWI_t = RawRingWidth_t / G(t)

where ``G(t)`` is the fitted negative exponential growth curve evaluated
at year *t*.

A well-detrended series has a mean RWI near 1.0.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from treering.model import GROWTH_FLOOR

logger = logging.getLogger(__name__)


class RWIError(Exception):
    """Raised when RWI calculation encounters invalid data."""


def calculate_rwi(
    raw_ring_widths: np.ndarray,
    fitted_growth: np.ndarray,
    *,
    series_id: str = "<unknown>",
) -> np.ndarray:
    """Calculate Ring Width Index (RWI) by dividing raw by fitted growth.

    Parameters
    ----------
    raw_ring_widths : array_like
        Observed ring-width measurements.
    fitted_growth : array_like
        Fitted biological growth curve ``G(t)``.
    series_id : str
        Series identifier for error messages.

    Returns
    -------
    np.ndarray
        RWI values (``raw / fitted``).

    Raises
    ------
    RWIError
        If inputs have mismatched lengths, contain non-finite values,
        or if fitted growth contains values at or below the safety floor.
    """
    raw = np.asarray(raw_ring_widths, dtype=np.float64)
    fitted = np.asarray(fitted_growth, dtype=np.float64)

    if len(raw) != len(fitted):
        raise RWIError(
            f"Series '{series_id}': raw_ring_widths length ({len(raw)}) != "
            f"fitted_growth length ({len(fitted)})"
        )

    if not np.all(np.isfinite(raw)):
        raise RWIError(
            f"Series '{series_id}': raw_ring_widths contain non-finite values"
        )

    if not np.all(np.isfinite(fitted)):
        raise RWIError(
            f"Series '{series_id}': fitted_growth contain non-finite values"
        )

    # Guard against division by zero / near-zero
    if np.any(fitted < GROWTH_FLOOR):
        bad_count = int(np.sum(fitted < GROWTH_FLOOR))
        raise RWIError(
            f"Series '{series_id}': {bad_count} fitted growth value(s) "
            f"below tolerance {GROWTH_FLOOR}; cannot safely compute RWI"
        )

    rwi = raw / fitted

    if not np.all(np.isfinite(rwi)):
        raise RWIError(
            f"Series '{series_id}': computed RWI contains non-finite values"
        )

    logger.debug(
        "Series '%s': RWI mean=%.4f, std=%.4f",
        series_id,
        rwi.mean(),
        rwi.std(),
    )

    return rwi
