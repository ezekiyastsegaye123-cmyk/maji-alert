"""Negative exponential growth model and curve fitting.

Growth Model
------------
The biological growth model is:

    G(t) = a * exp(-b * t) + c

where:

* ``t`` — age index (years since first measurement, starting from 0).
* ``a`` — amplitude of the decaying component (must be > 0).
* ``b`` — decay rate (must be > 0).
* ``c`` — asymptotic minimum growth (must be > 0).
* ``G(t)`` — expected biological ring-width growth.

Fitting
-------
We use ``scipy.optimize.curve_fit`` with bounded parameters to ensure
the fitted curve is biologically plausible (positive amplitude, positive
decay, positive asymptote).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

logger = logging.getLogger(__name__)

# Minimum number of observations needed to fit three parameters (a, b, c).
MIN_OBSERVATIONS = 10

# Tolerance below which a fitted growth value is considered too close to
# zero for safe division.  Units match the ring-width measurement scale.
GROWTH_FLOOR = 1e-6


class FitResult(NamedTuple):
    """Container for curve-fit output.

    Attributes
    ----------
    a : float
        Amplitude of the decaying component.
    b : float
        Decay rate.
    c : float
        Asymptotic minimum growth.
    fitted_values : np.ndarray
        G(t) evaluated at every supplied *t* value.
    """

    a: float
    b: float
    c: float
    fitted_values: np.ndarray


class FittingError(Exception):
    """Raised when curve fitting fails or produces invalid results."""


def negative_exponential(
    t: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """Evaluate the negative exponential growth model.

    Parameters
    ----------
    t : array_like
        Age index values (≥ 0).
    a, b, c : float
        Model parameters.

    Returns
    -------
    np.ndarray
        G(t) = a * exp(-b * t) + c
    """
    t = np.asarray(t, dtype=np.float64)
    return a * np.exp(-b * t) + c


def fit_growth_curve(
    years: np.ndarray,
    ring_widths: np.ndarray,
    *,
    series_id: str = "<unknown>",
) -> FitResult:
    """Fit the negative exponential growth model to a single series.

    Parameters
    ----------
    years : array_like
        Calendar years (used only for ordering; internally converted to
        a 0-based age index ``t = year - min(year)``).
    ring_widths : array_like
        Measured ring-width values corresponding to *years*.
    series_id : str
        Identifier for error messages.

    Returns
    -------
    FitResult
        Fitted parameters and evaluated growth curve.

    Raises
    ------
    FittingError
        If there are too few observations, if input contains non-finite
        values, or if ``curve_fit`` cannot converge.
    """
    years = np.asarray(years, dtype=np.float64)
    ring_widths = np.asarray(ring_widths, dtype=np.float64)

    # --- Input validation ---------------------------------------------------
    if len(years) != len(ring_widths):
        raise FittingError(
            f"Series '{series_id}': years length ({len(years)}) != "
            f"ring_widths length ({len(ring_widths)})"
        )

    if len(years) < MIN_OBSERVATIONS:
        raise FittingError(
            f"Series '{series_id}': only {len(years)} observations; "
            f"need at least {MIN_OBSERVATIONS} for curve fitting"
        )

    if not np.all(np.isfinite(years)):
        raise FittingError(
            f"Series '{series_id}': years contain non-finite values"
        )

    if not np.all(np.isfinite(ring_widths)):
        raise FittingError(
            f"Series '{series_id}': ring_widths contain non-finite values"
        )

    if np.any(ring_widths < 0):
        raise FittingError(
            f"Series '{series_id}': ring_widths contain negative values"
        )

    # --- Convert to age index -----------------------------------------------
    t = years - years.min()

    # --- Initial parameter estimates ----------------------------------------
    rw_max = float(ring_widths.max())
    rw_min = float(ring_widths.min())
    rw_mean = float(ring_widths.mean())

    # a ≈ range of data (amplitude of decay)
    a0 = max(rw_max - rw_min, 1.0)
    # b ≈ modest decay rate
    t_range = float(t.max()) if t.max() > 0 else 1.0
    b0 = 2.0 / t_range
    # c ≈ minimum observed value (asymptote)
    c0 = max(rw_min, 0.1)

    p0 = [a0, b0, c0]

    # Parameter bounds: all positive, reasonable upper limits
    bounds = (
        [0.0, 0.0, 0.0],                  # lower bounds
        [rw_max * 10, 10.0, rw_max * 5],  # upper bounds
    )

    # --- Fit ----------------------------------------------------------------
    try:
        popt, _ = curve_fit(
            negative_exponential,
            t,
            ring_widths,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )
    except (RuntimeError, ValueError, OptimizeWarning) as exc:
        raise FittingError(
            f"Series '{series_id}': curve fitting failed — {exc}"
        ) from exc

    a_fit, b_fit, c_fit = popt

    # --- Evaluate fitted curve ----------------------------------------------
    fitted = negative_exponential(t, a_fit, b_fit, c_fit)

    # --- Post-fit validation ------------------------------------------------
    if not np.all(np.isfinite(fitted)):
        raise FittingError(
            f"Series '{series_id}': fitted growth curve contains "
            f"non-finite values (a={a_fit:.4f}, b={b_fit:.4f}, "
            f"c={c_fit:.4f})"
        )

    if np.any(fitted < GROWTH_FLOOR):
        raise FittingError(
            f"Series '{series_id}': fitted growth curve contains values "
            f"below tolerance {GROWTH_FLOOR} (min={fitted.min():.6f}). "
            f"Division would be numerically unstable."
        )

    logger.info(
        "Series '%s': fitted a=%.4f, b=%.6f, c=%.4f",
        series_id,
        a_fit,
        b_fit,
        c_fit,
    )

    return FitResult(
        a=a_fit,
        b=b_fit,
        c=c_fit,
        fitted_values=fitted,
    )
