"""Tests for treering.model — negative exponential model and curve fitting."""

from __future__ import annotations

import numpy as np
import pytest

from treering.model import (
    FitResult,
    FittingError,
    GROWTH_FLOOR,
    MIN_OBSERVATIONS,
    fit_growth_curve,
    negative_exponential,
)


class TestNegativeExponential:
    """Unit tests for the mathematical model function."""

    def test_at_t_zero(self) -> None:
        """G(0) = a + c."""
        result = negative_exponential(np.array([0.0]), a=10.0, b=0.5, c=3.0)
        np.testing.assert_allclose(result, [13.0])

    def test_large_t(self) -> None:
        """For very large t, G(t) ≈ c (the asymptote)."""
        result = negative_exponential(np.array([1000.0]), a=10.0, b=0.5, c=3.0)
        np.testing.assert_allclose(result, [3.0], atol=1e-10)

    def test_vector_input(self) -> None:
        """Vectorized evaluation."""
        t = np.array([0.0, 1.0, 2.0])
        result = negative_exponential(t, a=10.0, b=1.0, c=5.0)
        expected = 10.0 * np.exp(-1.0 * t) + 5.0
        np.testing.assert_allclose(result, expected)

    def test_known_values(self) -> None:
        """Manually computed values."""
        # G(1) = 2 * exp(-0.5 * 1) + 1 = 2 * 0.60653... + 1 = 2.21306...
        result = negative_exponential(np.array([1.0]), a=2.0, b=0.5, c=1.0)
        expected = 2.0 * np.exp(-0.5) + 1.0
        np.testing.assert_allclose(result, [expected])

    def test_zero_decay_rate(self) -> None:
        """b=0 means no decay: G(t) = a + c for all t."""
        t = np.array([0.0, 10.0, 100.0])
        result = negative_exponential(t, a=5.0, b=0.0, c=2.0)
        np.testing.assert_allclose(result, [7.0, 7.0, 7.0])


class TestFitGrowthCurve:
    """Tests for curve fitting."""

    def test_synthetic_recovery(self) -> None:
        """Fitting synthetic data should recover known parameters."""
        np.random.seed(42)
        a_true, b_true, c_true = 300.0, 0.05, 150.0
        t = np.arange(0, 80, dtype=np.float64)
        years = t + 1900
        truth = negative_exponential(t, a_true, b_true, c_true)
        noise = np.random.normal(0, 5, size=len(t))
        ring_widths = truth + noise

        fit = fit_growth_curve(years, ring_widths, series_id="synth")

        assert isinstance(fit, FitResult)
        np.testing.assert_allclose(fit.a, a_true, rtol=0.15)
        np.testing.assert_allclose(fit.b, b_true, rtol=0.30)
        np.testing.assert_allclose(fit.c, c_true, rtol=0.15)

    def test_fitted_values_shape(self) -> None:
        """Fitted values array must match input length."""
        np.random.seed(0)
        t = np.arange(50, dtype=np.float64)
        rw = 200 * np.exp(-0.03 * t) + 100 + np.random.normal(0, 3, 50)
        years = t + 1950

        fit = fit_growth_curve(years, rw, series_id="test")
        assert len(fit.fitted_values) == 50

    def test_fitted_values_finite(self) -> None:
        """All fitted values must be finite."""
        np.random.seed(1)
        t = np.arange(30, dtype=np.float64)
        rw = 100 * np.exp(-0.05 * t) + 50 + np.random.normal(0, 2, 30)
        years = t + 2000

        fit = fit_growth_curve(years, rw, series_id="finite_test")
        assert np.all(np.isfinite(fit.fitted_values))

    def test_fitted_values_positive(self) -> None:
        """All fitted values must be above GROWTH_FLOOR."""
        np.random.seed(2)
        t = np.arange(40, dtype=np.float64)
        rw = 150 * np.exp(-0.04 * t) + 80 + np.random.normal(0, 3, 40)
        years = t + 1980

        fit = fit_growth_curve(years, rw, series_id="pos_test")
        assert np.all(fit.fitted_values >= GROWTH_FLOOR)

    def test_too_few_observations(self) -> None:
        """Should raise with fewer than MIN_OBSERVATIONS points."""
        with pytest.raises(FittingError, match="only"):
            fit_growth_curve(
                np.array([1900, 1901, 1902]),
                np.array([100.0, 90.0, 85.0]),
                series_id="short",
            )

    def test_non_finite_years(self) -> None:
        years = np.array([1900, np.nan, 1902] + [1903 + i for i in range(8)])
        rw = np.ones(11) * 100.0
        with pytest.raises(FittingError, match="non-finite"):
            fit_growth_curve(years, rw, series_id="nan_year")

    def test_non_finite_ring_widths(self) -> None:
        years = np.arange(1900, 1911, dtype=np.float64)
        rw = np.ones(11) * 100.0
        rw[5] = np.inf
        with pytest.raises(FittingError, match="non-finite"):
            fit_growth_curve(years, rw, series_id="inf_rw")

    def test_negative_ring_widths(self) -> None:
        years = np.arange(1900, 1911, dtype=np.float64)
        rw = np.ones(11) * 100.0
        rw[3] = -5.0
        with pytest.raises(FittingError, match="negative"):
            fit_growth_curve(years, rw, series_id="neg_rw")

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(FittingError, match="length"):
            fit_growth_curve(
                np.arange(10, dtype=np.float64),
                np.ones(5),
                series_id="mismatch",
            )
