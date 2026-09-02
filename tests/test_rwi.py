"""Tests for treering.rwi — Ring Width Index calculation."""

from __future__ import annotations

import numpy as np
import pytest

from treering.rwi import RWIError, calculate_rwi


class TestCalculateRWI:
    """Unit tests for RWI calculation."""

    def test_simple_division(self) -> None:
        """RWI = raw / fitted."""
        raw = np.array([5.0, 10.0, 15.0])
        fitted = np.array([4.0, 8.0, 10.0])
        rwi = calculate_rwi(raw, fitted, series_id="simple")
        expected = np.array([1.25, 1.25, 1.5])
        np.testing.assert_allclose(rwi, expected)

    def test_perfect_fit(self) -> None:
        """When raw == fitted, RWI should be 1.0 everywhere."""
        raw = np.array([100.0, 200.0, 300.0])
        fitted = np.array([100.0, 200.0, 300.0])
        rwi = calculate_rwi(raw, fitted, series_id="perfect")
        np.testing.assert_allclose(rwi, 1.0)

    def test_result_is_finite(self) -> None:
        raw = np.array([10.0, 20.0, 30.0, 40.0])
        fitted = np.array([12.0, 18.0, 25.0, 35.0])
        rwi = calculate_rwi(raw, fitted, series_id="finite")
        assert np.all(np.isfinite(rwi))

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(RWIError, match="length"):
            calculate_rwi(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                series_id="mismatch",
            )

    def test_non_finite_raw(self) -> None:
        raw = np.array([1.0, np.nan, 3.0])
        fitted = np.array([1.0, 2.0, 3.0])
        with pytest.raises(RWIError, match="non-finite"):
            calculate_rwi(raw, fitted, series_id="nan_raw")

    def test_non_finite_fitted(self) -> None:
        raw = np.array([1.0, 2.0, 3.0])
        fitted = np.array([1.0, np.inf, 3.0])
        with pytest.raises(RWIError, match="non-finite"):
            calculate_rwi(raw, fitted, series_id="inf_fitted")

    def test_near_zero_fitted(self) -> None:
        """Fitted values near zero should be rejected."""
        raw = np.array([1.0, 2.0, 3.0])
        fitted = np.array([1.0, 1e-10, 3.0])
        with pytest.raises(RWIError, match="below tolerance"):
            calculate_rwi(raw, fitted, series_id="nearzero")

    def test_zero_fitted(self) -> None:
        raw = np.array([1.0, 2.0, 3.0])
        fitted = np.array([1.0, 0.0, 3.0])
        with pytest.raises(RWIError, match="below tolerance"):
            calculate_rwi(raw, fitted, series_id="zero")
