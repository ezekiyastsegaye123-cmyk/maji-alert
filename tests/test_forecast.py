"""
Test suite for tree-ring & solar-cycle Random Forest drought forecasting pipeline.
===================================================================================

Covers:
1. SolarCyclePhaseCalculator (monotonicity, harmonic identities, edge bounds)
2. DroughtFeatureEngineer (lags, growth memory, schema validation, merge checks)
3. DroughtForecaster (Random Forest fitting, continuous/classification predictions, CI bounds)
4. Cross-Validation (KFold, TimeSeriesSplit metrics computation)
5. Forward 11-Year Forecast (2025-2035 projection, probability bounds, risk tiers)
6. Pipeline Execution & File Exporters
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from treering.forecast import (
    CrossValidationMetrics,
    DroughtFeatureEngineer,
    DroughtForecaster,
    FeatureEngineeringError,
    ForecastError,
    ForecastYearResult,
    ModelTrainingError,
    ProjectionError,
    SolarCyclePhaseCalculator,
    run_drought_forecasting_pipeline,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def synthetic_sunspot_df() -> pd.DataFrame:
    """Generates 120 years (1900-2019) of synthetic sunspot data with ~11-year cycle."""
    years = np.arange(1900, 2020)
    # Sine wave with 11-year period + noise
    sn = 75.0 + 65.0 * np.sin(2.0 * np.pi * (years - 1902) / 11.0) + np.random.normal(0, 5, len(years))
    sn = np.clip(sn, 0, 250)
    return pd.DataFrame({"year": years, "sunspot": sn})


@pytest.fixture
def synthetic_tree_ring_df() -> pd.DataFrame:
    """Generates 120 years (1900-2019) of synthetic RWI data."""
    years = np.arange(1900, 2020)
    rwi = 1.0 + 0.2 * np.sin(2.0 * np.pi * (years - 1902) / 11.0) + np.random.normal(0, 0.1, len(years))
    return pd.DataFrame({"year": years, "rwi": rwi})


@pytest.fixture
def synthetic_spei_df() -> pd.DataFrame:
    """Generates 120 years (1900-2019) of synthetic SPEI data."""
    years = np.arange(1900, 2020)
    spei = 0.0 - 0.25 * np.sin(2.0 * np.pi * (years - 1902) / 11.0) + np.random.normal(0, 0.2, len(years))
    return pd.DataFrame({"year": years, "spei": spei})


# =====================================================================
# Test SolarCyclePhaseCalculator
# =====================================================================


class TestSolarCyclePhaseCalculator:
    def test_default_minima_phase_bounds(self):
        calc = SolarCyclePhaseCalculator()
        for yr in range(1900, 2040):
            phase = calc.compute_phase(yr)
            assert 0.0 <= phase < 1.0, f"Phase {phase} out of [0, 1) for year {yr}"

    def test_phase_harmonic_identity(self):
        calc = SolarCyclePhaseCalculator()
        for yr in [1902, 1913, 1950, 1986, 2019, 2025, 2030]:
            phase, sin_p, cos_p = calc.compute_harmonics(yr)
            assert 0.0 <= phase < 1.0
            # sin^2 + cos^2 == 1
            assert np.isclose(sin_p**2 + cos_p**2, 1.0, atol=1e-6)

    def test_known_minima_phase_zero(self):
        calc = SolarCyclePhaseCalculator()
        # Years that match minima should have phase ~0.0
        assert np.isclose(calc.compute_phase(1913), 0.0, atol=1e-5)
        assert np.isclose(calc.compute_phase(2019), 0.0, atol=1e-5)
        assert np.isclose(calc.compute_phase(2030), 0.0, atol=1e-5)

    def test_invalid_minima_raises(self):
        with pytest.raises(ValueError):
            SolarCyclePhaseCalculator(minima=(2000,))
        with pytest.raises(ValueError):
            # Non-monotonic
            SolarCyclePhaseCalculator(minima=(2010, 2005, 2020))


# =====================================================================
# Test DroughtFeatureEngineer
# =====================================================================


class TestDroughtFeatureEngineer:
    def test_build_solar_feature_table(self, synthetic_sunspot_df):
        engineer = DroughtFeatureEngineer()
        df_feats = engineer.build_solar_feature_table(synthetic_sunspot_df)

        expected_cols = [
            "year", "sunspot", "sunspot_lag1", "sunspot_lag2", "sunspot_lag3",
            "sunspot_lag4", "sunspot_lag5", "sunspot_smooth11", "sunspot_diff1",
            "sunspot_diff3", "solar_phase", "solar_phase_sin", "solar_phase_cos"
        ]
        for c in expected_cols:
            assert c in df_feats.columns
            assert not df_feats[c].isna().any(), f"NaN in {c}"

    def test_build_solar_missing_columns_raises(self):
        engineer = DroughtFeatureEngineer()
        with pytest.raises(FeatureEngineeringError):
            engineer.build_solar_feature_table(pd.DataFrame({"invalid": [1, 2, 3]}))

    def test_build_tree_ring_chronology(self, synthetic_tree_ring_df):
        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)

        assert "year" in df_chron.columns
        assert "rwi" in df_chron.columns
        assert "rwi_lag1" in df_chron.columns
        assert "rwi_diff1" in df_chron.columns
        assert "rwi_smooth5" in df_chron.columns
        assert not df_chron.isna().any().any()

    def test_build_training_dataset_merge(
        self, synthetic_tree_ring_df, synthetic_sunspot_df, synthetic_spei_df
    ):
        engineer = DroughtFeatureEngineer(drought_threshold=-0.25)
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)
        df_solar = engineer.build_solar_feature_table(synthetic_sunspot_df)
        df_train = engineer.build_training_dataset(df_chron, df_solar, synthetic_spei_df)

        assert len(df_train) == 120
        assert "drought_class" in df_train.columns
        assert set(df_train["drought_class"].unique()).issubset({0, 1})
        for f in DroughtFeatureEngineer.FEATURE_NAMES:
            assert f in df_train.columns
            assert not df_train[f].isna().any()


# =====================================================================
# Test DroughtForecaster & Cross-Validation
# =====================================================================


class TestDroughtForecaster:
    def test_fit_and_feature_importances(
        self, synthetic_tree_ring_df, synthetic_sunspot_df, synthetic_spei_df
    ):
        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)
        df_solar = engineer.build_solar_feature_table(synthetic_sunspot_df)
        df_train = engineer.build_training_dataset(df_chron, df_solar, synthetic_spei_df)

        forecaster = DroughtForecaster(n_estimators=50, random_state=42)
        forecaster.fit(df_train)

        assert forecaster.classifier is not None
        assert forecaster.regressor is not None
        assert forecaster.feature_importances_ is not None

        # Feature importances sum to 1
        total_imp = sum(forecaster.feature_importances_.values())
        assert np.isclose(total_imp, 1.0, atol=1e-4)

    def test_kfold_cross_validation_metrics(
        self, synthetic_tree_ring_df, synthetic_sunspot_df, synthetic_spei_df
    ):
        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)
        df_solar = engineer.build_solar_feature_table(synthetic_sunspot_df)
        df_train = engineer.build_training_dataset(df_chron, df_solar, synthetic_spei_df)

        forecaster = DroughtForecaster(n_estimators=30, random_state=42)
        forecaster.fit(df_train)

        metrics = forecaster.evaluate_cross_validation(df_train, n_splits=3, cv_type="kfold")
        assert isinstance(metrics, CrossValidationMetrics)
        assert 0.0 <= metrics.classification_accuracy <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.regression_mae > 0.0

    def test_timeseries_cross_validation_metrics(
        self, synthetic_tree_ring_df, synthetic_sunspot_df, synthetic_spei_df
    ):
        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)
        df_solar = engineer.build_solar_feature_table(synthetic_sunspot_df)
        df_train = engineer.build_training_dataset(df_chron, df_solar, synthetic_spei_df)

        forecaster = DroughtForecaster(n_estimators=30, random_state=42)
        forecaster.fit(df_train)

        metrics = forecaster.evaluate_cross_validation(df_train, n_splits=3, cv_type="timeseries")
        assert metrics.cv_method == "timeseries"
        assert metrics.n_splits == 3

    def test_forecast_without_fit_raises(self):
        forecaster = DroughtForecaster()
        with pytest.raises(ModelTrainingError):
            forecaster.forecast_solar_cycle(2025, 2035)

    def test_forecast_solar_cycle_outputs(
        self, synthetic_tree_ring_df, synthetic_sunspot_df, synthetic_spei_df
    ):
        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(synthetic_tree_ring_df)
        df_solar = engineer.build_solar_feature_table(synthetic_sunspot_df)
        df_train = engineer.build_training_dataset(df_chron, df_solar, synthetic_spei_df)

        forecaster = DroughtForecaster(n_estimators=50, random_state=42)
        forecaster.fit(df_train)

        results = forecaster.forecast_solar_cycle(
            start_year=2025,
            end_year=2035,
            df_historical_sun=synthetic_sunspot_df,
        )

        assert len(results) == 11
        for idx, res in enumerate(results):
            expected_yr = 2025 + idx
            assert res.year == expected_yr
            assert 0.0 <= res.solar_phase < 1.0
            assert 0.0 <= res.drought_probability <= 1.0
            assert res.risk_level in {"Low", "Guarded", "Elevated", "High"}
            assert res.spei_ci_lower_95 <= res.predicted_spei <= res.spei_ci_upper_95


# =====================================================================
# End-to-End Pipeline & Real Data Tests
# =====================================================================


class TestEndToEndPipeline:
    def test_real_pipeline_execution(self, tmp_path):
        out_f = tmp_path / "test_forecast.csv"
        out_m = tmp_path / "test_metrics.json"
        out_b = tmp_path / "test_backtest.csv"

        df_fc, metrics = run_drought_forecasting_pipeline(
            processed_lagged_data_path="results/processed_lagged_data.csv",
            spei_csv_path="results/spei_debrebirkan.csv",
            sunspot_csv_path="SN_y_tot_V2.0.csv",
            output_forecast_csv=out_f,
            output_metrics_json=out_m,
            output_backtesting_csv=out_b,
            drought_threshold=-0.25,
            overwrite=True,
        )

        assert len(df_fc) == 11
        assert out_f.exists()
        assert out_m.exists()
        assert out_b.exists()

        # Check forecast schema
        assert list(df_fc.columns) == [
            "year", "projected_sunspot", "solar_phase", "predicted_spei",
            "spei_ci_lower_95", "spei_ci_upper_95", "drought_probability",
            "risk_level", "predicted_class"
        ]
        assert df_fc["year"].min() == 2025
        assert df_fc["year"].max() == 2035

        # Check metrics payload
        with open(out_m, "r") as f:
            data = json.load(f)
        assert data["n_training_samples"] == 114
        assert "kfold_cv_metrics" in data
        assert "feature_importances" in data
