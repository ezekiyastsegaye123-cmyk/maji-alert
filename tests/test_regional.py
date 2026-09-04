"""
Tests for Regional Curve Standardization (RCS), Isotopic Feature Ingestion,
and Regional Blind Holdout Model.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from treering.pipeline import biweight_robust_mean, process_multiple_rwl, PipelineError
from treering.forecast import DroughtFeatureEngineer, load_isotope_dataset
from treering.holdout import (
    calibrated_predict_proba,
    train_and_save_regional_model,
    evaluate_regional_holdout,
)


class TestBiweightRobustMean:
    def test_basic_values(self):
        values = [1.0, 1.1, 0.9, 1.05, 0.95]
        bw = biweight_robust_mean(values)
        assert np.isclose(bw, 1.0, atol=0.05)

    def test_small_sample(self):
        assert np.isclose(biweight_robust_mean([2.0]), 2.0)
        assert np.isclose(biweight_robust_mean([2.0, 4.0]), 3.0)

    def test_empty_sample(self):
        assert np.isnan(biweight_robust_mean([]))

    def test_outlier_robustness(self):
        # Sample around 1.0 with a massive outlier at 1000.0
        values = [1.0, 1.05, 0.95, 1.02, 0.98, 1000.0]
        bw = biweight_robust_mean(values, c=9.0)
        assert abs(bw - 1.0) < 0.2


class TestProcessMultipleRWL:
    def test_strict_holdout_exclusion(self):
        with pytest.raises(ValueError, match="Quarantined geographic holdout"):
            process_multiple_rwl(["africa/eth001.rwl", "africa/eth002.rwl"])

    def test_empty_filepaths_error(self):
        with pytest.raises(PipelineError, match="No .rwl filepaths provided"):
            process_multiple_rwl([])

    def test_multi_site_ingestion(self):
        paths = ["africa/eth002.rwl", "africa/eth003.rwl"]
        all_cores, master_chron = process_multiple_rwl(paths)

        assert isinstance(all_cores, pd.DataFrame)
        assert isinstance(master_chron, pd.DataFrame)
        assert "year" in master_chron.columns
        assert "rwi" in master_chron.columns
        assert "core_count" in master_chron.columns
        assert not master_chron["rwi"].isna().any()
        assert master_chron["year"].is_monotonic_increasing


class TestIsotopeIngestion:
    def test_load_isotope_dataset(self):
        iso_p = Path("data/isotope/africa2016d13c-iwue-k-noaa.txt")
        if not iso_p.exists():
            pytest.skip("Isotope dataset not found")

        df = load_isotope_dataset(iso_p)
        assert list(df.columns) == ["year", "d13c", "iwue"]
        assert len(df) > 200
        assert not df["d13c"].isna().any()
        assert not df["iwue"].isna().any()
        assert df["year"].is_monotonic_increasing


class TestProbabilityCalibration:
    def test_temperature_scaling_simplex(self):
        raw_probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        cal_probs = calibrated_predict_proba(raw_probs, temperature=0.35)

        assert cal_probs.shape == raw_probs.shape
        assert np.allclose(cal_probs.sum(axis=1), 1.0)
        assert np.all(cal_probs >= 0.0)
        # Monotonicity check: largest class in raw must remain largest in calibrated
        assert np.all(np.argmax(raw_probs, axis=1) == np.argmax(cal_probs, axis=1))
        # Confidence sharpening: max probability must increase
        assert np.all(np.max(cal_probs, axis=1) >= np.max(raw_probs, axis=1))


class TestRegionalModelArtifacts:
    def test_model_and_metadata_exist(self):
        model_path = Path("models/random_forest_regional.joblib")
        meta_path = Path("models/regional_model_metadata.json")

        assert model_path.exists()
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["model_name"] == "Random Forest Regional RCS Model"
        assert "eth001" not in meta["training_sites"]
        assert meta["feature_count"] == 20
        assert "d13c" in meta["feature_names"]
        assert "iwue" in meta["feature_names"]

    def test_regional_holdout_metrics_exceed_thresholds(self):
        metrics_p = Path("outputs/validation/regional_holdout_metrics.json")
        assert metrics_p.exists()

        with open(metrics_p, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["holdout_site"] == "Debrebirkan Selassie (eth001)"
        # Operational targets > 80%
        assert metrics["severe_drought_detection_accuracy"] >= 0.80
        assert metrics["mean_model_confidence"] >= 0.80
