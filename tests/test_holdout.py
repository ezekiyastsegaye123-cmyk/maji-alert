"""
Test suite for Phase 4 Geographic Holdout Validation and Model Provenance.
==========================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from treering.forecast import DroughtFeatureEngineer
from treering.holdout import (
    CLASS_NAMES_3,
    classify_spei_calibrated_3class,
    classify_spei_strict_3class,
    evaluate_geographic_holdout,
    train_and_save_gondar_model,
)


class TestTargetClassification:
    def test_strict_3class_boundaries(self):
        # Class 0: SPEI > -1.0
        assert classify_spei_strict_3class(0.5) == 0
        assert classify_spei_strict_3class(-0.5) == 0
        assert classify_spei_strict_3class(-0.99) == 0

        # Exact boundary -1.0 -> Class 1
        assert classify_spei_strict_3class(-1.0) == 1

        # Class 1: -1.5 < SPEI <= -1.0
        assert classify_spei_strict_3class(-1.2) == 1
        assert classify_spei_strict_3class(-1.49) == 1

        # Exact boundary -1.5 -> Class 2
        assert classify_spei_strict_3class(-1.5) == 2

        # Class 2: SPEI <= -1.5
        assert classify_spei_strict_3class(-2.0) == 2
        assert classify_spei_strict_3class(-3.5) == 2

    def test_calibrated_3class_boundaries(self):
        # Class 0: SPEI > -0.10
        assert classify_spei_calibrated_3class(0.2) == 0
        assert classify_spei_calibrated_3class(-0.05) == 0

        # Class 1: -0.35 < SPEI <= -0.10
        assert classify_spei_calibrated_3class(-0.10) == 1
        assert classify_spei_calibrated_3class(-0.25) == 1

        # Class 2: SPEI <= -0.35
        assert classify_spei_calibrated_3class(-0.35) == 2
        assert classify_spei_calibrated_3class(-0.80) == 2


class TestGondarModelTrainingAndPersistence:
    def test_train_and_save_gondar_model(self, tmp_path):
        model_out = tmp_path / "test_gondar_rf.joblib"
        meta_out = tmp_path / "test_gondar_meta.json"

        rf, meta = train_and_save_gondar_model(
            rwl_path="africa/eth007.rwl",
            sunspot_path="SN_y_tot_V2.0.csv",
            netcdf_path="data/spei01.nc",
            model_output_path=model_out,
            metadata_output_path=meta_out,
            n_estimators=30,
            max_depth=3,
            overwrite=True,
        )

        assert isinstance(rf, RandomForestClassifier)
        assert model_out.exists()
        assert meta_out.exists()
        assert meta["training_site"] == "Gondar (eth007)"
        assert meta["n_samples"] == 114
        assert meta["training_period"] == [1901, 2014]
        assert set(meta["feature_names"]) == set(DroughtFeatureEngineer.FEATURE_NAMES)


class TestGeographicHoldoutEvaluation:
    def test_evaluate_geographic_holdout_execution(self, tmp_path):
        out_dir = tmp_path / "outputs"

        metrics = evaluate_geographic_holdout(
            holdout_rwl_path="africa/eth001.rwl",
            sunspot_path="SN_y_tot_V2.0.csv",
            netcdf_path="data/spei01.nc",
            model_path="models/random_forest_eth007.joblib",
            output_dir=out_dir,
        )

        assert metrics["holdout_site"] == "Debrebirkan Selassie (eth001)"
        assert metrics["n_holdout_samples"] == 106
        assert metrics["holdout_period"] == [1901, 2006]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert np.array(metrics["confusion_matrix"]).shape == (3, 3)

        # Verify output files
        assert (out_dir / "validation" / "holdout_validation_results.csv").exists()
        assert (out_dir / "validation" / "holdout_classification_report.json").exists()
        assert (out_dir / "validation" / "holdout_confusion_matrix.csv").exists()
        assert (out_dir / "validation" / "holdout_metrics.json").exists()
        assert (out_dir / "figures" / "solar_rwi_hypothesis.png").exists()
        assert (out_dir / "figures" / "holdout_confusion_matrix.png").exists()
        assert (out_dir / "figures" / "feature_importance.png").exists()
        assert (out_dir / "metadata" / "phase4_metadata.json").exists()

    def test_holdout_csv_validation(self, tmp_path):
        out_dir = tmp_path / "outputs"
        evaluate_geographic_holdout(
            holdout_rwl_path="africa/eth001.rwl",
            sunspot_path="SN_y_tot_V2.0.csv",
            netcdf_path="data/spei01.nc",
            model_path="models/random_forest_eth007.joblib",
            output_dir=out_dir,
        )

        csv_p = out_dir / "validation" / "holdout_validation_results.csv"
        df_val = pd.read_csv(csv_p)

        assert len(df_val) == 106
        assert df_val["year"].is_monotonic_increasing
        assert not df_val.isna().any().any()

        # Probabilities sum to 1
        prob_sums = df_val["prob_class_0"] + df_val["prob_class_1"] + df_val["prob_class_2"]
        assert np.allclose(prob_sums, 1.0, atol=1e-3)


class TestLeakageRedTeam:
    def test_no_holdout_data_in_training_metadata(self):
        meta_p = Path("models/eth007_model_metadata.json")
        assert meta_p.exists()
        with open(meta_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["training_site"] == "Gondar (eth007)"
        assert "eth001" not in str(data)
        assert data["training_coordinates"]["latitude"] == 13.01
        assert data["training_coordinates"]["longitude"] == 37.80
