import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from scripts.recalibrate_phase1 import predict_with_threshold_rule

def test_candidate_artifact_verification():
    p = Path("models/random_forest_eth007.joblib")
    assert p.exists()
    rf = joblib.load(p)
    assert rf.class_weight is None
    assert np.array_equal(rf.classes_, np.array([0, 1, 2]))
    assert rf.n_features_in_ == 16

def test_threshold_edge_cases():
    classes = np.array([0, 1, 2])
    # 0.5999 -> not Class 2
    assert predict_with_threshold_rule(np.array([0.2001, 0.2000, 0.5999]), classes, 0.60) != 2
    # 0.6000 -> not Class 2 (strict >)
    assert predict_with_threshold_rule(np.array([0.2000, 0.2000, 0.6000]), classes, 0.60) != 2
    # 0.6001 -> Class 2
    assert predict_with_threshold_rule(np.array([0.1999, 0.2000, 0.6001]), classes, 0.60) == 2
    # Class 0 > Class 1
    assert predict_with_threshold_rule(np.array([0.50, 0.20, 0.30]), classes, 0.60) == 0
    # Class 1 > Class 0
    assert predict_with_threshold_rule(np.array([0.20, 0.50, 0.30]), classes, 0.60) == 1
    # Class 0 == Class 1
    assert predict_with_threshold_rule(np.array([0.35, 0.35, 0.30]), classes, 0.60) == 0

def test_handoff_metadata():
    p = Path("model_recalibration_metadata.json")
    assert p.exists()
    with open(p, "r") as f:
        meta = json.load(f)
    assert meta["model_path"] == "models/random_forest_eth007.joblib"
    assert meta["classes"] == [0, 1, 2]
    assert meta["feature_count"] == 16
    assert meta["class_2_threshold"] == 0.60
    assert meta["threshold_operator"] == ">"
    assert meta["class_2_rule"] == "P(Class 2) > 0.60"
    assert meta["training_site"] == "eth007"
    assert meta["holdout_site"] == "eth001"
    assert meta["phase_1_status"] == "PHASE 1 READY FOR PHASE 2"

def test_deliverable_files_exist():
    assert Path("models/random_forest_eth007.joblib").exists()
    assert Path("models/random_forest_eth007_balanced_baseline.joblib").exists()
    assert Path("holdout_confusion_matrix.png").exists()
    assert Path("outputs/figures/holdout_confusion_matrix.png").exists()
    assert Path("results/baseline_holdout_classification_report.json").exists()
    assert Path("results/candidate_raw_holdout_classification_report.json").exists()
    assert Path("results/candidate_thresholded_holdout_classification_report.json").exists()
    assert Path("results/threshold_analysis_eth007.json").exists()
    assert Path("results/baseline_vs_candidate_comparison.json").exists()
    assert Path("model_recalibration_metadata.json").exists()
