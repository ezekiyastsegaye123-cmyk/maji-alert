"""
FRADSCR — Phase 1 Model Recalibration & Validation Pipeline
=============================================================

Executes:
1. Baseline reproduction (class_weight='balanced') on untouched eth001 holdout.
2. Candidate training (class_weight=None) exclusively on eth007 (Gondar).
3. Artifact persistence to models/random_forest_eth007.joblib & verification.
4. Threshold analysis on eth007 across [0.50, 0.55, 0.60, 0.65, 0.70].
5. Production decision rule evaluation & edge-case unit tests.
6. Geographic holdout evaluation on eth001.
7. Production-relevant holdout confusion matrix figure generation.
8. Deliverables & Handoff metadata (model_recalibration_metadata.json).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit

from treering.forecast import DroughtFeatureEngineer
from treering.holdout import CLASS_NAMES_3, classify_spei_calibrated_3class
from treering.pipeline import process_rwl
from treering.spei import extract_annual_spei

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recalibrate_phase1")


def compute_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def predict_with_threshold_rule(
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float = 0.60,
) -> np.ndarray:
    """
    Verified Phase 2 Decision Rule:
    IF P(Class 2) > 0.60:
        final_class = 2
    ELSE:
        final_class = argmax(P(Class 0), P(Class 1))  # tie-breaker: Class 0 if equal
    """
    class_list = [int(c) for c in classes]
    idx_0 = class_list.index(0)
    idx_1 = class_list.index(1)
    idx_2 = class_list.index(2)

    probs = np.asarray(probabilities)
    single = probs.ndim == 1
    if single:
        probs = probs.reshape(1, -1)

    final_classes = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        p0 = p[idx_0]
        p1 = p[idx_1]
        p2 = p[idx_2]

        if p2 > threshold:
            final_classes[i] = 2
        else:
            final_classes[i] = 0 if p0 >= p1 else 1

    return int(final_classes[0]) if single else final_classes


def run_threshold_edge_case_tests(classes: np.ndarray) -> None:
    logger.info("Executing threshold edge-case tests...")

    # 1. 0.5999 -> not Class 2
    p_5999 = np.array([0.2001, 0.2000, 0.5999])
    res_5999 = predict_with_threshold_rule(p_5999, classes, threshold=0.60)
    assert res_5999 != 2, f"Expected not Class 2 for 0.5999, got {res_5999}"

    # 2. 0.6000 -> not Class 2 (strict >)
    p_6000 = np.array([0.2000, 0.2000, 0.6000])
    res_6000 = predict_with_threshold_rule(p_6000, classes, threshold=0.60)
    assert res_6000 != 2, f"Expected not Class 2 for 0.6000, got {res_6000}"

    # 3. 0.6001 -> Class 2 is eligible
    p_6001 = np.array([0.1999, 0.2000, 0.6001])
    res_6001 = predict_with_threshold_rule(p_6001, classes, threshold=0.60)
    assert res_6001 == 2, f"Expected Class 2 for 0.6001, got {res_6001}"

    # 4. Class 0 > Class 1 (fallback rule)
    p_c0_gt_c1 = np.array([0.50, 0.20, 0.30])
    res_c0 = predict_with_threshold_rule(p_c0_gt_c1, classes, threshold=0.60)
    assert res_c0 == 0, f"Expected Class 0, got {res_c0}"

    # 5. Class 1 > Class 0 (fallback rule)
    p_c1_gt_c0 = np.array([0.20, 0.50, 0.30])
    res_c1 = predict_with_threshold_rule(p_c1_gt_c0, classes, threshold=0.60)
    assert res_c1 == 1, f"Expected Class 1, got {res_c1}"

    # 6. Class 0 == Class 1 (fallback rule tie-break)
    p_equal = np.array([0.35, 0.35, 0.30])
    res_eq = predict_with_threshold_rule(p_equal, classes, threshold=0.60)
    assert res_eq == 0, f"Expected Class 0 on tie-break, got {res_eq}"

    logger.info("All threshold edge-case tests PASSED.")


def main():
    logger.info("Starting FRADSCR Phase 1 Model Recalibration...")

    # Paths
    eth007_rwl = Path("africa/eth007.rwl")
    eth001_rwl = Path("africa/eth001.rwl")
    sunspot_csv = Path("SN_y_tot_V2.0.csv")
    spei_nc = Path("data/spei01.nc")

    model_dir = Path("models")
    results_dir = Path("results")
    figures_dir = Path("outputs/figures")

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_artifact_path = model_dir / "random_forest_eth007_balanced_baseline.joblib"
    candidate_artifact_path = model_dir / "random_forest_eth007.joblib"
    candidate_metadata_path = model_dir / "eth007_model_metadata.json"

    # Step 1: Load and Evaluate Baseline on eth001
    logger.info("--- Step 1: Baseline Reproduction ---")
    if not baseline_artifact_path.exists():
        if candidate_artifact_path.exists():
            import shutil
            shutil.copy(candidate_artifact_path, baseline_artifact_path)
            shutil.copy(candidate_metadata_path, model_dir / "eth007_model_metadata_balanced_baseline.json")

    baseline_hash = compute_sha256(baseline_artifact_path)
    baseline_model: RandomForestClassifier = joblib.load(baseline_artifact_path)
    logger.info("Baseline artifact SHA256: %s", baseline_hash)
    logger.info("Baseline class_weight: %s", baseline_model.class_weight)

    # Ingest Sunspots
    df_sun = pd.read_csv(sunspot_csv, sep=";", header=None, usecols=[0, 1])
    df_sun.columns = ["year_dec", "sunspot"]
    df_sun["year"] = df_sun["year_dec"].astype(int)
    df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

    engineer = DroughtFeatureEngineer()
    df_solar = engineer.build_solar_feature_table(df_sun)

    # Prepare eth001 Holdout Data
    df_rwl_001 = process_rwl(eth001_rwl)
    chron_001 = df_rwl_001.groupby("year")[["rwi"]].mean().reset_index()
    debre_spei = extract_annual_spei(spei_nc, lat=9.63, lon=39.53).annual_df

    df_chron_001 = engineer.build_tree_ring_chronology(chron_001)
    df_holdout = engineer.build_training_dataset(df_chron_001, df_solar, debre_spei)
    df_holdout["actual_class_calibrated"] = [classify_spei_calibrated_3class(s) for s in df_holdout["spei"]]

    X_holdout = df_holdout[DroughtFeatureEngineer.FEATURE_NAMES].values
    y_holdout = df_holdout["actual_class_calibrated"].values

    # Evaluate Baseline
    base_preds = baseline_model.predict(X_holdout)
    base_acc = float(accuracy_score(y_holdout, base_preds))
    base_bal_acc = float(balanced_accuracy_score(y_holdout, base_preds))
    base_macro_f1 = float(f1_score(y_holdout, base_preds, average="macro", zero_division=0))
    base_weighted_f1 = float(f1_score(y_holdout, base_preds, average="weighted", zero_division=0))
    base_cm = confusion_matrix(y_holdout, base_preds, labels=[0, 1, 2])
    base_rep = classification_report(y_holdout, base_preds, target_names=CLASS_NAMES_3, output_dict=True, zero_division=0)

    base_c2_fp = int(base_cm[0, 2] + base_cm[1, 2])
    base_c2_tp = int(base_cm[2, 2])
    base_c2_fn = int(base_cm[2, 0] + base_cm[2, 1])

    baseline_metrics = {
        "model_artifact": str(baseline_artifact_path),
        "sha256": baseline_hash,
        "class_weight": baseline_model.class_weight,
        "accuracy": base_acc,
        "balanced_accuracy": base_bal_acc,
        "macro_f1": base_macro_f1,
        "weighted_f1": base_weighted_f1,
        "confusion_matrix": base_cm.tolist(),
        "class_0": {
            "precision": float(base_rep["Normal / Wet"]["precision"]),
            "recall": float(base_rep["Normal / Wet"]["recall"]),
            "f1": float(base_rep["Normal / Wet"]["f1-score"]),
            "support": int(base_rep["Normal / Wet"]["support"]),
        },
        "class_1": {
            "precision": float(base_rep["Moderate Drought"]["precision"]),
            "recall": float(base_rep["Moderate Drought"]["recall"]),
            "f1": float(base_rep["Moderate Drought"]["f1-score"]),
            "support": int(base_rep["Moderate Drought"]["support"]),
        },
        "class_2": {
            "precision": float(base_rep["Severe Drought"]["precision"]),
            "recall": float(base_rep["Severe Drought"]["recall"]),
            "f1": float(base_rep["Severe Drought"]["f1-score"]),
            "support": int(base_rep["Severe Drought"]["support"]),
            "tp": base_c2_tp,
            "fp": base_c2_fp,
            "fn": base_c2_fn,
        },
    }
    with open(results_dir / "baseline_holdout_classification_report.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    # Step 2: Ingest eth007 and Train Candidate Model (class_weight=None)
    logger.info("--- Step 2: Training Candidate Model (class_weight=None) on eth007 ---")
    df_rwl_007 = process_rwl(eth007_rwl)
    chron_007 = df_rwl_007.groupby("year")[["rwi"]].mean().reset_index()
    gondar_spei_res = extract_annual_spei(spei_nc, lat=13.01, lon=37.80)

    df_chron_007 = engineer.build_tree_ring_chronology(chron_007)
    df_train_007 = engineer.build_training_dataset(df_chron_007, df_solar, gondar_spei_res.annual_df)
    df_train_007["target_3class"] = [classify_spei_calibrated_3class(s) for s in df_train_007["spei"]]

    X_train = df_train_007[DroughtFeatureEngineer.FEATURE_NAMES].values
    y_train = df_train_007["target_3class"].values

    candidate_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        class_weight=None,
        random_state=42,
    )
    candidate_model.fit(X_train, y_train)

    # Step 3: Model Artifact Persistence & Verification
    logger.info("--- Step 3: Saving and Verifying Candidate Artifact ---")
    joblib.dump(candidate_model, candidate_artifact_path)
    cand_hash = compute_sha256(candidate_artifact_path)

    # Reload from disk to guarantee independent persistence
    reloaded_model: RandomForestClassifier = joblib.load(candidate_artifact_path)
    assert reloaded_model.class_weight is None
    assert np.array_equal(reloaded_model.classes_, np.array([0, 1, 2]))
    assert reloaded_model.n_features_in_ == 16
    logger.info("Candidate model verified. SHA256: %s", cand_hash)

    feature_importances = {
        feat: float(imp)
        for feat, imp in zip(DroughtFeatureEngineer.FEATURE_NAMES, reloaded_model.feature_importances_)
    }

    import sklearn
    import sys
    candidate_metadata = {
        "model_name": "Random Forest Gondar Training Model",
        "model_type": "RandomForestClassifier",
        "training_site": "Gondar (eth007)",
        "training_coordinates": {"latitude": 13.01, "longitude": 37.80},
        "selected_spei_grid_cell": {
            "latitude": gondar_spei_res.grid_metadata.selected_lat,
            "longitude": gondar_spei_res.grid_metadata.selected_lon,
            "distance_km": gondar_spei_res.grid_metadata.spatial_distance_km,
        },
        "training_period": [int(df_train_007["year"].min()), int(df_train_007["year"].max())],
        "n_samples": len(df_train_007),
        "class_distribution": pd.Series(y_train).value_counts().to_dict(),
        "classes_": [int(c) for c in reloaded_model.classes_],
        "feature_names": DroughtFeatureEngineer.FEATURE_NAMES,
        "feature_count": len(DroughtFeatureEngineer.FEATURE_NAMES),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 4,
            "class_weight": None,
            "random_state": 42,
        },
        "feature_importances": feature_importances,
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "sha256": cand_hash,
    }
    with open(candidate_metadata_path, "w") as f:
        json.dump(candidate_metadata, f, indent=2)

    # Step 4: Class Mapping Verification
    logger.info("--- Step 4: Class Mapping Verification ---")
    logger.info("model.classes_: %s", reloaded_model.classes_.tolist())
    dummy_probs = reloaded_model.predict_proba(X_train[:5])
    assert dummy_probs.shape[1] == 3
    logger.info("model.predict_proba shape: (N, %d) mapping directly to classes [0, 1, 2]", dummy_probs.shape[1])

    # Step 5: Threshold Edge Case Verification
    run_threshold_edge_case_tests(reloaded_model.classes_)

    # Step 6: Threshold Analysis on eth007 (Validation / CV Framework)
    logger.info("--- Step 6: Threshold Analysis on eth007 ---")
    train_probs_c2 = reloaded_model.predict_proba(X_train)[:, 2]

    # In-sample threshold analysis on eth007
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    thresh_insample = []
    for thresh in thresholds:
        pred_c2 = (train_probs_c2 > thresh).astype(int)
        actual_c2 = (y_train == 2).astype(int)
        tp = int(((pred_c2 == 1) & (actual_c2 == 1)).sum())
        fp = int(((pred_c2 == 1) & (actual_c2 == 0)).sum())
        fn = int(((pred_c2 == 0) & (actual_c2 == 1)).sum())
        pred_cnt = int(pred_c2.sum())
        p = float(tp / pred_cnt) if pred_cnt > 0 else 0.0
        r = float(tp / actual_c2.sum()) if actual_c2.sum() > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        thresh_insample.append({
            "threshold": thresh,
            "predicted_count": pred_cnt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    # TimeSeriesSplit CV on eth007
    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = np.zeros(len(y_train))
    oof_counts = np.zeros(len(y_train))
    for train_idx, val_idx in tscv.split(X_train):
        fold_rf = RandomForestClassifier(n_estimators=300, max_depth=4, class_weight=None, random_state=42)
        fold_rf.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_rf.predict_proba(X_train[val_idx])[:, 2]
        oof_counts[val_idx] += 1
    val_mask = oof_counts > 0

    thresh_cv = []
    for thresh in thresholds:
        pred_c2 = (oof_probs[val_mask] > thresh).astype(int)
        actual_c2 = (y_train[val_mask] == 2).astype(int)
        tp = int(((pred_c2 == 1) & (actual_c2 == 1)).sum())
        fp = int(((pred_c2 == 1) & (actual_c2 == 0)).sum())
        fn = int(((pred_c2 == 0) & (actual_c2 == 1)).sum())
        pred_cnt = int(pred_c2.sum())
        p = float(tp / pred_cnt) if pred_cnt > 0 else 0.0
        r = float(tp / actual_c2.sum()) if actual_c2.sum() > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        thresh_cv.append({
            "threshold": thresh,
            "predicted_count": pred_cnt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    thresh_analysis_data = {
        "site": "Gondar (eth007)",
        "in_sample_evaluation": thresh_insample,
        "timeseries_cv_evaluation": thresh_cv,
        "production_decision_rule": "IF P(Class 2) > 0.60: Class 2; ELSE: argmax(P(Class 0), P(Class 1))",
    }
    with open(results_dir / "threshold_analysis_eth007.json", "w") as f:
        json.dump(thresh_analysis_data, f, indent=2)

    # Step 7: Evaluate Candidate Model on Untouched Geographic Holdout (eth001)
    logger.info("--- Step 7: Candidate Evaluation on eth001 Holdout ---")
    cand_probs = reloaded_model.predict_proba(X_holdout)

    # Raw Model Predict
    cand_raw_preds = reloaded_model.predict(X_holdout)
    raw_acc = float(accuracy_score(y_holdout, cand_raw_preds))
    raw_bal_acc = float(balanced_accuracy_score(y_holdout, cand_raw_preds))
    raw_macro_f1 = float(f1_score(y_holdout, cand_raw_preds, average="macro", zero_division=0))
    raw_weighted_f1 = float(f1_score(y_holdout, cand_raw_preds, average="weighted", zero_division=0))
    raw_cm = confusion_matrix(y_holdout, cand_raw_preds, labels=[0, 1, 2])
    raw_rep = classification_report(y_holdout, cand_raw_preds, target_names=CLASS_NAMES_3, output_dict=True, zero_division=0)

    raw_c2_fp = int(raw_cm[0, 2] + raw_cm[1, 2])
    raw_c2_tp = int(raw_cm[2, 2])
    raw_c2_fn = int(raw_cm[2, 0] + raw_cm[2, 1])

    cand_raw_metrics = {
        "model_artifact": str(candidate_artifact_path),
        "class_weight": None,
        "rule": "model.predict() [raw argmax]",
        "accuracy": raw_acc,
        "balanced_accuracy": raw_bal_acc,
        "macro_f1": raw_macro_f1,
        "weighted_f1": raw_weighted_f1,
        "confusion_matrix": raw_cm.tolist(),
        "class_0": {
            "precision": float(raw_rep["Normal / Wet"]["precision"]),
            "recall": float(raw_rep["Normal / Wet"]["recall"]),
            "f1": float(raw_rep["Normal / Wet"]["f1-score"]),
            "support": int(raw_rep["Normal / Wet"]["support"]),
        },
        "class_1": {
            "precision": float(raw_rep["Moderate Drought"]["precision"]),
            "recall": float(raw_rep["Moderate Drought"]["recall"]),
            "f1": float(raw_rep["Moderate Drought"]["f1-score"]),
            "support": int(raw_rep["Moderate Drought"]["support"]),
        },
        "class_2": {
            "precision": float(raw_rep["Severe Drought"]["precision"]),
            "recall": float(raw_rep["Severe Drought"]["recall"]),
            "f1": float(raw_rep["Severe Drought"]["f1-score"]),
            "support": int(raw_rep["Severe Drought"]["support"]),
            "tp": raw_c2_tp,
            "fp": raw_c2_fp,
            "fn": raw_c2_fn,
        },
    }
    with open(results_dir / "candidate_raw_holdout_classification_report.json", "w") as f:
        json.dump(cand_raw_metrics, f, indent=2)

    # Thresholded Production Rule Predict
    cand_thresh_preds = predict_with_threshold_rule(cand_probs, reloaded_model.classes_, threshold=0.60)
    thresh_acc = float(accuracy_score(y_holdout, cand_thresh_preds))
    thresh_bal_acc = float(balanced_accuracy_score(y_holdout, cand_thresh_preds))
    thresh_macro_f1 = float(f1_score(y_holdout, cand_thresh_preds, average="macro", zero_division=0))
    thresh_weighted_f1 = float(f1_score(y_holdout, cand_thresh_preds, average="weighted", zero_division=0))
    thresh_cm = confusion_matrix(y_holdout, cand_thresh_preds, labels=[0, 1, 2])
    thresh_rep = classification_report(y_holdout, cand_thresh_preds, target_names=CLASS_NAMES_3, output_dict=True, zero_division=0)

    thresh_c2_fp = int(thresh_cm[0, 2] + thresh_cm[1, 2])
    thresh_c2_tp = int(thresh_cm[2, 2])
    thresh_c2_fn = int(thresh_cm[2, 0] + thresh_cm[2, 1])

    cand_thresh_metrics = {
        "model_artifact": str(candidate_artifact_path),
        "class_weight": None,
        "rule": "P(Class 2) > 0.60, fallback argmax(P(Class 0), P(Class 1))",
        "accuracy": thresh_acc,
        "balanced_accuracy": thresh_bal_acc,
        "macro_f1": thresh_macro_f1,
        "weighted_f1": thresh_weighted_f1,
        "confusion_matrix": thresh_cm.tolist(),
        "class_0": {
            "precision": float(thresh_rep["Normal / Wet"]["precision"]),
            "recall": float(thresh_rep["Normal / Wet"]["recall"]),
            "f1": float(thresh_rep["Normal / Wet"]["f1-score"]),
            "support": int(thresh_rep["Normal / Wet"]["support"]),
        },
        "class_1": {
            "precision": float(thresh_rep["Moderate Drought"]["precision"]),
            "recall": float(thresh_rep["Moderate Drought"]["recall"]),
            "f1": float(thresh_rep["Moderate Drought"]["f1-score"]),
            "support": int(thresh_rep["Moderate Drought"]["support"]),
        },
        "class_2": {
            "precision": float(thresh_rep["Severe Drought"]["precision"]),
            "recall": float(thresh_rep["Severe Drought"]["recall"]),
            "f1": float(thresh_rep["Severe Drought"]["f1-score"]),
            "support": int(thresh_rep["Severe Drought"]["support"]),
            "tp": thresh_c2_tp,
            "fp": thresh_c2_fp,
            "fn": thresh_c2_fn,
        },
    }
    with open(results_dir / "candidate_thresholded_holdout_classification_report.json", "w") as f:
        json.dump(cand_thresh_metrics, f, indent=2)

    # Save holdout validation results CSV
    df_holdout_out = pd.DataFrame({
        "year": df_holdout["year"].astype(int),
        "actual_spei": df_holdout["spei"].astype(float),
        "actual_class": y_holdout,
        "predicted_class_raw": cand_raw_preds,
        "predicted_class_thresholded": cand_thresh_preds,
        "prob_class_0": cand_probs[:, 0],
        "prob_class_1": cand_probs[:, 1],
        "prob_class_2": cand_probs[:, 2],
    })
    df_holdout_out.to_csv(Path("outputs/validation/holdout_validation_results.csv"), index=False)

    # Step 8: Before/After Comparison Table
    logger.info("--- Step 8: Before/After Comparison Table ---")
    comparison = {
        "metrics": [
            {
                "metric": "Class 2 Precision",
                "baseline": base_rep["Severe Drought"]["precision"],
                "candidate_raw": raw_rep["Severe Drought"]["precision"],
                "candidate_thresholded": thresh_rep["Severe Drought"]["precision"],
                "change_vs_baseline": thresh_rep["Severe Drought"]["precision"] - base_rep["Severe Drought"]["precision"],
            },
            {
                "metric": "Class 2 Recall",
                "baseline": base_rep["Severe Drought"]["recall"],
                "candidate_raw": raw_rep["Severe Drought"]["recall"],
                "candidate_thresholded": thresh_rep["Severe Drought"]["recall"],
                "change_vs_baseline": thresh_rep["Severe Drought"]["recall"] - base_rep["Severe Drought"]["recall"],
            },
            {
                "metric": "Class 2 F1",
                "baseline": base_rep["Severe Drought"]["f1-score"],
                "candidate_raw": raw_rep["Severe Drought"]["f1-score"],
                "candidate_thresholded": thresh_rep["Severe Drought"]["f1-score"],
                "change_vs_baseline": thresh_rep["Severe Drought"]["f1-score"] - base_rep["Severe Drought"]["f1-score"],
            },
            {
                "metric": "Class 2 False Positives",
                "baseline": base_c2_fp,
                "candidate_raw": raw_c2_fp,
                "candidate_thresholded": thresh_c2_fp,
                "change_vs_baseline": thresh_c2_fp - base_c2_fp,
            },
            {
                "metric": "Balanced Accuracy",
                "baseline": base_bal_acc,
                "candidate_raw": raw_bal_acc,
                "candidate_thresholded": thresh_bal_acc,
                "change_vs_baseline": thresh_bal_acc - base_bal_acc,
            },
            {
                "metric": "Macro F1",
                "baseline": base_macro_f1,
                "candidate_raw": raw_macro_f1,
                "candidate_thresholded": thresh_macro_f1,
                "change_vs_baseline": thresh_macro_f1 - base_macro_f1,
            },
            {
                "metric": "Weighted F1",
                "baseline": base_weighted_f1,
                "candidate_raw": raw_weighted_f1,
                "candidate_thresholded": thresh_weighted_f1,
                "change_vs_baseline": thresh_weighted_f1 - base_weighted_f1,
            },
            {
                "metric": "Accuracy",
                "baseline": base_acc,
                "candidate_raw": raw_acc,
                "candidate_thresholded": thresh_acc,
                "change_vs_baseline": thresh_acc - base_acc,
            },
        ]
    }
    with open(results_dir / "baseline_vs_candidate_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Step 9: Production-Relevant Confusion Matrix Figure (Thresholded)
    logger.info("--- Step 9: Generating Production-Relevant Confusion Matrix ---")
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        thresh_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES_3,
        yticklabels=CLASS_NAMES_3,
        cbar=True,
    )
    plt.title(
        "Candidate Geographic Holdout Confusion Matrix (P(Class 2) > 0.60)\nDebrebirkan Selassie (eth001)",
        fontsize=13,
        pad=12,
    )
    plt.xlabel("Predicted Drought Class (Thresholded Rule)", fontsize=11)
    plt.ylabel("Actual Drought Class (Ground Truth)", fontsize=11)
    plt.tight_layout()

    # Save to both required locations
    cm_path_1 = figures_dir / "holdout_confusion_matrix.png"
    cm_path_2 = Path("holdout_confusion_matrix.png")
    plt.savefig(cm_path_1, dpi=300)
    plt.savefig(cm_path_2, dpi=300)
    plt.close()
    logger.info("Saved holdout confusion matrix to %s and %s", cm_path_1, cm_path_2)

    # Step 10: Phase 1 Handoff Metadata File
    logger.info("--- Step 10: Creating model_recalibration_metadata.json Handoff File ---")
    handoff = {
        "model_path": str(candidate_artifact_path),
        "model_hash_sha256": cand_hash,
        "classes": [int(c) for c in reloaded_model.classes_],
        "feature_names": DroughtFeatureEngineer.FEATURE_NAMES,
        "feature_count": len(DroughtFeatureEngineer.FEATURE_NAMES),
        "class_2_threshold": 0.60,
        "threshold_operator": ">",
        "class_2_rule": "P(Class 2) > 0.60",
        "fallback_rule": "argmax over Class 0 and Class 1",
        "training_site": "eth007",
        "training_coordinates": {"latitude": 13.01, "longitude": 37.80},
        "holdout_site": "eth001",
        "holdout_coordinates": {"latitude": 9.63, "longitude": 39.53},
        "geographic_distance_km": 412.5,
        "training_samples": len(df_train_007),
        "holdout_samples": len(df_holdout),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "phase_1_status": "PHASE 1 READY FOR PHASE 2",
    }
    with open("model_recalibration_metadata.json", "w") as f:
        json.dump(handoff, f, indent=2)

    logger.info("Phase 1 recalibration completed successfully!")


if __name__ == "__main__":
    main()
