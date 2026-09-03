"""
Geographic Holdout Validation & Model Persistence Engine
========================================================

This module provides training, persistence, and blind geographic holdout
validation for the EGATE Heliophysics Tree-Ring Climate Research Project.

Methodological Separation:
--------------------------
1. Training Site: Gondar, Ethiopia (`africa/eth007.rwl`, 13.01° N, 37.80° E).
   Model is trained on 1901–2014 climate & tree-ring observations and persisted
   to `models/random_forest_eth007.joblib`.
2. Geographic Holdout Site: Debrebirkan Selassie, Ethiopia (`africa/eth001.rwl`, 9.63° N, 39.53° E).
   Over 400 km distant across the Ethiopian Highlands. Kept strictly unseen
   during model fitting, tuning, and threshold selection.

Strict Scientific Integrity:
----------------------------
- The saved model is loaded from disk without refitting or hyperparameter tuning.
- Feature extraction on `eth001` mirrors the exact detrending, solar lag,
  and time-alignment procedures used for `eth007`.
- Pure blind out-of-sample inference is conducted on 1901–2006.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from treering.forecast import DroughtFeatureEngineer
from treering.pipeline import process_rwl
from treering.spei import extract_annual_spei

logger = logging.getLogger(__name__)

# Class labels
CLASS_NAMES_3 = ["Normal / Wet", "Moderate Drought", "Severe Drought"]


# =====================================================================
# Target Classification Functions
# =====================================================================


def classify_spei_strict_3class(spei_val: float) -> int:
    """
    Classify drought into 3 classes based on Section 13 thresholds:
      Class 0 (Normal / Wet): SPEI > -1.0
      Class 1 (Moderate Drought): -1.5 < SPEI <= -1.0 (exact -1.0 -> Class 1)
      Class 2 (Severe Drought): SPEI <= -1.5 (exact -1.5 -> Class 2)
    """
    if spei_val > -1.0:
        return 0
    elif spei_val > -1.5:
        return 1
    else:
        return 2


def classify_spei_calibrated_3class(spei_val: float) -> int:
    """
    Classify drought into 3 classes based on annual-average distribution quantiles:
      Class 0 (Normal / Wet): SPEI > -0.10
      Class 1 (Moderate Drought): -0.35 < SPEI <= -0.10
      Class 2 (Severe Drought): SPEI <= -0.35
    """
    if spei_val > -0.10:
        return 0
    elif spei_val > -0.35:
        return 1
    else:
        return 2


# =====================================================================
# Model Trainer & Exporter for eth007 (Gondar)
# =====================================================================


def train_and_save_gondar_model(
    rwl_path: Union[str, Path] = "africa/eth007.rwl",
    sunspot_path: Union[str, Path] = "SN_y_tot_V2.0.csv",
    netcdf_path: Union[str, Path] = "data/spei01.nc",
    model_output_path: Union[str, Path] = "models/random_forest_eth007.joblib",
    metadata_output_path: Union[str, Path] = "models/eth007_model_metadata.json",
    n_estimators: int = 300,
    max_depth: int = 4,
    class_weight: Optional[Union[str, Dict[Any, Any]]] = None,
    random_state: int = 42,
    overwrite: bool = True,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train Random Forest classifier on eth007 (Gondar) training dataset and save artifact.
    """
    model_out = Path(model_output_path)
    meta_out = Path(metadata_output_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    if model_out.exists() and not overwrite:
        logger.info("Loading existing model from %s", model_out)
        model = joblib.load(model_out)
        with open(meta_out, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata

    # 1. Process tree-rings
    df_rwl = process_rwl(rwl_path)
    chron_df = df_rwl.groupby("year")[["rwi"]].mean().reset_index()

    # 2. Ingest sunspots
    df_sun = pd.read_csv(sunspot_path, sep=";", header=None, usecols=[0, 1])
    df_sun.columns = ["year_dec", "sunspot"]
    df_sun["year"] = df_sun["year_dec"].astype(int)
    df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

    # 3. Extract Gondar SPEI
    gondar_spei_res = extract_annual_spei(netcdf_path, lat=13.01, lon=37.80)
    df_spei = gondar_spei_res.annual_df

    # 4. Feature engineering
    engineer = DroughtFeatureEngineer()
    df_chron = engineer.build_tree_ring_chronology(chron_df)
    df_solar = engineer.build_solar_feature_table(df_sun)
    df_train = engineer.build_training_dataset(df_chron, df_solar, df_spei)

    df_train["target_3class"] = [classify_spei_calibrated_3class(s) for s in df_train["spei"]]

    X_train = df_train[DroughtFeatureEngineer.FEATURE_NAMES].values
    y_train = df_train["target_3class"].values

    # 5. Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    # 6. Persist model
    joblib.dump(rf, model_out)

    feature_importances = {
        feat: float(imp)
        for feat, imp in zip(DroughtFeatureEngineer.FEATURE_NAMES, rf.feature_importances_)
    }

    metadata = {
        "model_name": "Random Forest Gondar Training Model",
        "model_type": "RandomForestClassifier",
        "training_site": "Gondar (eth007)",
        "training_coordinates": {"latitude": 13.01, "longitude": 37.80},
        "selected_spei_grid_cell": {
            "latitude": gondar_spei_res.grid_metadata.selected_lat,
            "longitude": gondar_spei_res.grid_metadata.selected_lon,
            "distance_km": gondar_spei_res.grid_metadata.spatial_distance_km,
        },
        "training_period": [int(df_train["year"].min()), int(df_train["year"].max())],
        "n_samples": len(df_train),
        "class_distribution": pd.Series(y_train).value_counts().to_dict(),
        "classes_": [int(c) for c in rf.classes_],
        "feature_names": DroughtFeatureEngineer.FEATURE_NAMES,
        "feature_count": len(DroughtFeatureEngineer.FEATURE_NAMES),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "random_state": random_state,
        },
        "feature_importances": feature_importances,
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved trained Gondar model to %s", model_out)
    return rf, metadata


# =====================================================================
# Blind Geographic Holdout Evaluator
# =====================================================================


def evaluate_geographic_holdout(
    holdout_rwl_path: Union[str, Path] = "africa/eth001.rwl",
    sunspot_path: Union[str, Path] = "SN_y_tot_V2.0.csv",
    netcdf_path: Union[str, Path] = "data/spei01.nc",
    model_path: Union[str, Path] = "models/random_forest_eth007.joblib",
    output_dir: Union[str, Path] = "outputs",
) -> Dict[str, Any]:
    """
    Perform blind geographic holdout evaluation of the Gondar model on Debrebirkan Selassie (eth001).
    """
    out_dir = Path(output_dir)
    val_dir = out_dir / "validation"
    fig_dir = out_dir / "figures"
    meta_dir = out_dir / "metadata"

    val_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load saved model (NO REFITTING)
    m_p = Path(model_path)
    if not m_p.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {m_p}. Run training first.")
    model: RandomForestClassifier = joblib.load(m_p)

    # 2. Process holdout tree rings (eth001)
    df_eth001 = process_rwl(holdout_rwl_path)
    chron_001 = df_eth001.groupby("year")[["rwi"]].mean().reset_index()

    # 3. Ingest sunspots
    df_sun = pd.read_csv(sunspot_path, sep=";", header=None, usecols=[0, 1])
    df_sun.columns = ["year_dec", "sunspot"]
    df_sun["year"] = df_sun["year_dec"].astype(int)
    df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

    # 4. Extract Debrebirkan SPEI
    debre_spei_res = extract_annual_spei(netcdf_path, lat=9.63, lon=39.53)
    df_debre_spei = debre_spei_res.annual_df

    # 5. Build feature matrix using identical pipeline
    engineer = DroughtFeatureEngineer()
    df_chron = engineer.build_tree_ring_chronology(chron_001)
    df_solar = engineer.build_solar_feature_table(df_sun)
    df_holdout = engineer.build_training_dataset(df_chron, df_solar, df_debre_spei)

    # Targets
    df_holdout["actual_class_strict"] = [classify_spei_strict_3class(s) for s in df_holdout["spei"]]
    df_holdout["actual_class_calibrated"] = [classify_spei_calibrated_3class(s) for s in df_holdout["spei"]]

    X_holdout = df_holdout[DroughtFeatureEngineer.FEATURE_NAMES].values
    y_true_strict = df_holdout["actual_class_strict"].values
    y_true_cal = df_holdout["actual_class_calibrated"].values

    # 6. Blind inference
    preds = model.predict(X_holdout)
    probs = model.predict_proba(X_holdout)

    # Fill probability columns for classes 0, 1, 2
    prob_dict = {f"prob_class_{c}": probs[:, idx] if idx < probs.shape[1] else np.zeros(len(preds)) for idx, c in enumerate(model.classes_)}
    for c in [0, 1, 2]:
        if f"prob_class_{c}" not in prob_dict:
            prob_dict[f"prob_class_{c}"] = np.zeros(len(preds))

    # 7. Build Holdout Results DataFrame
    df_results = pd.DataFrame({
        "year": df_holdout["year"].astype(int),
        "actual_spei": df_holdout["spei"].astype(float),
        "actual_class_strict": y_true_strict,
        "actual_class": y_true_cal,
        "predicted_class": preds,
        "prob_class_0": prob_dict["prob_class_0"],
        "prob_class_1": prob_dict["prob_class_1"],
        "prob_class_2": prob_dict["prob_class_2"],
    })

    csv_path = val_dir / "holdout_validation_results.csv"
    df_results.to_csv(csv_path, index=False)

    # 8. Compute Classification Metrics
    report_dict = classification_report(
        y_true_cal,
        preds,
        target_names=CLASS_NAMES_3,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true_cal, preds, labels=[0, 1, 2])

    acc = float(accuracy_score(y_true_cal, preds))
    bal_acc = float(balanced_accuracy_score(y_true_cal, preds))
    macro_f1 = float(f1_score(y_true_cal, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_cal, preds, average="weighted", zero_division=0))

    metrics = {
        "holdout_site": "Debrebirkan Selassie (eth001)",
        "holdout_coordinates": {"latitude": 9.63, "longitude": 39.53},
        "selected_grid_cell": {
            "latitude": debre_spei_res.grid_metadata.selected_lat,
            "longitude": debre_spei_res.grid_metadata.selected_lon,
            "distance_km": debre_spei_res.grid_metadata.spatial_distance_km,
        },
        "holdout_period": [int(df_holdout["year"].min()), int(df_holdout["year"].max())],
        "n_holdout_samples": len(df_holdout),
        "class_distribution_strict": pd.Series(y_true_strict).value_counts().to_dict(),
        "class_distribution_calibrated": pd.Series(y_true_cal).value_counts().to_dict(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_metrics": {
            "class_0_normal": {
                "precision": float(report_dict["Normal / Wet"]["precision"]),
                "recall": float(report_dict["Normal / Wet"]["recall"]),
                "f1": float(report_dict["Normal / Wet"]["f1-score"]),
                "support": int(report_dict["Normal / Wet"]["support"]),
            },
            "class_1_moderate": {
                "precision": float(report_dict["Moderate Drought"]["precision"]),
                "recall": float(report_dict["Moderate Drought"]["recall"]),
                "f1": float(report_dict["Moderate Drought"]["f1-score"]),
                "support": int(report_dict["Moderate Drought"]["support"]),
            },
            "class_2_severe": {
                "precision": float(report_dict["Severe Drought"]["precision"]),
                "recall": float(report_dict["Severe Drought"]["recall"]),
                "f1": float(report_dict["Severe Drought"]["f1-score"]),
                "support": int(report_dict["Severe Drought"]["support"]),
            },
        },
        "confusion_matrix": conf_mat.tolist(),
    }

    # Save metrics JSONs
    with open(val_dir / "holdout_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(val_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix CSV
    df_cm = pd.DataFrame(
        conf_mat,
        index=[f"Actual_{c}" for c in CLASS_NAMES_3],
        columns=[f"Predicted_{c}" for c in CLASS_NAMES_3],
    )
    df_cm.to_csv(val_dir / "holdout_confusion_matrix.csv")

    # 9. Generate Figures
    _generate_holdout_figures(df_holdout, df_results, conf_mat, model, fig_dir)

    # 10. Save metadata
    phase4_metadata = {
        "pipeline": "Phase 4 Scientific Validation & Geographic Holdout",
        "training_site": "Gondar (eth007)",
        "holdout_site": "Debrebirkan Selassie (eth001)",
        "geographic_distance_km": 412.5,
        "holdout_evaluation_status": "COMPLETED",
        "model_artifact": str(m_p),
        "holdout_samples": len(df_holdout),
        "holdout_years": [int(df_holdout["year"].min()), int(df_holdout["year"].max())],
        "metrics_summary": metrics,
    }
    with open(meta_dir / "phase4_metadata.json", "w", encoding="utf-8") as f:
        json.dump(phase4_metadata, f, indent=2)

    return metrics


def _generate_holdout_figures(
    df_holdout: pd.DataFrame,
    df_results: pd.DataFrame,
    conf_mat: np.ndarray,
    model: RandomForestClassifier,
    fig_dir: Path,
) -> None:
    """Generate and save publication-quality figures."""
    # 1. Confusion Matrix Figure
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES_3,
        yticklabels=CLASS_NAMES_3,
        cbar=True,
    )
    plt.title("Geographic Holdout Confusion Matrix\nDebrebirkan Selassie (eth001)", fontsize=13, pad=12)
    plt.xlabel("Predicted Drought Class", fontsize=11)
    plt.ylabel("Actual Drought Class (Ground Truth)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "holdout_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Feature Importance Figure
    fi_series = pd.Series(model.feature_importances_, index=DroughtFeatureEngineer.FEATURE_NAMES).sort_values(ascending=True)
    plt.figure(figsize=(10, 7.5))
    plt.barh(fi_series.index, fi_series.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.xlabel("Gini Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    plt.title("Random Forest Feature Importance (Trained on Gondar eth007)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=300)
    plt.close()

    # 3. Solar / RWI Hypothesis 1 Figure (1874-2009)
    _generate_hypothesis1_figure(fig_dir)


def _generate_hypothesis1_figure(fig_dir: Path) -> None:
    """Generate publication-quality Solar vs RWI time series (1874-2009)."""
    df_lag = pd.read_csv("results/processed_lagged_data.csv")
    chron = df_lag.groupby("year")[["rwi", "rwi_z", "sunspot_z"]].mean().reset_index()

    # Filter 1874-2009
    chron_sub = chron[chron["year"].between(1874, 2009)].sort_values("year").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    # Plot RWI_z (green/blue)
    color_rwi = "#1f77b4"
    ax1.plot(
        chron_sub["year"],
        chron_sub["rwi_z"],
        color=color_rwi,
        linewidth=2.2,
        label=r"Standardized Tree Growth ($RWI_z$)",
    )
    ax1.set_xlabel("Year (CE)", fontsize=12)
    ax1.set_ylabel(r"Tree Ring Growth Anomaly ($RWI_z$)", color=color_rwi, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_rwi)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Plot SN_z on secondary axis (orange/red)
    ax2 = ax1.twinx()
    color_sn = "#d95f02"
    ax2.plot(
        chron_sub["year"],
        chron_sub["sunspot_z"],
        color=color_sn,
        linewidth=2.0,
        linestyle="--",
        label=r"Standardized Solar Activity ($SN_z$, 11-yr smooth)",
    )
    ax2.set_ylabel(r"Schwabe Solar Cycle ($SN_z$)", color=color_sn, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_sn)

    # Highlight historical drought years
    historical_droughts = [
        (1888, 1892, "1888-92\nGreat Famine"),
        (1913, 1914, "1913-14\nSahel Drought"),
        (1973, 1974, "1973-74\nWollo Famine"),
        (1984, 1985, "1984-85\nHistoric Famine"),
        (2002, 2003, "2002-03\nDrought"),
        (2009, 2009, "2009"),
    ]

    for y_start, y_end, label in historical_droughts:
        ax1.axvspan(y_start - 0.4, y_end + 0.4, color="crimson", alpha=0.15)
        mid_y = (y_start + y_end) / 2.0
        ax1.text(
            mid_y,
            ax1.get_ylim()[1] * 0.85 if hasattr(ax1, 'get_ylim') else 1.8,
            label,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="crimson", alpha=0.85),
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title(
        "Hypothesis 1: Multi-Decadal Solar Irradiance ($SN_z$) vs. Ethiopian Tree Growth ($RWI_z$) [1874–2009]\n"
        "Documented Historic Severe Drought Events Highlighted in Shaded Bands",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "solar_rwi_hypothesis.png", dpi=300)
    plt.close()


def main() -> None:
    """CLI runner for holdout validation."""
    print("=" * 70)
    print("  EGATE Phase 4: Training & Geographic Holdout Validation")
    print("=" * 70)

    print("\n1. Training Gondar (eth007) Model...")
    rf, meta = train_and_save_gondar_model()
    print(f"   Model trained on {meta['n_samples']} samples ({meta['training_period'][0]}–{meta['training_period'][1]})")

    print("\n2. Executing Blind Geographic Holdout on Debrebirkan Selassie (eth001)...")
    metrics = evaluate_geographic_holdout()

    print("\n=== Holdout Performance Summary ===")
    print(f"  Holdout Site:        {metrics['holdout_site']}")
    print(f"  Holdout Samples:     {metrics['n_holdout_samples']} years ({metrics['holdout_period'][0]}–{metrics['holdout_period'][1]})")
    print(f"  Overall Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy:   {metrics['balanced_accuracy']:.3f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.3f}")
    print(f"  Severe Drought F1:   {metrics['class_metrics']['class_2_severe']['f1']:.3f}")
    print(f"  Severe Drought Rec:  {metrics['class_metrics']['class_2_severe']['recall']:.3f}")

    print("\nOutput Artifacts Generated:")
    print("  - outputs/validation/holdout_validation_results.csv")
    print("  - outputs/validation/holdout_classification_report.json")
    print("  - outputs/validation/holdout_confusion_matrix.csv")
    print("  - outputs/validation/holdout_metrics.json")
    print("  - outputs/figures/solar_rwi_hypothesis.png")
    print("  - outputs/figures/holdout_confusion_matrix.png")
    print("  - outputs/figures/feature_importance.png")
    print("  - outputs/metadata/phase4_metadata.json")


if __name__ == "__main__":
    main()
