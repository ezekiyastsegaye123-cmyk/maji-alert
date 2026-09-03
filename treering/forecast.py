"""
Tree-Ring & Solar-Cycle Random Forest Drought Forecasting Module
================================================================

This module implements a production machine learning pipeline to predict
and forecast drought occurrence and Standardized Precipitation-Evapotranspiration
Index (SPEI) values across multi-year and decadal timescales (11-year Schwabe cycle).

Scientific Foundations:
-----------------------
1. Solar Cycle Teleconnection:
   The ~11-year Schwabe solar cycle modulates regional hydroclimate through
   stratospheric ozone heating, Walker circulation adjustments, and regional
   monsoon dynamics in East Africa (Ethiopian Highlands / Upper Blue Nile).
2. Tree-Ring Growth Memory:
   Standardized Ring Width Index (RWI) captures biological growth persistence,
   soil moisture memory, and cambial growth responses.
3. Ground-Truth Climatology:
   Standardized Precipitation-Evapotranspiration Index (SPEI-1 annual average)
   provides verified drought ground truth from SPEIbase v2.11.

Architecture:
-------------
- `SolarCyclePhaseCalculator`: Phase calculation along the 11-year Schwabe cycle.
- `DroughtFeatureEngineer`: Causal feature generation from solar & RWI records.
- `DroughtForecaster`: Ensemble Random Forest classifier and regressor with
  walk-forward cross validation and 11-year multi-step forward forecasting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit

logger = logging.getLogger(__name__)

# Known historical solar cycle minima (SILSO / NOAA solar cycle consensus)
DEFAULT_SOLAR_MINIMA: Tuple[int, ...] = (
    1700, 1712, 1723, 1734, 1745, 1755, 1766, 1775, 1784, 1798,
    1810, 1823, 1833, 1843, 1856, 1867, 1878, 1889, 1902, 1913,
    1923, 1933, 1944, 1954, 1964, 1976, 1986, 1996, 2008, 2019,
    2030, 2041,
)

# Projected Sunspot Numbers for Solar Cycle 25 (declining) and Solar Cycle 26 (rising)
DEFAULT_PROJECTED_SUNSPOTS: Dict[int, float] = {
    2025: 123.2,  # Observed/provisional SILSO 2025
    2026: 95.0,   # Declining phase Solar Cycle 25
    2027: 65.0,   # Declining phase
    2028: 40.0,   # Approaching solar minimum
    2029: 20.0,   # Low solar activity
    2030: 8.0,    # Solar Minimum SC25/SC26 transition
    2031: 15.0,   # Early ascending phase Solar Cycle 26
    2032: 45.0,   # Ascending phase
    2033: 85.0,   # Rapid ascent
    2034: 125.0,  # High solar activity
    2035: 145.0,  # Solar Maximum Solar Cycle 26
}


# =====================================================================
# Custom Exceptions
# =====================================================================


class ForecastError(Exception):
    """Base exception for all drought forecasting errors."""


class FeatureEngineeringError(ForecastError):
    """Raised when feature engineering or alignment fails."""


class ModelTrainingError(ForecastError):
    """Raised when model training or evaluation fails."""


class ProjectionError(ForecastError):
    """Raised when forward solar projection or forecasting fails."""


# =====================================================================
# Metadata Containers
# =====================================================================


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Evaluation metrics from time-series / k-fold cross validation."""

    cv_method: str
    n_splits: int
    classification_accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    regression_r2: float
    regression_mae: float
    regression_rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastYearResult:
    """Individual annual forecast prediction result."""

    year: int
    projected_sunspot: float
    solar_phase: float
    predicted_spei: float
    spei_ci_lower_95: float
    spei_ci_upper_95: float
    drought_probability: float
    risk_level: str
    predicted_class: int  # 1 = Drought, 0 = Normal/Wet

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Solar Cycle Phase Calculator
# =====================================================================


class SolarCyclePhaseCalculator:
    """Calculates normalized solar cycle phase theta in [0, 1) and harmonics."""

    def __init__(self, minima: Optional[Tuple[int, ...]] = None):
        self.minima = np.array(minima or DEFAULT_SOLAR_MINIMA, dtype=int)
        if len(self.minima) < 2 or not np.all(np.diff(self.minima) > 0):
            raise ValueError("Solar minima must be a strictly increasing sequence of at least 2 years.")

    def compute_phase(self, year: int) -> float:
        """Calculate normalized phase theta in [0, 1) for a given calendar year."""
        prev_mins = self.minima[self.minima <= year]
        if len(prev_mins) == 0:
            t0 = self.minima[0] - 11
            period = self.minima[0] - t0
        else:
            t0 = prev_mins[-1]
            next_mins = self.minima[self.minima > year]
            t1 = next_mins[0] if len(next_mins) > 0 else t0 + 11
            period = max(1, t1 - t0)

        phase = (year - t0) / period
        return float(phase % 1.0)

    def compute_harmonics(self, year: int) -> Tuple[float, float, float]:
        """Returns (phase, sin(2*pi*phase), cos(2*pi*phase))."""
        phase = self.compute_phase(year)
        sin_p = float(np.sin(2.0 * np.pi * phase))
        cos_p = float(np.cos(2.0 * np.pi * phase))
        return phase, sin_p, cos_p


# =====================================================================
# Feature Engineering
# =====================================================================


class DroughtFeatureEngineer:
    """Engineers causal predictors from Sunspot, Tree-Ring (RWI), Ocean Indices (ENSO/IOD), and SPEI records."""

    BASE_FEATURE_NAMES: List[str] = [
        "sunspot",
        "sunspot_lag1",
        "sunspot_lag2",
        "sunspot_lag3",
        "sunspot_lag4",
        "sunspot_lag5",
        "sunspot_smooth11",
        "sunspot_diff1",
        "sunspot_diff3",
        "solar_phase",
        "solar_phase_sin",
        "solar_phase_cos",
        "rwi",
        "rwi_lag1",
        "rwi_diff1",
        "rwi_smooth5",
    ]

    OCEAN_FEATURE_NAMES: List[str] = [
        "nino34_mean",
        "dmi_mean",
    ]

    FEATURE_NAMES: List[str] = BASE_FEATURE_NAMES + OCEAN_FEATURE_NAMES

    def __init__(
        self,
        phase_calculator: Optional[SolarCyclePhaseCalculator] = None,
        drought_threshold: float = -0.25,
    ):
        self.phase_calc = phase_calculator or SolarCyclePhaseCalculator()
        self.drought_threshold = float(drought_threshold)

    def build_solar_feature_table(self, df_sun: pd.DataFrame) -> pd.DataFrame:
        """
        Build full solar feature series from historical sunspot records.
        Requires columns: ['year', 'sunspot'].
        """
        if "year" not in df_sun.columns or "sunspot" not in df_sun.columns:
            raise FeatureEngineeringError("Sunspot DataFrame must contain 'year' and 'sunspot' columns.")

        df = df_sun.copy().sort_values("year").drop_duplicates("year").reset_index(drop=True)
        df["year"] = df["year"].astype(int)
        df["sunspot"] = df["sunspot"].astype(float)

        # Lags
        for l in range(1, 6):
            df[f"sunspot_lag{l}"] = df["sunspot"].shift(l)

        # Differentials & moving averages
        df["sunspot_diff1"] = df["sunspot"].diff(1)
        df["sunspot_diff3"] = df["sunspot"].diff(3)
        df["sunspot_smooth11"] = df["sunspot"].rolling(11, center=True, min_periods=3).mean()

        # Fill edges
        df["sunspot_smooth11"] = df["sunspot_smooth11"].bfill().ffill()
        df["sunspot_diff1"] = df["sunspot_diff1"].bfill().ffill()
        df["sunspot_diff3"] = df["sunspot_diff3"].bfill().ffill()
        for l in range(1, 6):
            df[f"sunspot_lag{l}"] = df[f"sunspot_lag{l}"].bfill().ffill()

        # Phases
        phases = [self.phase_calc.compute_harmonics(int(y)) for y in df["year"]]
        df["solar_phase"] = [p[0] for p in phases]
        df["solar_phase_sin"] = [p[1] for p in phases]
        df["solar_phase_cos"] = [p[2] for p in phases]

        return df

    def build_tree_ring_chronology(self, df_lag: pd.DataFrame) -> pd.DataFrame:
        """
        Build site-average chronology and growth dynamics from tree-ring dataset.
        Requires columns: ['year', 'rwi'].
        """
        if "year" not in df_lag.columns or "rwi" not in df_lag.columns:
            raise FeatureEngineeringError("Tree-ring DataFrame must contain 'year' and 'rwi' columns.")

        chronology = df_lag.groupby("year")[["rwi"]].mean().reset_index().sort_values("year")
        chronology["year"] = chronology["year"].astype(int)
        chronology["rwi"] = chronology["rwi"].astype(float)

        chronology["rwi_lag1"] = chronology["rwi"].shift(1).bfill()
        chronology["rwi_lag2"] = chronology["rwi"].shift(2).bfill()
        chronology["rwi_diff1"] = chronology["rwi"].diff(1).bfill()
        chronology["rwi_smooth5"] = chronology["rwi"].rolling(5, center=True, min_periods=2).mean().bfill().ffill()

        return chronology

    def build_training_dataset(
        self,
        df_chronology: pd.DataFrame,
        df_solar: pd.DataFrame,
        df_spei: pd.DataFrame,
        df_ocean: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge chronology, solar features, SPEI ground-truth, and ocean indices into aligned training set.
        """
        if "year" not in df_spei.columns or "spei" not in df_spei.columns:
            raise FeatureEngineeringError("SPEI DataFrame must contain 'year' and 'spei' columns.")

        df_spei_clean = df_spei[["year", "spei"]].copy()
        df_spei_clean["year"] = df_spei_clean["year"].astype(int)
        df_spei_clean["spei"] = df_spei_clean["spei"].astype(float)

        # Merge chronology, solar, and SPEI
        merged = pd.merge(df_chronology, df_solar, on="year", how="inner")
        merged = pd.merge(merged, df_spei_clean, on="year", how="inner").sort_values("year").reset_index(drop=True)

        # Merge ocean teleconnection features if provided
        if df_ocean is not None:
            merged = pd.merge(merged, df_ocean, on="year", how="left")

        # Fill neutral climatology (0.0 anomaly) for any missing ocean features
        for f in self.OCEAN_FEATURE_NAMES:
            if f not in merged.columns:
                merged[f] = 0.0
            else:
                merged[f] = merged[f].fillna(0.0)

        if len(merged) < 20:
            raise FeatureEngineeringError(f"Insufficient merged training samples: {len(merged)} rows found.")

        # Target classification
        merged["drought_class"] = (merged["spei"] < self.drought_threshold).astype(int)

        # Validate no NaNs in required features
        for f in self.FEATURE_NAMES:
            if f not in merged.columns:
                raise FeatureEngineeringError(f"Missing required feature column: {f}")
            if merged[f].isna().any():
                merged[f] = merged[f].bfill().ffill().fillna(0.0)

        return merged


# =====================================================================
# Drought Forecaster Model Suite
# =====================================================================


class DroughtForecaster:
    """
    Random Forest Drought Forecasting Engine.
    Combines Random Forest Classification (drought risk probability) and
    Random Forest Regression (continuous SPEI index estimation with confidence intervals).
    """

    def __init__(
        self,
        drought_threshold: float = -0.25,
        n_estimators: int = 300,
        max_depth: int = 4,
        min_samples_split: int = 4,
        random_state: int = 42,
    ):
        self.drought_threshold = drought_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.feature_engineer = DroughtFeatureEngineer(drought_threshold=drought_threshold)
        self.classifier: Optional[RandomForestClassifier] = None
        self.regressor: Optional[RandomForestRegressor] = None
        self.training_data_: Optional[pd.DataFrame] = None
        self.feature_importances_: Optional[Dict[str, float]] = None

    def fit(self, df_train: pd.DataFrame) -> DroughtForecaster:
        """Fit classifier and regressor on training dataset."""
        self.training_data_ = df_train.copy()
        X = df_train[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_clf = df_train["drought_class"].values
        y_reg = df_train["spei"].values

        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.classifier.fit(X, y_clf)

        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )
        self.regressor.fit(X, y_reg)

        # Calculate feature importances
        importances = self.classifier.feature_importances_
        self.feature_importances_ = {
            f: float(imp) for f, imp in zip(DroughtFeatureEngineer.FEATURE_NAMES, importances)
        }

        return self

    def evaluate_cross_validation(
        self,
        df_train: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
        cv_type: str = "kfold",
    ) -> CrossValidationMetrics:
        """
        Perform rigorous cross-validation (KFold or TimeSeriesSplit) and return metrics.
        """
        data = df_train if df_train is not None else self.training_data_
        if data is None:
            raise ModelTrainingError("No training data available. Call fit() or provide df_train.")

        X = data[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_clf = data["drought_class"].values
        y_reg = data["spei"].values

        if cv_type.lower() == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        accs, b_accs, precs, recs, f1s, aucs, briers = [], [], [], [], [], [], []
        r2s, maes, rmses = [], [], []

        for tr_idx, te_idx in cv.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_c_tr, y_c_te = y_clf[tr_idx], y_clf[te_idx]
            y_r_tr, y_r_te = y_reg[tr_idx], y_reg[te_idx]

            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                class_weight="balanced",
                random_state=self.random_state,
            )
            clf.fit(X_tr, y_c_tr)

            reg = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            reg.fit(X_tr, y_r_tr)

            probs = clf.predict_proba(X_te)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_c_te))
            preds_c = clf.predict(X_te)
            preds_r = reg.predict(X_te)

            accs.append(accuracy_score(y_c_te, preds_c))
            b_accs.append(balanced_accuracy_score(y_c_te, preds_c))
            precs.append(precision_score(y_c_te, preds_c, zero_division=0))
            recs.append(recall_score(y_c_te, preds_c, zero_division=0))
            f1s.append(f1_score(y_c_te, preds_c, zero_division=0))
            briers.append(brier_score_loss(y_c_te, probs))

            if len(np.unique(y_c_te)) > 1:
                aucs.append(roc_auc_score(y_c_te, probs))

            r2s.append(r2_score(y_r_te, preds_r))
            maes.append(mean_absolute_error(y_r_te, preds_r))
            rmses.append(float(np.sqrt(mean_squared_error(y_r_te, preds_r))))

        return CrossValidationMetrics(
            cv_method=cv_type,
            n_splits=n_splits,
            classification_accuracy=float(np.mean(accs)),
            balanced_accuracy=float(np.mean(b_accs)),
            precision=float(np.mean(precs)),
            recall=float(np.mean(recs)),
            f1=float(np.mean(f1s)),
            roc_auc=float(np.mean(aucs)) if aucs else 0.5,
            brier_score=float(np.mean(briers)),
            regression_r2=float(np.mean(r2s)),
            regression_mae=float(np.mean(maes)),
            regression_rmse=float(np.mean(rmses)),
        )

    def forecast_solar_cycle(
        self,
        start_year: int = 2025,
        end_year: int = 2035,
        projected_sunspots: Optional[Dict[int, float]] = None,
        df_historical_sun: Optional[pd.DataFrame] = None,
    ) -> List[ForecastYearResult]:
        """
        Generate 11-year forward drought forecast across target solar cycle.
        """
        if self.classifier is None or self.regressor is None or self.training_data_ is None:
            raise ModelTrainingError("Model must be fitted before running forecast.")

        proj_sn = projected_sunspots or DEFAULT_PROJECTED_SUNSPOTS
        years = list(range(start_year, end_year + 1))

        # Build combined sunspot table
        if df_historical_sun is not None:
            df_sun_base = df_historical_sun.copy()
        else:
            df_sun_base = self.training_data_[["year", "sunspot"]].copy()

        df_proj = pd.DataFrame([{"year": y, "sunspot": proj_sn.get(y, 50.0)} for y in years])
        df_sun_full = (
            pd.concat([df_sun_base[df_sun_base["year"] < start_year], df_proj], ignore_index=True)
            .sort_values("year")
            .drop_duplicates("year")
            .reset_index(drop=True)
        )

        # Recompute full solar features
        df_solar_features = self.feature_engineer.build_solar_feature_table(df_sun_full)

        # Growth autoregressive state
        curr_rwi = float(self.training_data_["rwi"].iloc[-1])
        curr_rwi_lag1 = float(self.training_data_["rwi"].iloc[-2])
        recent_rwis = list(self.training_data_["rwi"].iloc[-4:])

        results: List[ForecastYearResult] = []

        for y in years:
            sun_rows = df_solar_features[df_solar_features["year"] == y]
            if len(sun_rows) == 0:
                raise ProjectionError(f"No solar feature data available for forecast year {y}")
            sun_row = sun_rows.iloc[0]

            rwi_diff = curr_rwi - curr_rwi_lag1
            rwi_smooth5 = float(np.mean(recent_rwis[-4:] + [curr_rwi]))

            feat_dict = {
                "sunspot": float(sun_row["sunspot"]),
                "sunspot_lag1": float(sun_row["sunspot_lag1"]),
                "sunspot_lag2": float(sun_row["sunspot_lag2"]),
                "sunspot_lag3": float(sun_row["sunspot_lag3"]),
                "sunspot_lag4": float(sun_row["sunspot_lag4"]),
                "sunspot_lag5": float(sun_row["sunspot_lag5"]),
                "sunspot_smooth11": float(sun_row["sunspot_smooth11"]),
                "sunspot_diff1": float(sun_row["sunspot_diff1"]),
                "sunspot_diff3": float(sun_row["sunspot_diff3"]),
                "solar_phase": float(sun_row["solar_phase"]),
                "solar_phase_sin": float(sun_row["solar_phase_sin"]),
                "solar_phase_cos": float(sun_row["solar_phase_cos"]),
                "rwi": curr_rwi,
                "rwi_lag1": curr_rwi_lag1,
                "rwi_diff1": rwi_diff,
                "rwi_smooth5": rwi_smooth5,
                "nino34_mean": 0.0,
                "dmi_mean": 0.0,
            }

            x_vec = np.array([[feat_dict[f] for f in DroughtFeatureEngineer.FEATURE_NAMES]])

            prob_drought = float(self.classifier.predict_proba(x_vec)[0, 1])
            pred_class = int(prob_drought >= 0.50)
            pred_spei = float(self.regressor.predict(x_vec)[0])

            # Ensemble prediction bounds
            tree_preds = [tree.predict(x_vec)[0] for tree in self.regressor.estimators_]
            spei_std = float(np.std(tree_preds))
            ci_lower = float(pred_spei - 1.96 * spei_std)
            ci_upper = float(pred_spei + 1.96 * spei_std)

            # Climatological risk assignment
            if prob_drought >= 0.65:
                risk = "High"
            elif prob_drought >= 0.45:
                risk = "Elevated"
            elif prob_drought >= 0.30:
                risk = "Guarded"
            else:
                risk = "Low"

            results.append(
                ForecastYearResult(
                    year=y,
                    projected_sunspot=float(sun_row["sunspot"]),
                    solar_phase=float(sun_row["solar_phase"]),
                    predicted_spei=pred_spei,
                    spei_ci_lower_95=ci_lower,
                    spei_ci_upper_95=ci_upper,
                    drought_probability=prob_drought,
                    risk_level=risk,
                    predicted_class=pred_class,
                )
            )

            # Autoregressive forward propagation of tree growth
            next_rwi = float(0.70 * curr_rwi + 0.30 * (1.0 + pred_spei * 0.15))
            curr_rwi_lag1 = curr_rwi
            curr_rwi = next_rwi
            recent_rwis.append(curr_rwi)

        return results


# =====================================================================
# Pipeline Orchestrator & Exporters
# =====================================================================


def run_drought_forecasting_pipeline(
    processed_lagged_data_path: Union[str, Path] = "results/processed_lagged_data.csv",
    spei_csv_path: Union[str, Path] = "results/spei_debrebirkan.csv",
    sunspot_csv_path: Union[str, Path] = "SN_y_tot_V2.0.csv",
    output_forecast_csv: Union[str, Path] = "results/drought_forecast_2025_2035.csv",
    output_metrics_json: Union[str, Path] = "results/drought_model_metrics.json",
    output_backtesting_csv: Union[str, Path] = "results/drought_backtesting_historical.csv",
    drought_threshold: float = -0.25,
    overwrite: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete end-to-end training, validation, and 11-year forward forecasting.
    """
    lag_p = Path(processed_lagged_data_path)
    spei_p = Path(spei_csv_path)
    sun_p = Path(sunspot_csv_path)

    if not lag_p.exists():
        raise FileNotFoundError(f"Processed lagged data not found: {lag_p}")
    if not spei_p.exists():
        raise FileNotFoundError(f"SPEI CSV not found: {spei_p}")
    if not sun_p.exists():
        raise FileNotFoundError(f"Sunspot CSV not found: {sun_p}")

    # 1. Ingest datasets
    df_lag = pd.read_csv(lag_p)
    df_spei = pd.read_csv(spei_p)

    # Ingest SILSO sunspots
    df_sun = pd.read_csv(sun_p, sep=";", header=None, usecols=[0, 1])
    df_sun.columns = ["year_dec", "sunspot"]
    df_sun["year"] = df_sun["year_dec"].astype(int)
    df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

    # 2. Feature engineering
    engineer = DroughtFeatureEngineer(drought_threshold=drought_threshold)
    df_chronology = engineer.build_tree_ring_chronology(df_lag)
    df_solar = engineer.build_solar_feature_table(df_sun)
    df_train = engineer.build_training_dataset(df_chronology, df_solar, df_spei)

    # 3. Model training & Cross-Validation
    forecaster = DroughtForecaster(drought_threshold=drought_threshold)
    forecaster.fit(df_train)

    cv_kfold = forecaster.evaluate_cross_validation(df_train, n_splits=5, cv_type="kfold")
    cv_timeseries = forecaster.evaluate_cross_validation(df_train, n_splits=5, cv_type="timeseries")

    # 4. Out-of-fold historical backtesting
    X_train = df_train[DroughtFeatureEngineer.FEATURE_NAMES].values
    y_clf_train = df_train["drought_class"].values
    y_reg_train = df_train["spei"].values

    hist_preds = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in kf.split(X_train):
        clf_fold = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
        clf_fold.fit(X_train[tr_idx], y_clf_train[tr_idx])
        probs = clf_fold.predict_proba(X_train[te_idx])[:, 1]
        preds = clf_fold.predict(X_train[te_idx])

        reg_fold = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        reg_fold.fit(X_train[tr_idx], y_reg_train[tr_idx])
        preds_spei = reg_fold.predict(X_train[te_idx])

        for idx, p, pr, ps in zip(te_idx, preds, probs, preds_spei):
            hist_preds.append({
                "year": int(df_train["year"].iloc[idx]),
                "actual_spei": float(df_train["spei"].iloc[idx]),
                "actual_drought": int(df_train["drought_class"].iloc[idx]),
                "predicted_drought": int(p),
                "drought_probability": float(pr),
                "predicted_spei": float(ps),
            })

    df_backtest = pd.DataFrame(hist_preds).sort_values("year").reset_index(drop=True)

    # 5. 11-Year forward forecast (2025-2035)
    forecast_results = forecaster.forecast_solar_cycle(
        start_year=2025,
        end_year=2035,
        df_historical_sun=df_sun,
    )
    df_forecast = pd.DataFrame([r.to_dict() for r in forecast_results])

    # 6. Export outputs
    out_f_p = Path(output_forecast_csv)
    out_m_p = Path(output_metrics_json)
    out_b_p = Path(output_backtesting_csv)

    for p in [out_f_p, out_m_p, out_b_p]:
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output file {p} exists and overwrite=False.")
        p.parent.mkdir(parents=True, exist_ok=True)

    df_forecast.to_csv(out_f_p, index=False)
    df_backtest.to_csv(out_b_p, index=False)

    metrics_payload = {
        "model_name": "Random Forest Solar-Tree Drought Forecaster",
        "drought_threshold_spei": drought_threshold,
        "n_training_samples": len(df_train),
        "training_year_range": [int(df_train["year"].min()), int(df_train["year"].max())],
        "kfold_cv_metrics": cv_kfold.to_dict(),
        "timeseries_cv_metrics": cv_timeseries.to_dict(),
        "feature_importances": forecaster.feature_importances_,
        "forecast_period": [2025, 2035],
    }

    with open(out_m_p, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return df_forecast, metrics_payload


# =====================================================================
# CLI Entry Point
# =====================================================================


def main() -> None:
    """CLI dispatcher for treering forecast."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Random Forest Solar-Cycle 11-Year Drought Forecasting Engine"
    )
    parser.add_argument(
        "--input-lagged",
        default="results/processed_lagged_data.csv",
        help="Path to processed lagged RWI data CSV",
    )
    parser.add_argument(
        "--input-spei",
        default="results/spei_debrebirkan.csv",
        help="Path to annual SPEI ground-truth CSV",
    )
    parser.add_argument(
        "--input-sunspot",
        default="SN_y_tot_V2.0.csv",
        help="Path to SILSO sunspot CSV",
    )
    parser.add_argument(
        "--output-forecast",
        default="results/drought_forecast_2025_2035.csv",
        help="Output 11-year forecast CSV path",
    )
    parser.add_argument(
        "--output-metrics",
        default="results/drought_model_metrics.json",
        help="Output model validation metrics JSON path",
    )
    parser.add_argument(
        "--drought-threshold",
        type=float,
        default=-0.25,
        help="SPEI threshold defining drought event (default: -0.25)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Random Forest Solar-Cycle 11-Year Drought Forecaster")
    print("=" * 70)

    df_forecast, metrics = run_drought_forecasting_pipeline(
        processed_lagged_data_path=args.input_lagged,
        spei_csv_path=args.input_spei,
        sunspot_csv_path=args.input_sunspot,
        output_forecast_csv=args.output_forecast,
        output_metrics_json=args.output_metrics,
        drought_threshold=args.drought_threshold,
        overwrite=args.overwrite,
    )

    print("\nModel Training & Cross-Validation Summary:")
    print(f"  Training Samples: {metrics['n_training_samples']} years ({metrics['training_year_range'][0]}–{metrics['training_year_range'][1]})")
    print(f"  K-Fold CV ROC-AUC: {metrics['kfold_cv_metrics']['roc_auc']:.3f} | Accuracy: {metrics['kfold_cv_metrics']['classification_accuracy']:.3f}")
    print(f"  TimeSeries CV ROC-AUC: {metrics['timeseries_cv_metrics']['roc_auc']:.3f} | Accuracy: {metrics['timeseries_cv_metrics']['classification_accuracy']:.3f}")

    print("\n=== 11-Year Drought Forecast Schedule (2025–2035) ===")
    print(df_forecast[["year", "projected_sunspot", "solar_phase", "predicted_spei", "drought_probability", "risk_level"]].to_string(index=False))

    print(f"\nForecast exported to: {args.output_forecast}")
    print(f"Metrics exported to:  {args.output_metrics}")


if __name__ == "__main__":
    main()
