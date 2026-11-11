"""
LightGBM Modeling Module for American Express Decision Science Challenge.
Implements:
1. Baseline LightGBM (Static & Rolling Features only)
2. Hybrid HMM + LightGBM (Engineered Features + HMM Latent Form Features)
3. Hyperparameter tuning via GridSearchCV with TimeSeriesSplit
"""

import os
import sys
import json
import ctypes
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any

# Safe macOS OpenMP library preloading for LightGBM
if sys.platform == "darwin":
    try:
        sklearn_path = os.path.abspath(".venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib")
        if os.path.exists(sklearn_path):
            ctypes.CDLL(sklearn_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

# Feature column definitions
BASE_FEATURES = [
    "is_toss_winner_team1",
    "toss_decision_bat",
    "venue_chase_win_rate",
    "h2h_t1_win_rate",
    "team1_win_rate_5",
    "team2_win_rate_5",
    "diff_win_rate_5",
    "team1_bat_runs_5",
    "team2_bat_runs_5",
    "diff_bat_avg_runs",
    "team1_bat_sr_5",
    "team2_bat_sr_5",
    "diff_bat_avg_sr",
    "team1_bowl_econ_5",
    "team2_bowl_econ_5",
    "diff_bowl_avg_econ",
    "team1_bowl_wkts_5",
    "team2_bowl_wkts_5",
    "diff_bowl_avg_wkts"
]

HMM_FEATURES = [
    "team1_bat_peak_prob",
    "team1_bat_slump_prob",
    "team2_bat_peak_prob",
    "team2_bat_slump_prob",
    "team1_bowl_peak_prob",
    "team1_bowl_slump_prob",
    "team2_bowl_peak_prob",
    "team2_bowl_slump_prob",
    "diff_bat_peak_prob",
    "diff_bat_slump_prob",
    "diff_bowl_peak_prob",
    "diff_bowl_slump_prob",
    "net_latent_form_index"
]

HYBRID_FEATURES = BASE_FEATURES + HMM_FEATURES
TARGET_COL = "team1_win"


def train_lightgbm_models(
    data_path: str = "data/processed/hybrid_dataset.csv",
    models_dir: str = "models",
    train_end_season: int = 2020
) -> Tuple[LGBMClassifier, LGBMClassifier, Dict[str, Any]]:
    """
    Trains Baseline and Hybrid LightGBM classifiers using GridSearchCV with TimeSeriesSplit.
    """
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Split chronologically
    df_train = df[df["season"] <= train_end_season].copy().reset_index(drop=True)
    df_test = df[df["season"] > 2022].copy().reset_index(drop=True)
    
    y_train = df_train[TARGET_COL].values
    y_test = df_test[TARGET_COL].values
    
    # Time-series cross validation (4 splits)
    tscv = TimeSeriesSplit(n_splits=4)
    
    param_grid = {
        "n_estimators": [60, 100],
        "learning_rate": [0.03, 0.08],
        "max_depth": [3, 5],
        "num_leaves": [15, 31],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    # ----------------------------------------------------
    # 1. Baseline LightGBM (No HMM Features)
    # ----------------------------------------------------
    print("\n--- Hyperparameter Tuning: Baseline LightGBM ---")
    X_train_base = df_train[BASE_FEATURES].values
    base_lgbm = LGBMClassifier(random_state=42, verbose=-1)
    grid_base = GridSearchCV(
        estimator=base_lgbm,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1
    )
    grid_base.fit(X_train_base, y_train)
    best_base_model = grid_base.best_estimator_
    print(f"Best Baseline Params: {grid_base.best_params_}")
    print(f"Best CV Neg Log Loss: {grid_base.best_score_:.4f}")
    
    # ----------------------------------------------------
    # 2. Hybrid HMM + LightGBM (Engineered + Latent Form)
    # ----------------------------------------------------
    print("\n--- Hyperparameter Tuning: Hybrid HMM-LightGBM ---")
    X_train_hybrid = df_train[HYBRID_FEATURES].values
    hybrid_lgbm = LGBMClassifier(random_state=42, verbose=-1)
    grid_hybrid = GridSearchCV(
        estimator=hybrid_lgbm,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1
    )
    grid_hybrid.fit(X_train_hybrid, y_train)
    best_hybrid_model = grid_hybrid.best_estimator_
    print(f"Best Hybrid Params: {grid_hybrid.best_params_}")
    print(f"Best CV Neg Log Loss: {grid_hybrid.best_score_:.4f}")
    
    # Quick Test Evaluation
    X_test_base = df_test[BASE_FEATURES].values
    X_test_hybrid = df_test[HYBRID_FEATURES].values
    
    p_base = best_base_model.predict_proba(X_test_base)[:, 1]
    p_hybrid = best_hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    
    print("\nPreliminary Out-of-Time Test Set Performance:")
    print(f"  Baseline LightGBM   -> ROC-AUC: {roc_auc_score(y_test, p_base):.4f} | Log Loss: {log_loss(y_test, p_base):.4f}")
    print(f"  Hybrid HMM-LightGBM -> ROC-AUC: {roc_auc_score(y_test, p_hybrid):.4f} | Log Loss: {log_loss(y_test, p_hybrid):.4f}")
    
    # Save models
    base_path = os.path.join(models_dir, "baseline_lightgbm.pkl")
    hybrid_path = os.path.join(models_dir, "hybrid_lightgbm.pkl")
    meta_path = os.path.join(models_dir, "feature_metadata.json")
    
    joblib.dump(best_base_model, base_path)
    joblib.dump(best_hybrid_model, hybrid_path)
    
    metadata = {
        "base_features": BASE_FEATURES,
        "hmm_features": HMM_FEATURES,
        "hybrid_features": HYBRID_FEATURES,
        "target_col": TARGET_COL,
        "best_base_params": grid_base.best_params_,
        "best_hybrid_params": grid_hybrid.best_params_
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nModels and metadata saved to {models_dir}/")
    return best_base_model, best_hybrid_model, metadata


if __name__ == "__main__":
    train_lightgbm_models()
