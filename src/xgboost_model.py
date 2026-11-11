"""
XGBoost Benchmarking Module for American Express Decision Science Challenge.
Implements independent XGBoost gradient boosted classifier to benchmark against
Hybrid HMM + LightGBM and Baseline LightGBM.
"""

import os
import sys
import ctypes
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any

# Safe macOS OpenMP library preloading
if sys.platform == "darwin":
    try:
        sklearn_path = os.path.abspath(".venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib")
        if os.path.exists(sklearn_path):
            ctypes.CDLL(sklearn_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from src.lightgbm_model import HYBRID_FEATURES, TARGET_COL


def train_xgboost_benchmark(
    data_path: str = "data/processed/hybrid_dataset.csv",
    models_dir: str = "models",
    train_end_season: int = 2020
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Trains XGBoost benchmark classifier using GridSearchCV with TimeSeriesSplit.
    """
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Chronological Split
    df_train = df[df["season"] <= train_end_season].copy().reset_index(drop=True)
    df_test = df[df["season"] > 2022].copy().reset_index(drop=True)
    
    X_train = df_train[HYBRID_FEATURES].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[HYBRID_FEATURES].values
    y_test = df_test[TARGET_COL].values
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    param_grid = {
        "n_estimators": [60, 100],
        "learning_rate": [0.03, 0.08],
        "max_depth": [3, 4],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    print("\n--- Hyperparameter Tuning: XGBoost Benchmark ---")
    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
    grid_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1
    )
    grid_xgb.fit(X_train, y_train)
    best_xgb_model = grid_xgb.best_estimator_
    
    print(f"Best XGBoost Params: {grid_xgb.best_params_}")
    print(f"Best CV Neg Log Loss: {grid_xgb.best_score_:.4f}")
    
    # Test evaluation
    p_xgb = best_xgb_model.predict_proba(X_test)[:, 1]
    print("\nXGBoost Out-of-Time Test Set Performance:")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, p_xgb):.4f}")
    print(f"  Log Loss : {log_loss(y_test, p_xgb):.4f}")
    print(f"  Accuracy : {accuracy_score(y_test, (p_xgb >= 0.5).astype(int)):.4f}")
    
    # Save model
    xgb_path = os.path.join(models_dir, "benchmark_xgboost.pkl")
    joblib.dump(best_xgb_model, xgb_path)
    print(f"Serialized XGBoost model to {xgb_path}")
    
    return best_xgb_model, grid_xgb.best_params_


if __name__ == "__main__":
    train_xgboost_benchmark()
