"""
Unit Test Suite for American Express Decision Science Challenge.
Tests data integrity, leakage prevention, HMM state normalization,
model predictions, and SHAP artifacts.
"""

import os
import sys
import json
import ctypes
import pytest
import numpy as np
import pandas as pd
import joblib

# Safe macOS OpenMP library preloading
if sys.platform == "darwin":
    try:
        sklearn_path = os.path.abspath(".venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib")
        if os.path.exists(sklearn_path):
            ctypes.CDLL(sklearn_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_raw_data, create_time_based_split
from src.hmm_model import PlayerFormHMM

# Register in __main__ for backwards unpickling compatibility
import __main__
__main__.PlayerFormHMM = PlayerFormHMM


def test_data_integrity():
    """Verify that raw dataset files exist and meet schema constraints."""
    df_m, df_bat, df_bowl = load_raw_data("data/raw")
    
    assert len(df_m) >= 500, "Matches dataset is unexpectedly small"
    assert len(df_bat) >= 5000, "Batting scorecard dataset is unexpectedly small"
    assert len(df_bowl) >= 4000, "Bowling scorecard dataset is unexpectedly small"
    
    assert "team1_win" in df_m.columns, "Target variable 'team1_win' missing"
    assert set(df_m["team1_win"].unique()).issubset({0, 1}), "Target variable contains invalid labels"
    
    # Class balance check: sports outcomes should not be severely skewed
    t1_win_rate = df_m["team1_win"].mean()
    assert 0.40 <= t1_win_rate <= 0.60, f"Unrealistic match outcome balance: {t1_win_rate:.3f}"


def test_leakage_guards():
    """Verify strict chronological splitting and causal ordering."""
    df_m, _, _ = load_raw_data("data/raw")
    train_m, val_m, test_m = create_time_based_split(df_m, train_end_season=2020, val_end_season=2022)
    
    max_train_date = train_m["date"].max()
    min_val_date = val_m["date"].min()
    max_val_date = val_m["date"].max()
    min_test_date = test_m["date"].min()
    
    assert max_train_date < min_val_date, "Leakage detected: Train date overlaps with validation date"
    assert max_val_date < min_test_date, "Leakage detected: Validation date overlaps with test date"


def test_hmm_models_and_normalization():
    """Verify HMM models output valid probability simplex vectors."""
    bat_hmm = joblib.load("models/hmm_batting.pkl")
    bowl_hmm = joblib.load("models/hmm_bowling.pkl")
    
    assert bat_hmm.is_fitted, "Batting HMM is not fitted"
    assert bowl_hmm.is_fitted, "Bowling HMM is not fitted"
    
    # Dummy player history
    dummy_bat = pd.DataFrame({
        "runs": [35, 12, 65, 8],
        "strike_rate": [130.0, 110.0, 155.0, 80.0],
        "boundary_rate": [0.15, 0.08, 0.22, 0.05]
    })
    
    bat_probs = bat_hmm.predict_next_state_probs(dummy_bat)
    assert len(bat_probs) == 3, "Batting HMM did not return 3 states"
    assert np.isclose(np.sum(bat_probs), 1.0, atol=1e-4), "State probabilities do not sum to 1.0"
    assert (bat_probs >= 0).all() and (bat_probs <= 1).all(), "Probabilities outside [0, 1]"


def test_trained_model_inference():
    """Verify that LightGBM and XGBoost models generate calibrated predictions in [0, 1]."""
    base_lgbm = joblib.load("models/baseline_lightgbm.pkl")
    hybrid_lgbm = joblib.load("models/hybrid_lightgbm.pkl")
    benchmark_xgb = joblib.load("models/benchmark_xgboost.pkl")
    
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
        
    df_hybrid = pd.read_csv("data/processed/hybrid_dataset.csv")
    test_sample = df_hybrid[df_hybrid["season"] >= 2023].iloc[:5]
    
    X_base = test_sample[meta["base_features"]].values
    X_hybrid = test_sample[meta["hybrid_features"]].values
    
    p_base = base_lgbm.predict_proba(X_base)[:, 1]
    p_hybrid = hybrid_lgbm.predict_proba(X_hybrid)[:, 1]
    p_xgb = benchmark_xgb.predict_proba(X_hybrid)[:, 1]
    
    for p_arr, name in [(p_base, "Baseline LightGBM"), (p_hybrid, "Hybrid LightGBM"), (p_xgb, "Benchmark XGBoost")]:
        assert len(p_arr) == 5, f"{name} returned incorrect prediction length"
        assert (p_arr >= 0.0).all() and (p_arr <= 1.0).all(), f"{name} predicted probabilities outside [0, 1]"
        assert not np.isnan(p_arr).any(), f"{name} produced NaN predictions"


def test_shap_artifacts():
    """Verify that SHAP explanation data exists and matches dimensions."""
    assert os.path.exists("visualizations/shap_data.pkl"), "SHAP data payload missing"
    shap_data = joblib.load("visualizations/shap_data.pkl")
    
    assert "shap_values" in shap_data, "Missing shap_values in payload"
    assert "feature_names" in shap_data, "Missing feature_names in payload"
    assert len(shap_data["feature_names"]) == len(shap_data["shap_values"][0]), "SHAP feature dimension mismatch"
