"""
Model Evaluation and SHAP Explainability Module for American Express Decision Science Challenge.

Computes rigorous out-of-time test performance metrics across:
1. Baseline LightGBM
2. Hybrid HMM + LightGBM
3. Benchmark XGBoost

Generates publication-quality charts and computes TreeExplainer SHAP values.
"""

import os
import sys
import json
import ctypes
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set non-interactive backend for server/script execution
plt.switch_backend("Agg")

# Safe macOS OpenMP library preloading
if sys.platform == "darwin":
    try:
        sklearn_path = os.path.abspath(".venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib")
        if os.path.exists(sklearn_path):
            ctypes.CDLL(sklearn_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def run_full_evaluation(
    data_path: str = "data/processed/hybrid_dataset.csv",
    models_dir: str = "models",
    output_dir: str = "visualizations",
    test_season_start: int = 2023
) -> pd.DataFrame:
    """
    Evaluates all trained models on held-out out-of-time test set (2023 season).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Load metadata
    meta_path = os.path.join(models_dir, "feature_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    base_features = meta["base_features"]
    hybrid_features = meta["hybrid_features"]
    target_col = meta["target_col"]
    
    # Load models
    base_lgbm = joblib.load(os.path.join(models_dir, "baseline_lightgbm.pkl"))
    hybrid_lgbm = joblib.load(os.path.join(models_dir, "hybrid_lightgbm.pkl"))
    benchmark_xgb = joblib.load(os.path.join(models_dir, "benchmark_xgboost.pkl"))
    
    # Strict Out-of-Time Test Set
    df_test = df[df["season"] >= test_season_start].copy().reset_index(drop=True)
    y_test = df_test[target_col].values
    
    X_test_base = df_test[base_features].values
    X_test_hybrid = df_test[hybrid_features].values
    
    # Model predictions
    models_dict = {
        "Baseline LightGBM": {
            "model": base_lgbm,
            "X": X_test_base,
            "color": "#1f77b4"
        },
        "Hybrid HMM-LightGBM": {
            "model": hybrid_lgbm,
            "X": X_test_hybrid,
            "color": "#2ca02c"
        },
        "Benchmark XGBoost": {
            "model": benchmark_xgb,
            "X": X_test_hybrid,
            "color": "#ff7f0e"
        }
    }
    
    results = []
    
    for name, info in models_dict.items():
        m = info["model"]
        X = info["X"]
        
        # Probabilities for class 1 (Team 1 wins)
        probs = m.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)
        info["probs"] = probs
        info["preds"] = preds
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        loss = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Log Loss": round(float(loss), 4),
            "Brier Score": round(float(brier), 4),
            "True Positives": int(tp),
            "False Positives": int(fp),
            "True Negatives": int(tn),
            "False Negatives": int(fn)
        })
        
    df_metrics = pd.DataFrame(results)
    csv_metrics_path = os.path.join(output_dir, "metrics_comparison.csv")
    json_metrics_path = os.path.join(output_dir, "metrics_summary.json")
    
    df_metrics.to_csv(csv_metrics_path, index=False)
    with open(json_metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n================ Out-of-Time Test Set Evaluation (Season 2023) ================")
    print(df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Log Loss", "Brier Score"]].to_string(index=False))
    print("================================================================================")
    
    # ----------------------------------------------------
    # Visualization 1: ROC Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    for name, info in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, info["probs"])
        auc = roc_auc_score(y_test, info["probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=info["color"], lw=2.2)
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance Level (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title("Out-of-Time ROC Curves: Model Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()
    
    # ----------------------------------------------------
    # Visualization 2: Precision-Recall Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    for name, info in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, info["probs"])
        plt.plot(rec, prec, label=f"{name}", color=info["color"], lw=2.2)
    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Precision-Recall Curves (Test Set 2023)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"))
    plt.close()
    
    # ----------------------------------------------------
    # Visualization 3: Confusion Matrices
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    for idx, (name, info) in enumerate(models_dict.items()):
        cm = confusion_matrix(y_test, info["preds"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Team 2 Win (0)", "Team 1 Win (1)"],
            yticklabels=["Team 2 Win (0)", "Team 1 Win (1)"],
            ax=axes[idx], annot_kws={"size": 14, "weight": "bold"}
        )
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontweight="bold")
        axes[idx].set_ylabel("True Label", fontweight="bold")
    plt.suptitle("Confusion Matrices on Held-out Out-of-Time Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()
    
    # ----------------------------------------------------
    # Visualization 4: Calibration Curves (Reliability Diagrams)
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    for name, info in models_dict.items():
        frac_pos, mean_pred = calibration_curve(y_test, info["probs"], n_bins=5, strategy="uniform")
        brier = brier_score_loss(y_test, info["probs"])
        plt.plot(mean_pred, frac_pos, "s-", label=f"{name} (Brier = {brier:.3f})", color=info["color"], lw=2.0)
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontweight="bold")
    plt.ylabel("Observed Fraction of Positives", fontweight="bold")
    plt.title("Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curves.png"))
    plt.close()
    
    # ----------------------------------------------------
    # Visualization 5: Comparative Metric Bar Chart
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=300)
    plot_df = df_metrics.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=["#1f77b4", "#2ca02c", "#ff7f0e"])
    plt.ylim(0, 1.0)
    plt.title("Comprehensive Metric Comparison Across Models", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Metric Score", fontweight="bold")
    plt.xlabel("Metric Type", fontweight="bold")
    plt.legend(title="", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_bar_chart.png"))
    plt.close()
    
    print(f"All evaluation plots saved to {output_dir}/")
    return df_metrics


def compute_shap_explainability(
    data_path: str = "data/processed/hybrid_dataset.csv",
    models_dir: str = "models",
    output_dir: str = "visualizations",
    test_season_start: int = 2023
):
    """
    Computes SHAP explanations on Hybrid HMM-LightGBM model using TreeExplainer.
    Generates summary bar, beeswarm, and dependence plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    meta_path = os.path.join(models_dir, "feature_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    hybrid_features = meta["hybrid_features"]
    hybrid_lgbm = joblib.load(os.path.join(models_dir, "hybrid_lightgbm.pkl"))
    
    df_test = df[df["season"] >= test_season_start].copy().reset_index(drop=True)
    X_test_df = df_test[hybrid_features]
    
    print("\nComputing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(hybrid_lgbm)
    shap_values = explainer.shap_values(X_test_df)
    
    # In binary classification, LightGBM shap_values may be 2D array or list of 2 arrays
    if isinstance(shap_values, list):
        sv = shap_values[1] # positive class
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
        
    # Create winner name string if not present
    if "winner" not in df_test.columns:
        df_test["winner"] = np.where(df_test["team1_win"] == 1, df_test["team1"], df_test["team2"])

    # Save SHAP data for Streamlit interactive plots
    shap_data_payload = {
        "shap_values": sv,
        "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
        "feature_names": hybrid_features,
        "X_test": X_test_df.values,
        "test_metadata": df_test[["match_id", "date", "team1", "team2", "winner", "team1_win"]].to_dict(orient="records")
    }
    joblib.dump(shap_data_payload, os.path.join(output_dir, "shap_data.pkl"))
    
    # ----------------------------------------------------
    # SHAP Plot 1: Summary Bar Chart (Global Feature Importance)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 7), dpi=300)
    shap.summary_plot(sv, X_test_df, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Win Odds)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"))
    plt.close()
    
    # ----------------------------------------------------
    # SHAP Plot 2: Beeswarm Plot (Feature Impact Distribution)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 7), dpi=300)
    shap.summary_plot(sv, X_test_df, show=False, max_display=15)
    plt.title("SHAP Beeswarm Plot: Directional Feature Impact on Win Probability", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"))
    plt.close()
    
    # ----------------------------------------------------
    # SHAP Plot 3: Dependence Plot for Net Latent Form Index
    # ----------------------------------------------------
    if "net_latent_form_index" in hybrid_features:
        idx = hybrid_features.index("net_latent_form_index")
        plt.figure(figsize=(8, 6), dpi=300)
        shap.dependence_plot(
            idx, sv, X_test_df.values,
            feature_names=hybrid_features,
            show=False
        )
        plt.title("SHAP Dependence Plot: Net HMM Latent Form Index", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_dependence_net_form.png"))
        plt.close()
        
    print(f"SHAP visualizations and artifacts saved to {output_dir}/")


if __name__ == "__main__":
    run_full_evaluation()
    compute_shap_explainability()
