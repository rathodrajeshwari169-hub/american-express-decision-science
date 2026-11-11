"""
Generates clean, production-grade Jupyter Notebooks for the American Express Decision Science Challenge.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def create_all_notebooks(output_dir="notebooks"):
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Notebook 01: EDA
    # ----------------------------------------------------
    nb01_cells = [
        markdown_cell("""# 01. Exploratory Data Analysis (EDA)
## American Express Decision Science Challenge – T20 Sports Analytics

### Objective:
- Understand dataset distributions, team performance, venue dynamics, and target class balance.
- Inspect player scorecards (batsmen and bowlers) to identify momentum and form patterns.
"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_matches = pd.read_csv('../data/raw/matches.csv')
df_bat = pd.read_csv('../data/raw/batsman_level_scorecard.csv')
df_bowl = pd.read_csv('../data/raw/bowler_level_scorecard.csv')

print(f"Total Matches: {len(df_matches)}")
print(f"Total Batsman Records: {len(df_bat)}")
print(f"Total Bowler Records: {len(df_bowl)}")"""),
        markdown_cell("""### 1. Target Variable Distribution: Match Outcomes
Examining the balance between Team 1 wins and Team 2 wins to ensure no synthetic bias or class skew.
"""),
        code_cell("""df_matches['team1_win'] = (df_matches['winner'] == df_matches['team1']).astype(int)
print(df_matches['team1_win'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(data=df_matches, x='team1_win', palette='Set2')
plt.title('Target Variable Distribution (Team 1 Wins: 1, Team 2 Wins: 0)')
plt.xlabel('Outcome')
plt.ylabel('Match Count')
plt.show()"""),
        markdown_cell("""### 2. Batting Performance Distributions
Analyzing runs scored, balls faced, and strike rates.
"""),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_bat['runs'], kde=True, bins=30, ax=axes[0], color='navy')
axes[0].set_title('Distribution of Batsman Runs')

sns.histplot(df_bat['strike_rate'], kde=True, bins=30, ax=axes[1], color='darkgreen')
axes[1].set_title('Distribution of Batsman Strike Rates')
plt.tight_layout()
plt.show()"""),
        markdown_cell("""### 3. Bowling Performance Distributions
Analyzing economy rates and wickets taken per match.
"""),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_bowl['economy'], kde=True, bins=25, ax=axes[0], color='crimson')
axes[0].set_title('Distribution of Bowler Economy Rates')

sns.countplot(data=df_bowl, x='wickets', ax=axes[1], palette='Blues_r')
axes[1].set_title('Wickets Taken per Spell')
plt.tight_layout()
plt.show()""")
    ]
    with open(os.path.join(output_dir, "01_EDA.ipynb"), "w") as f:
        json.dump(make_notebook(nb01_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 02: Preprocessing
    # ----------------------------------------------------
    nb02_cells = [
        markdown_cell("""# 02. Data Preprocessing & Leakage Prevention
## American Express Decision Science Challenge

### Objective:
- Enforce strict time-based train/val/test splitting to prevent future information leakage.
- Clean missing values and format chronological indices.
"""),
        code_cell("""import sys
sys.path.append('..')
from src.preprocessing import load_raw_data, create_time_based_split

df_m, df_bat, df_bowl = load_raw_data('../data/raw')
train_m, val_m, test_m = create_time_based_split(df_m, train_end_season=2020, val_end_season=2022)"""),
        markdown_cell("""### Chronological Integrity Verification
Verify that test matches occur strictly after training and validation matches.
"""),
        code_cell("""print("Max Training Date:", train_m['date'].max())
print("Min Validation Date:", val_m['date'].min())
print("Max Validation Date:", val_m['date'].max())
print("Min Test Date:", test_m['date'].min())
assert train_m['date'].max() < val_m['date'].min() < test_m['date'].min()
print("Temporal ordering verified successfully!")""")
    ]
    with open(os.path.join(output_dir, "02_Preprocessing.ipynb"), "w") as f:
        json.dump(make_notebook(nb02_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 03: Feature Engineering
    # ----------------------------------------------------
    nb03_cells = [
        markdown_cell("""# 03. Feature Engineering
## American Express Decision Science Challenge

### Objective:
- Compute pre-match available features: toss decisions, venue chasing records, rolling team win rates, head-to-head records, and roster rolling statistics.
"""),
        code_cell("""import pandas as pd
df_features = pd.read_csv('../data/processed/engineered_features.csv')
print("Engineered features shape:", df_features.shape)
df_features.head()"""),
        markdown_cell("""### Feature Correlations
Check correlations of engineered features with the match target.
"""),
        code_cell("""import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df_features.select_dtypes(include=['float64', 'int64']).columns
corrs = df_features[numeric_cols].corr()['team1_win'].sort_values()

plt.figure(figsize=(8, 6))
corrs.plot(kind='barh', color='teal')
plt.title('Feature Correlation with Target (team1_win)')
plt.xlabel('Correlation Coefficient')
plt.show()""")
    ]
    with open(os.path.join(output_dir, "03_Feature_Engineering.ipynb"), "w") as f:
        json.dump(make_notebook(nb03_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 04: HMM Model
    # ----------------------------------------------------
    nb04_cells = [
        markdown_cell("""# 04. Hidden Markov Model (HMM) for Sequential Latent Form
## American Express Decision Science Challenge

### Mathematical Concepts:
- **Hidden State $S_t \in \{0: \text{Slump}, 1: \text{Average}, 2: \text{Peak Form}\}$**
- **Transition Probabilities:** $A_{ij} = P(S_t = j \mid S_{t-1} = i)$
- **Continuous Gaussian Emissions:** $o_t \sim \mathcal{N}(\mu_k, \Sigma_k)$
- **Causal Inference:** Forward belief $\gamma_{t-1}$ projected via $\pi_t = A^\top \gamma_{t-1}$.
"""),
        code_cell("""import joblib
import numpy as np
import pandas as pd

bat_hmm = joblib.load('../models/hmm_batting.pkl')
bowl_hmm = joblib.load('../models/hmm_bowling.pkl')

print("Batting HMM Feature Columns:", bat_hmm.feature_cols)
print("Batting HMM Transition Matrix:\n", np.round(bat_hmm.model.transmat_, 3))
print("Bowling HMM Feature Columns:", bowl_hmm.feature_cols)
print("Bowling HMM Transition Matrix:\n", np.round(bowl_hmm.model.transmat_, 3))"""),
        markdown_cell("""### Inspect Learned State Emission Means
Examine the physical characteristics of each discovered state.
"""),
        code_cell("""for idx, internal_state in enumerate(bat_hmm.state_order):
    state_names = ["Slump", "Average", "Peak"]
    print(f"Batting State {state_names[idx]}: Runs={bat_hmm.model.means_[internal_state, 0]:.1f}, SR={bat_hmm.model.means_[internal_state, 1]:.1f}")
""")
    ]
    with open(os.path.join(output_dir, "04_HMM_Model.ipynb"), "w") as f:
        json.dump(make_notebook(nb04_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 05: LightGBM Model
    # ----------------------------------------------------
    nb05_cells = [
        markdown_cell("""# 05. LightGBM Modeling (Baseline vs. Hybrid HMM)
## American Express Decision Science Challenge

### Objective:
- Train and compare Baseline LightGBM (without HMM) and Hybrid HMM-LightGBM (with latent form).
- Optimize hyperparameters via GridSearchCV with TimeSeriesSplit.
"""),
        code_cell("""import joblib
import pandas as pd

base_lgbm = joblib.load('../models/baseline_lightgbm.pkl')
hybrid_lgbm = joblib.load('../models/hybrid_lightgbm.pkl')

print("Baseline LightGBM Params:", base_lgbm.get_params())
print("Hybrid LightGBM Params:", hybrid_lgbm.get_params())""")
    ]
    with open(os.path.join(output_dir, "05_LightGBM_Model.ipynb"), "w") as f:
        json.dump(make_notebook(nb05_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 06: XGBoost Model
    # ----------------------------------------------------
    nb06_cells = [
        markdown_cell("""# 06. XGBoost Benchmarking
## American Express Decision Science Challenge

### Objective:
- Benchmark an independent gradient boosted decision tree (XGBoost) on the hybrid feature space.
"""),
        code_cell("""import joblib
xgb_model = joblib.load('../models/benchmark_xgboost.pkl')
print("Tuned XGBoost Parameters:", xgb_model.get_params())""")
    ]
    with open(os.path.join(output_dir, "06_XGBoost_Model.ipynb"), "w") as f:
        json.dump(make_notebook(nb06_cells), f, indent=2)

    # ----------------------------------------------------
    # Notebook 07: SHAP Explainability
    # ----------------------------------------------------
    nb07_cells = [
        markdown_cell("""# 07. Model Explainability using SHAP
## American Express Decision Science Challenge

### Objective:
- Compute exact Shapley values using TreeExplainer.
- Visualize global feature ranking, bee-swarm distributions, and local prediction waterfalls.
"""),
        code_cell("""import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

shap_data = joblib.load('../visualizations/shap_data.pkl')
sv = shap_data['shap_values']
feat_names = shap_data['feature_names']
X_test = shap_data['X_test']

plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test, feature_names=feat_names, plot_type="bar")
plt.show()""")
    ]
    with open(os.path.join(output_dir, "07_SHAP_Explainability.ipynb"), "w") as f:
        json.dump(make_notebook(nb07_cells), f, indent=2)

    print("All 7 Jupyter Notebooks generated successfully in notebooks/!")


if __name__ == "__main__":
    create_all_notebooks()
