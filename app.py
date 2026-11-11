"""
American Express Decision Science Challenge: Hybrid HMM + LightGBM for Sports Prediction
Interactive Streamlit Application.
"""

import os
import sys
import json
import ctypes
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Safe macOS OpenMP library preloading
if sys.platform == "darwin":
    try:
        sklearn_path = os.path.abspath(".venv/lib/python3.14/site-packages/sklearn/.dylibs/libomp.dylib")
        if os.path.exists(sklearn_path):
            ctypes.CDLL(sklearn_path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

# Page Configuration
st.set_page_config(
    page_title="AmEx Decision Science: Hybrid HMM + LightGBM",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #002663, #0070d2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #002663;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Data & Model Loaders (Cached for performance)
@st.cache_data
def load_datasets():
    df_matches = pd.read_csv("data/raw/matches.csv")
    df_batting = pd.read_csv("data/raw/batsman_level_scorecard.csv")
    df_bowling = pd.read_csv("data/raw/bowler_level_scorecard.csv")
    df_hybrid = pd.read_csv("data/processed/hybrid_dataset.csv")
    df_metrics = pd.read_csv("visualizations/metrics_comparison.csv")
    return df_matches, df_batting, df_bowling, df_hybrid, df_metrics


@st.cache_resource
def load_models():
    base_lgbm = joblib.load("models/baseline_lightgbm.pkl")
    hybrid_lgbm = joblib.load("models/hybrid_lightgbm.pkl")
    benchmark_xgb = joblib.load("models/benchmark_xgboost.pkl")
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
    shap_data = joblib.load("visualizations/shap_data.pkl")
    return base_lgbm, hybrid_lgbm, benchmark_xgb, meta, shap_data


# Load assets
try:
    df_matches, df_batting, df_bowling, df_hybrid, df_metrics = load_datasets()
    base_lgbm, hybrid_lgbm, benchmark_xgb, meta, shap_data = load_models()
    base_features = meta["base_features"]
    hybrid_features = meta["hybrid_features"]
except Exception as e:
    st.error(f"Error loading models or dataset artifacts: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/American_Express_logo_%282018%29.svg", width=180)
st.sidebar.markdown("### Decision Science Challenge")
st.sidebar.markdown("**Track:** Sports Prediction (T20)")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "📊 Data Analysis (EDA)",
        "⚔️ Model Comparison Arena",
        "🎯 Match Predictor Simulator",
        "🔍 Model Explainability (SHAP)",
        "📘 About Project & Interview Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Portfolio Project: End-to-End Decision Science Pipeline")


# ==============================================================================
# Page 1: Home Dashboard
# ==============================================================================
if nav_choice == "🏠 Home Dashboard":
    st.markdown('<p class="main-header">American Express Decision Science Challenge</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Hidden Markov Model (HMM) + LightGBM for Sequential Sports Analytics</p>', unsafe_allow_html=True)
    
    # KPI metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", f"{len(df_matches):,}", "16 Seasons (2008–2023)")
    with col2:
        st.metric("Player Scorecards", f"{len(df_batting) + len(df_bowling):,}", "Batting & Bowling Spells")
    with col3:
        st.metric("HMM Hidden Regimes", "3 States", "Slump | Normal | Peak")
    with col4:
        st.metric("Benchmark Models", "3 Architectures", "LightGBM vs Hybrid vs XGBoost")
        
    st.markdown("---")
    
    st.subheader("📌 Project Motivation & Industry Relevance")
    st.markdown("""
    In high-stakes decision systems (such as financial credit authorization or live sports wagering odds calibration), **past performance is not static**. 
    
    Standard machine learning algorithms typically rely on **lagged static rolling averages** (e.g., 5-game moving averages). However, rolling averages:
    1. Suffer from high temporal delay.
    2. Treat fluky outliers identically to sustainable form.
    3. Fail to capture **unobservable psychological and physiological states** (confidence, rhythm, fatigue).
    
    **Our Solution:** A **Hybrid HMM + LightGBM architecture**:
    - **Hidden Markov Models (HMM):** Infer latent performance states (*Slump*, *Consolidating*, *Peak Form*) from multi-dimensional sequence logs without lookahead leakage.
    - **LightGBM Gradient Boosting:** Combines these dynamic state priors with contextual features (toss biases, venue records, head-to-head dynamics) to generate calibrated win probability estimates.
    """)
    
    st.subheader("🏗️ Architectural Pipeline")
    st.info("""
    **Pipeline Workflow:**
    `Raw Scorecards` ➡️ `Causal Preprocessing` ➡️ `Gaussian HMM (Batting & Bowling)` ➡️ `State Posterior Projection (A^T * gamma)` ➡️ `Roster Form Aggregation` ➡️ `LightGBM & XGBoost with GridSearchCV` ➡️ `SHAP TreeExplainer`
    """)


# ==============================================================================
# Page 2: Data Analysis (EDA)
# ==============================================================================
elif nav_choice == "📊 Data Analysis (EDA)":
    st.markdown('<p class="main-header">Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Inspect historical distributions, target balance, and scorecards.</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Match Level Analytics", "Batsman Scorecards", "Bowler Scorecards"])
    
    with tab1:
        st.subheader("Match Outcome & Class Distribution")
        colA, colB = st.columns([1, 2])
        with colA:
            win_counts = df_matches["winner"].value_counts()
            t1_wins = (df_matches["winner"] == df_matches["team1"]).sum()
            t2_wins = len(df_matches) - t1_wins
            st.write("**Target Variable Balance:**")
            st.dataframe(pd.DataFrame({
                "Outcome": ["Team 1 Win (Bat 1st)", "Team 2 Win (Field 1st)"],
                "Count": [t1_wins, t2_wins],
                "Percentage": [f"{(t1_wins/len(df_matches))*100:.1f}%", f"{(t2_wins/len(df_matches))*100:.1f}%"]
            }))
        with colB:
            st.bar_chart(win_counts.head(10))
            st.caption("Top 10 Most Successful Franchises (Total Historical Wins)")
            
        st.markdown("---")
        st.subheader("Venue Chasing Advantage")
        venue_stats = df_hybrid.groupby("venue")["venue_chase_win_rate"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(venue_stats, width=800)
        
    with tab2:
        st.subheader("Batsman Performance Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Top Batsmen by Career Runs:**")
            top_batsmen = df_batting.groupby("batsman_name").agg(
                innings=("match_id", "count"),
                total_runs=("runs", "sum"),
                avg_sr=("strike_rate", "mean")
            ).sort_values("total_runs", ascending=False).head(10)
            st.dataframe(top_batsmen)
        with c2:
            st.write("**Distribution of Batsman Runs:**")
            hist_values, bin_edges = np.histogram(df_batting["runs"], bins=20)
            st.bar_chart(pd.DataFrame({"Frequency": hist_values}, index=np.round(bin_edges[:-1], 0)))
            
    with tab3:
        st.subheader("Bowler Performance Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Top Wicket Takers:**")
            top_bowlers = df_bowling.groupby("bowler_name").agg(
                spells=("match_id", "count"),
                total_wickets=("wickets", "sum"),
                avg_economy=("economy", "mean")
            ).sort_values("total_wickets", ascending=False).head(10)
            st.dataframe(top_bowlers)
        with c2:
            st.write("**Bowler Economy Distribution:**")
            econ_hist, econ_bins = np.histogram(df_bowling["economy"], bins=20)
            st.bar_chart(pd.DataFrame({"Spells": econ_hist}, index=np.round(econ_bins[:-1], 1)))


# ==============================================================================
# Page 3: Model Comparison Arena
# ==============================================================================
elif nav_choice == "⚔️ Model Comparison Arena":
    st.markdown('<p class="main-header">Model Arena: Empirical Benchmark</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluating Baseline LightGBM vs. Hybrid HMM-LightGBM vs. Benchmark XGBoost on Held-out Out-of-Time Test Data (Season 2023).</p>', unsafe_allow_html=True)
    
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#d4edda")
                           .highlight_min(axis=0, subset=["Log Loss", "Brier Score"], color="#d4edda"), width=900)
                           
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        if os.path.exists("visualizations/roc_curves.png"):
            st.image("visualizations/roc_curves.png", use_container_width=True)
    with col2:
        st.subheader("Precision-Recall Curves")
        if os.path.exists("visualizations/precision_recall_curves.png"):
            st.image("visualizations/precision_recall_curves.png", use_container_width=True)
            
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Confusion Matrices (Test Set)")
        if os.path.exists("visualizations/confusion_matrices.png"):
            st.image("visualizations/confusion_matrices.png", use_container_width=True)
    with col4:
        st.subheader("Probability Calibration (Reliability Diagram)")
        if os.path.exists("visualizations/calibration_curves.png"):
            st.image("visualizations/calibration_curves.png", use_container_width=True)
            
    st.info("""
    **Analytical Takeaways for Placement Interviews:**
    1. **Strict Zero-Leakage Out-of-Time Protocol:** Unlike synthetic CV splits that shuffle matches across seasons, this benchmark evaluates strictly on future matches (2023 season), mimicking real-world production deployment.
    2. **Log Loss & Probability Calibration:** In sports wagering or odds-making, Log Loss and Brier Score are vastly more critical than raw accuracy because overconfident wrong predictions carry massive penalties.
    """)


# ==============================================================================
# Page 4: Match Predictor Simulator
# ==============================================================================
elif nav_choice == "🎯 Match Predictor Simulator":
    st.markdown('<p class="main-header">Interactive Match Simulator & Probability Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulate any upcoming match fixture with customizable team form, venue, and toss dynamics.</p>', unsafe_allow_html=True)
    
    teams = sorted(df_matches["team1"].unique())
    venues = sorted(df_matches["venue"].unique())
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Match Setup")
        team1 = st.selectbox("Team 1 (Batting First / Home)", teams, index=0)
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2 (Fielding First / Away)", team2_options, index=0)
        venue = st.selectbox("Venue / Stadium", venues, index=0)
        
    with colB:
        st.subheader("Toss & Historical Momentum")
        toss_winner = st.radio("Toss Winner", [team1, team2], horizontal=True)
        toss_decision = st.radio("Toss Decision", ["field", "bat"], horizontal=True)
        
        st.markdown("**Latent Form Scenario Preset:**")
        scenario = st.selectbox(
            "Select Team Latent Form Dynamic",
            ["Balanced Contest (Normal Form)", "Team 1 In-Form Surge", "Team 2 Momentum Surge", "Team 1 Slumping / Team 2 Peaking"]
        )
        
    # Scenario parameter adjustments
    if scenario == "Balanced Contest (Normal Form)":
        t1_win_rate_5 = 0.60
        t2_win_rate_5 = 0.60
        net_latent_form = 0.00
    elif scenario == "Team 1 In-Form Surge":
        t1_win_rate_5 = 0.80
        t2_win_rate_5 = 0.40
        net_latent_form = 0.12
    elif scenario == "Team 2 Momentum Surge":
        t1_win_rate_5 = 0.40
        t2_win_rate_5 = 0.80
        net_latent_form = -0.12
    else:
        t1_win_rate_5 = 0.20
        t2_win_rate_5 = 0.80
        net_latent_form = -0.22

    st.markdown("---")
    
    if st.button("🚀 Calculate Win Probability & Odds", type="primary"):
        # Construct feature vector
        is_toss_winner_t1 = 1 if toss_winner == team1 else 0
        toss_dec_bat = 1 if toss_decision == "bat" else 0
        diff_win_rate_5 = t1_win_rate_5 - t2_win_rate_5
        
        # Build synthetic feature row matching hybrid schema
        input_dict = {
            "is_toss_winner_team1": is_toss_winner_t1,
            "toss_decision_bat": toss_dec_bat,
            "venue_chase_win_rate": 0.52,
            "h2h_t1_win_rate": 0.55,
            "team1_win_rate_5": t1_win_rate_5,
            "team2_win_rate_5": t2_win_rate_5,
            "diff_win_rate_5": diff_win_rate_5,
            "team1_bat_runs_5": 28.5 + (net_latent_form * 10),
            "team2_bat_runs_5": 26.0 - (net_latent_form * 10),
            "diff_bat_avg_runs": 2.5 + (net_latent_form * 20),
            "team1_bat_sr_5": 138.0 + (net_latent_form * 15),
            "team2_bat_sr_5": 134.0 - (net_latent_form * 15),
            "diff_bat_avg_sr": 4.0 + (net_latent_form * 30),
            "team1_bowl_econ_5": 7.8 - (net_latent_form * 2),
            "team2_bowl_econ_5": 8.1 + (net_latent_form * 2),
            "diff_bowl_avg_econ": 0.3 + (net_latent_form * 4),
            "team1_bowl_wkts_5": 1.35 + (net_latent_form * 0.4),
            "team2_bowl_wkts_5": 1.20 - (net_latent_form * 0.4),
            "diff_bowl_avg_wkts": 0.15 + (net_latent_form * 0.8),
            "team1_bat_peak_prob": np.clip(0.33 + net_latent_form, 0.05, 0.90),
            "team1_bat_slump_prob": np.clip(0.25 - net_latent_form, 0.05, 0.90),
            "team2_bat_peak_prob": np.clip(0.33 - net_latent_form, 0.05, 0.90),
            "team2_bat_slump_prob": np.clip(0.25 + net_latent_form, 0.05, 0.90),
            "team1_bowl_peak_prob": np.clip(0.35 + net_latent_form, 0.05, 0.90),
            "team1_bowl_slump_prob": np.clip(0.20 - net_latent_form, 0.05, 0.90),
            "team2_bowl_peak_prob": np.clip(0.35 - net_latent_form, 0.05, 0.90),
            "team2_bowl_slump_prob": np.clip(0.20 + net_latent_form, 0.05, 0.90),
            "diff_bat_peak_prob": 2 * net_latent_form,
            "diff_bat_slump_prob": 2 * net_latent_form,
            "diff_bowl_peak_prob": 2 * net_latent_form,
            "diff_bowl_slump_prob": 2 * net_latent_form,
            "net_latent_form_index": net_latent_form
        }
        
        X_in_base = np.array([[input_dict[c] for c in base_features]])
        X_in_hybrid = np.array([[input_dict[c] for c in hybrid_features]])
        
        prob_base = base_lgbm.predict_proba(X_in_base)[0, 1]
        prob_hybrid = hybrid_lgbm.predict_proba(X_in_hybrid)[0, 1]
        prob_xgb = benchmark_xgb.predict_proba(X_in_hybrid)[0, 1]
        
        st.subheader("🏁 Calibrated Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(f"Hybrid HMM-LightGBM ({team1})", f"{prob_hybrid*100:.1f}%", f"{team2}: {(1-prob_hybrid)*100:.1f}%")
        with res_col2:
            st.metric(f"Baseline LightGBM ({team1})", f"{prob_base*100:.1f}%", f"{team2}: {(1-prob_base)*100:.1f}%")
        with res_col3:
            st.metric(f"Benchmark XGBoost ({team1})", f"{prob_xgb*100:.1f}%", f"{team2}: {(1-prob_xgb)*100:.1f}%")
            
        # Win Probability Bar
        st.write(f"**Visual Win Probability Gauge ({team1} vs. {team2}):**")
        st.progress(float(prob_hybrid), text=f"{team1}: {prob_hybrid*100:.1f}%  |  {team2}: {(1-prob_hybrid)*100:.1f}%")


# ==============================================================================
# Page 5: Model Explainability (SHAP)
# ==============================================================================
elif nav_choice == "🔍 Model Explainability (SHAP)":
    st.markdown('<p class="main-header">Model Explainability via SHAP (Shapley Additive exPlanations)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand global feature importance and granular match-level decision drivers.</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Global Feature Importance", "Individual Match Explanations"])
    
    with tab1:
        st.subheader("Global Feature Importance (TreeExplainer)")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("visualizations/shap_summary_bar.png"):
                st.image("visualizations/shap_summary_bar.png", caption="Mean |SHAP Value| across Test Matches", use_container_width=True)
        with col2:
            if os.path.exists("visualizations/shap_beeswarm.png"):
                st.image("visualizations/shap_beeswarm.png", caption="SHAP Beeswarm: Impact of Feature Value on Win Probability", use_container_width=True)
                
        st.markdown("---")
        st.subheader("SHAP Dependence on Net Latent Form Index")
        if os.path.exists("visualizations/shap_dependence_net_form.png"):
            st.image("visualizations/shap_dependence_net_form.png", width=650)
            
    with tab2:
        st.subheader("Local Match Explainability")
        test_matches = shap_data["test_metadata"]
        match_options = [f"Match {m['match_id']}: {m['team1']} vs {m['team2']} (Winner: {m['winner']})" for m in test_matches]
        selected_idx = st.selectbox("Select a Test Match to Inspect Local SHAP Attributions:", range(len(match_options)), format_func=lambda i: match_options[i])
        
        selected_meta = test_matches[selected_idx]
        local_shap = shap_data["shap_values"][selected_idx]
        feat_names = shap_data["feature_names"]
        feat_vals = shap_data["X_test"][selected_idx]
        
        st.write(f"**Match Overview:** {selected_meta['team1']} vs {selected_meta['team2']} | **Actual Winner:** `{selected_meta['winner']}`")
        
        # Top 5 positive drivers and top 5 negative drivers
        df_local = pd.DataFrame({
            "Feature": feat_names,
            "Feature Value": feat_vals,
            "SHAP Value": local_shap
        }).sort_values("SHAP Value", ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Top Features Boosting {selected_meta['team1']}'s Win Odds:**")
            st.dataframe(df_local[df_local["SHAP Value"] > 0].head(5), width=450)
        with c2:
            st.error(f"**Top Features Reducing {selected_meta['team1']}'s Win Odds:**")
            st.dataframe(df_local[df_local["SHAP Value"] < 0].tail(5).iloc[::-1], width=450)


# ==============================================================================
# Page 6: About Project & Interview Guide
# ==============================================================================
elif nav_choice == "📘 About Project & Interview Guide":
    st.markdown('<p class="main-header">Placement Interview Guide & Methodology Playbook</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Key technical answers and conceptual frameworks for Data Science & ML interviews.</p>', unsafe_allow_html=True)
    
    with st.expander("1. Why use a Hidden Markov Model (HMM) instead of just rolling averages?", expanded=True):
        st.markdown(r"""
        **The Interview Answer:**
        - A rolling average (e.g. last 5 games) is a simple moving average of *observed emissions*. It suffers from lag, treats every run/wicket equally, and cannot model the **regime-switching behavior** inherent in sports form.
        - An HMM assumes an underlying, unobservable Markovian latent state $S_t \in \{\text{Slump}, \text{Average}, \text{Peak Form}\}$.
        - When a player scores 85 off 40 balls, the HMM updates the forward state posterior $\gamma_t = P(S_t \mid \mathbf{o}_{1:t})$.
        - Multiplying by the learned transition matrix $\mathbf{A}$, we project the causal pre-match prior $\boldsymbol{\pi}_{t+1} = \mathbf{A}^\top \boldsymbol{\gamma}_t$, giving LightGBM a state-aware probabilistic prior rather than just a noisy lagged number.
        """)
        
    with st.expander("2. How did you ensure ZERO data leakage in the pipeline?", expanded=True):
        st.markdown("""
        **The Interview Answer:**
        - **Temporal Separation:** We split data strictly chronologically (Train: 2008–2020, Val: 2021–2022, Test: 2023). Random K-Fold CV was strictly forbidden because future matches would leak into past predictions.
        - **Pre-Match Availability Constraint:** For match $m$ on date $t_m$, feature engineering only aggregates scorecards with $\tau < t_m$. Match $m$'s own scores or wickets are completely masked.
        - **HMM Causal Filtering:** HMM forward probabilities are computed using observations strictly prior to $t_m$.
        """)
        
    with st.expander("3. Why benchmark LightGBM against XGBoost?", expanded=True):
        st.markdown("""
        **The Interview Answer:**
        - LightGBM utilizes **Histogram-based split finding** and **Leaf-wise (best-first) tree growth**, making it significantly faster and memory-efficient on tabular data.
        - XGBoost utilizes **Exact / Approx greedy splits** and **Level-wise (depth-first) growth**, which can provide different regularization characteristics on noisy sports data.
        - Benchmarking against both verifies whether the HMM latent form lift is model-agnostic or tied to a single gradient boosting implementation.
        """)

    with st.expander("4. Why evaluate on Log Loss and Brier Score rather than just Accuracy?", expanded=True):
        st.markdown(r"""
        **The Interview Answer:**
        - Sports predictions are fundamentally probabilistic—no model can guarantee a 100% win certainty in T20 cricket.
        - **Accuracy** only evaluates whether $\hat{p} \ge 0.50$, discarding whether the model was 51% confident or 99% confident.
        - **Log Loss** penalizes overconfident wrong predictions exponentially:
          $$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
        - **Brier Score** measures mean squared calibration error, essential for risk management and betting odds calibration.
        """)
