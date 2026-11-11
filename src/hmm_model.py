"""
Hidden Markov Model (HMM) Module for American Express Decision Science Challenge.

This module models sequential, unobserved player performance regimes (latent form)
using Gaussian Hidden Markov Models (GaussianHMM).

Mathematical Framework:
- States: S_t in {0: Slump, 1: Average, 2: Peak Form}
- Transitions: A_ij = P(S_t = j | S_{t-1} = i)
- Emissions: o_t ~ Normal(mu_k, Sigma_k)
- Inference: Projected pre-match prior pi_t = A^T * gamma_{t-1}
  where gamma_{t-1} is the forward state belief conditioned strictly on history prior to match t.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_raw_data


class PlayerFormHMM:
    """
    Gaussian Hidden Markov Model for sequential sports performance modeling.
    """
    def __init__(self, n_states: int = 3, feature_cols: List[str] = None, random_state: int = 42):
        self.n_states = n_states
        self.feature_cols = feature_cols
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=150,
            random_state=random_state,
            tol=1e-4
        )
        self.is_fitted = False
        self.state_order = None # Maps internal state to [0: Slump, 1: Average, 2: Peak]

    def fit(self, df_scorecards: pd.DataFrame, player_id_col: str, sort_by_feature_idx: int = 0):
        """
        Fits Gaussian HMM across all player sequences in the training set.
        
        Args:
            df_scorecards: Historical player scorecard dataframe.
            player_id_col: Column name identifying players (e.g. 'batsman_name').
            sort_by_feature_idx: Index of feature used to sort states into [Slump, Average, Peak].
        """
        sequences = []
        lengths = []
        
        # Build individual player sequences
        for player, group in df_scorecards.groupby(player_id_col):
            group_sorted = group.sort_values("date")
            feats = group_sorted[self.feature_cols].values
            if len(feats) >= 3: # Need at least 3 innings for sequential transitions
                sequences.append(feats)
                lengths.append(len(feats))
                
        if len(sequences) == 0:
            raise ValueError("No valid sequences found for HMM fitting.")
            
        X = np.concatenate(sequences)
        print(f"Fitting HMM with {self.n_states} states on {len(lengths)} sequences ({len(X)} total observations)...")
        self.model.fit(X, lengths)
        self.is_fitted = True
        
        # Canonical State Reordering:
        # Sort states so that state 0 is Slump, 1 is Average, 2 is Peak
        # sort_by_feature_idx: 0 for runs (ascending), 0 for wickets (ascending)
        means = self.model.means_[:, sort_by_feature_idx]
        self.state_order = np.argsort(means)
        print(f"Learned state means for '{self.feature_cols[sort_by_feature_idx]}': {means}")
        print(f"Ordered State Mapping (Slump -> Average -> Peak): {self.state_order}")
        return self

    def predict_next_state_probs(self, player_history_df: pd.DataFrame) -> np.ndarray:
        """
        Computes projected state probability distribution pi_t = A^T * gamma_{t-1}
        conditioned strictly on observations up to match t-1.
        
        Returns:
            np.ndarray of shape (3,) corresponding to [P(Slump), P(Average), P(Peak)].
        """
        if not self.is_fitted:
            raise RuntimeError("HMM must be fitted before computing state probabilities.")
            
        if len(player_history_df) == 0:
            # Unseen player / cold-start: return stationary prior (startprob)
            raw_probs = self.model.startprob_
        else:
            X_hist = player_history_df[self.feature_cols].values
            try:
                # Forward-backward posterior gamma over past observations
                posteriors = self.model.predict_proba(X_hist)
                last_gamma = posteriors[-1] # State belief after last observed match
                # Project forward one step: pi_t = A^T * gamma_{t-1}
                raw_probs = self.model.transmat_.T @ last_gamma
            except Exception:
                raw_probs = self.model.startprob_
                
        # Remap to canonical [Slump, Average, Peak] order
        canonical_probs = np.zeros(self.n_states)
        for canon_idx, internal_state in enumerate(self.state_order):
            canonical_probs[canon_idx] = raw_probs[internal_state]
            
        # Ensure proper normalization
        canonical_probs = np.clip(canonical_probs, 1e-4, 1.0)
        canonical_probs = canonical_probs / canonical_probs.sum()
        return canonical_probs


def train_and_evaluate_hmms(
    data_dir: str = "data/raw",
    models_dir: str = "models",
    train_end_season: int = 2020
) -> Tuple[PlayerFormHMM, PlayerFormHMM]:
    """
    Fits Batting HMM and Bowling HMM on training seasons (<= 2020) and serializes them.
    """
    os.makedirs(models_dir, exist_ok=True)
    df_matches, df_batting, df_bowling = load_raw_data(data_dir)
    
    # Strictly training split
    train_match_ids = set(df_matches[df_matches["season"] <= train_end_season]["match_id"])
    train_batting = df_batting[df_batting["match_id"].isin(train_match_ids)]
    train_bowling = df_bowling[df_bowling["match_id"].isin(train_match_ids)]
    
    print("\n--- Training Batting HMM ---")
    bat_features = ["runs", "strike_rate", "boundary_rate"]
    bat_hmm = PlayerFormHMM(n_states=3, feature_cols=bat_features, random_state=42)
    bat_hmm.fit(train_batting, player_id_col="batsman_name", sort_by_feature_idx=0) # sort by runs
    
    print("\n--- Training Bowling HMM ---")
    bowl_features = ["wickets", "economy", "dot_rate"]
    bowl_hmm = PlayerFormHMM(n_states=3, feature_cols=bowl_features, random_state=42)
    bowl_hmm.fit(train_bowling, player_id_col="bowler_name", sort_by_feature_idx=0) # sort by wickets
    
    # Save models
    bat_path = os.path.join(models_dir, "hmm_batting.pkl")
    bowl_path = os.path.join(models_dir, "hmm_bowling.pkl")
    joblib.dump(bat_hmm, bat_path)
    joblib.dump(bowl_hmm, bowl_path)
    print(f"\nSerialized HMM models to {models_dir}/")
    
    return bat_hmm, bowl_hmm


def generate_hybrid_features(
    features_csv: str = "data/processed/engineered_features.csv",
    data_dir: str = "data/raw",
    models_dir: str = "models",
    output_path: str = "data/processed/hybrid_dataset.csv"
) -> pd.DataFrame:
    """
    Computes HMM latent state features for every match and joins with engineered features.
    """
    df_features = pd.read_csv(features_csv)
    df_matches, df_batting, df_bowling = load_raw_data(data_dir)
    
    bat_hmm = joblib.load(os.path.join(models_dir, "hmm_batting.pkl"))
    bowl_hmm = joblib.load(os.path.join(models_dir, "hmm_bowling.pkl"))
    
    print(f"Generating HMM Latent Form vectors for {len(df_features)} matches...")
    
    hmm_records = []
    
    for idx, row in df_features.iterrows():
        m_id = row["match_id"]
        m_date = pd.to_datetime(row["date"])
        
        # Match roster from matches table
        match_info = df_matches[df_matches["match_id"] == m_id].iloc[0]
        t1_roster = match_info["team1_roster"].split(";") if pd.notna(match_info["team1_roster"]) else []
        t2_roster = match_info["team2_roster"].split(";") if pd.notna(match_info["team2_roster"]) else []
        
        # Causal player histories prior to match date
        prior_batting = df_batting[df_batting["date"] < m_date]
        prior_bowling = df_bowling[df_bowling["date"] < m_date]
        
        # Batting latent state aggregation for Team 1 and Team 2
        t1_bat_probs = []
        for p in t1_roster:
            p_hist = prior_batting[prior_batting["batsman_name"] == p].sort_values("date")
            t1_bat_probs.append(bat_hmm.predict_next_state_probs(p_hist))
            
        t2_bat_probs = []
        for p in t2_roster:
            p_hist = prior_batting[prior_batting["batsman_name"] == p].sort_values("date")
            t2_bat_probs.append(bat_hmm.predict_next_state_probs(p_hist))
            
        # Bowling latent state aggregation for Team 1 and Team 2
        t1_bowl_probs = []
        for p in t1_roster:
            p_hist = prior_bowling[prior_bowling["bowler_name"] == p].sort_values("date")
            t1_bowl_probs.append(bowl_hmm.predict_next_state_probs(p_hist))
            
        t2_bowl_probs = []
        for p in t2_roster:
            p_hist = prior_bowling[prior_bowling["bowler_name"] == p].sort_values("date")
            t2_bowl_probs.append(bowl_hmm.predict_next_state_probs(p_hist))
            
        t1_bat_mean = np.mean(t1_bat_probs, axis=0) if len(t1_bat_probs) > 0 else np.array([0.33, 0.34, 0.33])
        t2_bat_mean = np.mean(t2_bat_probs, axis=0) if len(t2_bat_probs) > 0 else np.array([0.33, 0.34, 0.33])
        t1_bowl_mean = np.mean(t1_bowl_probs, axis=0) if len(t1_bowl_probs) > 0 else np.array([0.33, 0.34, 0.33])
        t2_bowl_mean = np.mean(t2_bowl_probs, axis=0) if len(t2_bowl_probs) > 0 else np.array([0.33, 0.34, 0.33])
        
        # Extract specific state probabilities: 0: Slump, 1: Average, 2: Peak
        t1_bat_slump = t1_bat_mean[0]
        t1_bat_peak = t1_bat_mean[2]
        t2_bat_slump = t2_bat_mean[0]
        t2_bat_peak = t2_bat_mean[2]
        
        t1_bowl_slump = t1_bowl_mean[0]
        t1_bowl_peak = t1_bowl_mean[2]
        t2_bowl_slump = t2_bowl_mean[0]
        t2_bowl_peak = t2_bowl_mean[2]
        
        # Compute Latent Differentials
        diff_bat_peak = t1_bat_peak - t2_bat_peak
        diff_bat_slump = t2_bat_slump - t1_bat_slump # Positive if Team 2 is more in slump
        diff_bowl_peak = t1_bowl_peak - t2_bowl_peak
        diff_bowl_slump = t2_bowl_slump - t1_bowl_slump # Positive if Team 2 bowling is more in slump
        
        # Composite Latent Form Index
        net_latent_form_index = (diff_bat_peak + diff_bat_slump + diff_bowl_peak + diff_bowl_slump) / 4.0
        
        hmm_records.append({
            "match_id": m_id,
            "team1_bat_peak_prob": round(float(t1_bat_peak), 4),
            "team1_bat_slump_prob": round(float(t1_bat_slump), 4),
            "team2_bat_peak_prob": round(float(t2_bat_peak), 4),
            "team2_bat_slump_prob": round(float(t2_bat_slump), 4),
            "team1_bowl_peak_prob": round(float(t1_bowl_peak), 4),
            "team1_bowl_slump_prob": round(float(t1_bowl_slump), 4),
            "team2_bowl_peak_prob": round(float(t2_bowl_peak), 4),
            "team2_bowl_slump_prob": round(float(t2_bowl_slump), 4),
            "diff_bat_peak_prob": round(float(diff_bat_peak), 4),
            "diff_bat_slump_prob": round(float(diff_bat_slump), 4),
            "diff_bowl_peak_prob": round(float(diff_bowl_peak), 4),
            "diff_bowl_slump_prob": round(float(diff_bowl_slump), 4),
            "net_latent_form_index": round(float(net_latent_form_index), 4)
        })
        
    df_hmm = pd.DataFrame(hmm_records)
    df_hybrid = pd.merge(df_features, df_hmm, on="match_id")
    df_hybrid.to_csv(output_path, index=False)
    print(f"Hybrid dataset generated successfully: {df_hybrid.shape}")
    print(f"Saved to {output_path}")
    return df_hybrid


if __name__ == "__main__":
    train_and_evaluate_hmms()
    generate_hybrid_features()
