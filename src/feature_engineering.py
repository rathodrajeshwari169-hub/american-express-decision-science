"""
Feature Engineering Module for American Express Decision Science Challenge.
Extracts context, team momentum, head-to-head, and roster performance statistics
prior to match commencement (strictly pre-match availability).
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from src.preprocessing import load_raw_data


class PreMatchFeatureExtractor:
    """
    Extracts rolling historical features at match time with strict zero-leakage causal guards.
    """
    def __init__(self, df_matches: pd.DataFrame, df_batting: pd.DataFrame, df_bowling: pd.DataFrame):
        self.df_matches = df_matches.sort_values("date").reset_index(drop=True)
        self.df_batting = df_batting.sort_values("date").reset_index(drop=True)
        self.df_bowling = df_bowling.sort_values("date").reset_index(drop=True)

    def extract_features_for_all_matches(self) -> pd.DataFrame:
        """
        Iterates chronologically through matches, computing pre-match feature vectors.
        """
        records = []
        n_matches = len(self.df_matches)
        
        # Pre-index historical lookups for speed
        matches_by_date = self.df_matches.copy()
        
        for idx, row in self.df_matches.iterrows():
            m_id = row["match_id"]
            m_date = row["date"]
            t1 = row["team1"]
            t2 = row["team2"]
            venue = row["venue"]
            
            # --- Causal History Prior to this match ---
            prior_matches = matches_by_date[matches_by_date["date"] < m_date]
            prior_batting = self.df_batting[self.df_batting["date"] < m_date]
            prior_bowling = self.df_bowling[self.df_bowling["date"] < m_date]
            
            # 1. Context / Toss features
            is_toss_winner_team1 = 1 if row["toss_winner"] == t1 else 0
            toss_decision_bat = 1 if row["toss_decision"] == "bat" else 0
            
            # 2. Venue Features
            venue_prior = prior_matches[prior_matches["venue"] == venue]
            if len(venue_prior) > 0:
                # Chasing win rate at this venue
                # Chasing team won if (toss_decision == 'field' and winner == toss_winner) or (toss_decision == 'bat' and winner != toss_winner)
                chase_wins = (
                    ((venue_prior["toss_decision"] == "field") & (venue_prior["winner"] == venue_prior["toss_winner"])) |
                    ((venue_prior["toss_decision"] == "bat") & (venue_prior["winner"] != venue_prior["toss_winner"]))
                ).sum()
                venue_chase_win_rate = chase_wins / len(venue_prior)
            else:
                venue_chase_win_rate = 0.50
                
            # 3. Head-to-Head History
            h2h_prior = prior_matches[
                ((prior_matches["team1"] == t1) & (prior_matches["team2"] == t2)) |
                ((prior_matches["team1"] == t2) & (prior_matches["team2"] == t1))
            ]
            if len(h2h_prior) > 0:
                h2h_t1_wins = (h2h_prior["winner"] == t1).sum()
                h2h_t1_win_rate = h2h_t1_wins / len(h2h_prior)
            else:
                h2h_t1_win_rate = 0.50
                
            # 4. Team Rolling Form (Last 5 matches for each team)
            t1_prior = prior_matches[(prior_matches["team1"] == t1) | (prior_matches["team2"] == t1)].tail(5)
            t2_prior = prior_matches[(prior_matches["team1"] == t2) | (prior_matches["team2"] == t2)].tail(5)
            
            t1_win_rate_5 = (t1_prior["winner"] == t1).mean() if len(t1_prior) > 0 else 0.50
            t2_win_rate_5 = (t2_prior["winner"] == t2).mean() if len(t2_prior) > 0 else 0.50
            diff_win_rate_5 = t1_win_rate_5 - t2_win_rate_5
            
            # 5. Roster Rolling Performance (Aggregating active player stats over their last 5 innings/spells)
            t1_roster = row["team1_roster"].split(";") if pd.notna(row["team1_roster"]) else []
            t2_roster = row["team2_roster"].split(";") if pd.notna(row["team2_roster"]) else []
            
            t1_bat_stats = self._get_roster_batting_stats(t1_roster, prior_batting)
            t2_bat_stats = self._get_roster_batting_stats(t2_roster, prior_batting)
            t1_bowl_stats = self._get_roster_bowling_stats(t1_roster, prior_bowling)
            t2_bowl_stats = self._get_roster_bowling_stats(t2_roster, prior_bowling)
            
            # Differentials
            diff_bat_avg_runs = t1_bat_stats["avg_runs_last5"] - t2_bat_stats["avg_runs_last5"]
            diff_bat_avg_sr = t1_bat_stats["avg_sr_last5"] - t2_bat_stats["avg_sr_last5"]
            diff_bowl_avg_econ = t2_bowl_stats["avg_econ_last5"] - t1_bowl_stats["avg_econ_last5"] # Positive if Team 1 more economical
            diff_bowl_avg_wkts = t1_bowl_stats["avg_wkts_last5"] - t2_bowl_stats["avg_wkts_last5"]
            
            record = {
                "match_id": m_id,
                "season": row["season"],
                "date": m_date,
                "team1": t1,
                "team2": t2,
                "venue": venue,
                "is_toss_winner_team1": is_toss_winner_team1,
                "toss_decision_bat": toss_decision_bat,
                "venue_chase_win_rate": round(venue_chase_win_rate, 4),
                "h2h_t1_win_rate": round(h2h_t1_win_rate, 4),
                "team1_win_rate_5": round(t1_win_rate_5, 4),
                "team2_win_rate_5": round(t2_win_rate_5, 4),
                "diff_win_rate_5": round(diff_win_rate_5, 4),
                "team1_bat_runs_5": round(t1_bat_stats["avg_runs_last5"], 2),
                "team2_bat_runs_5": round(t2_bat_stats["avg_runs_last5"], 2),
                "diff_bat_avg_runs": round(diff_bat_avg_runs, 2),
                "team1_bat_sr_5": round(t1_bat_stats["avg_sr_last5"], 2),
                "team2_bat_sr_5": round(t2_bat_stats["avg_sr_last5"], 2),
                "diff_bat_avg_sr": round(diff_bat_avg_sr, 2),
                "team1_bowl_econ_5": round(t1_bowl_stats["avg_econ_last5"], 2),
                "team2_bowl_econ_5": round(t2_bowl_stats["avg_econ_last5"], 2),
                "diff_bowl_avg_econ": round(diff_bowl_avg_econ, 2),
                "team1_bowl_wkts_5": round(t1_bowl_stats["avg_wkts_last5"], 2),
                "team2_bowl_wkts_5": round(t2_bowl_stats["avg_wkts_last5"], 2),
                "diff_bowl_avg_wkts": round(diff_bowl_avg_wkts, 2),
                "team1_win": row["team1_win"]
            }
            records.append(record)
            
        df_features = pd.DataFrame(records)
        return df_features

    def _get_roster_batting_stats(self, roster: List[str], prior_batting: pd.DataFrame) -> Dict[str, float]:
        runs_list = []
        sr_list = []
        for p in roster:
            p_inngs = prior_batting[prior_batting["batsman_name"] == p].tail(5)
            if len(p_inngs) > 0:
                runs_list.append(p_inngs["runs"].mean())
                sr_list.append(p_inngs["strike_rate"].mean())
        return {
            "avg_runs_last5": np.mean(runs_list) if len(runs_list) > 0 else 24.0,
            "avg_sr_last5": np.mean(sr_list) if len(sr_list) > 0 else 130.0
        }

    def _get_roster_bowling_stats(self, roster: List[str], prior_bowling: pd.DataFrame) -> Dict[str, float]:
        econ_list = []
        wkts_list = []
        for p in roster:
            p_spells = prior_bowling[prior_bowling["bowler_name"] == p].tail(5)
            if len(p_spells) > 0:
                econ_list.append(p_spells["economy"].mean())
                wkts_list.append(p_spells["wickets"].mean())
        return {
            "avg_econ_last5": np.mean(econ_list) if len(econ_list) > 0 else 8.0,
            "avg_wkts_last5": np.mean(wkts_list) if len(wkts_list) > 0 else 1.10
        }


def extract_and_save_features(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Executes pre-match feature engineering and saves processed tabular features.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_m, df_bat, df_bowl = load_raw_data(raw_dir)
    
    print("Extracting pre-match features across all matches...")
    extractor = PreMatchFeatureExtractor(df_m, df_bat, df_bowl)
    df_features = extractor.extract_features_for_all_matches()
    
    output_path = os.path.join(output_dir, "engineered_features.csv")
    df_features.to_csv(output_path, index=False)
    print(f"Engineered features successfully extracted: {df_features.shape}")
    print(f"Saved to {output_path}")
    return df_features


if __name__ == "__main__":
    extract_and_save_features()
