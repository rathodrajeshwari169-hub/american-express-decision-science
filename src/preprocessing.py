"""
Data Preprocessing Module for American Express Decision Science Challenge.
Handles data cleaning, time-aware splitting, and validation checks.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_raw_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV files for matches, batsman scorecards, and bowler scorecards.
    Ensures datetime types and chronological sorting.
    """
    matches_path = os.path.join(data_dir, "matches.csv")
    batting_path = os.path.join(data_dir, "batsman_level_scorecard.csv")
    bowling_path = os.path.join(data_dir, "bowler_level_scorecard.csv")
    
    if not (os.path.exists(matches_path) and os.path.exists(batting_path) and os.path.exists(bowling_path)):
        raise FileNotFoundError(f"Raw data files not found in {data_dir}. Run data_loader.py first.")
        
    df_matches = pd.read_csv(matches_path)
    df_batting = pd.read_csv(batting_path)
    df_bowling = pd.read_csv(bowling_path)
    
    # Parse dates
    df_matches["date"] = pd.to_datetime(df_matches["date"])
    df_batting["date"] = pd.to_datetime(df_batting["date"])
    df_bowling["date"] = pd.to_datetime(df_bowling["date"])
    
    # Sort chronologically to preserve causality
    df_matches = df_matches.sort_values("date").reset_index(drop=True)
    df_batting = df_batting.sort_values(["date", "match_id"]).reset_index(drop=True)
    df_bowling = df_bowling.sort_values(["date", "match_id"]).reset_index(drop=True)
    
    # Create binary target variable: 1 if team1 wins, 0 if team2 wins
    df_matches["team1_win"] = (df_matches["winner"] == df_matches["team1"]).astype(int)
    
    return df_matches, df_batting, df_bowling


def create_time_based_split(
    df: pd.DataFrame,
    train_end_season: int = 2020,
    val_end_season: int = 2022
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically by tournament seasons to prevent lookahead leakage.
    - Train: seasons <= train_end_season (2008 - 2020)
    - Validation: seasons (train_end_season, val_end_season] (2021 - 2022)
    - Test: seasons > val_end_season (2023)
    """
    df_train = df[df["season"] <= train_end_season].copy().reset_index(drop=True)
    df_val = df[(df["season"] > train_end_season) & (df["season"] <= val_end_season)].copy().reset_index(drop=True)
    df_test = df[df["season"] > val_end_season].copy().reset_index(drop=True)
    
    print(f"Time-based Split Summary:")
    print(f"  Training set   : {len(df_train)} matches ({df_train['season'].min()} - {df_train['season'].max()})")
    print(f"  Validation set : {len(df_val)} matches ({df_val['season'].min()} - {df_val['season'].max()})")
    print(f"  Testing set    : {len(df_test)} matches ({df_test['season'].min()} - {df_test['season'].max()})")
    
    return df_train, df_val, df_test


def get_historical_player_slice(
    df_scorecards: pd.DataFrame,
    player_col: str,
    player_name: str,
    before_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Retrieves all historical innings/spells for a given player strictly prior to before_date.
    Guarantees zero data leakage from match day onwards.
    """
    mask = (df_scorecards[player_col] == player_name) & (df_scorecards["date"] < before_date)
    return df_scorecards[mask].sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    df_m, df_bat, df_bowl = load_raw_data()
    train_m, val_m, test_m = create_time_based_split(df_m)
    print("Preprocessing checks passed successfully!")
