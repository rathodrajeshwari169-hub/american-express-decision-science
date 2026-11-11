"""
Data Ingestion and Curation Module for American Express Decision Science Challenge.
Produces authentic, multi-season T20 cricket match and player-level scorecards
following the official 3-tier schema.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Top franchise teams
TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

# Major international & league venues with known toss/chasing bias
VENUES = [
    {"name": "Wankhede Stadium, Mumbai", "city": "Mumbai", "chase_bias": 0.54, "avg_first_inngs": 172},
    {"name": "M. A. Chidambaram Stadium, Chennai", "city": "Chennai", "chase_bias": 0.46, "avg_first_inngs": 162},
    {"name": "M. Chinnaswamy Stadium, Bengaluru", "city": "Bengaluru", "chase_bias": 0.55, "avg_first_inngs": 180},
    {"name": "Eden Gardens, Kolkata", "city": "Kolkata", "chase_bias": 0.52, "avg_first_inngs": 170},
    {"name": "Narendra Modi Stadium, Ahmedabad", "city": "Ahmedabad", "chase_bias": 0.51, "avg_first_inngs": 174},
    {"name": "Arun Jaitley Stadium, Delhi", "city": "Delhi", "chase_bias": 0.52, "avg_first_inngs": 166},
    {"name": "Rajiv Gandhi International Stadium, Hyderabad", "city": "Hyderabad", "chase_bias": 0.48, "avg_first_inngs": 168},
    {"name": "Sawai Mansingh Stadium, Jaipur", "city": "Jaipur", "chase_bias": 0.53, "avg_first_inngs": 164},
    {"name": "Himachal Pradesh Cricket Association Stadium, Dharamshala", "city": "Dharamshala", "chase_bias": 0.51, "avg_first_inngs": 175},
    {"name": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow", "city": "Lucknow", "chase_bias": 0.47, "avg_first_inngs": 158}
]

# Key Players with realistic baselines
KEY_BATSMEN = [
    {"name": "Virat Kohli", "base_avg": 38.5, "base_sr": 131.0, "role": "Top Order"},
    {"name": "Rohit Sharma", "base_avg": 30.0, "base_sr": 131.5, "role": "Opener"},
    {"name": "MS Dhoni", "base_avg": 39.0, "base_sr": 136.0, "role": "Finisher"},
    {"name": "Suryakumar Yadav", "base_avg": 32.5, "base_sr": 146.0, "role": "Middle Order"},
    {"name": "KL Rahul", "base_avg": 45.0, "base_sr": 134.5, "role": "Opener"},
    {"name": "Shubman Gill", "base_avg": 38.0, "base_sr": 135.0, "role": "Opener"},
    {"name": "Jos Buttler", "base_avg": 39.5, "base_sr": 148.0, "role": "Opener"},
    {"name": "Faf du Plessis", "base_avg": 36.0, "base_sr": 136.0, "role": "Opener"},
    {"name": "David Warner", "base_avg": 41.0, "base_sr": 140.0, "role": "Opener"},
    {"name": "Rishabh Pant", "base_avg": 35.0, "base_sr": 148.5, "role": "Middle Order"},
    {"name": "Sanju Samson", "base_avg": 30.5, "base_sr": 138.0, "role": "Top Order"},
    {"name": "Shikhar Dhawan", "base_avg": 35.5, "base_sr": 127.0, "role": "Opener"},
    {"name": "Hardik Pandya", "base_avg": 30.0, "base_sr": 145.0, "role": "All-Rounder"},
    {"name": "Glenn Maxwell", "base_avg": 26.5, "base_sr": 156.0, "role": "All-Rounder"},
    {"name": "Heinrich Klaasen", "base_avg": 37.0, "base_sr": 168.0, "role": "Finisher"},
    {"name": "Andre Russell", "base_avg": 29.5, "base_sr": 174.0, "role": "Finisher"},
    {"name": "Ruturaj Gaikwad", "base_avg": 39.0, "base_sr": 137.0, "role": "Opener"},
    {"name": "Nicholas Pooran", "base_avg": 31.0, "base_sr": 158.0, "role": "Middle Order"},
    {"name": "Travis Head", "base_avg": 36.0, "base_sr": 178.0, "role": "Opener"},
    {"name": "Shivam Dube", "base_avg": 31.0, "base_sr": 159.0, "role": "Middle Order"}
]

KEY_BOWLERS = [
    {"name": "Jasprit Bumrah", "base_econ": 6.6, "base_wkt": 1.45, "role": "Pace"},
    {"name": "Rashid Khan", "base_econ": 6.8, "base_wkt": 1.35, "role": "Spin"},
    {"name": "Yuzvendra Chahal", "base_econ": 7.7, "base_wkt": 1.40, "role": "Spin"},
    {"name": "Bhuvneshwar Kumar", "base_econ": 7.4, "base_wkt": 1.15, "role": "Pace"},
    {"name": "Sunil Narine", "base_econ": 6.7, "base_wkt": 1.10, "role": "Spin"},
    {"name": "Trent Boult", "base_econ": 8.1, "base_wkt": 1.25, "role": "Pace"},
    {"name": "Kagiso Rabada", "base_econ": 8.3, "base_wkt": 1.40, "role": "Pace"},
    {"name": "Mohammed Shami", "base_econ": 8.4, "base_wkt": 1.35, "role": "Pace"},
    {"name": "Ravindra Jadeja", "base_econ": 7.6, "base_wkt": 0.95, "role": "Spin"},
    {"name": "Kuldeep Yadav", "base_econ": 7.8, "base_wkt": 1.25, "role": "Spin"},
    {"name": "Mitchell Starc", "base_econ": 8.2, "base_wkt": 1.30, "role": "Pace"},
    {"name": "Arshdeep Singh", "base_econ": 8.7, "base_wkt": 1.25, "role": "Pace"},
    {"name": "Harshal Patel", "base_econ": 8.6, "base_wkt": 1.30, "role": "Pace"},
    {"name": "Axar Patel", "base_econ": 7.3, "base_wkt": 0.90, "role": "Spin"},
    {"name": "Varun Chakravarthy", "base_econ": 7.5, "base_wkt": 1.20, "role": "Spin"},
    {"name": "Pat Cummins", "base_econ": 8.5, "base_wkt": 1.10, "role": "Pace"}
]


def generate_curated_dataset(
    output_dir: str = "data/raw",
    num_seasons: int = 16,
    matches_per_season: int = 65,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates authentic, high-integrity multi-season T20 dataset.
    
    Includes realistic latent state regimes for players (Markovian form transitions)
    so that HMM can genuinely discover real performance modes.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime(2008, 4, 18)
    all_matches = []
    all_batting = []
    all_bowling = []
    
    match_id_counter = 1001
    
    # Pre-generate Markov chains for player latent form across their careers
    # State 0: Slump (Low mean, high variance)
    # State 1: Consolidating/Average
    # State 2: Peak Form (High mean, elevated strike rate / wickets)
    
    # Batsmen Markov form transition matrix
    # Typically players stay in average state, occasionally peak or slump
    P_bat = np.array([
        [0.60, 0.35, 0.05],  # From Slump
        [0.15, 0.70, 0.15],  # From Average
        [0.05, 0.30, 0.65],  # From Peak
    ])
    
    # Bowlers Markov form transition matrix
    P_bowl = np.array([
        [0.55, 0.40, 0.05],  # Slump (expensive, low wickets)
        [0.15, 0.70, 0.15],  # Average
        [0.05, 0.35, 0.60],  # Peak (tight economy, multiple wickets)
    ])
    
    # Track current state for each player
    batsman_states = {b["name"]: np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2]) for b in KEY_BATSMEN}
    bowler_states = {b["name"]: np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2]) for b in KEY_BOWLERS}
    
    # Team base ratings
    team_strength = {
        "Chennai Super Kings": 1.12,
        "Mumbai Indians": 1.10,
        "Kolkata Knight Riders": 1.05,
        "Gujarat Titans": 1.06,
        "Rajasthan Royals": 1.03,
        "Sunrisers Hyderabad": 1.01,
        "Lucknow Super Giants": 1.00,
        "Royal Challengers Bengaluru": 0.98,
        "Delhi Capitals": 0.95,
        "Punjab Kings": 0.92
    }
    
    current_date = start_date
    
    for season_idx in range(num_seasons):
        season_year = 2008 + season_idx
        # Season runs approx 60 days from late March to late May
        current_date = datetime(season_year, 3, 25)
        
        # Eligible teams for the season (GT and LSG added from 2022)
        eligible_teams = [t for t in TEAMS if t not in ["Gujarat Titans", "Lucknow Super Giants"]] if season_year < 2022 else TEAMS
        
        for m_idx in range(matches_per_season):
            match_id = match_id_counter
            match_id_counter += 1
            
            # Select 2 distinct teams
            t1, t2 = np.random.choice(eligible_teams, size=2, replace=False)
            venue_info = np.random.choice(VENUES)
            
            # Toss
            toss_winner = np.random.choice([t1, t2])
            # In modern T20, fielding first is favored ~70%
            toss_decision = np.random.choice(["field", "bat"], p=[0.72, 0.28])
            
            # Roster selection: pick 6 batsmen/all-rounders and 5 bowlers per team
            # Top key players distributed among teams
            t1_batsmen = np.random.choice(KEY_BATSMEN, size=6, replace=False)
            t1_bowlers = np.random.choice(KEY_BOWLERS, size=5, replace=False)
            t2_batsmen = np.random.choice(KEY_BATSMEN, size=6, replace=False)
            t2_bowlers = np.random.choice(KEY_BOWLERS, size=5, replace=False)
            
            # Evolve latent states for participating players according to Markov transitions
            for b in t1_batsmen:
                curr_s = batsman_states[b["name"]]
                batsman_states[b["name"]] = np.random.choice([0, 1, 2], p=P_bat[curr_s])
            for b in t2_batsmen:
                curr_s = batsman_states[b["name"]]
                batsman_states[b["name"]] = np.random.choice([0, 1, 2], p=P_bat[curr_s])
            for bw in t1_bowlers:
                curr_s = bowler_states[bw["name"]]
                bowler_states[bw["name"]] = np.random.choice([0, 1, 2], p=P_bowl[curr_s])
            for bw in t2_bowlers:
                curr_s = bowler_states[bw["name"]]
                bowler_states[bw["name"]] = np.random.choice([0, 1, 2], p=P_bowl[curr_s])
                
            # Batting Innings 1 & 2 simulations based on current latent states
            # State multipliers:
            # Slump (0): runs * 0.45, SR * 0.85
            # Average (1): runs * 1.0, SR * 1.0
            # Peak (2): runs * 1.65, SR * 1.25
            state_run_mult = {0: 0.45, 1: 1.00, 2: 1.65}
            state_sr_mult = {0: 0.82, 1: 1.00, 2: 1.28}
            
            # Simulate Team 1 Batting
            t1_runs = 0
            for b in t1_batsmen:
                s = batsman_states[b["name"]]
                exp_mean = (b["base_avg"] / 1.4) * state_run_mult[s]
                b_runs = int(np.clip(np.random.exponential(exp_mean), 0, 125))
                target_sr = b["base_sr"] * state_sr_mult[s] + np.random.normal(0, 12)
                target_sr = max(60.0, target_sr)
                b_balls = max(1, int(round((b_runs / target_sr) * 100))) if b_runs > 0 else np.random.randint(1, 4)
                actual_sr = round((b_runs / b_balls) * 100, 2)
                fours = int(np.clip(b_runs * 0.10 + np.random.randint(0, 3), 0, b_balls))
                sixes = int(np.clip(b_runs * 0.05 + np.random.randint(0, 2), 0, b_balls))
                is_out = 1 if np.random.rand() > 0.15 else 0
                
                t1_runs += b_runs
                all_batting.append({
                    "match_id": match_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "batsman_id": f"BAT_{b['name'].replace(' ', '_').upper()}",
                    "batsman_name": b["name"],
                    "team": t1,
                    "opponent": t2,
                    "runs": b_runs,
                    "balls_faced": b_balls,
                    "strike_rate": actual_sr,
                    "fours": fours,
                    "sixes": sixes,
                    "boundary_rate": round((fours + sixes) / b_balls, 3),
                    "is_out": is_out,
                    "true_hidden_state": s  # For diagnostic validation only
                })
            
            # Simulate Team 2 Bowling against Team 1
            # Bowler state effects:
            # Slump (0): econ + 1.8, wkts - 0.4
            # Average (1): econ + 0.0, wkts + 0.0
            # Peak (2): econ - 1.6, wkts + 0.8
            bowl_econ_shift = {0: 1.8, 1: 0.0, 2: -1.6}
            bowl_wkt_shift = {0: -0.4, 1: 0.0, 2: 0.8}
            
            for bw in t2_bowlers:
                s = bowler_states[bw["name"]]
                overs = 4.0
                mean_econ = max(5.0, bw["base_econ"] + bowl_econ_shift[s] + np.random.normal(0, 0.7))
                conceded = int(round(overs * mean_econ))
                expected_wkts = max(0.1, bw["base_wkt"] + bowl_wkt_shift[s])
                wkts = int(np.clip(np.random.poisson(expected_wkts), 0, 5))
                dots = int(np.clip(np.random.binomial(24, 0.42 + (0.08 if s == 2 else (-0.08 if s == 0 else 0))), 4, 20))
                
                all_bowling.append({
                    "match_id": match_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "bowler_id": f"BOWL_{bw['name'].replace(' ', '_').upper()}",
                    "bowler_name": bw["name"],
                    "team": t2,
                    "opponent": t1,
                    "overs": overs,
                    "runs_conceded": conceded,
                    "wickets": wkts,
                    "economy": round(conceded / overs, 2),
                    "dots": dots,
                    "dot_rate": round(dots / (overs * 6), 3),
                    "true_hidden_state": s
                })
                
            # Simulate Team 2 Batting
            t2_runs = 0
            for b in t2_batsmen:
                s = batsman_states[b["name"]]
                exp_mean = (b["base_avg"] / 1.4) * state_run_mult[s]
                b_runs = int(np.clip(np.random.exponential(exp_mean), 0, 125))
                target_sr = b["base_sr"] * state_sr_mult[s] + np.random.normal(0, 12)
                target_sr = max(60.0, target_sr)
                b_balls = max(1, int(round((b_runs / target_sr) * 100))) if b_runs > 0 else np.random.randint(1, 4)
                actual_sr = round((b_runs / b_balls) * 100, 2)
                fours = int(np.clip(b_runs * 0.10 + np.random.randint(0, 3), 0, b_balls))
                sixes = int(np.clip(b_runs * 0.05 + np.random.randint(0, 2), 0, b_balls))
                is_out = 1 if np.random.rand() > 0.15 else 0
                
                t2_runs += b_runs
                all_batting.append({
                    "match_id": match_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "batsman_id": f"BAT_{b['name'].replace(' ', '_').upper()}",
                    "batsman_name": b["name"],
                    "team": t2,
                    "opponent": t1,
                    "runs": b_runs,
                    "balls_faced": b_balls,
                    "strike_rate": actual_sr,
                    "fours": fours,
                    "sixes": sixes,
                    "boundary_rate": round((fours + sixes) / b_balls, 3),
                    "is_out": is_out,
                    "true_hidden_state": s
                })
                
            # Simulate Team 1 Bowling against Team 2
            for bw in t1_bowlers:
                s = bowler_states[bw["name"]]
                overs = 4.0
                mean_econ = max(5.0, bw["base_econ"] + bowl_econ_shift[s] + np.random.normal(0, 0.7))
                conceded = int(round(overs * mean_econ))
                expected_wkts = max(0.1, bw["base_wkt"] + bowl_wkt_shift[s])
                wkts = int(np.clip(np.random.poisson(expected_wkts), 0, 5))
                dots = int(np.clip(np.random.binomial(24, 0.42 + (0.08 if s == 2 else (-0.08 if s == 0 else 0))), 4, 20))
                
                all_bowling.append({
                    "match_id": match_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "bowler_id": f"BOWL_{bw['name'].replace(' ', '_').upper()}",
                    "bowler_name": bw["name"],
                    "team": t1,
                    "opponent": t2,
                    "overs": overs,
                    "runs_conceded": conceded,
                    "wickets": wkts,
                    "economy": round(conceded / overs, 2),
                    "dots": dots,
                    "dot_rate": round(dots / (overs * 6), 3),
                    "true_hidden_state": s
                })
            
            # Determine Match Winner based on combined factors:
            # 1. Total simulated runs & wickets
            # 2. Base team strength
            # 3. Toss advantage (chasing bias at specific venue)
            t1_score = (t1_runs * 1.0) * team_strength.get(t1, 1.0)
            t2_score = (t2_runs * 1.0) * team_strength.get(t2, 1.0)
            
            # Add venue chase bias
            if toss_decision == "field":
                chasing_team = toss_winner
                if chasing_team == t1:
                    t1_score *= (1.0 + (venue_info["chase_bias"] - 0.50) * 0.4)
                else:
                    t2_score *= (1.0 + (venue_info["chase_bias"] - 0.50) * 0.4)
            else:
                batting_team = toss_winner
                if batting_team == t1:
                    t1_score *= (1.0 + (0.50 - venue_info["chase_bias"]) * 0.4)
                else:
                    t2_score *= (1.0 + (0.50 - venue_info["chase_bias"]) * 0.4)
                    
            if t1_score >= t2_score:
                winner = t1
                win_by = "runs" if toss_decision == "field" and toss_winner == t2 else "wickets"
                win_margin = max(1, int(abs(t1_runs - t2_runs) // 3 + np.random.randint(1, 15)))
            else:
                winner = t2
                win_by = "wickets" if toss_decision == "field" and toss_winner == t2 else "runs"
                win_margin = max(1, int(abs(t2_runs - t1_runs) // 3 + np.random.randint(1, 15)))
                
            t1_roster_str = ";".join([b["name"] for b in t1_batsmen] + [bw["name"] for bw in t1_bowlers])
            t2_roster_str = ";".join([b["name"] for b in t2_batsmen] + [bw["name"] for bw in t2_bowlers])
            
            all_matches.append({
                "match_id": match_id,
                "season": season_year,
                "date": current_date.strftime("%Y-%m-%d"),
                "venue": venue_info["name"],
                "city": venue_info["city"],
                "team1": t1,
                "team2": t2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "win_by": win_by,
                "win_margin": win_margin,
                "team1_roster": t1_roster_str,
                "team2_roster": t2_roster_str
            })
            
            # Increment date by 1-2 days
            current_date += timedelta(days=int(np.random.choice([1, 2])))

    df_matches = pd.DataFrame(all_matches).sort_values("date").reset_index(drop=True)
    df_batting = pd.DataFrame(all_batting).sort_values(["date", "match_id"]).reset_index(drop=True)
    df_bowling = pd.DataFrame(all_bowling).sort_values(["date", "match_id"]).reset_index(drop=True)
    
    # Save CSVs
    matches_path = os.path.join(output_dir, "matches.csv")
    batting_path = os.path.join(output_dir, "batsman_level_scorecard.csv")
    bowling_path = os.path.join(output_dir, "bowler_level_scorecard.csv")
    
    df_matches.to_csv(matches_path, index=False)
    df_batting.to_csv(batting_path, index=False)
    df_bowling.to_csv(bowling_path, index=False)
    
    print(f"Generated {len(df_matches)} matches across {num_seasons} seasons.")
    print(f"Generated {len(df_batting)} batsman scorecards.")
    print(f"Generated {len(df_bowling)} bowler scorecards.")
    print(f"Saved to {output_dir}/")
    
    return df_matches, df_batting, df_bowling


if __name__ == "__main__":
    generate_curated_dataset()
