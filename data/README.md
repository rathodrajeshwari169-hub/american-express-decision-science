# Dataset Documentation: American Express Decision Science Challenge

## 1. Overview
This directory contains the authentic multi-season sports dataset configured in the exact schema of the **American Express Campus Challenge (Decision Science Track / Data Titans 2024)**.

Unlike synthetic toy datasets, this data is grounded in historical professional T20 cricket matches, capturing the real trajectories, player forms, and team dynamics of over 600 players across 1,000+ matches from 2008 to 2024.

---

## 2. File Schema & Descriptions

### 2.1 `raw/matches.csv` (Match Level Records)
Corresponds to `train_data.csv` in the AmEx competition.
- `match_id` (int): Unique match identifier.
- `season` (int): Tournament season year.
- `date` (str: YYYY-MM-DD): Date of the match (critical for zero-leakage temporal ordering).
- `venue` (str): Stadium / ground name.
- `city` (str): Match host city.
- `team1` (str): Batting first team / home team designation.
- `team2` (str): Fielding first team / away team designation.
- `toss_winner` (str): Team winning the toss.
- `toss_decision` (str): `bat` or `field`.
- `winner` (str): Match winner (Target variable source).
- `win_margin` (int/float): Margin of victory.
- `win_by` (str): `runs` or `wickets`.
- `team1_roster` (str): Semicolon-delimited player IDs/names in Team 1's Playing XI.
- `team2_roster` (str): Semicolon-delimited player IDs/names in Team 2's Playing XI.

### 2.2 `raw/batsman_level_scorecard.csv` (Batting Scorecards)
Granular player-innings performance records.
- `match_id` (int): Foreign key to `matches.csv`.
- `batsman_id` (str): Unique player identifier.
- `batsman_name` (str): Player full name.
- `team` (str): Player's team in this match.
- `opponent` (str): Opposing bowling team.
- `runs` (int): Total runs scored in innings.
- `balls_faced` (int): Total legal balls faced.
- `strike_rate` (float): Batting strike rate: $(runs / balls\_faced) \times 100$.
- `fours` (int): Number of 4s hit.
- `sixes` (int): Number of 6s hit.
- `boundary_rate` (float): Percentage of balls resulting in boundaries.
- `is_out` (int: 0/1): Binary indicator of whether the batsman was dismissed.

### 2.3 `raw/bowler_level_scorecard.csv` (Bowling Scorecards)
Granular bowler performance records per match.
- `match_id` (int): Foreign key to `matches.csv`.
- `bowler_id` (str): Unique player identifier.
- `bowler_name` (str): Player full name.
- `team` (str): Bowler's team in this match.
- `opponent` (str): Opposing batting team.
- `overs` (float): Overs bowled (e.g., 4.0).
- `runs_conceded` (int): Total runs conceded.
- `wickets` (int): Total wickets taken by bowler.
- `economy` (float): Runs conceded per 6 balls: $(runs\_conceded / overs)$.
- `dots` (int): Number of dot balls (0 runs scored).
- `dot_rate` (float): Percentage of deliveries that were dot balls.

---

## 3. Strict Pre-Match Availability Constraint (Zero-Leakage Protocol)
In production deployment and historical validation:
- All features calculated for match $m$ on date $t_m$ strictly utilize data from matches prior to $t_m$ ($\tau < t_m$).
- Live match statistics (runs scored in match $m$, wickets in match $m$) are completely masked at prediction time.
