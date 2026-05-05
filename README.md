# QB Clutch Optimality

**Research question:** Which quarterbacks stay closest to optimal pocket behavior when the game is on the line, and which ones break down?

## Overview

We train an XGBoost model on spatial features derived from NFL player tracking data to predict EPA (Expected Points Added) per play. The model learns what pocket behaviors historically produce good outcomes. For each play we compute an **Optimality Score** = Actual EPA − Predicted EPA, then compare each QB's score in clutch vs. non-clutch situations to produce a **Clutch Optimality Rating**.

Inspired by Brian Burke's DeepQB (ESPN / MIT Sloan 2019), extended to pocket behavior and clutch situational context.

---

## Clutch Definition

| Label | Win Probability |
|-------|----------------|
| Clutch | 40% – 60% |
| Non-clutch | < 20% or > 80% |
| Neutral | Everything else (excluded from primary analysis) |

**Clutch Optimality Rating** = mean(Optimality Score in clutch) − mean(Optimality Score in non-clutch)  
Positive → QB outperforms model expectations more when the game is close.

---

## Data Sources

| Source | Contents | Location |
|--------|----------|----------|
| nfl_data_py (nflfastR) | 2021 play-by-play, EPA, win probability | `data/raw/pbp_2021.parquet` |
| NFL Big Data Bowl 2023 (Kaggle) | 10 Hz player tracking, weeks 1–8 of 2021 | `data/raw/big_data_bowl_2023/` |

Big Data Bowl files: `games.csv`, `plays.csv`, `players.csv`, `pffScoutingData.csv`, `tracking_week_1.csv` … `tracking_week_8.csv`

> **Note:** `data/raw/` and `data/processed/` are git-ignored. See setup steps below.

---

## Directory Layout

```
qb-optimality/
├── data/
│   ├── raw/
│   │   ├── pbp_2021.parquet          # nflfastR play-by-play
│   │   └── big_data_bowl_2023/       # Kaggle tracking data
│   └── processed/                    # Merged, feature-engineered data
├── outputs/
│   ├── figures/                      # Plots and charts
│   └── tables/                       # CSV summaries and rankings
├── notebooks/                        # Exploratory analysis
├── 00_download_pbp.py                # Download nflfastR PBP data
├── 01_validate_data.py               # Validate all data files are present
├── 02_feature_engineering.py         # (upcoming) Build spatial features
├── 03_model_training.py              # (upcoming) Train XGBoost model
├── 04_analysis.py                    # (upcoming) Compute Optimality Scores
├── 05_clutch_rating.py               # (upcoming) QB Clutch Optimality Rankings
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download nflfastR play-by-play data
```bash
python 00_download_pbp.py
```
Saves filtered 2021 QB dropback data to `data/raw/pbp_2021.parquet`.

### 3. Download Big Data Bowl 2023 tracking data (Kaggle)

**One-time Kaggle credentials setup:**
1. Go to kaggle.com → Account → "Create New Token" → copies a `KGAT_...` token to your clipboard
2. Save it: `mkdir -p ~/.kaggle && echo "PASTE_YOUR_TOKEN_HERE" > ~/.kaggle/access_token`
3. Lock permissions: `chmod 600 ~/.kaggle/access_token`

**Download and unzip:**
```bash
kaggle competitions download -c nfl-big-data-bowl-2023 -p data/raw/big_data_bowl_2023/
unzip data/raw/big_data_bowl_2023/nfl-big-data-bowl-2023.zip -d data/raw/big_data_bowl_2023/
```
> You must accept the competition rules on Kaggle before the download will work.

### 4. Validate all data
```bash
python 01_validate_data.py
```
All 13 file checks should pass before proceeding.

---

## Engineered Features (at moment of pass release)

| Category | Features |
|----------|---------|
| QB | Displacement from snap, speed at release, orientation at release, time from snap to release |
| Pass rushers | Distance of nearest rusher, approach speed of nearest 2 rushers, rushers within 3 yards |
| Pocket | Pocket area (convex hull of OL), pocket collapse rate (Δarea snap→release) |
| Context | Down, distance, yardline, score differential, half_seconds_remaining |

---

## Model

- **Algorithm:** XGBoost regressor
- **Target:** EPA per play
- **Split:** Train weeks 1–6 / Validate week 7 / Test week 8
- **Sample size:** ~5,000–8,000 QB dropback plays (weeks 1–8, 2021)

Neural networks were ruled out — sample size is too small for reliable generalization at this scope.

---

## Pipeline (in order)

```
00_download_pbp.py          →  data/raw/pbp_2021.parquet
kaggle download + unzip     →  data/raw/big_data_bowl_2023/
01_validate_data.py         →  confirms all inputs present
02_feature_engineering.py   →  data/processed/features.parquet
03_model_training.py        →  outputs/model.json + validation metrics
04_analysis.py              →  outputs/tables/play_optimality.csv
05_clutch_rating.py         →  outputs/tables/qb_clutch_ratings.csv
```

---

## Team Tasks

Feature engineering (script `02`) parallelizes across three people. All three write standalone functions with the same interface — takes a DataFrame of tracking frames for one play, returns a dict of features. Eshaan integrates them. Model and analysis are sequential after that.

```
Abhi ──┐
Gonzalo─┼──▶ Eshaan integrates ──▶ Andrew trains model ──▶ Keith analyzes ──▶ Dillon visualizes
Dillon ─┘
```

---

### Eshaan — Integration & Merge
**Script:** `02_feature_engineering.py`  
**Depends on:** everyone's feature functions  
**Difficulty:** ★★★★★

The glue that holds everything together. Every downstream script depends on this being correct.

**Tasks:**
- [ ] Join Big Data Bowl `games.csv` to nflfastR PBP — translate integer `gameId` to nflfastR `game_id` format (e.g. `2021_01_ARI_TEN`)
- [ ] Filter tracking data to QB dropback plays only (cross-reference `plays.csv`)
- [ ] For each play, isolate the `ball_snap` frame and `pass_forward` frame from tracking data
- [ ] Call `get_qb_features()`, `get_rusher_features()`, and `get_pocket_features()` per play
- [ ] Merge all feature dicts with context features from PBP (down, distance, yardline, score differential, half_seconds_remaining)
- [ ] Save final feature table to `data/processed/features.parquet`
- [ ] Handle edge cases: plays missing snap/release frames, plays with no OL detected, sacks

**Output columns:** `game_id`, `play_id`, `passer_player_name`, `week`, `situation`, `epa`, + all engineered features

---

### Abhi — Pocket Geometry Features
**Script:** `features/pocket_features.py`  
**Depends on:** nothing (standalone)  
**Difficulty:** ★★★★☆

Hardest feature module. Requires geometric reasoning and correct player identification — wrong OL tagging silently corrupts every downstream model.

**Tasks:**
- [ ] Write `get_pocket_features(snap_frame, release_frame, players_df)` → dict
- [ ] Identify the 5 offensive linemen using position data from `players.csv` (positions: `T`, `G`, `C`)
- [ ] Compute convex hull area of OL positions at snap using `scipy.spatial.ConvexHull`
- [ ] Compute convex hull area of OL positions at release
- [ ] Derive `pocket_area_at_release` and `pocket_collapse_rate` = (snap_area − release_area) / time_elapsed
- [ ] Handle edge cases: fewer than 3 OL detected (can't form hull), OL tagged as eligible receiver and split out

**Returns:** `{"pocket_area_at_release": float, "pocket_collapse_rate": float}`

---

### Andrew — Model Training
**Script:** `03_model_training.py`  
**Depends on:** `data/processed/features.parquet` (Eshaan's output)  
**Difficulty:** ★★★☆☆

**Tasks:**
- [ ] Load `data/processed/features.parquet`, drop rows with any null features
- [ ] Split by week: train = weeks 1–6, val = week 7, test = week 8
- [ ] Define feature columns (exclude `game_id`, `play_id`, `passer_player_name`, `situation`, `epa`)
- [ ] Train `xgboost.XGBRegressor` predicting `epa`
- [ ] Tune at minimum: `n_estimators`, `max_depth`, `learning_rate`, `subsample` using val set RMSE
- [ ] Report RMSE on val and test sets; print feature importances ranked
- [ ] Save trained model to `outputs/model.json` using `model.save_model()`
- [ ] Save predictions on full dataset to `outputs/tables/play_predictions.csv` (columns: `game_id`, `play_id`, `epa`, `predicted_epa`)

---

### Keith — Analysis & Optimality Scores
**Script:** `04_analysis.py`  
**Depends on:** `outputs/tables/play_predictions.csv` (Andrew's output)  
**Difficulty:** ★★☆☆☆

**Tasks:**
- [ ] Load `play_predictions.csv` and merge back with PBP to get `passer_player_name`, `situation`, `week`
- [ ] Compute `optimality_score` = `epa` − `predicted_epa` per play
- [ ] Group by `passer_player_name` × `situation` (clutch / non_clutch), compute mean optimality score
- [ ] Filter to QBs with at least 50 clutch plays and 50 non-clutch plays
- [ ] Compute **Clutch Optimality Rating** = mean(clutch optimality) − mean(non-clutch optimality) per QB
- [ ] Save to `outputs/tables/qb_clutch_ratings.csv` with columns: `passer_player_name`, `clutch_optimality`, `non_clutch_optimality`, `clutch_rating`, `n_clutch_plays`, `n_non_clutch_plays`

---

### Dillon — QB Spatial Features
**Script:** `features/qb_features.py`  
**Depends on:** nothing (standalone)  
**Difficulty:** ★★☆☆☆

**Tasks:**
- [ ] Write `get_qb_features(snap_frame, release_frame, qb_nfl_id)` → dict
- [ ] Look up QB's `(x, y)` position at snap frame and at release frame
- [ ] Compute `qb_displacement` = Euclidean distance between snap and release positions
- [ ] Extract `qb_speed_at_release` and `qb_orientation_at_release` directly from `s` and `o` columns at release frame
- [ ] Compute `time_to_throw` = release frame timestamp − snap frame timestamp (in seconds)
- [ ] Handle edge cases: QB not found in frame (use `nflId` from `plays.csv`), missing release frame

**Returns:** `{"qb_displacement": float, "qb_speed_at_release": float, "qb_orientation_at_release": float, "time_to_throw": float}`

---

### Gonzalo — Pass Rusher Features
**Script:** `features/rusher_features.py`  
**Depends on:** nothing (standalone)  
**Difficulty:** ★☆☆☆☆

Most mechanical module. Spatial lookups using PFF role tags.

**Tasks:**
- [ ] Write `get_rusher_features(release_frame, qb_nfl_id, pff_play_df)` → dict
- [ ] Filter `pff_play_df` to rows where `pff_role == "Pass Rush"` to identify rushers
- [ ] For each rusher, compute distance to QB at release frame using Euclidean distance on `(x, y)`
- [ ] Return `nearest_rusher_dist` = minimum distance across all rushers
- [ ] Compute approach speed for the 2 nearest rushers using their `s` (speed) column at release frame; return as `rusher_1_approach_speed`, `rusher_2_approach_speed`
- [ ] Return `rushers_within_3yds` = count of rushers with distance < 3.0 yards
- [ ] Handle edge cases: no rushers tagged (set all to `NaN`), fewer than 2 rushers (second approach speed = `NaN`)

**Returns:** `{"nearest_rusher_dist": float, "rusher_1_approach_speed": float, "rusher_2_approach_speed": float, "rushers_within_3yds": int}`
