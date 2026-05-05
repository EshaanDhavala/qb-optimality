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
1. Go to kaggle.com → Account → "Create New API Token" → downloads `kaggle.json`
2. Move it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
3. Lock permissions: `chmod 600 ~/.kaggle/kaggle.json`

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
