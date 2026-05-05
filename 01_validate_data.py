"""
01_validate_data.py
-------------------
Run this after manually downloading the Big Data Bowl 2023 data from Kaggle.
Checks all required files are present, prints row counts and column names,
and flags any issues before you spend time on feature engineering.

Usage:
    python src/01_validate_data.py
"""

import os
import pandas as pd

BDB_DIR = "data/raw/big_data_bowl_2023"
RAW_DIR = "data/raw"

REQUIRED_FILES = {
    # Core files
    "games.csv":           "Game metadata (gameId, week, teams)",
    "plays.csv":           "Play metadata (down, distance, QB, outcome)",
    "players.csv":         "Player metadata (name, position, height, weight)",
    "pffScoutingData.csv": "PFF pressure grades per player per play",

    # Tracking files — weeks 1-8 of 2021 season
    "tracking_week_1.csv": "Player tracking week 1",
    "tracking_week_2.csv": "Player tracking week 2",
    "tracking_week_3.csv": "Player tracking week 3",
    "tracking_week_4.csv": "Player tracking week 4",
    "tracking_week_5.csv": "Player tracking week 5",
    "tracking_week_6.csv": "Player tracking week 6",
    "tracking_week_7.csv": "Player tracking week 7",
    "tracking_week_8.csv": "Player tracking week 8",
}

REQUIRED_PBP = "pbp_2021.parquet"

print("=" * 60)
print("QB Optimality Project — Data Validation")
print("=" * 60)

all_good = True

# ── Check PBP ─────────────────────────────────────────────────────────────
print("\n── 1. nflfastR Play-by-Play ──")
pbp_path = f"{RAW_DIR}/{REQUIRED_PBP}"
if os.path.exists(pbp_path):
    pbp = pd.read_parquet(pbp_path)
    print(f"  ✓ pbp_2021.parquet — {len(pbp):,} rows × {len(pbp.columns)} cols")
    print(f"    Columns: {', '.join(pbp.columns[:8])} ...")
    print(f"    Situations: {pbp['situation'].value_counts().to_dict()}")
else:
    print(f"  ✗ MISSING: {pbp_path}")
    print(f"    Run: python src/00_download_pbp.py")
    all_good = False

# ── Check Big Data Bowl files ─────────────────────────────────────────────
print("\n── 2. Big Data Bowl 2023 Files ──")
print(f"   Expected location: {BDB_DIR}/")

if not os.path.exists(BDB_DIR):
    print(f"  ✗ Directory not found: {BDB_DIR}")
    print(f"    Create it and place Kaggle files inside")
    all_good = False
else:
    for filename, description in REQUIRED_FILES.items():
        filepath = f"{BDB_DIR}/{filename}"
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1e6
            # Quick row count without loading full file
            try:
                if filename.endswith(".csv"):
                    # Count lines fast
                    with open(filepath, 'r') as f:
                        nrows = sum(1 for _ in f) - 1  # subtract header
                    print(f"  ✓ {filename:<30} {nrows:>10,} rows  ({size_mb:.0f} MB)")
                    print(f"    {description}")
            except Exception as e:
                print(f"  ✓ {filename:<30} ({size_mb:.0f} MB) — could not count rows: {e}")
        else:
            print(f"  ✗ MISSING: {filename}")
            print(f"    {description}")
            all_good = False

# ── Spot check tracking columns ───────────────────────────────────────────
print("\n── 3. Tracking Data Column Check ──")
tracking_path = f"{BDB_DIR}/tracking_week_1.csv"
if os.path.exists(tracking_path):
    sample = pd.read_csv(tracking_path, nrows=5)
    print(f"  Columns found: {list(sample.columns)}")

    required_tracking_cols = ["gameId", "playId", "nflId", "frameId",
                               "x", "y", "s", "a", "dis", "o", "dir", "event"]
    missing_cols = [c for c in required_tracking_cols if c not in sample.columns]
    if missing_cols:
        print(f"  ✗ Missing expected columns: {missing_cols}")
        all_good = False
    else:
        print(f"  ✓ All required tracking columns present")

    # Check event types
    full_week1 = pd.read_csv(tracking_path, usecols=["event"])
    events = full_week1["event"].dropna().unique()
    print(f"\n  Event types found: {sorted(events)}")
    print(f"  Key events needed: 'ball_snap', 'pass_forward' (or 'pass_released')")

# ── Check plays.csv columns ───────────────────────────────────────────────
print("\n── 4. Plays Data Column Check ──")
plays_path = f"{BDB_DIR}/plays.csv"
if os.path.exists(plays_path):
    plays = pd.read_csv(plays_path, nrows=5)
    print(f"  Columns found: {list(plays.columns)}")

# ── Check PFF scouting columns ────────────────────────────────────────────
print("\n── 5. PFF Scouting Data Column Check ──")
pff_path = f"{BDB_DIR}/pffScoutingData.csv"
if os.path.exists(pff_path):
    pff = pd.read_csv(pff_path, nrows=5)
    print(f"  Columns found: {list(pff.columns)}")
    print(f"  (Looking for: pff_hit, pff_hurry, pff_sack, pff_beatenByDefender)")

# ── Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_good:
    print("✓ All data present. Ready to run:")
    print("  python src/02_feature_engineering.py")
else:
    print("✗ Missing files — fix issues above before proceeding")
    print("\n  Kaggle download URL:")
    print("  https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data")
print("=" * 60)
