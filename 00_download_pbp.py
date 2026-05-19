"""
00_download_pbp.py
------------------
Downloads 2021 NFL play-by-play data from nflverse via nfl_data_py.
This is the automated half of the data acquisition.

The Big Data Bowl 2023 tracking data covers the 2021 season (weeks 1-8),
so we pull PBP for 2021 only.

We need PBP for two things:
  1. Win probability (wp) per play — defines clutch vs non-clutch
  2. EPA per play — this is our outcome variable for the model

Usage:
    python src/00_download_pbp.py
"""

import os
import pandas as pd
import nfl_data_py as nfl

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Download 2021 play-by-play ─────────────────────────────────────────────
print("Downloading 2021 NFL play-by-play from nflverse...")
pbp = nfl.import_pbp_data([2021], downcast=True)

print(f"  Raw: {len(pbp):,} rows × {len(pbp.columns)} columns")

# ── Keep only columns we actually need ────────────────────────────────────
# Keeps file size small and merging simple
KEEP = [
    # Keys to merge with tracking data
    "game_id",          # format: 2021_01_ATL_PHI
    "play_id",          # matches playId in tracking

    # Outcome variable
    "epa",              # expected points added — model target
    "qb_epa",           # QB-credited EPA

    # Situation — defines clutch vs non-clutch
    "wp",               # win probability at start of play
    "wpa",              # win probability added

    # Game context — model features / controls
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "qtr",
    "season_type",

    # QB identity
    "passer_player_name",
    "passer_player_id",

    # Play type filters
    "play_type",
    "pass",
    "sack",
    "qb_scramble",
    "interception",
    "complete_pass",
    "air_yards",
    "cpoe",

    # Identifiers
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "week",
    "season",
]

# Only keep columns that exist
keep_cols = [c for c in KEEP if c in pbp.columns]
pbp_slim  = pbp[keep_cols].copy()

# ── Filter to dropback plays only ──────────────────────────────────────────
# Big Data Bowl 2023 covers QB dropbacks — keep only those
pbp_drops = pbp_slim[
    (pbp_slim["pass"] == 1) &
    (pbp_slim["play_type"].isin(["pass"])) &
    (pbp_slim["passer_player_name"].notna()) &
    (pbp_slim["epa"].notna()) &
    (pbp_slim["wp"].notna())
].copy()

print(f"  After dropback filter: {len(pbp_drops):,} plays")

# ── Add clutch flag ────────────────────────────────────────────────────────
# Clutch captures three late-game high-pressure scenarios (applied last so
# they overwrite non_clutch when WP is low but the play is genuinely critical):
#   1. Close game (WP 40–60%) in Q4/OT
#   2. 4th down within one score (|score_diff| ≤ 8) in Q4/OT — must-convert plays
#   3. 2-minute drill: trailing by ≤8 with ≤120 seconds left in Q4/OT
# Non-clutch (blowouts) is set first so clutch can overwrite it where needed.
pbp_drops["situation"] = "neutral"

pbp_drops.loc[
    (pbp_drops["wp"] < 0.20) | (pbp_drops["wp"] > 0.80),
    "situation"
] = "non_clutch"

clutch_mask = (
    # 1. Classic close game, late quarter
    ((pbp_drops["wp"] >= 0.40) & (pbp_drops["wp"] <= 0.60) & (pbp_drops["qtr"] >= 4))
    |
    # 2. Must-convert: 4th down within one score in Q4/OT
    ((pbp_drops["qtr"] >= 4) & (pbp_drops["down"] == 4) &
     (pbp_drops["score_differential"].abs() <= 8))
    |
    # 3. 2-minute drill: trailing by one score or less with ≤2 minutes left
    ((pbp_drops["qtr"] >= 4) & (pbp_drops["game_seconds_remaining"] <= 120) &
     (pbp_drops["score_differential"] >= -8) & (pbp_drops["score_differential"] < 0))
)
pbp_drops.loc[clutch_mask, "situation"] = "clutch"

counts = pbp_drops["situation"].value_counts()
print(f"\n  Clutch plays:     {counts.get('clutch', 0):,}")
print(f"  Non-clutch plays: {counts.get('non_clutch', 0):,}")
print(f"  Neutral plays:    {counts.get('neutral', 0):,}")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = f"{RAW_DIR}/pbp_2021.parquet"
pbp_drops.to_parquet(out_path, index=False)
print(f"\n✓ Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1e6:.1f} MB")
print("\nNext: manually download Big Data Bowl 2023 data from Kaggle")
print("      then run: python src/01_validate_data.py")
