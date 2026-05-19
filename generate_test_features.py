"""
generate_test_features.py
--------------------------
Generates a realistic mock features.parquet for testing 03_model_training.py
WITHOUT needing 02_feature_engineering.py to be complete.

Uses the real PBP data (pbp_2021.parquet) for authentic EPA/situation values,
and synthesizes spatial features with realistic distributions based on
NFL tracking data research.

Usage:
    python generate_test_features.py

Output:
    data/processed/features.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

def main():
    print("Loading real PBP data...")
    pbp = pd.read_parquet(RAW_DIR / "pbp_2021.parquet")
    print(f"  {len(pbp):,} plays loaded")

    # Only keep plays that would realistically have tracking data (weeks 1-8)
    # The PBP week column may be stored differently, so we derive it from game_id
    pbp["week"] = pbp["game_id"].str.extract(r"2021_(\d+)_").astype(int)
    pbp = pbp[pbp["week"] <= 8].copy()
    print(f"  {len(pbp):,} plays in weeks 1-8")

    n = len(pbp)
    print(f"\nSynthesizing spatial features for {n:,} plays...")

    # ── QB Features ───────────────────────────────────────────────────────
    # qb_displacement: QBs typically move 0-5 yards from snap spot
    qb_displacement = rng.gamma(shape=1.5, scale=1.2, size=n).clip(0, 8)

    # qb_speed_at_release: 0-8 mph range, most around 1-3
    qb_speed_at_release = rng.gamma(shape=1.8, scale=1.0, size=n).clip(0, 9)

    # qb_orientation: encode as sin/cos to handle circular nature (0° ≈ 360°)
    qb_orientation_raw = rng.uniform(0, 2 * np.pi, size=n)
    qb_orientation_sin = np.sin(qb_orientation_raw)
    qb_orientation_cos = np.cos(qb_orientation_raw)

    # time_to_throw: typically 2.0-4.0 seconds, avg ~2.7
    time_to_throw = rng.normal(loc=2.7, scale=0.6, size=n).clip(1.0, 6.0)

    # ── Rusher Features ───────────────────────────────────────────────────
    # nearest_rusher_dist: 2-10 yards at release
    nearest_rusher_dist = rng.gamma(shape=3.0, scale=1.5, size=n).clip(0.5, 15)

    # rusher approach speeds
    rusher_1_approach_speed = rng.gamma(shape=2.0, scale=1.2, size=n).clip(0, 10)
    rusher_2_approach_speed = rng.gamma(shape=1.8, scale=1.1, size=n).clip(0, 10)
    # ~5% of plays have no second rusher
    no_second_rusher = rng.random(n) < 0.05
    rusher_2_approach_speed[no_second_rusher] = np.nan

    # rushers within 3 yards: usually 0-2
    rushers_within_3yds = rng.choice([0, 1, 2, 3], size=n, p=[0.55, 0.30, 0.12, 0.03])

    # ── Pocket Features ───────────────────────────────────────────────────
    # pocket_area_at_release: convex hull area of OL, typically 15-40 sq yards
    pocket_area_at_release = rng.normal(loc=25, scale=7, size=n).clip(5, 55)

    # pocket_collapse_rate: positive = pocket shrinking, negative = expanding
    pocket_collapse_rate = rng.normal(loc=3.0, scale=4.0, size=n).clip(-10, 20)

    # ── Correlate features with EPA slightly (model needs signal to learn) ─
    # Better pocket + more time = higher EPA tendency
    epa_signal = (
        - 0.15 * rushers_within_3yds
        - 0.08 * np.where(np.isnan(rusher_2_approach_speed), 0, rusher_2_approach_speed)
        + 0.05 * pocket_area_at_release / 25
        - 0.10 * np.maximum(0, time_to_throw - 3.5)  # penalty for holding too long
        + 0.08 * nearest_rusher_dist / 5
    )
    # Blend signal into EPA (keep mostly real EPA values)
    epa_adjusted = pbp["epa"].values + 0.3 * epa_signal

    # ── Assemble DataFrame ────────────────────────────────────────────────
    features_df = pd.DataFrame({
        # Identifiers
        "game_id"                   : pbp["game_id"].values,
        "play_id"                   : pbp["play_id"].values,
        "passer_player_name"        : pbp["passer_player_name"].values,
        "week"                      : pbp["week"].values,
        "situation"                 : pbp["situation"].values,
        "epa"                       : epa_adjusted,

        # QB spatial features
        "qb_displacement"           : qb_displacement,
        "qb_speed_at_release"       : qb_speed_at_release,
        "qb_orientation_sin"        : qb_orientation_sin,
        "qb_orientation_cos"        : qb_orientation_cos,
        "time_to_throw"             : time_to_throw,

        # Rusher features
        "nearest_rusher_dist"       : nearest_rusher_dist,
        "rusher_1_approach_speed"   : rusher_1_approach_speed,
        "rusher_2_approach_speed"   : rusher_2_approach_speed,
        "rushers_within_3yds"       : rushers_within_3yds,

        # Pocket features
        "pocket_area_at_release"    : pocket_area_at_release,
        "pocket_collapse_rate"      : pocket_collapse_rate,

        # Context features from real PBP
        "down"                      : pbp["down"].values,
        "ydstogo"                   : pbp["ydstogo"].values,
        "yardline_100"              : pbp["yardline_100"].values if "yardline_100" in pbp.columns else np.nan,
        "score_differential"        : pbp["score_differential"].values if "score_differential" in pbp.columns else np.nan,
        "half_seconds_remaining"    : pbp["half_seconds_remaining"].values if "half_seconds_remaining" in pbp.columns else np.nan,
        "wp"                        : pbp["wp"].values,
    })

    # ── Introduce ~5% nulls in spatial features (realistic for edge cases) ─
    spatial_cols = [
        "qb_displacement", "qb_speed_at_release", "qb_orientation_sin", "qb_orientation_cos",
        "time_to_throw", "nearest_rusher_dist", "rusher_1_approach_speed",
        "pocket_area_at_release", "pocket_collapse_rate"
    ]
    for col in spatial_cols:
        null_mask = rng.random(n) < 0.04
        features_df.loc[null_mask, col] = np.nan

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✓ Saved mock features to {out_path}")
    print(f"  Shape : {features_df.shape[0]:,} rows × {features_df.shape[1]} columns")
    print(f"  Size  : {size_mb:.1f} MB")
    print(f"\n  Situation breakdown:")
    for sit, cnt in features_df["situation"].value_counts().items():
        print(f"    {sit:<12} {cnt:>5,}")
    print(f"\n  Null rates in spatial features:")
    for col in spatial_cols + ["rusher_2_approach_speed"]:
        pct = features_df[col].isnull().mean() * 100
        print(f"    {col:<35} {pct:.1f}%")
    print(f"\n  Week distribution:")
    for wk, cnt in features_df["week"].value_counts().sort_index().items():
        print(f"    Week {wk}: {cnt:>4,} plays")

    print(f"\nNow run: python 03_model_training.py")


if __name__ == "__main__":
    main()
