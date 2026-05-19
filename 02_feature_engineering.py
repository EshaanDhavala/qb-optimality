"""
02_feature_engineering.py
--------------------------
Builds the feature table for the XGBoost model.

For each QB dropback play in weeks 1–8 of the 2021 season this script:
  1. Loads Big Data Bowl 2023 tracking data one week at a time
  2. Isolates ball_snap and pass_forward tracking frames for each play
  3. Calls get_qb_features(), get_rusher_features(), get_pocket_features()
  4. Merges spatial features with PBP context (EPA, win probability, game situation)
  5. Saves data/processed/features.parquet

Depends on:
  data/raw/pbp_2021.parquet                       (00_download_pbp.py)
  data/raw/big_data_bowl_2023/games.csv
  data/raw/big_data_bowl_2023/plays.csv
  data/raw/big_data_bowl_2023/players.csv
  data/raw/big_data_bowl_2023/pffScoutingData.csv
  data/raw/big_data_bowl_2023/tracking_week_1.csv … tracking_week_8.csv

Usage:
    python 02_feature_engineering.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from features.qb_features import get_qb_features
from features.rusher_features import get_rusher_features
from features.pocket_features import get_pocket_features

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
BDB_DIR       = RAW_DIR / "big_data_bowl_2023"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Team abbreviation normalisation: BDB → nflfastR ────────────────────────
# nflfastR uses JAX; some BDB files use JAC. Everything else is the same for
# the 2021 season teams that appear in weeks 1–8.
_TEAM_MAP = {"JAC": "JAX"}

def _norm_team(abbr):
    return _TEAM_MAP.get(str(abbr).upper(), str(abbr).upper())


# ── Game-ID mapping ────────────────────────────────────────────────────────

def build_game_id_map(games_df):
    """
    Map BDB integer gameId → nflfastR string game_id.

    nflfastR format: '2021_01_PHI_ATL'  (visitor_home, zero-padded week).
    BDB games.csv columns used: gameId, week, visitorTeamAbbr, homeTeamAbbr.
    """
    mapping = {}
    for _, row in games_df.iterrows():
        week    = int(row["week"])
        visitor = _norm_team(row["visitorTeamAbbr"])
        home    = _norm_team(row["homeTeamAbbr"])
        mapping[int(row["gameId"])] = f"2021_{week:02d}_{visitor}_{home}"
    return mapping


# ── QB identification ──────────────────────────────────────────────────────

def resolve_qb_nfl_id(game_id_bdb, play_id, plays_df, pff_play_df,
                       players_df, play_tracking):
    """
    Find the QB's nflId for a play using three fallback strategies:
      1. plays.csv 'ballCarrierId' column (present in some BDB releases)
      2. pffScoutingData row where pff_role is a passer role
      3. Tracking data cross-referenced with players.csv QB positions

    Returns int nflId or None if unresolvable.
    """
    mask = (plays_df["gameId"] == game_id_bdb) & (plays_df["playId"] == play_id)
    play_row = plays_df[mask]

    # Strategy 1 — direct column in plays.csv
    for col in ("ballCarrierId", "passerId", "qbNflId"):
        if not play_row.empty and col in play_row.columns:
            val = play_row[col].iloc[0]
            if pd.notna(val):
                return int(val)

    # Strategy 2 — pff_role passer tag
    if not pff_play_df.empty and "pff_role" in pff_play_df.columns:
        passer_rows = pff_play_df[
            pff_play_df["pff_role"].str.lower().isin(["pass", "passer"])
        ]
        if not passer_rows.empty and "nflId" in passer_rows.columns:
            return int(passer_rows["nflId"].iloc[0])

    # Strategy 3 — QB-position players in tracking data
    qb_ids      = set(players_df[players_df["position"] == "QB"]["nflId"].dropna().astype(int))
    tracked_ids = set(play_tracking["nflId"].dropna().astype(int))
    qbs_on_field = qb_ids & tracked_ids

    if len(qbs_on_field) == 1:
        return next(iter(qbs_on_field))

    if len(qbs_on_field) > 1:
        # Multiple QBs tracked (wildcat, injury sub, etc.) — pick the one
        # nearest to the ball at snap.
        snap_rows = play_tracking[play_tracking["event"] == "ball_snap"]
        ball_rows = snap_rows[snap_rows["nflId"].isna()]  # football row
        if not ball_rows.empty:
            bx, by   = ball_rows.iloc[0]["x"], ball_rows.iloc[0]["y"]
            best_id  = min(
                qbs_on_field,
                key=lambda qid: _dist(snap_rows, qid, bx, by),
            )
            return best_id

    return None


def _dist(frame_df, nfl_id, ref_x, ref_y):
    row = frame_df[frame_df["nflId"] == nfl_id]
    if row.empty:
        return float("inf")
    dx = row.iloc[0]["x"] - ref_x
    dy = row.iloc[0]["y"] - ref_y
    return (dx**2 + dy**2) ** 0.5


# ── Frame extraction ───────────────────────────────────────────────────────

def get_snap_and_release_frames(play_tracking):
    """
    Return (snap_frame_df, release_frame_df) for a single play.

    Release frame priority:
      1. pass_forward       (standard pass)
      2. pass_released      (alternate BDB event name)
      3. qb_sacked          (sack plays — pocket collapsed before throw)

    Returns (None, None) if either frame is missing or snap >= release.
    """
    snap_rows    = play_tracking[play_tracking["event"] == "ball_snap"]
    release_rows = play_tracking[play_tracking["event"] == "pass_forward"]
    if release_rows.empty:
        release_rows = play_tracking[play_tracking["event"] == "pass_released"]
    if release_rows.empty:
        release_rows = play_tracking[play_tracking["event"] == "qb_sacked"]

    if snap_rows.empty or release_rows.empty:
        return None, None

    snap_fid    = int(snap_rows["frameId"].min())
    release_fid = int(release_rows["frameId"].min())

    if snap_fid >= release_fid:
        return None, None

    snap_frame    = play_tracking[play_tracking["frameId"] == snap_fid]
    release_frame = play_tracking[play_tracking["frameId"] == release_fid]
    return snap_frame, release_frame


# ── Per-play feature extraction ────────────────────────────────────────────

_QB_DEFAULTS      = {"qb_displacement": np.nan, "qb_speed_at_release": np.nan,
                     "qb_orientation_sin": np.nan, "qb_orientation_cos": np.nan,
                     "time_to_throw": np.nan}
_RUSHER_DEFAULTS  = {"nearest_rusher_dist": np.nan, "rusher_1_approach_speed": np.nan,
                     "rusher_2_approach_speed": np.nan, "rushers_within_3yds": np.nan}
_POCKET_DEFAULTS  = {"pocket_area_at_release": np.nan, "pocket_collapse_rate": np.nan}


def process_play(play_tracking, qb_nfl_id, pff_play_df):
    """
    Extract all spatial features for one play.
    Returns a feature dict, or None if snap/release frames are missing.
    """
    snap_frame, release_frame = get_snap_and_release_frames(play_tracking)
    if snap_frame is None:
        return None

    feats = {}

    try:
        feats.update(get_qb_features(snap_frame, release_frame, qb_nfl_id))
    except Exception:
        feats.update(_QB_DEFAULTS)

    try:
        feats.update(get_rusher_features(release_frame, qb_nfl_id, pff_play_df))
    except Exception:
        feats.update(_RUSHER_DEFAULTS)

    try:
        feats.update(get_pocket_features(snap_frame, release_frame, pff_play_df))
    except Exception:
        feats.update(_POCKET_DEFAULTS)

    return feats


# ── Per-week processing ────────────────────────────────────────────────────

def process_week(week_num, tracking_df, plays_df, pff_df, players_df, game_id_map):
    """
    Process every play in one week's tracking file.
    Returns a list of feature dicts (one per successfully processed play).
    """
    # Index tracking and PFF data by (gameId, playId) for O(1) lookup
    tracking_groups = {
        key: grp.reset_index(drop=True)
        for key, grp in tracking_df.groupby(["gameId", "playId"])
    }
    pff_groups = {
        key: grp.reset_index(drop=True)
        for key, grp in pff_df.groupby(["gameId", "playId"])
    }

    week_game_ids = set(tracking_df["gameId"].unique())
    week_plays    = plays_df[plays_df["gameId"].isin(week_game_ids)]
    total         = len(week_plays)
    print(f"  Week {week_num}: {total} candidate plays")

    records, skipped = [], 0

    for _, play_row in week_plays.iterrows():
        game_id_bdb = int(play_row["gameId"])
        play_id     = int(play_row["playId"])

        nflfastr_id = game_id_map.get(game_id_bdb)
        if nflfastr_id is None:
            skipped += 1
            continue

        play_tracking = tracking_groups.get((game_id_bdb, play_id))
        if play_tracking is None or play_tracking.empty:
            skipped += 1
            continue

        pff_play_df = pff_groups.get((game_id_bdb, play_id), pd.DataFrame())

        qb_nfl_id = resolve_qb_nfl_id(
            game_id_bdb, play_id, plays_df, pff_play_df, players_df, play_tracking
        )
        if qb_nfl_id is None:
            skipped += 1
            continue

        feats = process_play(play_tracking, qb_nfl_id, pff_play_df)
        if feats is None:
            skipped += 1
            continue

        feats["game_id"] = nflfastr_id
        feats["play_id"] = play_id
        feats["week"]    = week_num
        records.append(feats)

    print(f"    → {len(records)} extracted, {skipped} skipped")
    return records


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("02_feature_engineering.py")
    print("=" * 60)

    # ── Validate inputs ────────────────────────────────────────────────────
    required = [
        RAW_DIR / "pbp_2021.parquet",
        BDB_DIR / "games.csv",
        BDB_DIR / "plays.csv",
        BDB_DIR / "players.csv",
        BDB_DIR / "pffScoutingData.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nERROR: Missing required files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun 00_download_pbp.py then download Big Data Bowl data from Kaggle.")
        sys.exit(1)

    # ── Load static tables ─────────────────────────────────────────────────
    print("\nLoading static tables...")
    games_df   = pd.read_csv(BDB_DIR / "games.csv")
    plays_bdb  = pd.read_csv(BDB_DIR / "plays.csv")
    players_df = pd.read_csv(BDB_DIR / "players.csv")
    pff_df     = pd.read_csv(BDB_DIR / "pffScoutingData.csv")
    pbp_df     = pd.read_parquet(RAW_DIR / "pbp_2021.parquet")

    print(f"  games:   {len(games_df):>6,} rows")
    print(f"  plays:   {len(plays_bdb):>6,} rows")
    print(f"  players: {len(players_df):>6,} rows")
    print(f"  PFF:     {len(pff_df):>6,} rows")
    print(f"  PBP:     {len(pbp_df):>6,} rows")

    # ── Build game-ID mapping ──────────────────────────────────────────────
    game_id_map = build_game_id_map(games_df)
    print(f"\n  Built game_id map for {len(game_id_map)} games")
    for bdb_id, nfl_id in list(game_id_map.items())[:3]:
        print(f"    {bdb_id} → {nfl_id}")

    # ── Cross-reference: keep only plays present in both BDB and PBP ──────
    plays_bdb["game_id"] = plays_bdb["gameId"].map(game_id_map)

    pbp_keys = set(zip(pbp_df["game_id"].astype(str), pbp_df["play_id"].astype(int)))

    def in_pbp(row):
        if pd.isna(row["game_id"]):
            return False
        return (str(row["game_id"]), int(row["playId"])) in pbp_keys

    plays_filtered = plays_bdb[plays_bdb.apply(in_pbp, axis=1)].copy()
    print(f"\n  Plays in both BDB + PBP: {len(plays_filtered):,}")

    unmatched = plays_bdb[plays_bdb["game_id"].isna()]
    if not unmatched.empty:
        print(f"  Warning: {len(unmatched)} BDB plays could not be mapped to nflfastR game_id")
        print(f"    (likely team abbreviation mismatch — check _TEAM_MAP at top of script)")

    # ── Process weeks 1–8 ─────────────────────────────────────────────────
    all_records = []

    for week in range(1, 9):
        tracking_path = BDB_DIR / f"tracking_week_{week}.csv"
        if not tracking_path.exists():
            print(f"\nWeek {week}: tracking file not found, skipping")
            continue

        print(f"\nLoading tracking_week_{week}.csv ...")
        tracking_df = pd.read_csv(tracking_path)
        print(f"  {len(tracking_df):,} rows")

        week_records = process_week(
            week, tracking_df, plays_filtered, pff_df, players_df, game_id_map
        )
        all_records.extend(week_records)
        del tracking_df  # free memory before loading next week

    print(f"\nTotal plays processed: {len(all_records):,}")

    if not all_records:
        print("ERROR: No plays were successfully processed.")
        print("Check that tracking files exist and data validation passes.")
        sys.exit(1)

    # ── Assemble feature DataFrame ─────────────────────────────────────────
    features_df = pd.DataFrame(all_records)

    # ── Merge PBP context ─────────────────────────────────────────────────
    print("\nMerging PBP context features...")
    context_cols = [
        "game_id", "play_id",
        "passer_player_name", "passer_player_id",
        "epa", "qb_epa",
        "wp", "wpa",
        "down", "ydstogo", "yardline_100",
        "score_differential", "half_seconds_remaining", "game_seconds_remaining",
        "qtr", "season_type", "situation",
        "posteam", "defteam",
        "sack", "qb_scramble", "interception", "complete_pass",
        "air_yards", "cpoe",
    ]
    context_cols = [c for c in context_cols if c in pbp_df.columns]
    pbp_context  = pbp_df[context_cols].copy()
    pbp_context["play_id"]   = pbp_context["play_id"].astype(int)
    features_df["play_id"]   = features_df["play_id"].astype(int)

    merged = features_df.merge(pbp_context, on=["game_id", "play_id"], how="inner")
    print(f"  {len(features_df):,} plays → {len(merged):,} after PBP merge")

    drop_count = len(features_df) - len(merged)
    if drop_count > 0:
        print(f"  Warning: {drop_count} plays lost in merge — possible play_id mismatch")

    # ── Derive situation from wp (nflfastR doesn't include this column) ───
    # Mirrors the logic in 00_download_pbp.py. Non-clutch is set first so
    # clutch can overwrite it for genuinely critical low-WP plays.
    if "situation" not in merged.columns and "wp" in merged.columns:
        has_qtr   = "qtr" in merged.columns
        has_down  = "down" in merged.columns
        has_sdiff = "score_differential" in merged.columns
        has_gsecs = "game_seconds_remaining" in merged.columns

        merged["situation"] = "neutral"
        merged.loc[
            (merged["wp"] < 0.2) | (merged["wp"] > 0.8), "situation"
        ] = "non_clutch"

        late = merged["qtr"] >= 4 if has_qtr else pd.Series(True, index=merged.index)
        cond1 = (merged["wp"] >= 0.4) & (merged["wp"] <= 0.6) & late
        cond2 = (
            late & (merged["down"] == 4) & (merged["score_differential"].abs() <= 8)
            if (has_qtr and has_down and has_sdiff) else pd.Series(False, index=merged.index)
        )
        cond3 = (
            late & (merged["game_seconds_remaining"] <= 120) &
            (merged["score_differential"] >= -8) & (merged["score_differential"] < 0)
            if (has_qtr and has_gsecs and has_sdiff) else pd.Series(False, index=merged.index)
        )
        merged.loc[cond1 | cond2 | cond3, "situation"] = "clutch"

    # ── Column ordering (matches README spec) ────────────────────────────
    id_cols = ["game_id", "play_id", "passer_player_name", "week", "situation", "epa"]
    spatial_cols = [
        "qb_displacement", "qb_speed_at_release", "qb_orientation_sin", "qb_orientation_cos", "time_to_throw",
        "nearest_rusher_dist", "rusher_1_approach_speed", "rusher_2_approach_speed", "rushers_within_3yds",
        "pocket_area_at_release", "pocket_collapse_rate",
    ]
    context_feature_cols = [
        "down", "ydstogo", "yardline_100", "score_differential",
        "half_seconds_remaining", "game_seconds_remaining",
        "qtr", "wp", "wpa",
        "sack", "qb_scramble", "complete_pass", "air_yards",
    ]
    ordered = [c for c in id_cols + spatial_cols + context_feature_cols if c in merged.columns]
    rest    = [c for c in merged.columns if c not in ordered]
    merged  = merged[ordered + rest]

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "features.parquet"
    merged.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✓ Saved: {out_path}")
    print(f"  Shape:  {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    print(f"  Size:   {size_mb:.1f} MB")

    # ── Situation breakdown ────────────────────────────────────────────────
    if "situation" in merged.columns:
        print(f"\n  Situation breakdown:")
        for sit, n in merged["situation"].value_counts().items():
            print(f"    {sit:<12} {n:>5,}")

    # ── Null report for feature columns ───────────────────────────────────
    null_pct = merged[spatial_cols].isnull().mean() * 100
    high_null = null_pct[null_pct > 10].sort_values(ascending=False)
    if not high_null.empty:
        print(f"\n  Features with >10% nulls (stub columns are expected):")
        for col, pct in high_null.items():
            print(f"    {col:<35} {pct:.0f}%")

    print(f"\nNext: python 03_model_training.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
