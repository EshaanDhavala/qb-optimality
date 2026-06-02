"""
04_analysis.py
--------------
Compute per-play Optimality Scores and per-QB Clutch Optimality Ratings.

Optimality Score    = epa - predicted_epa                    (per play)
Clutch Rating       = mean(optimality | clutch plays)
                    − mean(optimality | all plays for that QB)

The baseline is the QB's OWN overall mean optimality, not the league average.
This measures: "Does QB X perform above their own typical level in clutch
situations?" rather than "Does QB X outperform blowout-game QB X?"

Inputs:
    outputs/tables/play_predictions.csv   (produced by 03_model_training.py)
    data/raw/pbp_2021.parquet             (produced by 00_download_pbp.py)

Outputs:
    outputs/tables/qb_clutch_ratings.csv

Pipeline position:
    03_model_training.py  →  [THIS SCRIPT]  →  05_clutch_rating.py
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = "outputs/tables/play_predictions.csv"
PBP_PATH         = "data/raw/pbp_2021.parquet"
OUTPUT_PATH      = "outputs/tables/qb_clutch_ratings.csv"

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_CLUTCH_PLAYS = 10
MIN_TOTAL_PLAYS  = 50


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Predictions file not found at '{path}'. "
            "Run 03_model_training.py first."
        )
    df = pd.read_csv(path)
    required = {"game_id", "play_id", "epa", "predicted_epa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"play_predictions.csv missing columns: {missing}")
    df["play_id"] = df["play_id"].astype(int)
    print(f"Loaded {len(df):,} predictions from {path}")
    return df


def load_pbp_context(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PBP file not found at '{path}'. Run 00_download_pbp.py first."
        )
    pbp = pd.read_parquet(path)
    required = {"game_id", "play_id", "passer_player_name", "situation", "week"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"pbp_2021.parquet missing columns: {missing}")
    pbp = pbp[["game_id", "play_id", "passer_player_name", "situation", "week"]].copy()
    pbp["play_id"] = pbp["play_id"].astype(int)
    print(f"Loaded {len(pbp):,} PBP context rows from {path}")
    return pbp


# ── Transformations ────────────────────────────────────────────────────────────

def merge_context(preds: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(pbp, on=["game_id", "play_id"], how="inner")
    lost = len(preds) - len(merged)
    if lost:
        print(f"  Warning: {lost} prediction rows had no PBP match (dropped)")
    print(f"  After PBP merge: {len(merged):,} plays")
    return merged


def compute_optimality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["optimality_score"] = df["epa"] - df["predicted_epa"]
    return df


def aggregate_by_qb(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each QB compute:
      - overall_optimality : mean optimality across ALL their plays
      - clutch_optimality  : mean optimality in clutch plays only
      - play counts for both
    """
    overall = (
        df.groupby("passer_player_name")
          .agg(overall_optimality=("optimality_score", "mean"),
               n_total_plays     =("optimality_score", "size"))
          .reset_index()
    )
    clutch = (
        df[df["situation"] == "clutch"]
          .groupby("passer_player_name")
          .agg(clutch_optimality=("optimality_score", "mean"),
               n_clutch_plays   =("optimality_score", "size"))
          .reset_index()
    )
    return overall.merge(clutch, on="passer_player_name", how="inner")


def compute_clutch_rating(grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimum-sample filters and compute clutch_rating.

    clutch_rating = clutch_optimality − overall_optimality
    Positive = QB performs above their own average in clutch situations.
    """
    before = len(grouped)
    out = grouped[
        (grouped["n_clutch_plays"] >= MIN_CLUTCH_PLAYS) &
        (grouped["n_total_plays"]  >= MIN_TOTAL_PLAYS)
    ].copy()
    print(f"  QBs before threshold: {before}")
    print(f"  QBs after  threshold (≥{MIN_CLUTCH_PLAYS} clutch, "
          f"≥{MIN_TOTAL_PLAYS} total): {len(out)}")

    out["clutch_rating"] = out["clutch_optimality"] - out["overall_optimality"]
    out = out[[
        "passer_player_name",
        "clutch_optimality", "overall_optimality",
        "clutch_rating",
        "n_clutch_plays", "n_total_plays",
    ]]
    return out.sort_values("clutch_rating", ascending=False).reset_index(drop=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_top_bottom(ratings: pd.DataFrame, n: int = 10):
    if ratings.empty:
        print("\n  (no QBs cleared the threshold — nothing to display)")
        return
    n = min(n, len(ratings))
    print(f"\nTop {n} QBs by Clutch Optimality Rating:")
    print(ratings.head(n).to_string(index=False))
    print(f"\nBottom {n} QBs by Clutch Optimality Rating:")
    print(ratings.tail(n).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    preds = load_predictions(PREDICTIONS_PATH)
    pbp   = load_pbp_context(PBP_PATH)

    merged   = merge_context(preds, pbp)
    scored   = compute_optimality(merged)
    grouped  = aggregate_by_qb(scored)
    ratings  = compute_clutch_rating(grouped)

    ratings.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved: {OUTPUT_PATH}  ({len(ratings):,} QBs)")

    print_top_bottom(ratings)


if __name__ == "__main__":
    main()
