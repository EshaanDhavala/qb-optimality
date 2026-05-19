"""
03_model_training.py
--------------------
Train an XGBoost regressor to predict EPA per play from spatial pocket features.

Predictions are generated via leave-one-week-out (LOO) cross-validation so that
every play's predicted_epa comes from a model that never saw that play during
training. This eliminates data leakage and ensures unbiased optimality scores
in 04_analysis.py.

Inputs:
    data/processed/features.parquet   (produced by 02_feature_engineering.py)

Outputs:
    outputs/model.json                     final model trained on all data (for archiving)
    outputs/tables/play_predictions.csv    game_id, play_id, epa, predicted_epa (LOO)
    outputs/tables/model_summary.json      hyperparameters and per-week RMSE

Pipeline position:
    02_feature_engineering.py  →  [THIS SCRIPT]  →  04_analysis.py
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from itertools import product

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH   = "data/processed/features.parquet"
MODEL_OUT       = "outputs/model.json"
PREDICTIONS_OUT = "outputs/tables/play_predictions.csv"

# Columns excluded from model features.
# Identifiers and the target are obvious. Post-play outcomes must be excluded
# because they are only known AFTER the play resolves — using them would mean
# the model predicts EPA from EPA-correlated outcomes, not from the pocket situation.
NON_FEATURE_COLS = {
    # Identifiers
    "game_id", "play_id", "passer_player_name", "passer_player_id",
    "posteam", "defteam", "season_type",
    # Target
    "epa",
    # Grouping metadata
    "situation", "week",
    # Post-play outcomes — known only after the play, must not be features
    "qb_epa",        # QB-credited EPA — directly correlated with the target
    "wpa",           # win probability added — computed from the play's outcome
    "sack",          # whether the QB was sacked
    "qb_scramble",   # whether the QB scrambled
    "interception",  # whether the pass was intercepted
    "complete_pass", # whether the pass was completed
    "air_yards",     # yards the ball travelled in the air — measured post-throw
    "cpoe",          # completion % over expected — outcome metric
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Features file not found at '{path}'. "
            "Run 02_feature_engineering.py first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {path}")

    missing_required = NON_FEATURE_COLS - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=["epa"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with null EPA ({dropped/before:.1%})")

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ── Hyperparameter Tuning ──────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    Grid search over hyperparameter space using weeks 1-6 (train) and week 7 (val).
    Returns the best parameter dict by validation RMSE.
    """
    param_grid = {
        "n_estimators" : [200, 400, 600],
        "max_depth"    : [3, 5, 7],
        "learning_rate": [0.05, 0.10],
        "subsample"    : [0.7, 1.0],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    total  = 1
    for v in values:
        total *= len(v)
    print(f"\nTuning over {total} hyperparameter combinations…")

    best_rmse   = float("inf")
    best_params = {}

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model  = xgb.XGBRegressor(
            **params,
            objective    = "reg:squarederror",
            random_state = 42,
            n_jobs       = -1,
            verbosity    = 0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_rmse = rmse(y_val, model.predict(X_val))

        if val_rmse < best_rmse:
            best_rmse   = val_rmse
            best_params = params
            print(f"  ✓ New best  val RMSE = {val_rmse:.4f}  |  {params}")

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best val RMSE        : {best_rmse:.4f}")
    return best_params


# ── Leave-One-Week-Out Predictions ─────────────────────────────────────────────

def generate_oof_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
) -> pd.DataFrame:
    """
    For each week w, train on all other weeks and predict week w.
    Every play's predicted_epa is produced by a model that never saw it —
    no data leakage into the optimality scores used by 04_analysis.py.

    XGBoost handles NaN feature values natively (learns optimal split direction
    for missing values), so plays with partial features (e.g. screen passes with
    no pocket area) are still scored rather than dropped.
    """
    weeks  = sorted(df["week"].unique())
    chunks = []

    print(f"\nLeave-one-week-out predictions ({len(weeks)} folds):")
    for held_week in weeks:
        train_df = df[df["week"] != held_week]
        held_df  = df[df["week"] == held_week]

        model = xgb.XGBRegressor(
            **best_params,
            objective    = "reg:squarederror",
            random_state = 42,
            n_jobs       = -1,
            verbosity    = 0,
        )
        model.fit(train_df[feature_cols], train_df["epa"])
        preds = model.predict(held_df[feature_cols])

        chunk = held_df[["game_id", "play_id", "epa", "week"]].copy()
        chunk["predicted_epa"] = preds
        chunks.append(chunk)

        week_rmse = rmse(held_df["epa"], preds)
        print(f"  Week {held_week}  RMSE = {week_rmse:.4f}  ({len(held_df):,} plays)")

    return pd.concat(chunks, ignore_index=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_feature_importances(model: xgb.XGBRegressor, feature_cols: list[str]):
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("\nFeature importances (ranked):")
    for feat, score in importances.items():
        bar = "█" * int(score * 60)
        print(f"  {feat:<35} {score:.4f}  {bar}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs/tables", exist_ok=True)

    # 1. Load data — XGBoost handles NaN features natively, so we only drop
    #    rows with missing EPA (the target); partial-feature plays are kept.
    df = load_and_validate(FEATURES_PATH)
    feature_cols = get_feature_cols(df)
    print(f"\nFeature columns ({len(feature_cols)}):\n  {feature_cols}")

    # 2. Tune hyperparameters using weeks 1-6 (train) and week 7 (val)
    train = df[df["week"] <= 6]
    val   = df[df["week"] == 7]
    test  = df[df["week"] == 8]
    print(f"\nHyperparameter tuning split:")
    print(f"  Train (weeks 1–6) : {len(train):,} plays")
    print(f"  Val   (week 7)    : {len(val):,} plays")
    print(f"  Test  (week 8)    : {len(test):,} plays")

    best_params = tune_hyperparameters(
        train[feature_cols], train["epa"],
        val[feature_cols],   val["epa"],
    )

    # 3. Generate LOO predictions — every play predicted by a model that
    #    never saw it, eliminating data leakage into optimality scores
    oof_df = generate_oof_predictions(df, feature_cols, best_params)

    # 4. Report metrics
    week8_oof    = oof_df[oof_df["week"] == 8]
    test_rmse    = rmse(week8_oof["epa"], week8_oof["predicted_epa"])
    overall_rmse = rmse(oof_df["epa"], oof_df["predicted_epa"])

    print(f"\n{'='*45}")
    print(f"  Test RMSE (week 8 LOO)  : {test_rmse:.4f}")
    print(f"  Overall RMSE (all LOO)  : {overall_rmse:.4f}")
    print(f"{'='*45}")

    # 5. Train a final model on ALL data for feature importances + archiving.
    #    This model is NOT used for predictions — LOO predictions are used instead.
    print("\nTraining final model on all data (for feature importances)…")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective    = "reg:squarederror",
        random_state = 42,
        n_jobs       = -1,
        verbosity    = 0,
    )
    final_model.fit(df[feature_cols], df["epa"])
    print_feature_importances(final_model, feature_cols)
    final_model.save_model(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    # 6. Save LOO predictions (used by 04_analysis.py)
    oof_out = oof_df[["game_id", "play_id", "epa", "predicted_epa"]]
    oof_out.to_csv(PREDICTIONS_OUT, index=False)
    print(f"LOO predictions saved → {PREDICTIONS_OUT}  ({len(oof_out):,} rows)")

    # 7. Summary report with per-week RMSE breakdown
    per_week_rmse = {
        f"week_{int(w)}": round(rmse(g["epa"], g["predicted_epa"]), 4)
        for w, g in oof_df.groupby("week")
    }
    summary = {
        "best_hyperparameters": best_params,
        "test_rmse_loo"       : round(test_rmse, 4),
        "overall_rmse_loo"    : round(overall_rmse, 4),
        "per_week_rmse_loo"   : per_week_rmse,
        "n_total_plays"       : len(df),
        "features"            : feature_cols,
    }
    summary_path = "outputs/tables/model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
