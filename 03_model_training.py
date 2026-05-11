"""
03_model_training.py
--------------------
Train an XGBoost regressor to predict EPA per play from spatial pocket features.

Inputs:
    data/processed/features.parquet   (produced by 02_feature_engineering.py)

Outputs:
    outputs/model.json                          trained XGBoost model
    outputs/tables/play_predictions.csv         game_id, play_id, epa, predicted_epa

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

# Columns that are identifiers / targets, not features
NON_FEATURE_COLS = {
    "game_id",
    "play_id",
    "passer_player_name",
    "situation",
    "epa",
    "week",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    """Load the feature parquet and do basic sanity checks."""
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

    # Drop rows with any null feature values
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=feature_cols + ["epa"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with null feature/epa values ({dropped/before:.1%})")

    return df


def split_by_week(df: pd.DataFrame):
    """
    Train  : weeks 1–6
    Val    : week 7
    Test   : week 8
    """
    train = df[df["week"] <= 6].copy()
    val   = df[df["week"] == 7].copy()
    test  = df[df["week"] == 8].copy()
    print(f"\nSplit summary:")
    print(f"  Train (weeks 1–6) : {len(train):,} plays")
    print(f"  Val   (week 7)    : {len(val):,} plays")
    print(f"  Test  (week 8)    : {len(test):,} plays")
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of feature columns (everything that isn't metadata/target)."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return cols


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
    Grid search over a small but meaningful hyperparameter space.
    Uses validation RMSE as the selection criterion.

    Returns the best hyperparameter dict.
    """
    param_grid = {
        "n_estimators" : [200, 400, 600],
        "max_depth"    : [3, 5, 7],
        "learning_rate": [0.05, 0.10],
        "subsample"    : [0.7, 1.0],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    best_rmse   = float("inf")
    best_params = {}
    results     = []

    total = 1
    for v in values:
        total *= len(v)
    print(f"\nTuning over {total} hyperparameter combinations…")

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model  = xgb.XGBRegressor(
            **params,
            objective    = "reg:squarederror",
            random_state = 42,
            n_jobs       = -1,
            verbosity    = 0,
        )
        model.fit(
            X_train, y_train,
            eval_set              = [(X_val, y_val)],
            verbose               = False,
        )
        val_rmse = rmse(y_val, model.predict(X_val))
        results.append({**params, "val_rmse": val_rmse})

        if val_rmse < best_rmse:
            best_rmse   = val_rmse
            best_params = params
            print(f"  ✓ New best  val RMSE = {val_rmse:.4f}  |  {params}")

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best val RMSE        : {best_rmse:.4f}")
    return best_params


# ── Training ───────────────────────────────────────────────────────────────────

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict,
) -> xgb.XGBRegressor:
    """Retrain on train+val combined with the best hyperparameters."""
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    model = xgb.XGBRegressor(
        **best_params,
        objective    = "reg:squarederror",
        random_state = 42,
        n_jobs       = -1,
        verbosity    = 0,
    )
    model.fit(X_combined, y_combined)
    return model


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

    # 1. Load data
    df = load_and_validate(FEATURES_PATH)

    # 2. Split
    train, val, test = split_by_week(df)
    feature_cols     = get_feature_cols(df)
    print(f"\nFeature columns ({len(feature_cols)}):\n  {feature_cols}")

    X_train, y_train = train[feature_cols], train["epa"]
    X_val,   y_val   = val[feature_cols],   val["epa"]
    X_test,  y_test  = test[feature_cols],  test["epa"]

    # 3. Tune hyperparameters on validation set
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # 4. Retrain on train + val combined
    print("\nRetraining on train + val with best hyperparameters…")
    final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # 5. Evaluate on held-out test set
    test_preds = final_model.predict(X_test)
    test_rmse  = rmse(y_test, test_preds)
    val_preds  = final_model.predict(X_val)
    val_rmse   = rmse(y_val, val_preds)

    print(f"\n{'='*45}")
    print(f"  Val  RMSE : {val_rmse:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")
    print(f"{'='*45}")

    # 6. Feature importances
    print_feature_importances(final_model, feature_cols)

    # 7. Save model
    final_model.save_model(MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    # 8. Generate predictions for ALL plays (needed by 04_analysis.py)
    all_preds = final_model.predict(df[feature_cols])
    predictions_df = pd.DataFrame({
        "game_id"       : df["game_id"].values,
        "play_id"       : df["play_id"].values,
        "epa"           : df["epa"].values,
        "predicted_epa" : all_preds,
    })
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    print(f"Predictions saved → {PREDICTIONS_OUT}  ({len(predictions_df):,} rows)")

    # 9. Summary report
    summary = {
        "best_hyperparameters": best_params,
        "val_rmse" : round(val_rmse,  4),
        "test_rmse": round(test_rmse, 4),
        "n_train"  : len(train),
        "n_val"    : len(val),
        "n_test"   : len(test),
        "features" : feature_cols,
    }
    summary_path = "outputs/tables/model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved  → {summary_path}")


if __name__ == "__main__":
    main()
