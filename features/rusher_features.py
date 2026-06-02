"""
features/rusher_features.py
---------------------------
Pass rusher spatial features for one play.

Interface contract:
    get_rusher_features(release_frame, qb_nfl_id, pff_play_df) -> dict

Args:
    release_frame : pd.DataFrame — all tracking rows at the pass_forward frame for
                    this play (columns: nflId, x, y, s, a, o, dir, frameId, event, ...)
    qb_nfl_id     : int — the QB's nflId
    pff_play_df   : pd.DataFrame — rows from pffScoutingData.csv for this play only
                    (columns include: nflId, pff_role, pff_hit, pff_hurry, pff_sack, ...)

Returns dict with exactly these keys:
    nearest_rusher_dist     — float, yards from QB to closest pass rusher at release
    rusher_1_approach_speed — float, speed (yards/s) of the nearest rusher
    rusher_2_approach_speed — float, speed (yards/s) of the 2nd nearest rusher (NaN if < 2)
    rushers_within_3yds     — int (or float), count of rushers within 3.0 yards of QB

Edge cases:
    - No players tagged pff_role == "Pass Rush"  →  all NaN (rushers_within_3yds = NaN)
    - Fewer than 2 rushers                       →  rusher_2_approach_speed = NaN
    - Rusher nflId not found in release_frame     →  skip that rusher
"""

import numpy as np


def get_rusher_features(release_frame, qb_nfl_id, pff_play_df):
    nan_result = {
        "nearest_rusher_dist":      np.nan,
        "rusher_1_approach_speed":  np.nan,
        "rusher_2_approach_speed":  np.nan,
        "rushers_within_3yds":      np.nan,
    }

    qb_row = release_frame[release_frame["nflId"] == qb_nfl_id]
    if qb_row.empty:
        return nan_result
    qb_x = qb_row.iloc[0]["x"]
    qb_y = qb_row.iloc[0]["y"]

    rusher_ids = set(pff_play_df[pff_play_df["pff_role"] == "Pass Rush"]["nflId"].tolist())
    if not rusher_ids:
        return nan_result

    rushers = release_frame[release_frame["nflId"].isin(rusher_ids)].copy()
    if rushers.empty:
        return nan_result

    rushers["dist"] = np.sqrt((rushers["x"] - qb_x) ** 2 + (rushers["y"] - qb_y) ** 2)
    rushers = rushers.sort_values("dist").reset_index(drop=True)

    nearest          = rushers.iloc[0]
    nearest_dist     = float(nearest["dist"])
    rusher_1_speed   = float(nearest["s"])
    rusher_2_speed   = float(rushers.iloc[1]["s"]) if len(rushers) >= 2 else np.nan
    within_3         = int((rushers["dist"] <= 3.0).sum())

    return {
        "nearest_rusher_dist":      round(nearest_dist,   2),
        "rusher_1_approach_speed":  round(rusher_1_speed, 2),
        "rusher_2_approach_speed":  np.nan if np.isnan(rusher_2_speed) else round(rusher_2_speed, 2),
        "rushers_within_3yds":      within_3,
    }
