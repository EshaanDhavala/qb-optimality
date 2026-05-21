import math
from operator import add
import pandas as pd
import numpy as np
 
 
def get_rusher_features(
    release_frame: pd.DataFrame,
    qb_nfl_id: int,
    pff_play_df: pd.DataFrame,
) -> dict:
    nan = float("nan")
 
    # 1. Locate the QB in the release frame
    qb_rows = release_frame[release_frame["nflId"] == qb_nfl_id]
    if qb_rows.empty:
        raise ValueError(
            f"QB nflId {qb_nfl_id} not found in release_frame. "
            "Check that the correct frame and player ID were passed."
        )
    qb = qb_rows.iloc[0]
    qb_x, qb_y = float(qb["x"]), float(qb["y"])

    # 2. Identify pass rushers via PFF role tags
    rusher_ids = pff_play_df.loc[
        pff_play_df["pff_role"] == "Pass Rush", "nflId"
    ].unique()
    if len(rusher_ids) == 0:
        # No rushers tagged — return all-NaN / zero result
        return {
            "nearest_rusher_dist": nan,
            "rusher_1_approach_speed": nan,
            "rusher_2_approach_speed": nan,
            "rushers_within_3yds": 0,
        }

    # 3. Pull rusher rows from the release frame and compute distances
    rusher_frame = release_frame[release_frame["nflId"].isin(rusher_ids)].copy()
    if rusher_frame.empty:
        # Rushers exist in PFF data but not in tracking snapshot (edge case)
        return {
            "nearest_rusher_dist": nan,
            "rusher_1_approach_speed": nan,
            "rusher_2_approach_speed": nan,
            "rushers_within_3yds": 0,
        }
    rusher_frame = rusher_frame.copy()
    rusher_frame["dist_to_qb"] = rusher_frame.apply(
        lambda row: math.hypot(float(row["x"]) - qb_x, float(row["y"]) - qb_y),
        axis=1,
    )
    # Sort ascending by distance so index 0 = nearest
    rusher_frame = rusher_frame.sort_values("dist_to_qb").reset_index(drop=True)

    # 4. Derive the four return features
    nearest_rusher_dist = float(rusher_frame.loc[0, "dist_to_qb"])
    rusher_1_approach_speed = float(rusher_frame.loc[0, "s"])
    if len(rusher_frame) >= 2:
        rusher_2_approach_speed = float(rusher_frame.loc[1, "s"])
    else:
        rusher_2_approach_speed = nan
    rushers_within_3yds = int((rusher_frame["dist_to_qb"] < 3.0).sum())

    return {
        "nearest_rusher_dist": nearest_rusher_dist,
        "rusher_1_approach_speed": rusher_1_approach_speed,
        "rusher_2_approach_speed": rusher_2_approach_speed,
        "rushers_within_3yds": rushers_within_3yds,
    }