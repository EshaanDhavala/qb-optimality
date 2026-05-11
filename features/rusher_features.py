"""
features/rusher_features.py
---------------------------
STUB — Gonzalo replaces this with the real implementation.

Interface contract (do not change the signature or return keys):
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

Edge cases to handle:
    - No players tagged pff_role == "Pass Rush"  →  all NaN (rushers_within_3yds = NaN)
    - Fewer than 2 rushers                       →  rusher_2_approach_speed = NaN
    - Rusher nflId not found in release_frame     →  skip that rusher
"""

import numpy as np


def get_rusher_features(release_frame, qb_nfl_id, pff_play_df):
    # TODO (Gonzalo): replace stub body with real implementation
    return {
        "nearest_rusher_dist":      np.nan,
        "rusher_1_approach_speed":  np.nan,
        "rusher_2_approach_speed":  np.nan,
        "rushers_within_3yds":      np.nan,
    }
