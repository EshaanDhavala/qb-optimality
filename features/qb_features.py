"""
features/qb_features.py
-----------------------
STUB — Dillon replaces this with the real implementation.

Interface contract (do not change the signature or return keys):
    get_qb_features(snap_frame, release_frame, qb_nfl_id) -> dict

Args:
    snap_frame    : pd.DataFrame — all tracking rows at the ball_snap frame for
                    this play (columns: nflId, x, y, s, a, o, dir, frameId, event, ...)
    release_frame : pd.DataFrame — all tracking rows at the pass_forward frame
    qb_nfl_id     : int — the QB's nflId (from players.csv)

Returns dict with exactly these keys (float, NaN if uncomputable):
    qb_displacement           — Euclidean distance (yards) from snap pos to release pos
    qb_speed_at_release       — QB speed (yards/s) from `s` column at release frame
    qb_orientation_at_release — QB body orientation (degrees) from `o` column at release frame
    time_to_throw             — seconds from snap frame to release frame

Edge cases to handle:
    - QB not found in snap_frame or release_frame  →  return NaN for affected features
    - release frameId <= snap frameId              →  time_to_throw = NaN
"""

import numpy as np


def get_qb_features(snap_frame, release_frame, qb_nfl_id):
    # TODO (Dillon): replace stub body with real implementation
    return {
        "qb_displacement":           np.nan,
        "qb_speed_at_release":       np.nan,
        "qb_orientation_at_release": np.nan,
        "time_to_throw":             np.nan,
    }
