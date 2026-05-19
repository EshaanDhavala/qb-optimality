"""
features/qb_features.py
-----------------------
QB spatial features for one play.
"""

import numpy as np


def get_qb_features(snap_frame, release_frame, qb_nfl_id):
    features = {
        "qb_displacement":     np.nan,
        "qb_speed_at_release": np.nan,
        "qb_orientation_sin":  np.nan,  # sin/cos encode circular orientation correctly:
        "qb_orientation_cos":  np.nan,  # 1° and 359° are similar, 358 apart as raw degrees
        "time_to_throw":       np.nan,
    }

    qb_snap    = snap_frame[snap_frame["nflId"] == qb_nfl_id]
    qb_release = release_frame[release_frame["nflId"] == qb_nfl_id]

    if qb_snap.empty or qb_release.empty:
        return features

    qb_snap    = qb_snap.iloc[0]
    qb_release = qb_release.iloc[0]

    dx = qb_release["x"] - qb_snap["x"]
    dy = qb_release["y"] - qb_snap["y"]

    # np.minimum propagates NaN correctly; Python's min() does not (min(nan,15)→15)
    # Clip extremes that indicate bad frame data (realistic max ~12 yards pocket movement)
    features["qb_displacement"]     = float(np.minimum(np.sqrt(dx**2 + dy**2), 15.0))
    # BDB speed column `s` is yards/second; realistic QB max ~12 yds/s sprinting
    features["qb_speed_at_release"] = float(np.minimum(qb_release["s"], 15.0))

    orientation_rad = float(qb_release["o"]) * np.pi / 180.0
    features["qb_orientation_sin"]  = float(np.sin(orientation_rad))
    features["qb_orientation_cos"]  = float(np.cos(orientation_rad))

    snap_frame_id    = qb_snap["frameId"]
    release_frame_id = qb_release["frameId"]

    if release_frame_id > snap_frame_id:
        raw_time = float((release_frame_id - snap_frame_id) / 10)
        # Clip to 10s max — values above this indicate corrupted frame IDs
        features["time_to_throw"] = min(raw_time, 10.0)

    return features
