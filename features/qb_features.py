"""
features/qb_features.py
-----------------------
QB spatial features for one play.
"""

import numpy as np


def get_qb_features(snap_frame, release_frame, qb_nfl_id):
    features = {
        "qb_displacement": np.nan,
        "qb_speed_at_release": np.nan,
        "qb_orientation_at_release": np.nan,
        "time_to_throw": np.nan,
    }

    qb_snap = snap_frame[snap_frame["nflId"] == qb_nfl_id]
    qb_release = release_frame[release_frame["nflId"] == qb_nfl_id]

    if qb_snap.empty or qb_release.empty:
        return features

    qb_snap = qb_snap.iloc[0]
    qb_release = qb_release.iloc[0]

    dx = qb_release["x"] - qb_snap["x"]
    dy = qb_release["y"] - qb_snap["y"]

    features["qb_displacement"] = float(np.sqrt(dx**2 + dy**2))
    features["qb_speed_at_release"] = float(qb_release["s"])
    features["qb_orientation_at_release"] = float(qb_release["o"])

    snap_frame_id = qb_snap["frameId"]
    release_frame_id = qb_release["frameId"]

    if release_frame_id > snap_frame_id:
        features["time_to_throw"] = float((release_frame_id - snap_frame_id) / 10)

    return features
