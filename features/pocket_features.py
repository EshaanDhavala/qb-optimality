"""
features/pocket_features.py
---------------------------
STUB — Abhi replaces this with the real implementation.

Interface contract (do not change the signature or return keys):
    get_pocket_features(snap_frame, release_frame, players_df) -> dict

Args:
    snap_frame    : pd.DataFrame — all tracking rows at the ball_snap frame for
                    this play (columns: nflId, x, y, s, a, o, dir, frameId, event, ...)
    release_frame : pd.DataFrame — all tracking rows at the pass_forward frame
    players_df    : pd.DataFrame — full players.csv table
                    (columns: nflId, displayName, position, ...)
                    Used to identify offensive linemen by position in ("T", "G", "C")

Returns dict with exactly these keys (float, NaN if uncomputable):
    pocket_area_at_release — convex hull area (sq yards) of OL positions at release frame
    pocket_collapse_rate   — (snap_area - release_area) / time_elapsed  (sq yards/s)

Edge cases to handle:
    - Fewer than 3 OL detected in frame  →  both NaN (can't form a convex hull)
    - OL tagged as eligible receiver and split wide  →  exclude from hull
    - snap_area == release_area (no collapse)  →  pocket_collapse_rate = 0.0
    - time_elapsed == 0  →  pocket_collapse_rate = NaN
"""

import numpy as np


def get_pocket_features(snap_frame, release_frame, players_df):
    # TODO (Abhi): replace stub body with real implementation
    return {
        "pocket_area_at_release": np.nan,
        "pocket_collapse_rate":   np.nan,
    }
