"""
features/pocket_features.py
---------------------------
Pocket spatial features for one play.

Interface contract:
    get_pocket_features(snap_frame, release_frame, players_df) -> dict

Args:
    snap_frame    : pd.DataFrame — all tracking rows at the ball_snap frame for
                    this play (columns: nflId, x, y, s, a, o, dir, frameId, event, ...)
    release_frame : pd.DataFrame — all tracking rows at the pass_forward frame

Returns dict with exactly these keys (float, NaN if uncomputable):
    pocket_area_at_release — convex hull area (sq yards) of blocker positions at release frame
    pocket_collapse_rate   — (snap_area - release_area) / time_elapsed  (sq yards/s)
                             positive = pocket shrinking, negative = expanding, NaN if uncomputable

Edge cases:
    - Fewer than 3 blockers detected  →  both NaN
    - time_elapsed == 0               →  pocket_collapse_rate = NaN
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# Lazy-loaded fallback so the module can be imported without the data file present.
# 02_feature_engineering.py passes pff_play_df directly, so this is only used
# when running pocket_features.py standalone (the __main__ block below).
_PFF_DF_CACHE: pd.DataFrame | None = None

def _get_pff_df() -> pd.DataFrame:
    global _PFF_DF_CACHE
    if _PFF_DF_CACHE is None:
        _PFF_DF_CACHE = pd.read_csv("data/raw/big_data_bowl_2023/pffScoutingData.csv")
    return _PFF_DF_CACHE


def get_pocket_features(snap_frame, release_frame, pff_play_df=None):
    """
    pff_play_df: pre-filtered PFF rows for this play (gameId+playId already filtered).
                 Pass this from 02_feature_engineering.py to avoid reloading the full
                 PFF file. If None, the full file is loaded and filtered internally
                 (slower; only happens in standalone/test usage).
    """
    if pff_play_df is None:
        game_id  = snap_frame["gameId"].iloc[0]
        play_id  = snap_frame["playId"].iloc[0]
        full_pff = _get_pff_df()
        play_pff = full_pff[(full_pff["gameId"] == game_id) & (full_pff["playId"] == play_id)]
    else:
        play_pff = pff_play_df
    # "Pass" is the QB/passer role — including it would add the QB himself to the
    # pocket hull and inflate the area. Only "Pass Block" tags OL and blocking TEs.
    blocker_ids = set(play_pff[play_pff["pff_role"] == "Pass Block"]["nflId"].tolist())

    filtered_snaps    = snap_frame[snap_frame["nflId"].isin(blocker_ids)]
    filtered_releases = release_frame[release_frame["nflId"].isin(blocker_ids)]

    def convex_hull_area(df):
        coords = df[["x", "y"]].values
        if len(coords) < 3:
            return np.nan
        try:
            return float(ConvexHull(coords).volume)
        except Exception:
            return np.nan

    snap_area    = convex_hull_area(filtered_snaps)
    release_area = convex_hull_area(filtered_releases)

    pocket_collapse_rate = np.nan
    if (not filtered_snaps.empty and not filtered_releases.empty
            and not np.isnan(snap_area) and not np.isnan(release_area)):
        t1 = pd.to_datetime(filtered_snaps["time"].iloc[0])
        t2 = pd.to_datetime(filtered_releases["time"].iloc[0])
        seconds_elapsed = (t2 - t1).total_seconds()
        if seconds_elapsed > 0:
            # Positive = pocket shrinking (snap > release); negative = expanding
            pocket_collapse_rate = (snap_area - release_area) / seconds_elapsed

    return {
        "pocket_area_at_release": np.nan if np.isnan(release_area) else round(release_area, 2),
        "pocket_collapse_rate":   np.nan if np.isnan(pocket_collapse_rate) else round(pocket_collapse_rate, 2),
    }


if __name__ == "__main__":
    tracking1 = pd.read_csv("data/raw/big_data_bowl_2023/tracking_week_1.csv")

    snaps    = tracking1[tracking1["event"] == "ball_snap"]
    releases = tracking1[tracking1["event"] == "pass_forward"]

    common_plays = list(
        set(zip(snaps["gameId"], snaps["playId"])) &
        set(zip(releases["gameId"], releases["playId"]))
    )

    if not common_plays:
        print("Warning: no matching plays found — check data filters.")
    else:
        print("--- Testing 10 plays ---")
        for i, (game_id, play_id) in enumerate(common_plays[:10]):
            snap_f    = tracking1[(tracking1["gameId"] == game_id) & (tracking1["playId"] == play_id) & (tracking1["event"] == "ball_snap")]
            release_f = tracking1[(tracking1["gameId"] == game_id) & (tracking1["playId"] == play_id) & (tracking1["event"] == "pass_forward")]
            result = get_pocket_features(snap_f, release_f)
            print(f"Play #{i+1} | Game: {game_id} | Play: {play_id} | {result}")
