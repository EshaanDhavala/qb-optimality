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
import pandas as pd
from scipy.spatial import ConvexHull

players_df = pd.read_csv("data/raw/big_data_bowl_2023/players.csv")
plays_df = pd.read_csv("data/raw/big_data_bowl_2023/plays.csv")
tracking1 = pd.read_csv("data/raw/big_data_bowl_2023/tracking_week_1.csv")
PFF_DF = pd.read_csv("data/raw/big_data_bowl_2023/pffScoutingData.csv")


snaps = tracking1[tracking1['event'] == 'ball_snap']
releases = tracking1[tracking1['event'] == 'pass_forward']

snap_plays = set(zip(snaps['gameId'], snaps['playId']))
release_plays = set(zip(releases['gameId'], releases['playId']))
common_plays = list(snap_plays.intersection(release_plays))

if not common_plays:
    print("Warning: No matching plays found between snaps and releases. Double-check your data filters.")
else:
    # 2. Grab the very first matching play (Game ID, Play ID) tuple
    sample_game_id, sample_play_id = common_plays[0]
    
    print(f"Testing with Game ID: {sample_game_id}, Play ID: {sample_play_id}")

    # 3. Filter the tracking data to get EVERY row belonging to that exact play moment
    mock_snap = tracking1[
        (tracking1['gameId'] == sample_game_id) & 
        (tracking1['playId'] == sample_play_id) & 
        (tracking1['event'] == 'ball_snap')
    ]
    
    mock_releases = tracking1[
        (tracking1['gameId'] == sample_game_id) & 
        (tracking1['playId'] == sample_play_id) & 
        (tracking1['event'] == 'pass_forward')
    ]


def get_pocket_features(snap_frame, release_frame, players_df):
    # TODO (Abhi): replace stub body with real implementation

    import pandas as pd
    import numpy as np
    from scipy.spatial import ConvexHull
    # Load the PFF Scouting Data

    game_id = snap_frame['gameId'].iloc[0]
    play_id = snap_frame['playId'].iloc[0]
    
    play_pff = PFF_DF[(PFF_DF['gameId'] == game_id) & (PFF_DF['playId'] == play_id)]
    valid_ids = play_pff[play_pff['pff_role'].isin(['Pass Block', 'Pass'])]['nflId'].tolist()

    blockers = players_df[players_df['nflId'].isin(valid_ids)]
    print(blockers.shape)
    filtered_snaps = snap_frame.merge(blockers[['nflId']], on='nflId', how='inner').sort_values(by  = ['playId'])
    filtered_releases = release_frame.merge(blockers[['nflId']], on='nflId', how='inner').sort_values(by  = ['playId'])

    def calculate_area(df):
        coords = df[['x', 'y']].values
        if len(coords) < 3:
            return np.nan
        try:
            return ConvexHull(coords).volume 
        except:
            return np.nan

    # Calculate Areas
    snap_area = calculate_area(filtered_snaps)
    release_area = calculate_area(filtered_releases)
    
    # Calculate Time Elapsed using timestamps
    t1 = pd.to_datetime(filtered_snaps['time'].iloc[0])
    t2 = pd.to_datetime(filtered_releases['time'].iloc[0])
    
    # .total_seconds() 
    seconds_elapsed = (t2 - t1).total_seconds()
    
    # Calculate Rate
    pocket_collapse_rate = np.nan
    if seconds_elapsed > 0 and not np.isnan(snap_area) and not np.isnan(release_area):
        # Rate of change: (Final - Initial) / Time
        # A negative number means the pocket is shrinking
        pocket_collapse_rate = (release_area - snap_area) / seconds_elapsed

    return {
        "pocket_area_at_snap": np.round(float(snap_area), 2),
        "pocket_area_at_release": np.round(float(release_area), 2),
        "pocket_collapse_rate": np.round(float(pocket_collapse_rate), 2)
    }
    


# 4. Loop through the first 10 plays to see a variety of results
print("\n--- Testing a Batch of 10 Plays ---")

for i, (game_id, play_id) in enumerate(common_plays[:10]):
    # Extract frames for this specific play
    snap_f = tracking1[(tracking1['gameId'] == game_id) & (tracking1['playId'] == play_id) & (tracking1['event'] == 'ball_snap')]
    release_f = tracking1[(tracking1['gameId'] == game_id) & (tracking1['playId'] == play_id) & (tracking1['event'] == 'pass_forward')]
    
    # Run the function
    result = get_pocket_features(snap_f, release_f, players_df)
    
    print(f"Play #{i+1} | Game: {game_id} | Play ID: {play_id}")
    print(f"  -> {result}\n")


    
    