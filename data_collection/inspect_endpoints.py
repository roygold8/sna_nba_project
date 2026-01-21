from nba_api.stats.endpoints import playerdashptpass, teamdashptpass
import pandas as pd
import time

# Set up pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def inspect_endpoint(name, endpoint_class, **kwargs):
    print(f"\n--- {name} ---")
    try:
        # Some endpoints might require specific parameters
        req = endpoint_class(**kwargs)
        # The 'get_data_frames' method returns a list of dfs
        dfs = req.get_data_frames()
        for i, df in enumerate(dfs):
            print(f"DataFrame {i}: Shape {df.shape}")
            print(f"Columns: {list(df.columns)}")
            if not df.empty:
                print(df.head(2))
    except Exception as e:
        print(f"Error fetching {name}: {e}")

# 1. Inspect PlayerDashPtPass (known good, but expensive)
# Need a valid PlayerID. LeBron James = 2544
inspect_endpoint("PlayerDashPtPass", playerdashptpass.PlayerDashPtPass, team_id=0, player_id=2544, season='2023-24')

# 2. Inspect TeamDashPtPass (potential optimization)
# Need a valid TeamID. Lakers = 1610612747
inspect_endpoint("TeamDashPtPass", teamdashptpass.TeamDashPtPass, team_id=1610612747, season='2023-24')
