import os
import json
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass
from requests.exceptions import ReadTimeout, ConnectionError

# Configuration
START_SEASON = 2014  # Start year (e.g., 2020 means 2020-21 season)
END_SEASON = 2024   # End year (e.g., 2024 means 2024-25 season)
DATA_DIR = 'data'
MIN_GAMES = 20
MIN_MINUTES = 10  # MPG threshold

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_seasons(start_year, end_year):
    """
    Generates a list of NBA season strings from start_year to end_year.
    E.g., start=2020, end=2023 -> ['2020-21', '2021-22', '2022-23', '2023-24']
    """
    seasons = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]  # Last 2 digits of next year
        seasons.append(f"{year}-{next_year}")
    return seasons

def get_filtered_players(season):
    """
    Fetches all players for the season and filters them by Games and MPG.
    Returns a DataFrame of filtered players.
    """
    print(f"Fetching player list for season {season}...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
        df = stats.get_data_frames()[0]
        
        original_count = len(df)
        
        # Calculate average minutes per game
        df['MPG'] = df['MIN'] / df['GP']
        
        df_filtered = df[
            (df['GP'] > MIN_GAMES) & 
            (df['MPG'] >= MIN_MINUTES)
        ].copy()
        
        print(f"  Filtered players: {len(df_filtered)} / {original_count} (GP > {MIN_GAMES}, MPG >= {MIN_MINUTES})")
        return df_filtered
    except Exception as e:
        print(f"  Error fetching player list: {e}")
        return pd.DataFrame()

def fetch_passing_data(player_id, player_name, season, retry_count=3):
    """
    Fetches passing data for a specific player in a specific season.
    """
    file_path = os.path.join(DATA_DIR, season, f"passing_{player_id}.json")
    
    # Check cache
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True  # Already cached
        except:
            print(f"Corrupt cache for {player_name}, re-fetching.")
    
    print(f"  Fetching: {player_name} ({player_id})...")
    
    for attempt in range(retry_count):
        try:
            passing_stats = playerdashptpass.PlayerDashPtPass(
                player_id=player_id, 
                season=season, 
                team_id=0,
                timeout=10
            )
            
            data = passing_stats.get_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f)
                
            time.sleep(0.600)  # Rate limit
            return True
            
        except (ReadTimeout, ConnectionError) as e:
            print(f"  Timeout for {player_name} (Attempt {attempt+1}/{retry_count})")
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"  Error fetching {player_name}: {e}")
            break
            
    return False

def process_season(season):
    """Fetches all data for a single season."""
    print(f"\n{'='*50}")
    print(f"Processing season: {season}")
    print(f"{'='*50}")
    
    ensure_dir(os.path.join(DATA_DIR, season))
    
    players_df = get_filtered_players(season)
    if players_df.empty:
        print(f"  No players found for {season}.")
        return
    
    # Save player list
    players_df.to_csv(os.path.join(DATA_DIR, season, "filtered_players.csv"), index=False)
    
    total = len(players_df)
    cached = 0
    fetched = 0
    
    for i, row in players_df.iterrows():
        pid = row['PLAYER_ID']
        name = row['PLAYER_NAME']
        
        file_path = os.path.join(DATA_DIR, season, f"passing_{pid}.json")
        if os.path.exists(file_path):
            cached += 1
        else:
            fetched += 1
            
        fetch_passing_data(pid, name, season)
        
    print(f"  Done: {cached} cached, {fetched} fetched")

def main():
    seasons = generate_seasons(START_SEASON, END_SEASON)
    print(f"Will process {len(seasons)} seasons: {seasons}")
    
    for season in seasons:
        process_season(season)
        
    print(f"\n{'='*50}")
    print("All seasons complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
