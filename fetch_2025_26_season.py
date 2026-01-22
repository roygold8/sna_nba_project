"""
FETCH 2025-26 NBA SEASON DATA
==============================
Fetches real passing data for the current 2025-26 season.
Data current as of January 21, 2026.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass, leaguestandings
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("[WARNING] nba_api not available")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# Configuration
SEASON = '2025-26'
DATA_DIR = Path(f"data/{SEASON}")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_team_standings():
    """Fetch current team standings for 2025-26 season."""
    print(f"\n[1/3] Fetching {SEASON} team standings...")
    
    if not NBA_API_AVAILABLE:
        print("  [ERROR] NBA API not available")
        return None
    
    try:
        standings = leaguestandings.LeagueStandings(
            season=SEASON,
            season_type='Regular Season'
        )
        time.sleep(0.6)
        
        df = standings.get_data_frames()[0]
        
        # Select relevant columns
        cols_to_keep = ['TeamID', 'TeamCity', 'TeamName', 'TeamSlug', 
                        'WINS', 'LOSSES', 'WinPCT', 'Conference', 'PlayoffRank']
        
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols]
        
        # Save standings
        df.to_csv(DATA_DIR / 'standings.csv', index=False)
        print(f"  [OK] Fetched standings for {len(df)} teams")
        print(f"  [OK] Saved to {DATA_DIR}/standings.csv")
        
        return df
        
    except Exception as e:
        print(f"  [ERROR] Failed to fetch standings: {e}")
        return None


def fetch_player_stats():
    """Fetch player stats for 2025-26 season (GP >= 15, MIN >= 10)."""
    print(f"\n[2/3] Fetching {SEASON} player stats...")
    
    if not NBA_API_AVAILABLE:
        print("  [ERROR] NBA API not available")
        return None
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        time.sleep(0.6)
        
        df = stats.get_data_frames()[0]
        
        # Filter: GP >= 15 (mid-season), MIN >= 10
        df_filtered = df[(df['GP'] >= 15) & (df['MIN'] >= 10)].copy()
        
        # Save filtered players
        df_filtered.to_csv(DATA_DIR / 'filtered_players.csv', index=False)
        print(f"  [OK] Fetched {len(df_filtered)} qualified players (GP>=15, MIN>=10)")
        print(f"  [OK] Saved to {DATA_DIR}/filtered_players.csv")
        
        return df_filtered
        
    except Exception as e:
        print(f"  [ERROR] Failed to fetch player stats: {e}")
        return None


def fetch_passing_data(players_df):
    """Fetch passing data for each player."""
    print(f"\n[3/3] Fetching passing data for {len(players_df)} players...")
    
    if not NBA_API_AVAILABLE:
        print("  [ERROR] NBA API not available")
        return
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, row in tqdm(players_df.iterrows(), total=len(players_df), desc="Fetching passes"):
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER_NAME']
        team_id = row['TEAM_ID']
        
        filepath = DATA_DIR / f"passing_{player_id}.json"
        
        # Skip if already exists
        if filepath.exists():
            skip_count += 1
            continue
        
        try:
            passing = playerdashptpass.PlayerDashPtPass(
                player_id=player_id,
                team_id=team_id,
                season=SEASON,
                season_type_all_star='Regular Season'
            )
            time.sleep(0.6)  # Rate limiting
            
            data = passing.get_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  [ERROR] {player_name}: {e}")
    
    print(f"\n  [OK] Fetched: {success_count}, Cached: {skip_count}, Errors: {error_count}")


def main():
    print("="*70)
    print(f"FETCHING {SEASON} NBA SEASON DATA")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    if not NBA_API_AVAILABLE:
        print("\n[ERROR] NBA API is not available. Cannot fetch data.")
        print("Install with: pip install nba_api")
        return
    
    # Step 1: Fetch standings
    standings_df = fetch_team_standings()
    
    # Step 2: Fetch player stats
    players_df = fetch_player_stats()
    
    # Step 3: Fetch passing data
    if players_df is not None:
        fetch_passing_data(players_df)
    
    print("\n" + "="*70)
    print("[COMPLETE] 2025-26 season data fetched")
    print(f"Data saved to: {DATA_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
