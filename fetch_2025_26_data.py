"""
FETCH 2025-26 NBA SEASON DATA
=============================
Fetches current season passing data and team standings
as of January 21, 2026.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from nba_api.stats.endpoints import (
        leaguedashplayerstats,
        playerdashptpass,
        leaguestandings,
        teamdashboardbygeneralsplits
    )
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("[WARNING] nba_api not available")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

OUTPUT_DIR = Path("data/2025-26")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON = "2025-26"


def fetch_team_standings():
    """Fetch current NBA standings for 2025-26 season."""
    print("\n[1/3] FETCHING TEAM STANDINGS...")
    
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
        cols = ['TeamID', 'TeamCity', 'TeamName', 'TeamSlug', 'Conference', 
                'WINS', 'LOSSES', 'WinPCT', 'HOME', 'ROAD', 'L10', 'strCurrentStreak']
        
        available_cols = [c for c in cols if c in df.columns]
        df_clean = df[available_cols].copy()
        
        # Save
        df_clean.to_csv(OUTPUT_DIR / 'team_standings.csv', index=False)
        print(f"  [OK] Saved standings for {len(df_clean)} teams")
        print(f"  Top 5 teams:")
        for _, row in df_clean.head(5).iterrows():
            print(f"    {row.get('TeamSlug', 'N/A')}: {row.get('WINS', 0)}-{row.get('LOSSES', 0)} ({row.get('WinPCT', 0):.3f})")
        
        return df_clean
        
    except Exception as e:
        print(f"  [ERROR] Failed to fetch standings: {e}")
        return None


def fetch_player_stats():
    """Fetch player stats for 2025-26 season."""
    print("\n[2/3] FETCHING PLAYER STATS...")
    
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
        
        # Filter: GP >= 10, MIN >= 10
        df_filtered = df[(df['GP'] >= 10) & (df['MIN'] >= 10)].copy()
        
        # Save
        df_filtered.to_csv(OUTPUT_DIR / 'player_stats.csv', index=False)
        print(f"  [OK] Saved stats for {len(df_filtered)} qualified players")
        
        return df_filtered
        
    except Exception as e:
        print(f"  [ERROR] Failed to fetch player stats: {e}")
        return None


def fetch_passing_data(player_df, max_players=100):
    """Fetch passing data for top players."""
    print(f"\n[3/3] FETCHING PASSING DATA (top {max_players} players)...")
    
    if not NBA_API_AVAILABLE:
        print("  [ERROR] NBA API not available")
        return None
    
    if player_df is None or len(player_df) == 0:
        print("  [ERROR] No player data available")
        return None
    
    # Sort by minutes and take top players
    top_players = player_df.nlargest(max_players, 'MIN')
    
    passing_data = []
    
    for _, player in tqdm(top_players.iterrows(), total=len(top_players), desc="  Fetching passes"):
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER_NAME']
        team_id = player['TEAM_ID']
        team_abbr = player.get('TEAM_ABBREVIATION', 'UNK')
        
        try:
            passes = playerdashptpass.PlayerDashPtPass(
                player_id=player_id,
                season=SEASON,
                season_type_all_star='Regular Season'
            )
            time.sleep(0.5)  # Rate limiting
            
            # Get passes made
            passes_made = passes.passes_made.get_data_frame()
            passes_received = passes.passes_received.get_data_frame()
            
            # Store data
            player_pass_data = {
                'player_id': player_id,
                'player_name': player_name,
                'team_id': team_id,
                'team_abbr': team_abbr,
                'passes_made': passes_made.to_dict('records') if not passes_made.empty else [],
                'passes_received': passes_received.to_dict('records') if not passes_received.empty else [],
                'total_passes_made': passes_made['PASS'].sum() if 'PASS' in passes_made.columns else 0,
                'total_passes_received': passes_received['PASS'].sum() if 'PASS' in passes_received.columns else 0
            }
            
            passing_data.append(player_pass_data)
            
            # Save individual player file
            with open(OUTPUT_DIR / f'passing_{player_id}.json', 'w') as f:
                json.dump(player_pass_data, f, indent=2)
                
        except Exception as e:
            print(f"    [WARN] Failed for {player_name}: {e}")
            continue
    
    print(f"  [OK] Fetched passing data for {len(passing_data)} players")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'player_id': p['player_id'],
        'player_name': p['player_name'],
        'team_abbr': p['team_abbr'],
        'total_passes_made': p['total_passes_made'],
        'total_passes_received': p['total_passes_received']
    } for p in passing_data])
    
    summary_df.to_csv(OUTPUT_DIR / 'passing_summary.csv', index=False)
    
    return passing_data


def main():
    """Main execution."""
    print("="*70)
    print("FETCHING 2025-26 NBA SEASON DATA")
    print(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    if not NBA_API_AVAILABLE:
        print("\n[ERROR] NBA API is not available!")
        print("Please install: pip install nba_api")
        return
    
    # Fetch data
    standings = fetch_team_standings()
    players = fetch_player_stats()
    passing = fetch_passing_data(players, max_players=50)  # Start with top 50
    
    print("\n" + "="*70)
    print("DATA FETCH COMPLETE")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("="*70)
    
    if standings is not None:
        print(f"\nTeam Standings: {len(standings)} teams")
    if players is not None:
        print(f"Player Stats: {len(players)} players")
    if passing is not None:
        print(f"Passing Data: {len(passing)} players")


if __name__ == "__main__":
    main()
