"""
01_fetch_data.py - NBA Passing Data Fetcher

Downloads raw passing data for NBA players who meet specific criteria.
Implements caching to avoid redundant API calls and rate limiting to
prevent timeouts.

Usage:
    python 01_fetch_data.py

Author: NBA Network Analysis Project
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Set, Dict, List

# NBA API imports
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass
from nba_api.stats.library.parameters import SeasonType

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")


# =============================================================================
# CONFIGURATION
# =============================================================================
SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24'
]

# Filter criteria
MIN_GAMES_PLAYED = 20
MIN_MINUTES_PER_GAME = 10

# API rate limiting (seconds between requests)
API_SLEEP_TIME = 0.6

# Data directory
DATA_DIR = Path("data")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def format_player_name(name: str) -> str:
    """
    Convert "Last, First" to "First Last" format.
    
    Args:
        name: Name in "Last, First" format (or any format)
        
    Returns:
        Name in "First Last" format
        
    Examples:
        >>> format_player_name("James, LeBron")
        'LeBron James'
        >>> format_player_name("Giannis Antetokounmpo")
        'Giannis Antetokounmpo'
    """
    if not name or not isinstance(name, str):
        return str(name) if name else "Unknown"
    
    name = name.strip()
    
    if ',' in name:
        parts = name.split(', ', 1)  # Split only on first comma
        if len(parts) == 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    
    return name


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def progress_bar(iterable, desc: str, total: int = None):
    """
    Wrapper for progress bar that falls back to simple iteration.
    
    Args:
        iterable: Items to iterate over
        desc: Description for the progress bar
        total: Total number of items (optional)
        
    Yields:
        Items from the iterable
    """
    if HAS_TQDM:
        yield from tqdm(iterable, desc=desc, total=total)
    else:
        print(f"Processing: {desc}...")
        yield from iterable


def safe_api_call(func, *args, max_retries: int = 3, **kwargs):
    """
    Safely call an API function with retry logic.
    
    Args:
        func: Function to call
        *args: Positional arguments
        max_retries: Maximum number of retries
        **kwargs: Keyword arguments
        
    Returns:
        Result of the function call or None on failure
    """
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            time.sleep(API_SLEEP_TIME)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = API_SLEEP_TIME * (attempt + 2)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    return None


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================
def fetch_player_stats(season: str) -> pd.DataFrame:
    """
    Fetch league-wide player statistics for a season.
    
    Args:
        season: Season string (e.g., '2023-24')
        
    Returns:
        DataFrame with player statistics, empty DataFrame on error
    """
    print(f"\n{'='*60}")
    print(f"Fetching player stats for season: {season}")
    print(f"{'='*60}")
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SeasonType.regular,
            per_mode_detailed='PerGame'
        )
        
        time.sleep(API_SLEEP_TIME)
        
        data_frames = stats.get_data_frames()
        if not data_frames or len(data_frames) == 0:
            print(f"Warning: No data frames returned for season {season}")
            return pd.DataFrame()
        
        df = data_frames[0]
        
        if df.empty:
            print(f"Warning: Empty data returned for season {season}")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['PLAYER_ID', 'PLAYER_NAME', 'GP', 'MIN', 'TEAM_ID', 'TEAM_ABBREVIATION']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        print(f"[OK] Fetched {len(df)} players")
        return df
        
    except Exception as e:
        print(f"[ERROR] Error fetching player stats for {season}: {e}")
        return pd.DataFrame()


def filter_players(df: pd.DataFrame, min_gp: int, min_mpg: float) -> pd.DataFrame:
    """
    Filter players based on games played and minutes per game.
    
    Args:
        df: DataFrame with player statistics
        min_gp: Minimum games played
        min_mpg: Minimum minutes per game
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # The MIN column is already per-game when using per_mode_detailed='PerGame'
    # Create MPG column for clarity
    if 'MPG' not in df.columns:
        df = df.copy()
        df['MPG'] = df['MIN']
    
    # Apply filters
    mask = (df['GP'] >= min_gp) & (df['MPG'] >= min_mpg)
    filtered = df[mask].copy()
    
    print(f"[OK] Filtered to {len(filtered)} players (GP >= {min_gp}, MIN >= {min_mpg})")
    
    return filtered


def fetch_player_passing(player_id: int, season: str) -> Optional[dict]:
    """
    Fetch passing data for a specific player.
    
    Args:
        player_id: NBA player ID
        season: Season string (e.g., '2023-24')
        
    Returns:
        Dictionary with passing data or None on error
    """
    try:
        passing_data = playerdashptpass.PlayerDashPtPass(
            player_id=player_id,
            season=season,
            season_type_all_star=SeasonType.regular,
            per_mode_simple='Totals'
        )
        
        time.sleep(API_SLEEP_TIME)
        
        # Get raw response as dictionary
        data = passing_data.get_dict()
        
        # Validate that we got actual data
        if not data or 'resultSets' not in data:
            return None
        
        return data
        
    except Exception as e:
        print(f"  [ERROR] Error fetching passing data for player {player_id}: {e}")
        return None


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================
def save_filtered_players(df: pd.DataFrame, season_dir: Path) -> bool:
    """
    Save filtered players list to CSV.
    
    Args:
        df: DataFrame with filtered players
        season_dir: Directory to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = season_dir / "filtered_players.csv"
        df.to_csv(filepath, index=False)
        print(f"[OK] Saved filtered players to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving filtered players: {e}")
        return False


def save_passing_data(data: dict, player_id: int, season_dir: Path) -> bool:
    """
    Save player passing data to JSON.
    
    Args:
        data: Passing data dictionary
        player_id: Player ID
        season_dir: Directory to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = season_dir / f"passing_{player_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  [ERROR] Error saving passing data for player {player_id}: {e}")
        return False


def load_existing_player_ids(season_dir: Path) -> Set[int]:
    """
    Get set of player IDs that already have passing data (for caching).
    
    Args:
        season_dir: Directory containing passing JSON files
        
    Returns:
        Set of player IDs with existing data
    """
    existing = set()
    
    for file in season_dir.glob("passing_*.json"):
        try:
            # Extract player ID from filename
            player_id = int(file.stem.replace("passing_", ""))
            
            # Optionally verify file is valid JSON
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and 'resultSets' in data:
                    existing.add(player_id)
        except (ValueError, json.JSONDecodeError):
            continue
    
    return existing


def load_filtered_players(season_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load existing filtered players CSV if available.
    
    Args:
        season_dir: Directory containing filtered_players.csv
        
    Returns:
        DataFrame if file exists and is valid, None otherwise
    """
    filepath = season_dir / "filtered_players.csv"
    
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_csv(filepath)
        if 'PLAYER_ID' in df.columns and len(df) > 0:
            return df
    except Exception as e:
        print(f"Warning: Error loading {filepath}: {e}")
    
    return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def process_season(season: str, force_refresh: bool = False) -> Dict[str, int]:
    """
    Process a single season: fetch stats, filter players, download passing data.
    
    Args:
        season: Season string (e.g., '2023-24')
        force_refresh: If True, re-download even if cached
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {'players_total': 0, 'players_fetched': 0, 'success': 0, 'errors': 0, 'cached': 0}
    
    season_dir = DATA_DIR / season
    ensure_directory(season_dir)
    
    # Step 1: Get filtered players list
    filtered_df = None
    
    if not force_refresh:
        filtered_df = load_filtered_players(season_dir)
        if filtered_df is not None:
            print(f"Loaded existing filtered players from cache ({len(filtered_df)} players)")
    
    if filtered_df is None:
        # Fetch and filter player stats
        df = fetch_player_stats(season)
        
        if df.empty:
            print(f"Skipping season {season} - no data available")
            return stats
        
        filtered_df = filter_players(df, MIN_GAMES_PLAYED, MIN_MINUTES_PER_GAME)
        
        if filtered_df.empty:
            print(f"Skipping season {season} - no players met filter criteria")
            return stats
        
        # Save filtered players
        save_filtered_players(filtered_df, season_dir)
    
    stats['players_total'] = len(filtered_df)
    
    # Step 2: Get existing passing data files (caching)
    existing_ids = load_existing_player_ids(season_dir)
    stats['cached'] = len(existing_ids)
    print(f"Found {len(existing_ids)} cached passing data files")
    
    # Step 3: Determine which players need fetching
    player_ids = filtered_df['PLAYER_ID'].tolist()
    players_to_fetch = [pid for pid in player_ids if pid not in existing_ids]
    stats['players_fetched'] = len(players_to_fetch)
    
    print(f"Need to fetch passing data for {len(players_to_fetch)} players")
    
    if not players_to_fetch:
        print("All passing data already cached!")
        stats['success'] = stats['cached']
        return stats
    
    # Create player name lookup for progress display
    player_names = dict(zip(filtered_df['PLAYER_ID'], filtered_df['PLAYER_NAME']))
    
    # Step 4: Fetch passing data for each player
    success_count = 0
    error_count = 0
    
    for player_id in progress_bar(players_to_fetch, desc=f"Fetching {season}"):
        player_name = player_names.get(player_id, f"ID:{player_id}")
        
        data = fetch_player_passing(player_id, season)
        
        if data and save_passing_data(data, player_id, season_dir):
            success_count += 1
        else:
            error_count += 1
    
    stats['success'] = success_count + stats['cached']
    stats['errors'] = error_count
    
    print(f"\n[OK] Season {season} complete: {success_count} fetched, {stats['cached']} cached, {error_count} errors")
    
    return stats


def main():
    """Main entry point for the data fetching pipeline."""
    print("\n" + "="*60)
    print("NBA PASSING DATA FETCHER")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Seasons: {SEASONS[0]} to {SEASONS[-1]}")
    print(f"  - Min Games Played: {MIN_GAMES_PLAYED}")
    print(f"  - Min Minutes/Game: {MIN_MINUTES_PER_GAME}")
    print(f"  - API Sleep Time: {API_SLEEP_TIME}s")
    print(f"  - Data Directory: {DATA_DIR}")
    
    # Ensure base data directory exists
    ensure_directory(DATA_DIR)
    
    # Track overall statistics
    total_stats = {
        'seasons_processed': 0,
        'total_players': 0,
        'total_success': 0,
        'total_errors': 0
    }
    
    # Process each season
    for season in SEASONS:
        try:
            stats = process_season(season)
            total_stats['seasons_processed'] += 1
            total_stats['total_players'] += stats.get('players_total', 0)
            total_stats['total_success'] += stats.get('success', 0)
            total_stats['total_errors'] += stats.get('errors', 0)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Partial data saved.")
            break
        except Exception as e:
            print(f"\n[ERROR] Fatal error processing season {season}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("DATA FETCHING COMPLETE")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Seasons processed: {total_stats['seasons_processed']}")
    print(f"  - Total players: {total_stats['total_players']}")
    print(f"  - Successful downloads: {total_stats['total_success']}")
    print(f"  - Errors: {total_stats['total_errors']}")
    print(f"\nData saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
