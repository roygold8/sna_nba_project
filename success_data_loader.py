"""
Success Data Loader
Helper module to load success metrics (standings, scoring leaders, playoff scores)
from data directories.
"""

import os
import pandas as pd
from typing import Union, List, Dict, Optional

# Default data directory pointing to the 'data' folder relative to project root
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def get_available_seasons(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """
    Get a list of all available seasons in the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of season strings (e.g. ['2014-15', '2015-16']) sorted chronologically
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist.")
        return []
        
    seasons = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and '-' in d]
    return sorted(seasons)


def _normalize_seasons(seasons: Union[str, List[str], None], data_dir: str) -> List[str]:
    """Helper to normalize season argument to a list of seasons."""
    if seasons is None:
        return get_available_seasons(data_dir)
    elif isinstance(seasons, str):
        return [seasons]
    else:
        return seasons


def _load_season_csv(
    season: str, 
    filename: str, 
    data_dir: str, 
    required_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Helper to load a specific CSV for a season."""
    file_path = os.path.join(data_dir, season, filename)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        
        # Validation
        if required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"Warning: {filename} for {season} missing columns: {missing}")
                return pd.DataFrame()
                
        # Add season column
        df['Season'] = season
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def load_team_standings(
    seasons: Union[str, List[str], None] = None, 
    data_dir: str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load team standings (WinPCT, PlayoffRank) for specified seasons.
    
    Args:
        seasons: Single season string, list of seasons, or None for all available
        data_dir: Path to data directory
        
    Returns:
        DataFrame with stacked data from all requested seasons
    """
    target_seasons = _normalize_seasons(seasons, data_dir)
    dfs = []
    
    print(f"Loading team standings for {len(target_seasons)} seasons...")
    
    for season in target_seasons:
        df = _load_season_csv(season, 'team_standings.csv', data_dir, 
                             ['TeamID', 'TeamName', 'WinPCT', 'PlayoffRank'])
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)


def load_scoring_leaders(
    seasons: Union[str, List[str], None] = None, 
    data_dir: str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load top 100 scoring leaders for specified seasons.
    
    Args:
        seasons: Single season string, list of seasons, or None for all available
        data_dir: Path to data directory
        
    Returns:
        DataFrame with stacked data from all requested seasons
    """
    target_seasons = _normalize_seasons(seasons, data_dir)
    dfs = []
    
    print(f"Loading scoring leaders for {len(target_seasons)} seasons...")
    
    for season in target_seasons:
        df = _load_season_csv(season, 'scoring_leaders.csv', data_dir)
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)


def load_playoff_scores(
    seasons: Union[str, List[str], None] = None, 
    data_dir: str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load playoff scores for specified seasons.
    
    Args:
        seasons: Single season string, list of seasons, or None for all available
        data_dir: Path to data directory
        
    Returns:
        DataFrame with stacked data from all requested seasons
    """
    target_seasons = _normalize_seasons(seasons, data_dir)
    dfs = []
    
    print(f"Loading playoff scores for {len(target_seasons)} seasons...")
    
    for season in target_seasons:
        df = _load_season_csv(season, 'playoff_scores.csv', data_dir,
                             ['TeamID', 'TeamName', 'PlayoffScore'])
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)


def load_all_success_data(
    seasons: Union[str, List[str], None] = None,
    data_dir: str = DEFAULT_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load all success metrics types for specified seasons.
    
    Args:
        seasons: Single season string, list of seasons, or None for all available
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing DataFrames for 'standings', 'scoring_leaders', 'playoff_scores'
    """
    return {
        'standings': load_team_standings(seasons, data_dir),
        'scoring_leaders': load_scoring_leaders(seasons, data_dir),
        'playoff_scores': load_playoff_scores(seasons, data_dir)
    }


def load_season_success_summary(season: str, data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all metrics for a single specific season.
    
    Args:
        season: Season string (e.g. '2023-24')
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing DataFrames
    """
    return load_all_success_data(season, data_dir)


if __name__ == "__main__":
    # Simple test when run directly
    seasons = get_available_seasons()
    print(f"Found {len(seasons)} seasons: {seasons}")
    
    if seasons:
        latest = seasons[-1]
        print(f"\nTesting load for {latest}...")
        results = load_season_success_summary(latest)
        
        for key, df in results.items():
            print(f"  {key}: {len(df)} rows")
