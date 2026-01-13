"""
02_build_metrics.py - NBA Network Metrics Builder

Constructs network graphs from passing data JSON files and calculates
Social Network Analysis (SNA) metrics for both players and teams.

Metrics Calculated:
- Player Level: Degree, Weighted Degree, Betweenness Centrality, Eigenvector Centrality
- Team Level: Density, Gini Coefficient, Degree Centralization, Top2 Concentration

Usage:
    python 02_build_metrics.py

Author: NBA Network Analysis Project
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Suppress warnings
warnings.filterwarnings('ignore')

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

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def format_player_name(name: str) -> str:
    """
    Convert "Last, First" to "First Last" format.
    
    Args:
        name: Name in any format
        
    Returns:
        Name in "First Last" format
    """
    if not name or not isinstance(name, str):
        return str(name) if name else "Unknown"
    
    name = str(name).strip()
    
    if ',' in name:
        parts = name.split(', ', 1)
        if len(parts) == 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    
    return name


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def progress_bar(iterable, desc: str, total: int = None):
    """Wrapper for progress bar that falls back to simple iteration."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    else:
        print(f"Processing: {desc}...")
        return iterable


def calculate_gini(values: List[float]) -> float:
    """
    Calculate Gini coefficient for a list of values.
    Measures inequality (0 = perfect equality, 1 = maximum inequality).
    
    Args:
        values: List of numerical values
        
    Returns:
        Gini coefficient between 0 and 1
    """
    if not values or len(values) < 2:
        return 0.0
    
    # Remove negative values and convert to array
    values = np.array([v for v in values if v >= 0], dtype=float)
    
    if len(values) < 2 or values.sum() == 0:
        return 0.0
    
    # Sort values
    values = np.sort(values)
    n = len(values)
    
    # Gini coefficient formula using the relative mean absolute difference
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def calculate_degree_centralization(G: nx.DiGraph) -> float:
    """
    Calculate degree centralization for a directed graph.
    Measures how much the network is dominated by a single node.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Degree centralization between 0 and 1
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    
    # Get total degree (in + out) for each node
    degrees = [G.in_degree(node) + G.out_degree(node) for node in G.nodes()]
    
    if not degrees:
        return 0.0
    
    max_degree = max(degrees)
    
    # Theoretical maximum for a star graph
    # For directed graph with n nodes, max centralization occurs in star topology
    max_possible = 2 * (n - 1) * (n - 1)
    
    if max_possible == 0:
        return 0.0
    
    # Sum of differences from max degree
    sum_diff = sum(max_degree - d for d in degrees)
    
    return float(sum_diff / max_possible)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_filtered_players(season_dir: Path) -> pd.DataFrame:
    """
    Load filtered players CSV for a season.
    
    Args:
        season_dir: Directory containing filtered_players.csv
        
    Returns:
        DataFrame with player data, empty DataFrame if not found
    """
    filepath = season_dir / "filtered_players.csv"
    
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"Warning: Missing columns in {filepath}: {missing}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def load_passing_data(season_dir: Path) -> Dict[int, dict]:
    """
    Load all passing JSON files for a season.
    
    Args:
        season_dir: Directory containing passing_*.json files
        
    Returns:
        Dictionary mapping player_id -> passing data dict
    """
    passing_data = {}
    json_files = list(season_dir.glob("passing_*.json"))
    
    for filepath in json_files:
        try:
            # Extract player ID from filename
            player_id = int(filepath.stem.replace("passing_", ""))
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if data and 'resultSets' in data:
                passing_data[player_id] = data
                
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Error loading {filepath}: {e}")
            continue
    
    return passing_data


def extract_passes_made(passing_data: dict) -> List[dict]:
    """
    Extract passes made from passing data structure.
    
    The NBA API returns passing data with 'PassesMade' and 'PassesReceived' result sets.
    This function extracts the passes made by a player.
    
    Args:
        passing_data: Raw passing data dictionary from API
        
    Returns:
        List of pass records with standardized keys
    """
    passes = []
    
    try:
        result_sets = passing_data.get('resultSets', [])
        
        for result_set in result_sets:
            if result_set.get('name') == 'PassesMade':
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                
                for row in rows:
                    if len(row) == len(headers):
                        record = dict(zip(headers, row))
                        passes.append(record)
                break
                
    except Exception as e:
        # Silently handle errors - empty passes list will be returned
        pass
    
    return passes


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================
def build_team_network(
    team_id: int,
    team_abbr: str,
    passing_data: Dict[int, dict],
    player_info: pd.DataFrame
) -> nx.DiGraph:
    """
    Build a directed network graph for a team's passing.
    
    Nodes represent players, edges represent passes between players.
    Edge weight = total number of passes from passer to receiver.
    
    Args:
        team_id: NBA team ID
        team_abbr: Team abbreviation (e.g., 'LAL')
        passing_data: Dictionary of all player passing data
        player_info: DataFrame with player metadata
        
    Returns:
        NetworkX DiGraph with players as nodes, passes as weighted edges
    """
    G = nx.DiGraph()
    
    # Get players on this team
    team_players = player_info[player_info['TEAM_ID'] == team_id]
    team_player_ids = set(team_players['PLAYER_ID'].tolist())
    
    if not team_player_ids:
        return G
    
    # Create player name lookup
    player_names = {}
    for _, row in team_players.iterrows():
        player_id = row['PLAYER_ID']
        name = format_player_name(row['PLAYER_NAME'])
        player_names[player_id] = name
    
    # Add nodes for all team players
    for player_id in team_player_ids:
        name = player_names.get(player_id, f"Player_{player_id}")
        G.add_node(player_id, name=name, team=team_abbr)
    
    # Add edges from passing data
    for player_id in team_player_ids:
        if player_id not in passing_data:
            continue
        
        passes_made = extract_passes_made(passing_data[player_id])
        
        for pass_record in passes_made:
            # Get teammate ID who received the pass
            teammate_id = pass_record.get('PASS_TEAMMATE_PLAYER_ID')
            pass_count = pass_record.get('PASS', 0)
            
            # Only include passes to teammates on the same team
            if teammate_id and teammate_id in team_player_ids and pass_count > 0:
                # Add or update edge weight
                if G.has_edge(player_id, teammate_id):
                    G[player_id][teammate_id]['weight'] += pass_count
                else:
                    G.add_edge(player_id, teammate_id, weight=pass_count)
    
    return G


def build_all_team_networks(
    season: str,
    passing_data: Dict[int, dict],
    player_info: pd.DataFrame
) -> Dict[int, nx.DiGraph]:
    """
    Build network graphs for all teams in a season.
    
    Args:
        season: Season string (for logging)
        passing_data: Dictionary of all player passing data
        player_info: DataFrame with player metadata
        
    Returns:
        Dictionary mapping team_id -> NetworkX DiGraph
    """
    team_networks = {}
    
    # Get unique teams
    teams = player_info[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
    
    for _, row in teams.iterrows():
        team_id = row['TEAM_ID']
        team_abbr = row['TEAM_ABBREVIATION']
        
        G = build_team_network(team_id, team_abbr, passing_data, player_info)
        
        # Only include teams with actual passing data (edges)
        if G.number_of_edges() > 0:
            team_networks[team_id] = G
    
    return team_networks


# =============================================================================
# METRICS CALCULATION - PLAYER LEVEL
# =============================================================================
def calculate_player_metrics(G: nx.DiGraph, player_id: int) -> dict:
    """
    Calculate network metrics for a single player.
    
    Metrics:
    - In/Out/Total Degree: Number of unique passing connections
    - Weighted Degree: Total pass volume (in + out)
    - Betweenness Centrality: How often player bridges other players' shortest paths
    - Eigenvector Centrality: Influence based on connections to well-connected players
    
    Args:
        G: Team network graph
        player_id: Player ID
        
    Returns:
        Dictionary of player network metrics
    """
    metrics = {
        'PLAYER_ID': player_id,
        'In_Degree': 0,
        'Out_Degree': 0,
        'Total_Degree': 0,
        'Weighted_In_Degree': 0,
        'Weighted_Out_Degree': 0,
        'Weighted_Degree': 0,
        'Betweenness_Centrality': 0.0,
        'Eigenvector_Centrality': 0.0,
    }
    
    if player_id not in G.nodes():
        return metrics
    
    # Degree metrics (number of connections)
    metrics['In_Degree'] = G.in_degree(player_id)
    metrics['Out_Degree'] = G.out_degree(player_id)
    metrics['Total_Degree'] = metrics['In_Degree'] + metrics['Out_Degree']
    
    # Weighted degree (total pass volume)
    metrics['Weighted_In_Degree'] = G.in_degree(player_id, weight='weight')
    metrics['Weighted_Out_Degree'] = G.out_degree(player_id, weight='weight')
    metrics['Weighted_Degree'] = metrics['Weighted_In_Degree'] + metrics['Weighted_Out_Degree']
    
    # Betweenness centrality (computed for the full graph, then extracted)
    try:
        if G.number_of_nodes() > 1:
            bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
            metrics['Betweenness_Centrality'] = bc.get(player_id, 0.0)
    except Exception:
        pass
    
    # Eigenvector centrality (convert to undirected for calculation)
    try:
        if G.number_of_nodes() > 1:
            G_undirected = G.to_undirected()
            ec = nx.eigenvector_centrality(G_undirected, max_iter=1000, weight='weight')
            metrics['Eigenvector_Centrality'] = ec.get(player_id, 0.0)
    except Exception:
        # Eigenvector centrality may not converge for some graphs
        pass
    
    return metrics


# =============================================================================
# METRICS CALCULATION - TEAM LEVEL
# =============================================================================
def calculate_team_metrics(
    G: nx.DiGraph,
    team_id: int,
    team_abbr: str,
    player_info: pd.DataFrame,
    season: str
) -> dict:
    """
    Calculate network metrics for a team.
    
    Metrics:
    - Density: Proportion of possible connections that exist
    - Gini Coefficient: Inequality of pass distribution (0=equal, 1=concentrated)
    - Degree Centralization: How star-shaped is the network
    - Top2 Concentration: % of passes involving the top 2 most-connected players
    
    Args:
        G: Team network graph
        team_id: Team ID
        team_abbr: Team abbreviation
        player_info: DataFrame with player metadata
        season: Season string
        
    Returns:
        Dictionary of team metrics
    """
    metrics = {
        'TEAM_ID': team_id,
        'TEAM_ABBREVIATION': team_abbr,
        'SEASON': season,
        'Num_Players': G.number_of_nodes(),
        'Num_Edges': G.number_of_edges(),
        'Total_Passes': 0,
        'Density': 0.0,
        'Gini_Coefficient': 0.0,
        'Degree_Centralization': 0.0,
        'Top2_Concentration': 0.0,
        'Star_Player_ID': None,
        'Star_Player_Name': None,
        'Star_Weighted_Degree': 0,
        'W_PCT': 0.0,
        'WINS': 0,
        'LOSSES': 0,
    }
    
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return metrics
    
    # Total passes (sum of all edge weights)
    metrics['Total_Passes'] = int(sum(d['weight'] for _, _, d in G.edges(data=True)))
    
    # Density: ratio of actual edges to possible edges
    try:
        metrics['Density'] = nx.density(G)
    except Exception:
        pass
    
    # Calculate weighted degrees for all nodes
    weighted_degrees = {}
    for node in G.nodes():
        wd = G.in_degree(node, weight='weight') + G.out_degree(node, weight='weight')
        weighted_degrees[node] = wd
    
    degree_values = list(weighted_degrees.values())
    
    # Gini coefficient of degree distribution (measure of inequality)
    metrics['Gini_Coefficient'] = calculate_gini(degree_values)
    
    # Degree centralization (how star-shaped is the network)
    metrics['Degree_Centralization'] = calculate_degree_centralization(G)
    
    # Top 2 Concentration: percentage of total pass involvement by top 2 players
    if degree_values and metrics['Total_Passes'] > 0:
        sorted_degrees = sorted(degree_values, reverse=True)
        top2_sum = sum(sorted_degrees[:2]) if len(sorted_degrees) >= 2 else sum(sorted_degrees)
        # Total involvement = 2 * total_passes (each pass has a passer and receiver)
        total_involvement = metrics['Total_Passes'] * 2
        metrics['Top2_Concentration'] = top2_sum / total_involvement if total_involvement > 0 else 0.0
    
    # Identify star player (max weighted degree)
    if weighted_degrees:
        star_id = max(weighted_degrees, key=weighted_degrees.get)
        metrics['Star_Player_ID'] = star_id
        metrics['Star_Weighted_Degree'] = weighted_degrees[star_id]
        
        # Get star player name
        star_info = player_info[player_info['PLAYER_ID'] == star_id]
        if not star_info.empty:
            metrics['Star_Player_Name'] = format_player_name(star_info.iloc[0]['PLAYER_NAME'])
        else:
            # Fallback to node attribute
            metrics['Star_Player_Name'] = G.nodes[star_id].get('name', f"Player_{star_id}")
    
    # Team success metrics from player data
    team_players = player_info[player_info['TEAM_ID'] == team_id]
    
    if not team_players.empty:
        # Get W_PCT (Win Percentage)
        if 'W_PCT' in team_players.columns:
            w_pct_values = team_players['W_PCT'].dropna()
            if len(w_pct_values) > 0:
                # Use the mode (most common value) since all players on same team should have same W_PCT
                metrics['W_PCT'] = float(w_pct_values.mode().iloc[0]) if len(w_pct_values.mode()) > 0 else float(w_pct_values.mean())
        
        # Get wins and losses
        if 'W' in team_players.columns and 'L' in team_players.columns:
            # Take the max as team totals (players have team W/L in their stats)
            w_values = team_players['W'].dropna()
            l_values = team_players['L'].dropna()
            
            if len(w_values) > 0:
                metrics['WINS'] = int(w_values.max())
            if len(l_values) > 0:
                metrics['LOSSES'] = int(l_values.max())
            
            # Calculate W_PCT if not available or was 0
            total_games = metrics['WINS'] + metrics['LOSSES']
            if metrics['W_PCT'] == 0 and total_games > 0:
                metrics['W_PCT'] = metrics['WINS'] / total_games
    
    return metrics


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def process_season(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single season and return player and team metrics.
    
    Args:
        season: Season string (e.g., '2023-24')
        
    Returns:
        Tuple of (player_metrics_df, team_metrics_df)
    """
    print(f"\n{'='*60}")
    print(f"Processing season: {season}")
    print(f"{'='*60}")
    
    season_dir = DATA_DIR / season
    
    # Load data
    player_info = load_filtered_players(season_dir)
    if player_info.empty:
        print(f"No player data found for {season}")
        return pd.DataFrame(), pd.DataFrame()
    
    passing_data = load_passing_data(season_dir)
    if not passing_data:
        print(f"No passing data found for {season}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"[OK] Loaded {len(player_info)} players, {len(passing_data)} passing files")
    
    # Build team networks
    team_networks = build_all_team_networks(season, passing_data, player_info)
    print(f"[OK] Built networks for {len(team_networks)} teams")
    
    if not team_networks:
        print(f"No valid team networks for {season}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate metrics
    player_metrics_list = []
    team_metrics_list = []
    
    # Pre-compute centrality measures for efficiency
    centrality_cache = {}
    
    for team_id, G in team_networks.items():
        team_abbr = player_info[player_info['TEAM_ID'] == team_id]['TEAM_ABBREVIATION'].iloc[0]
        
        # Team metrics
        team_metrics = calculate_team_metrics(G, team_id, team_abbr, player_info, season)
        team_metrics_list.append(team_metrics)
        
        # Pre-compute betweenness and eigenvector centrality for the team
        bc = {}
        ec = {}
        try:
            if G.number_of_nodes() > 1:
                bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
        except Exception:
            pass
        
        try:
            if G.number_of_nodes() > 1:
                G_undirected = G.to_undirected()
                ec = nx.eigenvector_centrality(G_undirected, max_iter=1000, weight='weight')
        except Exception:
            pass
        
        # Player metrics for this team
        team_players = player_info[player_info['TEAM_ID'] == team_id]
        
        for _, player_row in team_players.iterrows():
            player_id = player_row['PLAYER_ID']
            
            # Calculate basic metrics
            player_metrics = {
                'PLAYER_ID': player_id,
                'PLAYER_NAME': format_player_name(player_row['PLAYER_NAME']),
                'TEAM_ID': team_id,
                'TEAM_ABBREVIATION': team_abbr,
                'SEASON': season,
                'In_Degree': 0,
                'Out_Degree': 0,
                'Total_Degree': 0,
                'Weighted_In_Degree': 0,
                'Weighted_Out_Degree': 0,
                'Weighted_Degree': 0,
                'Betweenness_Centrality': bc.get(player_id, 0.0),
                'Eigenvector_Centrality': ec.get(player_id, 0.0),
            }
            
            if player_id in G.nodes():
                player_metrics['In_Degree'] = G.in_degree(player_id)
                player_metrics['Out_Degree'] = G.out_degree(player_id)
                player_metrics['Total_Degree'] = player_metrics['In_Degree'] + player_metrics['Out_Degree']
                player_metrics['Weighted_In_Degree'] = G.in_degree(player_id, weight='weight')
                player_metrics['Weighted_Out_Degree'] = G.out_degree(player_id, weight='weight')
                player_metrics['Weighted_Degree'] = player_metrics['Weighted_In_Degree'] + player_metrics['Weighted_Out_Degree']
            
            # Add player stats from source data
            for col in ['GP', 'MIN', 'PTS', 'AST', 'REB']:
                if col in player_row:
                    player_metrics[col] = player_row[col]
            
            player_metrics_list.append(player_metrics)
    
    player_df = pd.DataFrame(player_metrics_list)
    team_df = pd.DataFrame(team_metrics_list)
    
    print(f"[OK] Calculated metrics: {len(player_df)} players, {len(team_df)} teams")
    
    return player_df, team_df


def main():
    """Main entry point for metrics building."""
    print("\n" + "="*60)
    print("NBA NETWORK METRICS BUILDER")
    print("="*60)
    
    ensure_directory(OUTPUT_DIR)
    
    all_player_metrics = []
    all_team_metrics = []
    
    # Process each season
    for season in progress_bar(SEASONS, desc="Processing seasons"):
        try:
            player_df, team_df = process_season(season)
            
            if not player_df.empty:
                all_player_metrics.append(player_df)
            
            if not team_df.empty:
                all_team_metrics.append(team_df)
                
        except Exception as e:
            print(f"[ERROR] Error processing season {season}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all seasons
    if all_player_metrics:
        final_player_df = pd.concat(all_player_metrics, ignore_index=True)
        
        # Reorder columns for better readability
        player_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON',
            'GP', 'MIN', 'PTS', 'AST', 'REB',
            'In_Degree', 'Out_Degree', 'Total_Degree',
            'Weighted_In_Degree', 'Weighted_Out_Degree', 'Weighted_Degree',
            'Betweenness_Centrality', 'Eigenvector_Centrality'
        ]
        player_cols = [c for c in player_cols if c in final_player_df.columns]
        final_player_df = final_player_df[player_cols]
        
        output_path = OUTPUT_DIR / "nba_player_metrics.csv"
        final_player_df.to_csv(output_path, index=False)
        print(f"\n[OK] Saved player metrics to {output_path}")
        print(f"  Shape: {final_player_df.shape}")
    else:
        print("\n[ERROR] No player metrics to save")
    
    if all_team_metrics:
        final_team_df = pd.concat(all_team_metrics, ignore_index=True)
        
        # Reorder columns for better readability
        team_cols = [
            'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON',
            'Num_Players', 'Num_Edges', 'Total_Passes',
            'Density', 'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration',
            'Star_Player_ID', 'Star_Player_Name', 'Star_Weighted_Degree',
            'W_PCT', 'WINS', 'LOSSES'
        ]
        team_cols = [c for c in team_cols if c in final_team_df.columns]
        final_team_df = final_team_df[team_cols]
        
        output_path = OUTPUT_DIR / "nba_team_metrics.csv"
        final_team_df.to_csv(output_path, index=False)
        print(f"\n[OK] Saved team metrics to {output_path}")
        print(f"  Shape: {final_team_df.shape}")
    else:
        print("\n[ERROR] No team metrics to save")
    
    print("\n" + "="*60)
    print("METRICS BUILDING COMPLETE")
    print("="*60)
    
    # Print summary statistics
    if all_team_metrics:
        print("\nSample Team Metrics Summary:")
        summary_cols = ['TEAM_ABBREVIATION', 'SEASON', 'Gini_Coefficient', 
                       'Top2_Concentration', 'W_PCT', 'Star_Player_Name']
        summary_cols = [c for c in summary_cols if c in final_team_df.columns]
        print(final_team_df[summary_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
