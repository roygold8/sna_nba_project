#!/usr/bin/env python3
"""
Build comprehensive player metrics DataFrame for 2023-24 season.
Combines network metrics with player performance data.

NETWORK EXPLANATION:
====================
This table is built from the NBA ASSIST NETWORK (nba_assist_network.gexf):
- NODES: Players (format: "Player Name, TEAM, Season")
- EDGES: Directed edges from Player A → Player B when A assists B
- EDGE WEIGHT: Number of assists A gave to B during the season

Network Metrics Calculated:
- Degree: Number of unique teammates a player connected with (in + out)
- In_Degree: Number of teammates who assisted to this player
- Out_Degree: Number of teammates this player assisted to
- Weighted_Degree: Total assists involved (given + received)
- Betweenness: How often a player lies on shortest paths between others
- Clustering: How interconnected a player's neighbors are with each other

WHY SOME PERFORMANCE DATA IS MISSING:
=====================================
The scoring_leaders.csv only contains TOP 100 scorers.
Players not in top 100 scorers will have NaN for PTS, AST, REB, etc.
The network has 465 players, but only 100 have scoring stats.
"""

import networkx as nx
import pandas as pd
import numpy as np
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
ASSIST_NETWORK_FILE = os.path.join(BASE_DIR, 'output', 'nba_assist_network.gexf')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'output', 'player_network_metrics_{season}.csv')

def get_weight(data, default=1):
    """Safely get edge weight from GEXF (stored as strings)."""
    w = data.get('weight', default)
    try:
        return float(w)
    except (TypeError, ValueError):
        return default

def get_team_from_node(node_name):
    """Extract team abbreviation from node name."""
    parts = node_name.split(', ')
    return parts[1] if len(parts) >= 2 else 'UNK'

def get_player_name_from_node(node_name):
    """Extract player name from node name."""
    return node_name.split(',')[0].strip()

def filter_by_season(G, season):
    """Filter network to specific season."""
    nodes_in_season = [n for n in G.nodes() if season in n]
    return G.subgraph(nodes_in_season).copy()

def calculate_team_density(G, team_nodes):
    """Calculate density for a team subgraph."""
    if len(team_nodes) < 2:
        return 0
    team_subgraph = G.subgraph(team_nodes)
    return nx.density(team_subgraph)

# Team abbreviation to full name mapping
TEAM_ABBR_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Reverse mapping
TEAM_NAME_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}

# Also create short name mapping (for team_standings.csv which uses short names)
TEAM_SHORT_TO_ABBR = {
    'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA', 
    'Bulls': 'CHI', 'Cavaliers': 'CLE', 'Mavericks': 'DAL', 'Nuggets': 'DEN',
    'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
    'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA',
    'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
    'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
    'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR',
    'Jazz': 'UTA', 'Wizards': 'WAS'
}

def build_player_metrics_df(season='2023-24'):
    """
    Build DataFrame with player network metrics and performance data.
    """
    
    print("="*70)
    print("BUILDING PLAYER METRICS DATAFRAME")
    print("="*70)
    print("\nNETWORK: NBA Assist Network (nba_assist_network.gexf)")
    print("- Nodes = Players")
    print("- Edges = Assist connections (A assists B = edge A→B)")
    print("- Edge Weight = Number of assists")
    print("="*70)
    
    print(f"\nLoading assist network...")
    G = nx.read_gexf('output/nba_assist_network.gexf')
    print(f"Full network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter to season
    G_season = filter_by_season(G, season)
    print(f"{season} season: {G_season.number_of_nodes()} nodes, {G_season.number_of_edges()} edges")
    
    # Group nodes by team
    teams = {}
    for node in G_season.nodes():
        team = get_team_from_node(node)
        if team not in teams:
            teams[team] = []
        teams[team].append(node)
    
    print(f"Found {len(teams)} teams: {sorted(teams.keys())}")
    
    # Calculate team densities
    team_densities = {}
    for team, team_nodes in teams.items():
        team_densities[team] = calculate_team_density(G_season, team_nodes)
    
    # Calculate betweenness centrality
    print("\nCalculating betweenness centrality...")
    G_undirected = G_season.to_undirected()
    betweenness = nx.betweenness_centrality(G_undirected, weight='weight')
    
    # Calculate clustering coefficient
    print("Calculating clustering coefficients...")
    clustering = nx.clustering(G_undirected, weight='weight')
    
    # Build player records
    print("Building player metrics...")
    records = []
    
    for node in G_season.nodes():
        player_name = get_player_name_from_node(node)
        team = get_team_from_node(node)
        
        # Degree metrics
        in_degree = G_season.in_degree(node)
        out_degree = G_season.out_degree(node)
        total_degree = in_degree + out_degree
        
        # Weighted degrees (actual assist counts)
        assists_given = sum(get_weight(d, 0) for _, _, d in G_season.out_edges(node, data=True))
        assists_received = sum(get_weight(d, 0) for _, _, d in G_season.in_edges(node, data=True))
        weighted_degree = assists_given + assists_received
        
        records.append({
            'Player_Name': player_name,
            'Team': team,
            'Team_FullName': TEAM_ABBR_TO_NAME.get(team, team),
            'Season': season,
            'Degree': total_degree,
            'In_Degree': in_degree,
            'Out_Degree': out_degree,
            'Weighted_Degree': weighted_degree,
            'Assists_Given': assists_given,
            'Assists_Received': assists_received,
            'Betweenness': betweenness.get(node, 0),
            'Clustering': clustering.get(node, 0),
            'Team_Density': team_densities.get(team, 0)
        })
    
    df = pd.DataFrame(records)
    
    # ========== ADD TEAM METRICS ==========
    print("\n" + "="*70)
    print("ADDING TEAM SUCCESS METRICS")
    print("="*70)
    
    # Load playoff scores
    print("\nLoading playoff scores...")
    try:
        playoff_df = pd.read_csv(f'data/{season}/playoff_scores.csv')
        # Map TeamName to abbreviation
        playoff_df['Team'] = playoff_df['TeamName'].map(TEAM_NAME_TO_ABBR)
        playoff_cols = playoff_df[['Team', 'PlayoffRank', 'PlayoffWins', 'PlayoffScore']].copy()
        playoff_cols = playoff_cols.rename(columns={
            'PlayoffRank': 'Team_PlayoffRank',
            'PlayoffWins': 'Team_PlayoffWins', 
            'PlayoffScore': 'Team_PlayoffScore'
        })
        df = df.merge(playoff_cols, on='Team', how='left')
        print(f"  ✓ Added PlayoffRank, PlayoffWins, PlayoffScore for {playoff_df['Team'].notna().sum()} teams")
    except Exception as e:
        print(f"  ✗ Could not load playoff scores: {e}")
    
    # Load team standings
    print("\nLoading team standings...")
    try:
        standings_df = pd.read_csv(f'data/{season}/team_standings.csv')
        # Map TeamName (short) to abbreviation
        standings_df['Team'] = standings_df['TeamName'].map(TEAM_SHORT_TO_ABBR)
        standings_cols = standings_df[['Team', 'WinPCT']].copy()
        standings_cols = standings_cols.rename(columns={'WinPCT': 'Team_WinPCT'})
        df = df.merge(standings_cols, on='Team', how='left')
        print(f"  ✓ Added WinPCT for {standings_df['Team'].notna().sum()} teams")
    except Exception as e:
        print(f"  ✗ Could not load team standings: {e}")
    
    # ========== ADD PLAYER PERFORMANCE ==========
    print("\n" + "="*70)
    print("ADDING PLAYER PERFORMANCE STATS")
    print("="*70)
    
    print("\nLoading scoring leaders (TOP 100 only)...")
    try:
        scoring_df = pd.read_csv(f'data/{season}/scoring_leaders.csv')
        
        # Select relevant columns
        perf_cols = ['PLAYER', 'GP', 'MIN', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'EFF']
        available_cols = [c for c in perf_cols if c in scoring_df.columns]
        perf_df = scoring_df[available_cols].copy()
        perf_df = perf_df.rename(columns={'PLAYER': 'Player_Name'})
        
        df = df.merge(perf_df, on='Player_Name', how='left')
        matched = df['PTS'].notna().sum()
        print(f"  ✓ Merged performance data for {matched}/{len(df)} players")
        print(f"  ⚠ {len(df) - matched} players have NO performance stats (not in top 100 scorers)")
    except Exception as e:
        print(f"  ✗ Could not load performance data: {e}")
    
    # Sort by weighted degree
    df = df.sort_values('Weighted_Degree', ascending=False)
    
    return df

def main():
    # Build the DataFrame
    df = build_player_metrics_df('2023-24')
    
    # Display summary
    print(f"\n{'='*70}")
    print(f"FINAL DATAFRAME SUMMARY")
    print(f"{'='*70}")
    print(f"Total Players: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")
    
    print(f"\n{'='*70}")
    print(f"TOP 15 PLAYERS BY NETWORK INVOLVEMENT")
    print(f"{'='*70}")
    display_cols = ['Player_Name', 'Team', 'Degree', 'Betweenness', 'Clustering', 
                    'Team_Density', 'Team_WinPCT', 'Team_PlayoffScore', 'PTS', 'AST']
    available_display = [c for c in display_cols if c in df.columns]
    print(df[available_display].head(15).to_string(index=False))
    
    # Save to CSV
    output_file = 'player_network_metrics_2023-24.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    
    # Column mapping
    print(f"\n{'='*70}")
    print("COLUMN REFERENCE (Hebrew)")
    print(f"{'='*70}")
    print("""
    Player_Name      = שם שחקן
    Team             = קבוצה (קיצור)
    Season           = עונה
    Degree           = דרגה (מספר קשרים)
    Betweenness      = מרכזיות ביניים
    Clustering       = Connectedness (מקדם אשכול)
    Team_Density     = צפיפות רשת הקבוצה
    Team_WinPCT      = אחוז ניצחונות הקבוצה
    Team_PlayoffRank = דירוג פלייאוף
    Team_PlayoffWins = ניצחונות בפלייאוף
    Team_PlayoffScore= ציון הצלחה בפלייאוף (0-6)
    Assists_Given    = אסיסטים שנתן
    Assists_Received = אסיסטים שקיבל
    PTS, AST, REB    = נתוני ביצועים (ממוצע למשחק)
    """)
    
    return df

if __name__ == "__main__":
    df = main()
