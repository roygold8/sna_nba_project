#!/usr/bin/env python3
"""
Analyze relationship between team network characteristics and team success.

Research Question:
Do teams with better assist networks achieve better results?

Network Features Analyzed:
- Team Density (how connected are teammates)
- Degree Centralization (are assists concentrated in few players or distributed?)
- Number of "Hubs" (players with high assist involvement)
- Average Clustering (how interconnected are player groups)
- Total Team Assists
- Assist Distribution (Gini coefficient - equality of assist sharing)
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_weight(data, default=1):
    """Safely get edge weight from GEXF."""
    w = data.get('weight', default)
    try:
        return float(w)
    except:
        return default

def get_team_from_node(node_name):
    parts = node_name.split(', ')
    return parts[1] if len(parts) >= 2 else 'UNK'

def filter_by_season(G, season):
    nodes = [n for n in G.nodes() if season in n]
    return G.subgraph(nodes).copy()

def calculate_gini(values):
    """Calculate Gini coefficient (0=equal, 1=concentrated)."""
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * np.sum(values))) - (n + 1) / n

def calculate_team_network_features(G, season='2023-24'):
    """Calculate network features for each team."""
    
    G_season = filter_by_season(G, season)
    
    # Group by team
    teams = {}
    for node in G_season.nodes():
        team = get_team_from_node(node)
        if team not in teams:
            teams[team] = []
        teams[team].append(node)
    
    team_features = []
    
    for team, players in teams.items():
        if len(players) < 3:
            continue
            
        # Team subgraph
        G_team = G_season.subgraph(players).copy()
        G_team_undirected = G_team.to_undirected()
        
        # Basic metrics
        n_players = len(players)
        n_edges = G_team.number_of_edges()
        
        # 1. DENSITY - How connected is the team
        density = nx.density(G_team)
        
        # 2. TOTAL ASSISTS
        total_assists = sum(get_weight(d, 0) for _, _, d in G_team.edges(data=True))
        avg_assists_per_player = total_assists / n_players if n_players > 0 else 0
        
        # 3. WEIGHTED DEGREES (assist involvement per player)
        weighted_degrees = []
        for node in players:
            out_w = sum(get_weight(d, 0) for _, _, d in G_team.out_edges(node, data=True))
            in_w = sum(get_weight(d, 0) for _, _, d in G_team.in_edges(node, data=True))
            weighted_degrees.append(out_w + in_w)
        
        # 4. DEGREE CENTRALIZATION - Are assists concentrated or distributed?
        # High = one player dominates, Low = distributed
        max_degree = max(weighted_degrees) if weighted_degrees else 0
        sum_diff = sum(max_degree - d for d in weighted_degrees)
        max_possible = (n_players - 1) * max_degree if n_players > 1 else 1
        degree_centralization = sum_diff / max_possible if max_possible > 0 else 0
        
        # 5. GINI COEFFICIENT - Inequality of assist distribution
        gini = calculate_gini(weighted_degrees)
        
        # 6. NUMBER OF HUBS - Players with >50% of max involvement
        hub_threshold = max_degree * 0.5 if max_degree > 0 else 0
        n_hubs = sum(1 for d in weighted_degrees if d >= hub_threshold)
        
        # 7. TOP 2 CONCENTRATION - What % of assists involve top 2 players
        sorted_degrees = sorted(weighted_degrees, reverse=True)
        top2_involvement = sum(sorted_degrees[:2]) / sum(weighted_degrees) if sum(weighted_degrees) > 0 else 0
        
        # 8. AVERAGE CLUSTERING
        try:
            avg_clustering = nx.average_clustering(G_team_undirected, weight='weight')
        except:
            avg_clustering = 0
        
        # 9. BETWEENNESS CENTRALIZATION
        betweenness = nx.betweenness_centrality(G_team_undirected, weight='weight')
        max_betweenness = max(betweenness.values()) if betweenness else 0
        avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0
        
        # 10. STRONGEST CONNECTION (max edge weight)
        edge_weights = [get_weight(d, 0) for _, _, d in G_team.edges(data=True)]
        max_edge_weight = max(edge_weights) if edge_weights else 0
        avg_edge_weight = np.mean(edge_weights) if edge_weights else 0
        
        team_features.append({
            'Team': team,
            'N_Players': n_players,
            'N_Connections': n_edges,
            'Density': density,
            'Total_Assists': total_assists,
            'Avg_Assists_Per_Player': avg_assists_per_player,
            'Degree_Centralization': degree_centralization,
            'Gini_Coefficient': gini,
            'N_Hubs': n_hubs,
            'Top2_Concentration': top2_involvement,
            'Avg_Clustering': avg_clustering,
            'Max_Betweenness': max_betweenness,
            'Avg_Betweenness': avg_betweenness,
            'Max_Edge_Weight': max_edge_weight,
            'Avg_Edge_Weight': avg_edge_weight
        })
    
    return pd.DataFrame(team_features)

def add_success_metrics(df, season='2023-24'):
    """Add team success metrics from CSV files."""
    
    # Team name mappings
    TEAM_NAME_TO_ABBR = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
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
    
    # Load playoff scores
    try:
        playoff_df = pd.read_csv(os.path.join(BASE_DIR, 'data', season, 'playoff_scores.csv'))
        playoff_df['Team'] = playoff_df['TeamName'].map(TEAM_NAME_TO_ABBR)
        playoff_cols = playoff_df[['Team', 'PlayoffRank', 'PlayoffWins', 'PlayoffScore']]
        df = df.merge(playoff_cols, on='Team', how='left')
    except Exception as e:
        print(f"Could not load playoff scores: {e}")
    
    # Load standings
    try:
        standings_df = pd.read_csv(os.path.join(BASE_DIR, 'data', season, 'team_standings.csv'))
        standings_df['Team'] = standings_df['TeamName'].map(TEAM_SHORT_TO_ABBR)
        standings_cols = standings_df[['Team', 'WinPCT']]
        df = df.merge(standings_cols, on='Team', how='left')
    except Exception as e:
        print(f"Could not load standings: {e}")
    
    return df

def analyze_correlations(df):
    """Analyze correlations between network features and success."""
    
    network_features = ['Density', 'Total_Assists', 'Avg_Assists_Per_Player', 
                        'Degree_Centralization', 'Gini_Coefficient', 'N_Hubs',
                        'Top2_Concentration', 'Avg_Clustering', 'Max_Betweenness',
                        'Avg_Edge_Weight']
    
    success_metrics = ['WinPCT', 'PlayoffWins', 'PlayoffScore']
    
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS: Network Features vs Team Success")
    print("="*70)
    
    results = []
    
    for success in success_metrics:
        if success not in df.columns:
            continue
        print(f"\n📊 {success}:")
        print("-" * 50)
        
        for feature in network_features:
            if feature not in df.columns:
                continue
            
            # Filter NaN
            valid = df[[feature, success]].dropna()
            if len(valid) < 5:
                continue
            
            # Calculate correlation
            corr, p_value = stats.pearsonr(valid[feature], valid[success])
            
            # Significance indicator
            sig = ""
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.1:
                sig = "*"
            
            results.append({
                'Feature': feature,
                'Success_Metric': success,
                'Correlation': corr,
                'P_Value': p_value,
                'Significant': sig
            })
            
            print(f"  {feature:25} r={corr:+.3f}  p={p_value:.3f} {sig}")
    
    return pd.DataFrame(results)

def find_patterns(df):
    """Find patterns in successful vs unsuccessful teams."""
    
    print("\n" + "="*70)
    print("PATTERN ANALYSIS: Successful vs Unsuccessful Teams")
    print("="*70)
    
    if 'WinPCT' not in df.columns:
        print("No WinPCT data available")
        return
    
    # Split into top and bottom teams
    median_win = df['WinPCT'].median()
    top_teams = df[df['WinPCT'] >= median_win]
    bottom_teams = df[df['WinPCT'] < median_win]
    
    print(f"\n📈 TOP TEAMS (WinPCT >= {median_win:.3f}): {len(top_teams)} teams")
    print(f"📉 BOTTOM TEAMS (WinPCT < {median_win:.3f}): {len(bottom_teams)} teams")
    
    features_to_compare = ['Density', 'Total_Assists', 'Degree_Centralization', 
                           'Gini_Coefficient', 'N_Hubs', 'Top2_Concentration',
                           'Avg_Clustering', 'Avg_Edge_Weight']
    
    print("\n" + "-"*70)
    print(f"{'Feature':<25} {'Top Teams':>12} {'Bottom Teams':>12} {'Diff':>10} {'Insight'}")
    print("-"*70)
    
    insights = []
    
    for feature in features_to_compare:
        if feature not in df.columns:
            continue
        
        top_mean = top_teams[feature].mean()
        bottom_mean = bottom_teams[feature].mean()
        diff_pct = ((top_mean - bottom_mean) / bottom_mean * 100) if bottom_mean != 0 else 0
        
        # Generate insight
        if abs(diff_pct) > 10:
            if diff_pct > 0:
                insight = "✅ Higher in successful teams"
            else:
                insight = "❌ Lower in successful teams"
        else:
            insight = "➖ Similar"
        
        print(f"{feature:<25} {top_mean:>12.3f} {bottom_mean:>12.3f} {diff_pct:>+9.1f}% {insight}")
        
        insights.append({
            'Feature': feature,
            'Top_Teams_Avg': top_mean,
            'Bottom_Teams_Avg': bottom_mean,
            'Diff_Percent': diff_pct
        })
    
    return pd.DataFrame(insights)

def visualize_relationships(df):
    """Create visualizations of key relationships."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Team Network Features vs Success (2023-24)', fontsize=14, fontweight='bold')
    
    if 'WinPCT' not in df.columns:
        print("No WinPCT for visualization")
        return
    
    # Color by playoff success
    colors = df['PlayoffScore'].fillna(0) if 'PlayoffScore' in df.columns else 'blue'
    
    # 1. Density vs WinPCT
    ax = axes[0, 0]
    ax.scatter(df['Density'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Network Density')
    ax.set_ylabel('Win %')
    ax.set_title('Density vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['Density'], row['WinPCT']), fontsize=8)
    
    # 2. Total Assists vs WinPCT
    ax = axes[0, 1]
    ax.scatter(df['Total_Assists'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Total Team Assists')
    ax.set_ylabel('Win %')
    ax.set_title('Total Assists vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['Total_Assists'], row['WinPCT']), fontsize=8)
    
    # 3. Gini (Equality) vs WinPCT
    ax = axes[0, 2]
    ax.scatter(df['Gini_Coefficient'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Gini Coefficient (0=Equal, 1=Concentrated)')
    ax.set_ylabel('Win %')
    ax.set_title('Assist Distribution vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['Gini_Coefficient'], row['WinPCT']), fontsize=8)
    
    # 4. Number of Hubs vs WinPCT
    ax = axes[1, 0]
    ax.scatter(df['N_Hubs'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Number of Hubs')
    ax.set_ylabel('Win %')
    ax.set_title('Number of Playmakers vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['N_Hubs'], row['WinPCT']), fontsize=8)
    
    # 5. Top2 Concentration vs WinPCT
    ax = axes[1, 1]
    ax.scatter(df['Top2_Concentration'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Top 2 Players Concentration')
    ax.set_ylabel('Win %')
    ax.set_title('Star Dependency vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['Top2_Concentration'], row['WinPCT']), fontsize=8)
    
    # 6. Clustering vs WinPCT
    ax = axes[1, 2]
    ax.scatter(df['Avg_Clustering'], df['WinPCT'], c=colors, cmap='RdYlGn', s=100, alpha=0.7)
    ax.set_xlabel('Average Clustering')
    ax.set_ylabel('Win %')
    ax.set_title('Team Cohesion vs Win %')
    for i, row in df.iterrows():
        ax.annotate(row['Team'], (row['Avg_Clustering'], row['WinPCT']), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('team_network_vs_success.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\n✓ Saved visualization to team_network_vs_success.png")
    plt.show()

def main():
    print("="*70)
    print("NBA TEAM NETWORK SUCCESS ANALYSIS")
    print("="*70)
    
    # Load network
    print("\nLoading assist network...")
    G = nx.read_gexf(os.path.join(BASE_DIR, 'output', 'nba_assist_network.gexf'))
    
    # Calculate team features
    print("Calculating team network features...")
    df = calculate_team_network_features(G, '2023-24')
    
    # Add success metrics
    print("Adding success metrics...")
    df = add_success_metrics(df, '2023-24')
    
    # Display team data
    print("\n" + "="*70)
    print("TEAM NETWORK FEATURES (2023-24)")
    print("="*70)
    print(df.sort_values('WinPCT', ascending=False).to_string(index=False))
    
    # Correlation analysis
    corr_df = analyze_correlations(df)
    
    # Pattern analysis
    patterns_df = find_patterns(df)
    
    # Key findings
    print("\n" + "="*70)
    print("🔑 KEY FINDINGS")
    print("="*70)
    
    if 'WinPCT' in df.columns:
        # Find strongest correlations
        if len(corr_df) > 0:
            win_corrs = corr_df[corr_df['Success_Metric'] == 'WinPCT'].sort_values('Correlation', key=abs, ascending=False)
            
            print("\nStrongest correlations with Win %:")
            for _, row in win_corrs.head(5).iterrows():
                direction = "positively" if row['Correlation'] > 0 else "negatively"
                print(f"  • {row['Feature']} is {direction} correlated (r={row['Correlation']:.3f})")
    
    # Save results
    df.to_csv('team_network_analysis_2023-24.csv', index=False)
    print("\n✓ Saved team analysis to team_network_analysis_2023-24.csv")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_relationships(df)
    
    return df, corr_df, patterns_df

if __name__ == "__main__":
    df, corr_df, patterns_df = main()

