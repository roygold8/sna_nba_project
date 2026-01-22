#!/usr/bin/env python3
"""
Comprehensive NBA Network Analysis - Hypothesis Testing

Tests 11 hypotheses on BOTH assist and pass networks:

ORIGINAL HYPOTHESES (H1-H8):
H1: Density → Success (positive)
H2: Equal Distribution → Success (Gini, negative)
H3: Multiple Hubs → Success (positive)
H4: Less Star Dependency → Success (Top2_Concentration, negative)
H5: Team Cohesion → Success (Avg_Clustering, positive)
H6: Higher Volume → Success (Total weight, positive)
H7: Distributed Centrality → Success (Degree_Centralization, negative)
H8: Distributed Bridging → Success (Max_Betweenness, negative)

NEW HYPOTHESES (H10-H12):
H10: Stronger Core → Success (K-core number, positive)
H11: Lower Diameter → Success (Network diameter, negative)
H12: Eigenvector Centrality Spread → Success (Std of eigenvector, positive)

SUCCESS METRICS:
- WinPCT (Regular season win percentage)
- PlayoffRank (Playoff seeding 1-10)
- PlayoffWins (Number of playoff wins)
- PlayoffScore (Custom 0-6 scale)
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIST_NETWORK_FILE = os.path.join(BASE_DIR, 'output', 'nba_assist_network.gexf')
PASS_NETWORK_FILE = os.path.join(BASE_DIR, 'output', 'nba_pass_network.gexf')
SEASON = '2023-24'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'hypothesis_results')

# Team name mappings
TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 
    'Washington Wizards': 'WAS'
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


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_weight(data, default=1):
    """Safely get edge weight from GEXF (stored as strings)."""
    w = data.get('weight', default)
    try:
        return float(w)
    except:
        return default


def get_team_from_node(node_name):
    """Extract team abbreviation from node name."""
    parts = node_name.split(', ')
    return parts[1] if len(parts) >= 2 else 'UNK'


def filter_by_season(G, season):
    """Filter network to specific season."""
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


def calculate_team_network_features(G, network_type='assist'):
    """
    Calculate comprehensive network features for each team.
    
    Returns DataFrame with all hypothesis-related metrics.
    """
    
    # Group by team
    teams = {}
    for node in G.nodes():
        team = get_team_from_node(node)
        if team not in teams:
            teams[team] = []
        teams[team].append(node)
    
    team_features = []
    
    for team, players in teams.items():
        if len(players) < 3:
            continue
            
        # Team subgraph
        G_team = G.subgraph(players).copy()
        G_team_undirected = G_team.to_undirected()
        
        n_players = len(players)
        n_edges = G_team.number_of_edges()
        
        # ===== H1: DENSITY =====
        density = nx.density(G_team)
        
        # ===== H6: TOTAL VOLUME (assists or passes) =====
        total_weight = sum(get_weight(d, 0) for _, _, d in G_team.edges(data=True))
        avg_weight_per_player = total_weight / n_players if n_players > 0 else 0
        
        # ===== WEIGHTED DEGREES =====
        weighted_degrees = []
        for node in players:
            out_w = sum(get_weight(d, 0) for _, _, d in G_team.out_edges(node, data=True))
            in_w = sum(get_weight(d, 0) for _, _, d in G_team.in_edges(node, data=True))
            weighted_degrees.append(out_w + in_w)
        
        # ===== H7: DEGREE CENTRALIZATION =====
        max_degree = max(weighted_degrees) if weighted_degrees else 0
        sum_diff = sum(max_degree - d for d in weighted_degrees)
        max_possible = (n_players - 1) * max_degree if n_players > 1 else 1
        degree_centralization = sum_diff / max_possible if max_possible > 0 else 0
        
        # ===== H2: GINI COEFFICIENT =====
        gini = calculate_gini(weighted_degrees)
        
        # ===== H3: NUMBER OF HUBS =====
        hub_threshold = max_degree * 0.5 if max_degree > 0 else 0
        n_hubs = sum(1 for d in weighted_degrees if d >= hub_threshold)
        
        # ===== H4: TOP 2 CONCENTRATION =====
        sorted_degrees = sorted(weighted_degrees, reverse=True)
        top2_concentration = sum(sorted_degrees[:2]) / sum(weighted_degrees) if sum(weighted_degrees) > 0 else 0
        
        # ===== H5: AVERAGE CLUSTERING =====
        try:
            avg_clustering = nx.average_clustering(G_team_undirected, weight='weight')
        except:
            avg_clustering = 0
        
        # ===== H8: MAX BETWEENNESS (bridging concentration) =====
        try:
            betweenness = nx.betweenness_centrality(G_team_undirected, weight='weight')
            max_betweenness = max(betweenness.values()) if betweenness else 0
            avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0
        except:
            max_betweenness = 0
            avg_betweenness = 0
        
        # ===== H10: K-CORE NUMBER (strongest core) =====
        try:
            core_numbers = nx.core_number(G_team_undirected)
            max_kcore = max(core_numbers.values()) if core_numbers else 0
            avg_kcore = np.mean(list(core_numbers.values())) if core_numbers else 0
        except:
            max_kcore = 0
            avg_kcore = 0
        
        # ===== H11: NETWORK DIAMETER =====
        try:
            if nx.is_connected(G_team_undirected):
                diameter = nx.diameter(G_team_undirected)
            else:
                # Use diameter of largest connected component
                largest_cc = max(nx.connected_components(G_team_undirected), key=len)
                diameter = nx.diameter(G_team_undirected.subgraph(largest_cc))
        except:
            diameter = n_players  # fallback to max possible
        
        # ===== H12: EIGENVECTOR CENTRALITY SPREAD =====
        try:
            eigenvector = nx.eigenvector_centrality(G_team_undirected, weight='weight', max_iter=1000)
            eigenvector_std = np.std(list(eigenvector.values()))
            eigenvector_max = max(eigenvector.values())
        except:
            eigenvector_std = 0
            eigenvector_max = 0
        
        # Edge statistics
        edge_weights = [get_weight(d, 0) for _, _, d in G_team.edges(data=True)]
        max_edge_weight = max(edge_weights) if edge_weights else 0
        avg_edge_weight = np.mean(edge_weights) if edge_weights else 0
        
        team_features.append({
            'Team': team,
            'Network_Type': network_type,
            'N_Players': n_players,
            'N_Connections': n_edges,
            # H1
            'Density': density,
            # H2
            'Gini_Coefficient': gini,
            # H3
            'N_Hubs': n_hubs,
            # H4
            'Top2_Concentration': top2_concentration,
            # H5
            'Avg_Clustering': avg_clustering,
            # H6
            'Total_Weight': total_weight,
            'Avg_Weight_Per_Player': avg_weight_per_player,
            # H7
            'Degree_Centralization': degree_centralization,
            # H8
            'Max_Betweenness': max_betweenness,
            'Avg_Betweenness': avg_betweenness,
            # H10
            'Max_Kcore': max_kcore,
            'Avg_Kcore': avg_kcore,
            # H11
            'Diameter': diameter,
            # H12
            'Eigenvector_Std': eigenvector_std,
            'Eigenvector_Max': eigenvector_max,
            # Additional
            'Max_Edge_Weight': max_edge_weight,
            'Avg_Edge_Weight': avg_edge_weight
        })
    
    return pd.DataFrame(team_features)


def add_success_metrics(df, season):
    """Add team success metrics (WinPCT, PlayoffRank, PlayoffWins, PlayoffScore)."""
    
    # Load playoff scores
    try:
        playoff_df = pd.read_csv(f'data/{season}/playoff_scores.csv')
        playoff_df['Team'] = playoff_df['TeamName'].map(TEAM_NAME_TO_ABBR)
        playoff_cols = playoff_df[['Team', 'PlayoffRank', 'PlayoffWins', 'PlayoffScore']]
        df = df.merge(playoff_cols, on='Team', how='left')
    except Exception as e:
        print(f"  Warning: Could not load playoff scores: {e}")
    
    # Load standings
    try:
        standings_df = pd.read_csv(f'data/{season}/team_standings.csv')
        standings_df['Team'] = standings_df['TeamName'].map(TEAM_SHORT_TO_ABBR)
        standings_cols = standings_df[['Team', 'WinPCT']]
        df = df.merge(standings_cols, on='Team', how='left')
    except Exception as e:
        print(f"  Warning: Could not load standings: {e}")
    
    return df


def run_correlation_analysis(df, network_type):
    """
    Run correlation analysis between network features and success metrics.
    Returns DataFrame with correlation results.
    """
    
    # Define hypotheses with expected directions
    hypotheses = {
        'H1_Density': ('Density', 'positive'),
        'H2_Gini': ('Gini_Coefficient', 'negative'),
        'H3_Hubs': ('N_Hubs', 'positive'),
        'H4_Top2': ('Top2_Concentration', 'negative'),
        'H5_Clustering': ('Avg_Clustering', 'positive'),
        'H6_Volume': ('Total_Weight', 'positive'),
        'H7_Centralization': ('Degree_Centralization', 'negative'),
        'H8_Betweenness': ('Max_Betweenness', 'negative'),
        'H10_Kcore': ('Max_Kcore', 'positive'),
        'H11_Diameter': ('Diameter', 'negative'),
        'H12_Eigenvector': ('Eigenvector_Std', 'positive'),
    }
    
    success_metrics = ['WinPCT', 'PlayoffRank', 'PlayoffWins', 'PlayoffScore']
    
    results = []
    
    for hyp_name, (feature, expected_dir) in hypotheses.items():
        if feature not in df.columns:
            continue
            
        for success in success_metrics:
            if success not in df.columns:
                continue
            
            # Filter NaN
            valid = df[[feature, success]].dropna()
            if len(valid) < 5:
                continue
            
            # Calculate correlation
            corr, p_value = stats.pearsonr(valid[feature], valid[success])
            
            # Check if direction matches hypothesis
            if expected_dir == 'positive':
                direction_matches = corr > 0
            else:
                direction_matches = corr < 0
            
            # Significance
            if p_value < 0.01:
                sig = '***'
            elif p_value < 0.05:
                sig = '**'
            elif p_value < 0.1:
                sig = '*'
            else:
                sig = ''
            
            # Hypothesis supported?
            supported = direction_matches and p_value < 0.1
            
            results.append({
                'Network_Type': network_type,
                'Hypothesis': hyp_name,
                'Feature': feature,
                'Expected_Direction': expected_dir,
                'Success_Metric': success,
                'Correlation': corr,
                'P_Value': p_value,
                'Significance': sig,
                'Direction_Matches': direction_matches,
                'Hypothesis_Supported': supported
            })
    
    return pd.DataFrame(results)


def print_results_summary(assist_results, pass_results):
    """Print formatted summary of hypothesis testing results."""
    
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING RESULTS SUMMARY")
    print("="*80)
    
    # Combine results
    all_results = pd.concat([assist_results, pass_results], ignore_index=True)
    
    # Group by hypothesis
    hypotheses = all_results['Hypothesis'].unique()
    
    for hyp in sorted(hypotheses):
        hyp_data = all_results[all_results['Hypothesis'] == hyp]
        
        print(f"\n{'─'*80}")
        print(f"📊 {hyp}")
        print(f"{'─'*80}")
        
        for network in ['assist', 'pass']:
            net_data = hyp_data[hyp_data['Network_Type'] == network]
            if net_data.empty:
                continue
                
            print(f"\n  [{network.upper()} NETWORK]")
            
            for _, row in net_data.iterrows():
                icon = "✅" if row['Hypothesis_Supported'] else "❌"
                dir_icon = "↑" if row['Correlation'] > 0 else "↓"
                print(f"    {row['Success_Metric']:15} r={row['Correlation']:+.3f} {dir_icon} "
                      f"p={row['P_Value']:.3f} {row['Significance']:3} {icon}")
    
    # Summary table
    print("\n" + "="*80)
    print("HYPOTHESIS SUPPORT SUMMARY")
    print("="*80)
    print(f"\n{'Hypothesis':<20} {'Assist Net':>12} {'Pass Net':>12} {'Overall':>12}")
    print("-"*60)
    
    for hyp in sorted(hypotheses):
        hyp_data = all_results[all_results['Hypothesis'] == hyp]
        
        assist_support = hyp_data[(hyp_data['Network_Type'] == 'assist') & 
                                   (hyp_data['Hypothesis_Supported'])].shape[0]
        assist_total = hyp_data[hyp_data['Network_Type'] == 'assist'].shape[0]
        
        pass_support = hyp_data[(hyp_data['Network_Type'] == 'pass') & 
                                 (hyp_data['Hypothesis_Supported'])].shape[0]
        pass_total = hyp_data[hyp_data['Network_Type'] == 'pass'].shape[0]
        
        total_support = hyp_data[hyp_data['Hypothesis_Supported']].shape[0]
        total = len(hyp_data)
        
        print(f"{hyp:<20} {assist_support}/{assist_total:>10} {pass_support}/{pass_total:>10} "
              f"{total_support}/{total:>10}")


def create_comparison_visualization(assist_df, pass_df, assist_results, pass_results):
    """Create visualization comparing assist vs pass network findings."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Network Features vs Win% - Assist (blue) vs Pass (orange) Networks ({SEASON})', 
                 fontsize=14, fontweight='bold')
    
    features_to_plot = [
        ('Density', 'H1: Density'),
        ('Gini_Coefficient', 'H2: Gini (Inequality)'),
        ('N_Hubs', 'H3: Number of Hubs'),
        ('Avg_Clustering', 'H5: Clustering'),
        ('Max_Kcore', 'H10: K-Core'),
        ('Diameter', 'H11: Diameter'),
    ]
    
    for idx, (feature, title) in enumerate(features_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        if feature in assist_df.columns and 'WinPCT' in assist_df.columns:
            ax.scatter(assist_df[feature], assist_df['WinPCT'], 
                      c='blue', alpha=0.6, label='Assist', s=80)
            
            # Add team labels
            for _, row in assist_df.iterrows():
                ax.annotate(row['Team'], (row[feature], row['WinPCT']), 
                           fontsize=7, alpha=0.7)
        
        if feature in pass_df.columns and 'WinPCT' in pass_df.columns:
            # Offset pass network points slightly for visibility
            ax.scatter(pass_df[feature], pass_df['WinPCT'] - 0.01, 
                      c='orange', alpha=0.6, label='Pass', s=80, marker='s')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Win %')
        ax.set_title(title)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    ensure_dir(OUTPUT_DIR)
    output_file = os.path.join(OUTPUT_DIR, 'network_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved comparison visualization to {output_file}")
    plt.close()


def main():
    print("="*80)
    print("COMPREHENSIVE NBA NETWORK HYPOTHESIS TESTING")
    print("="*80)
    print(f"\nSeason: {SEASON}")
    print("Testing 11 hypotheses on BOTH assist and pass networks")
    print("Success Metrics: WinPCT, PlayoffRank, PlayoffWins, PlayoffScore")
    
    ensure_dir(OUTPUT_DIR)
    
    # ===== LOAD NETWORKS =====
    print("\n" + "─"*40)
    print("Loading networks...")
    print("─"*40)
    
    G_assist = nx.read_gexf(ASSIST_NETWORK_FILE)
    G_assist_season = filter_by_season(G_assist, SEASON)
    print(f"Assist Network: {G_assist_season.number_of_nodes()} nodes, "
          f"{G_assist_season.number_of_edges()} edges")
    
    G_pass = nx.read_gexf(PASS_NETWORK_FILE)
    G_pass_season = filter_by_season(G_pass, SEASON)
    print(f"Pass Network: {G_pass_season.number_of_nodes()} nodes, "
          f"{G_pass_season.number_of_edges()} edges")
    
    # ===== CALCULATE FEATURES =====
    print("\n" + "─"*40)
    print("Calculating network features...")
    print("─"*40)
    
    print("  Processing assist network...")
    assist_df = calculate_team_network_features(G_assist_season, 'assist')
    assist_df = add_success_metrics(assist_df, SEASON)
    print(f"  → {len(assist_df)} teams analyzed")
    
    print("  Processing pass network...")
    pass_df = calculate_team_network_features(G_pass_season, 'pass')
    pass_df = add_success_metrics(pass_df, SEASON)
    print(f"  → {len(pass_df)} teams analyzed")
    
    # ===== RUN CORRELATION ANALYSIS =====
    print("\n" + "─"*40)
    print("Running hypothesis tests...")
    print("─"*40)
    
    assist_results = run_correlation_analysis(assist_df, 'assist')
    pass_results = run_correlation_analysis(pass_df, 'pass')
    
    # ===== PRINT RESULTS =====
    print_results_summary(assist_results, pass_results)
    
    # ===== SAVE RESULTS =====
    print("\n" + "─"*40)
    print("Saving results...")
    print("─"*40)
    
    # Save team features
    assist_df.to_csv(os.path.join(OUTPUT_DIR, 'assist_network_features.csv'), index=False)
    pass_df.to_csv(os.path.join(OUTPUT_DIR, 'pass_network_features.csv'), index=False)
    print(f"✓ Saved team features to {OUTPUT_DIR}/")
    
    # Save correlation results
    all_results = pd.concat([assist_results, pass_results], ignore_index=True)
    all_results.to_csv(os.path.join(OUTPUT_DIR, 'hypothesis_test_results.csv'), index=False)
    print(f"✓ Saved hypothesis results to {OUTPUT_DIR}/hypothesis_test_results.csv")
    
    # Create comparison table
    comparison = all_results.pivot_table(
        index=['Hypothesis', 'Success_Metric'],
        columns='Network_Type',
        values=['Correlation', 'P_Value', 'Hypothesis_Supported'],
        aggfunc='first'
    )
    comparison.to_csv(os.path.join(OUTPUT_DIR, 'hypothesis_comparison.csv'))
    print(f"✓ Saved comparison table to {OUTPUT_DIR}/hypothesis_comparison.csv")
    
    # ===== VISUALIZATION =====
    print("\n" + "─"*40)
    print("Creating visualizations...")
    print("─"*40)
    
    create_comparison_visualization(assist_df, pass_df, assist_results, pass_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return assist_df, pass_df, assist_results, pass_results


if __name__ == "__main__":
    assist_df, pass_df, assist_results, pass_results = main()
