#!/usr/bin/env python3
"""
Compare Top 1 vs Top 2 vs Top 3 player metrics as predictors of team success.
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
NETWORK_FILE = os.path.join(BASE_DIR, 'output', 'nba_pass_network.gexf')  # Using pass network like your graph
SEASON = '2023-24'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'slides')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Team mappings
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


def get_weight(data, default=1):
    w = data.get('weight', default)
    try:
        return float(w)
    except:
        return default


def get_team_from_node(node_name):
    parts = node_name.split(', ')
    return parts[1] if len(parts) >= 2 else 'UNK'


def calculate_top_n_metrics(G, season):
    """Calculate Top 1, Top 2, Top 3 average weighted degree for each team."""
    
    # Filter by season
    nodes = [n for n in G.nodes() if season in n]
    G_season = G.subgraph(nodes).copy()
    
    # Group by team
    teams = {}
    for node in G_season.nodes():
        team = get_team_from_node(node)
        if team not in teams:
            teams[team] = []
        teams[team].append(node)
    
    results = []
    
    for team, players in teams.items():
        if len(players) < 3:
            continue
        
        # Calculate weighted degree for each player
        G_team = G_season.subgraph(players).copy()
        
        player_degrees = []
        for node in players:
            out_w = sum(get_weight(d, 0) for _, _, d in G_team.out_edges(node, data=True))
            in_w = sum(get_weight(d, 0) for _, _, d in G_team.in_edges(node, data=True))
            player_degrees.append(out_w + in_w)
        
        # Sort descending and get top N
        sorted_degrees = sorted(player_degrees, reverse=True)
        
        top1_degree = sorted_degrees[0] if len(sorted_degrees) >= 1 else 0
        top2_avg_degree = np.mean(sorted_degrees[:2]) if len(sorted_degrees) >= 2 else 0
        top3_avg_degree = np.mean(sorted_degrees[:3]) if len(sorted_degrees) >= 3 else 0
        
        results.append({
            'Team': team,
            'Top1_Degree': top1_degree,
            'Top2_Avg_Degree': top2_avg_degree,
            'Top3_Avg_Degree': top3_avg_degree,
        })
    
    return pd.DataFrame(results)


def add_success_metrics(df, season):
    """Add WinPCT from standings."""
    try:
        standings_df = pd.read_csv(os.path.join(BASE_DIR, 'data', season, 'team_standings.csv'))
        standings_df['Team'] = standings_df['TeamName'].map(TEAM_SHORT_TO_ABBR)
        standings_cols = standings_df[['Team', 'WinPCT']]
        df = df.merge(standings_cols, on='Team', how='left')
    except Exception as e:
        print(f"Warning: Could not load standings: {e}")
    return df


def main():
    print("="*60)
    print("TOP 1 vs TOP 2 vs TOP 3 COMPARISON")
    print("="*60)
    
    # Load network
    print("\nLoading pass network...")
    G = nx.read_gexf(NETWORK_FILE)
    
    # Calculate metrics
    print("Calculating Top N metrics...")
    df = calculate_top_n_metrics(G, SEASON)
    df = add_success_metrics(df, SEASON)
    
    print(f"\nTeams analyzed: {len(df)}")
    
    # Calculate correlations
    print("\n" + "-"*40)
    print("CORRELATION RESULTS")
    print("-"*40)
    
    metrics = [
        ('Top1_Degree', 'THE STAR (Top 1)'),
        ('Top2_Avg_Degree', 'THE DUO (Top 2 Avg)'),
        ('Top3_Avg_Degree', 'THE TRIO (Top 3 Avg)'),
    ]
    
    correlations = {}
    
    for metric, label in metrics:
        valid = df[[metric, 'WinPCT']].dropna()
        corr, pval = stats.pearsonr(valid[metric], valid['WinPCT'])
        correlations[metric] = (corr, pval)
        
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"{label:25} r = {corr:+.3f}  p = {pval:.4f} {sig}")
    
    # Create comparison visualization
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#FF6B6B', '#FECA57', '#4ECDC4']
    
    for ax, (metric, label), color in zip(axes, metrics, colors):
        valid = df[[metric, 'WinPCT', 'Team']].dropna()
        
        # Scatter plot
        ax.scatter(valid[metric], valid['WinPCT'], c=color, s=80, alpha=0.7, edgecolor='white')
        
        # Regression line
        z = np.polyfit(valid[metric], valid['WinPCT'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='-')
        
        # Labels
        corr, pval = correlations[metric]
        ax.set_title(f'{label}\nr = {corr:.3f}, p = {pval:.4f}', fontsize=14, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' '), fontsize=12)
        ax.set_ylabel('Win %', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Top 1 vs Top 2 vs Top 3: Which Predicts Wins Best?', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'top123_comparison_scatter.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['THE STAR\n(Top 1)', 'THE DUO\n(Top 2 Avg)', 'THE TRIO\n(Top 3 Avg)']
    corr_values = [correlations[m][0] for m, _ in metrics]
    pvalues = [correlations[m][1] for m, _ in metrics]
    
    bars = ax.bar(labels, corr_values, color=colors, edgecolor='black', linewidth=2)
    
    # Add correlation values on bars
    for bar, corr, pval in zip(bars, corr_values, pvalues):
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'r = {corr:.3f}{sig}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Correlation with Win % (r)', fontsize=14)
    ax.set_title('Which Configuration Predicts Team Success Best?', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(corr_values) + 0.1)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Moderate threshold')
    
    output_file = os.path.join(OUTPUT_DIR, 'top123_comparison_bar.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Save data
    df.to_csv(os.path.join(OUTPUT_DIR, 'top123_data.csv'), index=False)
    print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'top123_data.csv')}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    winner_metric = max(correlations.items(), key=lambda x: x[1][0])
    print(f"\n🏆 BEST PREDICTOR: {winner_metric[0]} (r = {winner_metric[1][0]:.3f})")
    
    return df, correlations


if __name__ == "__main__":
    df, correlations = main()
