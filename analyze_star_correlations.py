"""
Star Player Correlations with Team Success
==========================================
1. Highest degree player vs winning
2. Top 2 players (duos) avg degree vs winning  
3. Clustering coefficient (triangles) vs winning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode
import networkx as nx
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("output_star_correlations")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load player and team metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    print(f"Loaded {len(player_df)} player-season records")
    print(f"Loaded {len(team_df)} team-season records")
    
    return player_df, team_df


def calculate_star_metrics(player_df):
    """Calculate star player metrics per team."""
    
    results = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        # Sort by weighted degree
        sorted_players = group.sort_values('Weighted_Degree', ascending=False)
        
        # Top 1 player (star)
        top1 = sorted_players.iloc[0]
        top1_degree = top1['Weighted_Degree']
        top1_name = top1['PLAYER_NAME']
        
        # Top 2 players (duo)
        if len(sorted_players) >= 2:
            top2 = sorted_players.iloc[:2]
            top2_avg_degree = top2['Weighted_Degree'].mean()
            top2_total_degree = top2['Weighted_Degree'].sum()
            top2_names = list(top2['PLAYER_NAME'])
        else:
            top2_avg_degree = top1_degree
            top2_total_degree = top1_degree
            top2_names = [top1_name]
        
        # Top 3 players
        if len(sorted_players) >= 3:
            top3 = sorted_players.iloc[:3]
            top3_avg_degree = top3['Weighted_Degree'].mean()
            top3_total_degree = top3['Weighted_Degree'].sum()
        else:
            top3_avg_degree = sorted_players['Weighted_Degree'].mean()
            top3_total_degree = sorted_players['Weighted_Degree'].sum()
        
        # Team total and percentages
        team_total_degree = group['Weighted_Degree'].sum()
        top1_pct = top1_degree / team_total_degree if team_total_degree > 0 else 0
        top2_pct = top2_total_degree / team_total_degree if team_total_degree > 0 else 0
        
        # Gap between top1 and top2
        if len(sorted_players) >= 2:
            gap_1_2 = top1_degree - sorted_players.iloc[1]['Weighted_Degree']
        else:
            gap_1_2 = 0
        
        results.append({
            'TEAM_ABBREVIATION': team,
            'SEASON': season,
            'Top1_Degree': top1_degree,
            'Top1_Name': top1_name,
            'Top2_Avg_Degree': top2_avg_degree,
            'Top2_Total_Degree': top2_total_degree,
            'Top2_Names': str(top2_names),
            'Top3_Avg_Degree': top3_avg_degree,
            'Top3_Total_Degree': top3_total_degree,
            'Top1_Pct_of_Team': top1_pct,
            'Top2_Pct_of_Team': top2_pct,
            'Gap_Top1_Top2': gap_1_2,
            'Team_Total_Degree': team_total_degree,
            'Team_Avg_Degree': group['Weighted_Degree'].mean(),
            'Team_Std_Degree': group['Weighted_Degree'].std(),
            'Num_Players': len(group)
        })
    
    return pd.DataFrame(results)


def calculate_clustering_coefficients(seasons_dir='data'):
    """Calculate clustering coefficient for each team from raw passing data.
    
    Since most NBA teams have nearly complete graphs (everyone passes to everyone),
    we use WEIGHTED clustering to find meaningful triangles.
    """
    
    results = []
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2024)]
    
    for season in seasons:
        season_path = Path(seasons_dir) / season
        if not season_path.exists():
            continue
            
        # Load filtered players
        players_file = season_path / 'filtered_players.csv'
        if not players_file.exists():
            continue
            
        players_df = pd.read_csv(players_file)
        
        # Group by team
        for team, team_players in players_df.groupby('TEAM_ABBREVIATION'):
            # Build graph for this team
            G = nx.DiGraph()
            
            # Add nodes
            for _, player in team_players.iterrows():
                G.add_node(player['PLAYER_ID'], name=player['PLAYER_NAME'])
            
            team_player_ids = set(team_players['PLAYER_ID'].values)
            
            # Add edges from passing data
            for _, player in team_players.iterrows():
                passing_file = season_path / f"passing_{player['PLAYER_ID']}.json"
                if not passing_file.exists():
                    continue
                
                try:
                    with open(passing_file, 'r') as f:
                        data = json.load(f)
                    
                    passes_made = data.get('PassesMade', [])
                    for pass_info in passes_made:
                        receiver_id = pass_info.get('PASS_TEAMMATE_PLAYER_ID')
                        passes = pass_info.get('PASS', 0)
                        if receiver_id in team_player_ids and passes > 0:
                            G.add_edge(player['PLAYER_ID'], receiver_id, weight=passes)
                except:
                    continue
            
            if len(G.nodes()) < 3:
                continue
            
            # Create undirected graph with combined weights
            G_undirected = nx.Graph()
            for u, v, d in G.edges(data=True):
                if G_undirected.has_edge(u, v):
                    G_undirected[u][v]['weight'] += d.get('weight', 1)
                else:
                    G_undirected.add_edge(u, v, weight=d.get('weight', 1))
            
            if len(G_undirected.nodes()) < 3 or len(G_undirected.edges()) < 3:
                continue
            
            # Calculate WEIGHTED clustering coefficient
            try:
                avg_clustering_weighted = nx.average_clustering(G_undirected, weight='weight')
            except:
                avg_clustering_weighted = 0
            
            # Calculate unweighted clustering
            try:
                avg_clustering = nx.average_clustering(G_undirected)
            except:
                avg_clustering = 0
            
            # Transitivity (global clustering)
            transitivity = nx.transitivity(G_undirected)
            
            # Number of triangles
            triangles = sum(nx.triangles(G_undirected).values()) // 3
            
            # Calculate "Strong Triangle" ratio - triangles with high-weight edges
            # A strong triangle is where all 3 edges have weight > median
            all_weights = [d['weight'] for _, _, d in G_undirected.edges(data=True)]
            if len(all_weights) > 0:
                median_weight = np.median(all_weights)
                
                # Create filtered graph with only strong edges
                G_strong = nx.Graph()
                for u, v, d in G_undirected.edges(data=True):
                    if d['weight'] > median_weight:
                        G_strong.add_edge(u, v, weight=d['weight'])
                
                strong_triangles = sum(nx.triangles(G_strong).values()) // 3 if len(G_strong.nodes()) >= 3 else 0
                strong_clustering = nx.average_clustering(G_strong) if len(G_strong.nodes()) >= 3 else 0
            else:
                strong_triangles = 0
                strong_clustering = 0
            
            # Calculate total edge weight variance (measure of pass distribution evenness)
            edge_weight_std = np.std(all_weights) if len(all_weights) > 0 else 0
            edge_weight_cv = edge_weight_std / np.mean(all_weights) if len(all_weights) > 0 and np.mean(all_weights) > 0 else 0
            
            results.append({
                'TEAM_ABBREVIATION': team,
                'SEASON': season,
                'Avg_Clustering': avg_clustering,
                'Avg_Clustering_Weighted': avg_clustering_weighted,
                'Transitivity': transitivity,
                'Num_Triangles': triangles,
                'Strong_Triangles': strong_triangles,
                'Strong_Clustering': strong_clustering,
                'Edge_Weight_Std': edge_weight_std,
                'Edge_Weight_CV': edge_weight_cv,
                'Num_Nodes': len(G.nodes()),
                'Num_Edges': len(G.edges())
            })
    
    return pd.DataFrame(results)


def merge_all_data(star_metrics, clustering_df, team_df):
    """Merge all metrics with team success."""
    
    # Merge star metrics with team success
    merged = star_metrics.merge(
        team_df[['TEAM_ABBREVIATION', 'SEASON', 'W_PCT', 'WINS', 'LOSSES', 
                 'Density', 'Gini_Coefficient', 'Degree_Centralization']],
        on=['TEAM_ABBREVIATION', 'SEASON'],
        how='left'
    )
    
    # Merge clustering if available
    if len(clustering_df) > 0:
        merged = merged.merge(
            clustering_df,
            on=['TEAM_ABBREVIATION', 'SEASON'],
            how='left'
        )
    
    return merged


def plot_top1_degree_vs_winning(df):
    """1. Highest degree player vs winning."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1a. Top1 Degree vs Win%
    ax1 = axes[0, 0]
    sns.regplot(data=df, x='Top1_Degree', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.5, 's': 40}, line_kws={'color': 'red', 'lw': 2})
    
    corr, pval = stats.pearsonr(df['Top1_Degree'].dropna(), df.loc[df['Top1_Degree'].notna(), 'W_PCT'])
    ax1.set_title(f'Star\'s Weighted Degree vs Win %\nr = {corr:.3f}, p = {pval:.4f}', 
                  fontsize=12, fontweight='bold', 
                  color='green' if pval < 0.05 else 'black')
    ax1.set_xlabel('Top Player Weighted Degree', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # Annotate top teams
    top_teams = df.nlargest(5, 'Top1_Degree')
    for _, row in top_teams.iterrows():
        name = unidecode(row['Top1_Name'].split(',')[0] if ',' in str(row['Top1_Name']) else str(row['Top1_Name']).split()[-1])
        ax1.annotate(f"{name}\n({row['SEASON'][-5:]})", 
                    xy=(row['Top1_Degree'], row['W_PCT']),
                    fontsize=7, alpha=0.8)
    
    # 1b. Top1 Percentage of Team vs Win%
    ax2 = axes[0, 1]
    sns.regplot(data=df, x='Top1_Pct_of_Team', y='W_PCT', ax=ax2,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'purple'}, 
                line_kws={'color': 'darkviolet', 'lw': 2})
    
    corr2, pval2 = stats.pearsonr(df['Top1_Pct_of_Team'].dropna(), 
                                   df.loc[df['Top1_Pct_of_Team'].notna(), 'W_PCT'])
    ax2.set_title(f'Star\'s Share of Team Passes vs Win %\nr = {corr2:.3f}, p = {pval2:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval2 < 0.05 else 'black')
    ax2.set_xlabel('Top Player % of Team Total Degree', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    
    # 1c. Categorized boxplot
    ax3 = axes[1, 0]
    df['Star_Level'] = pd.qcut(df['Top1_Degree'], q=4, labels=['Low', 'Medium', 'High', 'Elite'])
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
    sns.boxplot(data=df, x='Star_Level', y='W_PCT', ax=ax3, 
                order=['Low', 'Medium', 'High', 'Elite'],
                hue='Star_Level', palette=colors, legend=False)
    
    for i, level in enumerate(['Low', 'Medium', 'High', 'Elite']):
        subset = df[df['Star_Level'] == level]['W_PCT']
        ax3.annotate(f'n={len(subset)}\nμ={subset.mean():.3f}',
                    xy=(i, subset.mean()), xytext=(i, subset.mean() + 0.08),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_title('Win % by Star Player Quartile', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Star Player Weighted Degree Quartile', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 1d. Gap between Top1 and Top2 vs Win%
    ax4 = axes[1, 1]
    sns.regplot(data=df, x='Gap_Top1_Top2', y='W_PCT', ax=ax4,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'orange'},
                line_kws={'color': 'darkorange', 'lw': 2})
    
    corr3, pval3 = stats.pearsonr(df['Gap_Top1_Top2'].dropna(),
                                   df.loc[df['Gap_Top1_Top2'].notna(), 'W_PCT'])
    ax4.set_title(f'Gap (Top1 - Top2 Degree) vs Win %\nr = {corr3:.3f}, p = {pval3:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval3 < 0.05 else 'black')
    ax4.set_xlabel('Gap Between #1 and #2 Player', fontsize=11)
    ax4.set_ylabel('Win Percentage', fontsize=11)
    
    plt.suptitle('ANALYSIS 1: Star Player (Highest Degree) vs Team Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_top1_star_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_top1_star_vs_winning.png")
    
    return {'Top1_Degree': (corr, pval), 
            'Top1_Pct': (corr2, pval2),
            'Gap_1_2': (corr3, pval3)}


def plot_duo_vs_winning(df):
    """2. Top 2 players average degree vs winning."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2a. Top2 Average Degree vs Win%
    ax1 = axes[0, 0]
    sns.regplot(data=df, x='Top2_Avg_Degree', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.5, 's': 40}, line_kws={'color': 'red', 'lw': 2})
    
    corr1, pval1 = stats.pearsonr(df['Top2_Avg_Degree'].dropna(),
                                   df.loc[df['Top2_Avg_Degree'].notna(), 'W_PCT'])
    ax1.set_title(f'Duo Average Degree vs Win %\nr = {corr1:.3f}, p = {pval1:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval1 < 0.05 else 'black')
    ax1.set_xlabel('Average Degree of Top 2 Players', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # 2b. Top2 Total Degree vs Win%
    ax2 = axes[0, 1]
    sns.regplot(data=df, x='Top2_Total_Degree', y='W_PCT', ax=ax2,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'green'},
                line_kws={'color': 'darkgreen', 'lw': 2})
    
    corr2, pval2 = stats.pearsonr(df['Top2_Total_Degree'].dropna(),
                                   df.loc[df['Top2_Total_Degree'].notna(), 'W_PCT'])
    ax2.set_title(f'Duo Total Degree vs Win %\nr = {corr2:.3f}, p = {pval2:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval2 < 0.05 else 'black')
    ax2.set_xlabel('Combined Degree of Top 2 Players', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    
    # 2c. Top2 Share of Team vs Win%
    ax3 = axes[1, 0]
    sns.regplot(data=df, x='Top2_Pct_of_Team', y='W_PCT', ax=ax3,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'purple'},
                line_kws={'color': 'darkviolet', 'lw': 2})
    
    corr3, pval3 = stats.pearsonr(df['Top2_Pct_of_Team'].dropna(),
                                   df.loc[df['Top2_Pct_of_Team'].notna(), 'W_PCT'])
    ax3.set_title(f'Duo Share of Team Passes vs Win %\nr = {corr3:.3f}, p = {pval3:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval3 < 0.05 else 'black')
    ax3.set_xlabel('Top 2 Players % of Team Total', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 2d. Top3 Average vs Win%
    ax4 = axes[1, 1]
    sns.regplot(data=df, x='Top3_Avg_Degree', y='W_PCT', ax=ax4,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'teal'},
                line_kws={'color': 'darkcyan', 'lw': 2})
    
    corr4, pval4 = stats.pearsonr(df['Top3_Avg_Degree'].dropna(),
                                   df.loc[df['Top3_Avg_Degree'].notna(), 'W_PCT'])
    ax4.set_title(f'Top 3 Players Average Degree vs Win %\nr = {corr4:.3f}, p = {pval4:.4f}',
                  fontsize=12, fontweight='bold',
                  color='green' if pval4 < 0.05 else 'black')
    ax4.set_xlabel('Average Degree of Top 3 Players', fontsize=11)
    ax4.set_ylabel('Win Percentage', fontsize=11)
    
    plt.suptitle('ANALYSIS 2: Dynamic Duos (Top 2 Players) vs Team Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_duo_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_duo_vs_winning.png")
    
    return {'Top2_Avg': (corr1, pval1),
            'Top2_Total': (corr2, pval2),
            'Top2_Pct': (corr3, pval3),
            'Top3_Avg': (corr4, pval4)}


def plot_clustering_vs_winning(df):
    """3. Clustering coefficient (triangles) vs winning."""
    
    # Check what columns we have
    clustering_cols = ['Strong_Clustering', 'Strong_Triangles', 'Edge_Weight_CV', 'Num_Triangles']
    available_cols = [c for c in clustering_cols if c in df.columns and df[c].notna().any()]
    
    if len(available_cols) == 0:
        print("[SKIP] No clustering data available")
        return {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    results = {}
    
    # 3a. Strong Clustering (weighted) vs Win%
    ax1 = axes[0, 0]
    if 'Strong_Clustering' in df.columns:
        valid_df = df.dropna(subset=['Strong_Clustering', 'W_PCT']).copy()
        if len(valid_df) > 10 and valid_df['Strong_Clustering'].std() > 0.001:
            sns.regplot(data=valid_df, x='Strong_Clustering', y='W_PCT', ax=ax1,
                        scatter_kws={'alpha': 0.5, 's': 40}, line_kws={'color': 'red', 'lw': 2})
            
            corr1, pval1 = stats.pearsonr(valid_df['Strong_Clustering'], valid_df['W_PCT'])
            results['Strong_Clustering'] = (corr1, pval1)
            ax1.set_title(f'Strong Edge Clustering vs Win %\nr = {corr1:.3f}, p = {pval1:.4f}',
                          fontsize=12, fontweight='bold',
                          color='green' if pval1 < 0.05 else 'black')
        else:
            ax1.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_xlabel('Strong Edge Clustering Coefficient', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # 3b. Strong Triangles vs Win%
    ax2 = axes[0, 1]
    if 'Strong_Triangles' in df.columns:
        valid_df2 = df.dropna(subset=['Strong_Triangles', 'W_PCT']).copy()
        if len(valid_df2) > 10 and valid_df2['Strong_Triangles'].std() > 0.001:
            sns.regplot(data=valid_df2, x='Strong_Triangles', y='W_PCT', ax=ax2,
                        scatter_kws={'alpha': 0.5, 's': 40, 'color': 'green'},
                        line_kws={'color': 'darkgreen', 'lw': 2})
            
            corr2, pval2 = stats.pearsonr(valid_df2['Strong_Triangles'], valid_df2['W_PCT'])
            results['Strong_Triangles'] = (corr2, pval2)
            ax2.set_title(f'Strong Triangles (High-Weight Edges) vs Win %\nr = {corr2:.3f}, p = {pval2:.4f}',
                          fontsize=12, fontweight='bold',
                          color='green' if pval2 < 0.05 else 'black')
        else:
            ax2.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Number of Strong Triangles', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    
    # 3c. Total Triangles vs Win%
    ax3 = axes[1, 0]
    if 'Num_Triangles' in df.columns:
        valid_df3 = df.dropna(subset=['Num_Triangles', 'W_PCT']).copy()
        if len(valid_df3) > 10 and valid_df3['Num_Triangles'].std() > 0.001:
            sns.regplot(data=valid_df3, x='Num_Triangles', y='W_PCT', ax=ax3,
                        scatter_kws={'alpha': 0.5, 's': 40, 'color': 'orange'},
                        line_kws={'color': 'darkorange', 'lw': 2})
            
            corr3, pval3 = stats.pearsonr(valid_df3['Num_Triangles'], valid_df3['W_PCT'])
            results['Num_Triangles'] = (corr3, pval3)
            ax3.set_title(f'Total Triangles vs Win %\nr = {corr3:.3f}, p = {pval3:.4f}',
                          fontsize=12, fontweight='bold',
                          color='green' if pval3 < 0.05 else 'black')
        else:
            ax3.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlabel('Number of Triangles', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 3d. Edge Weight CV (evenness of pass distribution) vs Win%
    ax4 = axes[1, 1]
    if 'Edge_Weight_CV' in df.columns:
        valid_df4 = df.dropna(subset=['Edge_Weight_CV', 'W_PCT']).copy()
        if len(valid_df4) > 10 and valid_df4['Edge_Weight_CV'].std() > 0.001:
            sns.regplot(data=valid_df4, x='Edge_Weight_CV', y='W_PCT', ax=ax4,
                        scatter_kws={'alpha': 0.5, 's': 40, 'color': 'purple'},
                        line_kws={'color': 'darkviolet', 'lw': 2})
            
            corr4, pval4 = stats.pearsonr(valid_df4['Edge_Weight_CV'], valid_df4['W_PCT'])
            results['Edge_Weight_CV'] = (corr4, pval4)
            ax4.set_title(f'Pass Distribution Variance vs Win %\nr = {corr4:.3f}, p = {pval4:.4f}',
                          fontsize=12, fontweight='bold',
                          color='green' if pval4 < 0.05 else 'black')
        else:
            ax4.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel('Edge Weight CV (Higher = More Uneven)', fontsize=11)
    ax4.set_ylabel('Win Percentage', fontsize=11)
    
    plt.suptitle('ANALYSIS 3: Triangles & Pass Distribution vs Team Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_clustering_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_clustering_vs_winning.png")
    
    return results


def plot_summary_heatmap(df):
    """Create summary correlation heatmap."""
    
    metrics = ['Top1_Degree', 'Top1_Pct_of_Team', 'Top2_Avg_Degree', 'Top2_Total_Degree',
               'Top2_Pct_of_Team', 'Top3_Avg_Degree', 'Gap_Top1_Top2',
               'Team_Avg_Degree', 'Team_Std_Degree']
    
    if 'Strong_Clustering' in df.columns:
        metrics.extend(['Strong_Clustering', 'Strong_Triangles', 'Num_Triangles', 'Edge_Weight_CV'])
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Calculate correlations with W_PCT
    correlations = []
    for metric in available_metrics:
        valid = df[[metric, 'W_PCT']].dropna()
        if len(valid) > 10:
            corr, pval = stats.pearsonr(valid[metric], valid['W_PCT'])
            correlations.append({
                'Metric': metric,
                'Correlation': corr,
                'P_Value': pval,
                'Significant': pval < 0.05
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green' if sig else 'gray' for sig in corr_df['Significant']]
    bars = ax.barh(range(len(corr_df)), corr_df['Correlation'], color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['Metric'])
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    ax.set_xlabel('Correlation with Win %', fontsize=11)
    ax.set_title('Summary: All Star Metrics vs Win %\n(Green = p < 0.05, Gray = Not Significant)',
                 fontsize=12, fontweight='bold')
    
    # Add correlation values
    for i, (_, row) in enumerate(corr_df.iterrows()):
        offset = 0.02 if row['Correlation'] >= 0 else -0.02
        ha = 'left' if row['Correlation'] >= 0 else 'right'
        ax.text(row['Correlation'] + offset, i, f"{row['Correlation']:.3f}", 
                va='center', ha=ha, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_summary_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_summary_correlations.png")
    
    return corr_df


def print_summary(corr1, corr2, corr3, summary_df):
    """Print comprehensive summary."""
    
    print("\n" + "="*80)
    print("STAR PLAYER CORRELATIONS - SUMMARY")
    print("="*80)
    
    print("\n[1. HIGHEST DEGREE PLAYER (STAR) vs WINNING]")
    print("-" * 50)
    for name, (corr, pval) in corr1.items():
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name:20s}: r = {corr:+.3f}, p = {pval:.4f} {sig}")
    
    print("\n[2. TOP 2 PLAYERS (DUOS) vs WINNING]")
    print("-" * 50)
    for name, (corr, pval) in corr2.items():
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name:20s}: r = {corr:+.3f}, p = {pval:.4f} {sig}")
    
    if corr3:
        print("\n[3. CLUSTERING (TRIANGLES) vs WINNING]")
        print("-" * 50)
        for name, (corr, pval) in corr3.items():
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {name:20s}: r = {corr:+.3f}, p = {pval:.4f} {sig}")
    
    print("\n[OVERALL RANKING - Strongest Predictors of Winning]")
    print("-" * 50)
    for i, (_, row) in enumerate(summary_df.head(10).iterrows(), 1):
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else "*" if row['P_Value'] < 0.1 else ""
        print(f"  {i:2d}. {row['Metric']:25s}: r = {row['Correlation']:+.3f} {sig}")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("STAR PLAYER CORRELATIONS ANALYSIS")
    print("="*60)
    
    # Load data
    player_df, team_df = load_data()
    
    # Calculate star metrics
    print("\n[Calculating star player metrics...]")
    star_metrics = calculate_star_metrics(player_df)
    print(f"  Calculated metrics for {len(star_metrics)} team-seasons")
    
    # Calculate clustering coefficients
    print("\n[Calculating clustering coefficients...]")
    clustering_df = calculate_clustering_coefficients()
    print(f"  Calculated clustering for {len(clustering_df)} team-seasons")
    
    # Merge all data
    merged_df = merge_all_data(star_metrics, clustering_df, team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    print("-" * 40)
    
    # Generate plots
    corr1 = plot_top1_degree_vs_winning(merged_df)
    corr2 = plot_duo_vs_winning(merged_df)
    corr3 = plot_clustering_vs_winning(merged_df)
    summary_df = plot_summary_heatmap(merged_df)
    
    # Print summary
    print_summary(corr1, corr2, corr3, summary_df)
    
    # Save data
    merged_df.to_csv(OUTPUT_DIR / 'star_metrics_with_success.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'correlation_summary.csv', index=False)
    print(f"\n[OK] Saved data to {OUTPUT_DIR}/")
    
    print(f"\n[COMPLETE] All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
