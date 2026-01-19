"""
Deni Avdija Full Network Analysis
=================================
Comprehensive comparison with All-Star players using ALL network metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_deni_full_network")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_enhance_data():
    """Load data and calculate additional network metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    
    # Calculate additional network metrics
    # 1. In/Out Ratio (Ball retention)
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    
    # 2. Out/In Ratio (Passing tendency)  
    player_df['Out_In_Ratio'] = player_df['Weighted_Out_Degree'] / (player_df['Weighted_In_Degree'] + 1)
    
    # 3. Net Pass Flow
    player_df['Net_Pass_Flow'] = player_df['Weighted_In_Degree'] - player_df['Weighted_Out_Degree']
    
    # 4. Degree Per Game (Network involvement rate)
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    
    # 5. In-Degree Per Game
    player_df['In_Degree_Per_Game'] = player_df['Weighted_In_Degree'] / (player_df['GP'] + 1)
    
    # 6. Out-Degree Per Game  
    player_df['Out_Degree_Per_Game'] = player_df['Weighted_Out_Degree'] / (player_df['GP'] + 1)
    
    # 7. Connection Density (how many unique teammates)
    player_df['Avg_Pass_Per_Connection_In'] = player_df['Weighted_In_Degree'] / (player_df['In_Degree'] + 1)
    player_df['Avg_Pass_Per_Connection_Out'] = player_df['Weighted_Out_Degree'] / (player_df['Out_Degree'] + 1)
    
    # 8. Normalized Betweenness (scale 0-1)
    player_df['Betweenness_Normalized'] = player_df['Betweenness_Centrality'] / player_df['Betweenness_Centrality'].max()
    
    # 9. Normalized Eigenvector
    player_df['Eigenvector_Normalized'] = player_df['Eigenvector_Centrality'] / player_df['Eigenvector_Centrality'].max()
    
    # 10. Calculate team-level stats for concentration
    team_totals = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_Degree'].sum().reset_index()
    team_totals.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Team_Total_Degree']
    player_df = player_df.merge(team_totals, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    # 11. Degree Concentration (% of team's passes involving this player)
    player_df['Degree_Concentration'] = player_df['Weighted_Degree'] / player_df['Team_Total_Degree']
    
    # 12. In-Degree Concentration
    team_in_totals = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_In_Degree'].sum().reset_index()
    team_in_totals.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Team_Total_In_Degree']
    player_df = player_df.merge(team_in_totals, on=['TEAM_ABBREVIATION', 'SEASON'])
    player_df['In_Degree_Concentration'] = player_df['Weighted_In_Degree'] / player_df['Team_Total_In_Degree']
    
    # 13. Out-Degree Concentration
    team_out_totals = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_Out_Degree'].sum().reset_index()
    team_out_totals.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Team_Total_Out_Degree']
    player_df = player_df.merge(team_out_totals, on=['TEAM_ABBREVIATION', 'SEASON'])
    player_df['Out_Degree_Concentration'] = player_df['Weighted_Out_Degree'] / player_df['Team_Total_Out_Degree']
    
    # Calculate Z-scores within team
    team_stats = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std'],
        'Eigenvector_Centrality': ['mean', 'std'],
        'Betweenness_Centrality': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                          'Team_Degree_Mean', 'Team_Degree_Std',
                          'Team_Eigen_Mean', 'Team_Eigen_Std',
                          'Team_Between_Mean', 'Team_Between_Std']
    
    player_df = player_df.merge(team_stats, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    player_df['Degree_ZScore'] = (player_df['Weighted_Degree'] - player_df['Team_Degree_Mean']) / (player_df['Team_Degree_Std'] + 1)
    player_df['Eigen_ZScore'] = (player_df['Eigenvector_Centrality'] - player_df['Team_Eigen_Mean']) / (player_df['Team_Eigen_Std'] + 0.01)
    player_df['Between_ZScore'] = (player_df['Betweenness_Centrality'] - player_df['Team_Between_Mean']) / (player_df['Team_Between_Std'] + 0.001)
    
    # Network Star = top network position
    player_df['Is_Network_Star'] = (player_df['Degree_ZScore'] > 1.5) | (player_df['Eigen_ZScore'] > 1.5)
    
    print(f"Loaded {len(player_df)} player-seasons")
    print(f"Network Stars identified: {player_df['Is_Network_Star'].sum()}")
    
    return player_df


def get_deni_metrics():
    """Get Deni's estimated network metrics for 2025-26."""
    
    gp = 40
    
    # Based on his 26.1 PPG, 6.9 APG profile - elite playmaking forward
    passes_made_per_game = 85   # High playmaker (6.9 APG is elite for forward)
    passes_received_per_game = 100  # Primary option receives more
    
    deni_metrics = {
        'PLAYER_NAME': 'Deni Avdija',
        'TEAM_ABBREVIATION': 'POR',
        'SEASON': '2025-26',
        'GP': gp,
        
        # Core Degree Metrics
        'Weighted_Degree': gp * (passes_made_per_game + passes_received_per_game),  # 7400
        'Weighted_In_Degree': gp * passes_received_per_game,  # 4000
        'Weighted_Out_Degree': gp * passes_made_per_game,  # 3400
        'In_Degree': 12,  # Unique teammates receiving from
        'Out_Degree': 12,  # Unique teammates passing to
        'Total_Degree': 24,
        
        # Centrality Metrics (estimated based on similar players)
        'Eigenvector_Centrality': 0.42,  # High - connected to important players
        'Betweenness_Centrality': 0.008,  # Moderate - hub not bridge
        
        # Calculated Per-Game Metrics
        'Degree_Per_Game': passes_made_per_game + passes_received_per_game,  # 185
        'In_Degree_Per_Game': passes_received_per_game,  # 100
        'Out_Degree_Per_Game': passes_made_per_game,  # 85
        
        # Ratio Metrics
        'In_Out_Ratio': passes_received_per_game / passes_made_per_game,  # 1.18
        'Out_In_Ratio': passes_made_per_game / passes_received_per_game,  # 0.85
        'Net_Pass_Flow': gp * (passes_received_per_game - passes_made_per_game),  # +600
        
        # Connection Efficiency
        'Avg_Pass_Per_Connection_In': (gp * passes_received_per_game) / 12,  # 333
        'Avg_Pass_Per_Connection_Out': (gp * passes_made_per_game) / 12,  # 283
        
        # Concentration (estimated for primary option)
        'Degree_Concentration': 0.18,  # ~18% of team's passes
        'In_Degree_Concentration': 0.20,  # ~20% of passes received
        'Out_Degree_Concentration': 0.16,  # ~16% of passes made
        
        # Normalized Centralities
        'Eigenvector_Normalized': 0.42 / 0.65,  # ~65%
        'Betweenness_Normalized': 0.008 / 0.85,  # ~1%
    }
    
    return deni_metrics


def compare_to_allstars(deni_metrics, player_df):
    """Comprehensive comparison to All-Star network metrics."""
    
    network_stars = player_df[player_df['Is_Network_Star']]
    all_players = player_df
    
    # ALL network metrics
    metrics = [
        # Core Volume
        ('Weighted_Degree', 'Total Pass Volume'),
        ('Weighted_In_Degree', 'Passes Received'),
        ('Weighted_Out_Degree', 'Passes Made'),
        
        # Centrality
        ('Eigenvector_Centrality', 'Influence (Eigenvector)'),
        ('Betweenness_Centrality', 'Bridge Role (Betweenness)'),
        
        # Per-Game
        ('Degree_Per_Game', 'Ball Involvement/Game'),
        ('In_Degree_Per_Game', 'Received/Game'),
        ('Out_Degree_Per_Game', 'Made/Game'),
        
        # Ratios
        ('In_Out_Ratio', 'Receive vs Make Ratio'),
        ('Out_In_Ratio', 'Make vs Receive Ratio'),
        ('Net_Pass_Flow', 'Net Flow (+ = Receiver)'),
        
        # Concentration
        ('Degree_Concentration', 'Team Degree Share'),
        ('In_Degree_Concentration', 'Team In-Degree Share'),
        ('Out_Degree_Concentration', 'Team Out-Degree Share'),
        
        # Connection Efficiency
        ('Avg_Pass_Per_Connection_In', 'Avg Received per Teammate'),
        ('Avg_Pass_Per_Connection_Out', 'Avg Made per Teammate'),
        
        # Connection Count
        ('In_Degree', 'Unique Passers To'),
        ('Out_Degree', 'Unique Receivers From'),
    ]
    
    comparison = []
    
    for metric, description in metrics:
        if metric not in deni_metrics or metric not in player_df.columns:
            continue
        
        deni_val = deni_metrics[metric]
        
        star_mean = network_stars[metric].mean()
        star_std = network_stars[metric].std()
        star_median = network_stars[metric].median()
        all_mean = all_players[metric].mean()
        all_std = all_players[metric].std()
        
        pct_all = (all_players[metric] < deni_val).mean() * 100
        pct_stars = (network_stars[metric] < deni_val).mean() * 100
        
        z_all = (deni_val - all_mean) / all_std if all_std > 0 else 0
        z_star = (deni_val - star_mean) / star_std if star_std > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Description': description,
            'Deni': deni_val,
            'AllStar_Mean': star_mean,
            'AllStar_Median': star_median,
            'AllStar_Std': star_std,
            'All_Mean': all_mean,
            'Pct_vs_All': pct_all,
            'Pct_vs_AllStars': pct_stars,
            'Z_vs_All': z_all,
            'Z_vs_AllStars': z_star,
            'vs_AllStar_Avg': (deni_val / star_mean * 100) if star_mean != 0 else 0
        })
    
    return pd.DataFrame(comparison)


def plot_comprehensive_analysis(deni_metrics, comparison_df, player_df):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(20, 16))
    
    stars = player_df[player_df['Is_Network_Star']]
    others = player_df[~player_df['Is_Network_Star']]
    
    # 1. Radar Chart of key metrics vs All-Star avg
    ax1 = fig.add_subplot(2, 3, 1, polar=True)
    
    radar_metrics = ['Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
                     'Degree_Per_Game', 'In_Out_Ratio', 'Degree_Concentration']
    radar_labels = ['Total Degree', 'Eigenvector', 'Betweenness', 
                    'Degree/Game', 'In/Out Ratio', 'Team Share']
    
    # Normalize to All-Star average = 1
    deni_vals = []
    star_vals = []
    for m in radar_metrics:
        row = comparison_df[comparison_df['Metric'] == m]
        if len(row) > 0:
            deni_vals.append(row['vs_AllStar_Avg'].values[0] / 100)
            star_vals.append(1.0)
        else:
            deni_vals.append(0)
            star_vals.append(1.0)
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    deni_vals += deni_vals[:1]
    star_vals += star_vals[:1]
    angles += angles[:1]
    
    ax1.plot(angles, star_vals, 'o-', linewidth=2, label='All-Star Avg', color='gold')
    ax1.fill(angles, star_vals, alpha=0.25, color='gold')
    ax1.plot(angles, deni_vals, 'o-', linewidth=2, label='Deni', color='red')
    ax1.fill(angles, deni_vals, alpha=0.25, color='red')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(radar_labels, size=8)
    ax1.set_title('Deni vs All-Star Average\n(1.0 = All-Star Avg)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 2. Percentile Rankings Bar Chart
    ax2 = fig.add_subplot(2, 3, 2)
    
    key_metrics = comparison_df[comparison_df['Metric'].isin([
        'Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
        'Degree_Per_Game', 'In_Out_Ratio', 'Degree_Concentration',
        'Weighted_In_Degree', 'Weighted_Out_Degree'
    ])].sort_values('Pct_vs_All', ascending=True)
    
    colors = ['green' if p >= 75 else 'orange' if p >= 50 else 'red' for p in key_metrics['Pct_vs_All']]
    bars = ax2.barh(key_metrics['Description'], key_metrics['Pct_vs_All'], color=colors, edgecolor='black')
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Median')
    ax2.axvline(x=75, color='gold', linestyle='--', alpha=0.7, label='75th')
    ax2.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Elite (90th)')
    ax2.set_xlabel('Percentile (vs All Players)')
    ax2.set_title("Deni's Network Percentile Rankings", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(loc='lower right')
    
    for bar, pct in zip(bars, key_metrics['Pct_vs_All']):
        ax2.text(min(pct + 2, 95), bar.get_y() + bar.get_height()/2, f'{pct:.0f}%',
                va='center', fontsize=8, fontweight='bold')
    
    # 3. Eigenvector vs Betweenness Scatter
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(others['Eigenvector_Centrality'], others['Betweenness_Centrality'],
                alpha=0.3, s=20, c='gray', label='Role Players')
    ax3.scatter(stars['Eigenvector_Centrality'], stars['Betweenness_Centrality'],
                alpha=0.7, s=40, c='gold', edgecolors='black', linewidths=0.5, label='All-Stars')
    ax3.scatter(deni_metrics['Eigenvector_Centrality'], deni_metrics['Betweenness_Centrality'],
                c='red', s=200, marker='*', edgecolors='black', linewidths=2, 
                label='DENI', zorder=10)
    ax3.set_xlabel('Eigenvector Centrality (Influence)')
    ax3.set_ylabel('Betweenness Centrality (Bridge)')
    ax3.set_title('Centrality: Eigenvector vs Betweenness', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # Add quadrant labels
    ax3.text(0.5, 0.4, 'HUB\n(High Influence)', ha='center', fontsize=8, alpha=0.7)
    ax3.text(0.1, 0.4, 'BRIDGE\n(High Between)', ha='center', fontsize=8, alpha=0.7)
    
    # 4. In-Degree vs Out-Degree
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(others['Weighted_Out_Degree'], others['Weighted_In_Degree'],
                alpha=0.3, s=20, c='gray', label='Role Players')
    ax4.scatter(stars['Weighted_Out_Degree'], stars['Weighted_In_Degree'],
                alpha=0.7, s=40, c='gold', edgecolors='black', linewidths=0.5, label='All-Stars')
    ax4.scatter(deni_metrics['Weighted_Out_Degree'], deni_metrics['Weighted_In_Degree'],
                c='red', s=200, marker='*', edgecolors='black', linewidths=2, 
                label='DENI', zorder=10)
    
    max_val = max(player_df['Weighted_Out_Degree'].max(), player_df['Weighted_In_Degree'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='Equal In/Out')
    ax4.set_xlabel('Weighted Out-Degree (Passes Made)')
    ax4.set_ylabel('Weighted In-Degree (Passes Received)')
    ax4.set_title('Pass Flow: Made vs Received', fontsize=11, fontweight='bold')
    ax4.legend()
    
    # 5. Comparison Table as Heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    
    table_metrics = comparison_df[comparison_df['Metric'].isin([
        'Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
        'Degree_Per_Game', 'In_Out_Ratio', 'Degree_Concentration'
    ])][['Description', 'Deni', 'AllStar_Mean', 'vs_AllStar_Avg']]
    
    # Create comparison ratio heatmap
    table_for_heatmap = table_metrics.set_index('Description')['vs_AllStar_Avg'].to_frame()
    
    sns.heatmap(table_for_heatmap, annot=True, fmt='.0f', cmap='RdYlGn', 
                center=100, ax=ax5, cbar_kws={'label': '% of All-Star Avg'})
    ax5.set_title('Deni as % of All-Star Average', fontsize=11, fontweight='bold')
    ax5.set_ylabel('')
    
    # 6. Distribution comparison for key metric
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Degree Per Game distribution
    ax6.hist(others['Degree_Per_Game'], bins=50, alpha=0.5, label='Role Players', color='gray')
    ax6.hist(stars['Degree_Per_Game'], bins=30, alpha=0.7, label='All-Stars', color='gold')
    ax6.axvline(x=deni_metrics['Degree_Per_Game'], color='red', linestyle='--', lw=3,
                label=f"Deni: {deni_metrics['Degree_Per_Game']:.0f}")
    ax6.axvline(x=stars['Degree_Per_Game'].mean(), color='gold', linestyle='-', lw=2,
                label=f"All-Star Avg: {stars['Degree_Per_Game'].mean():.0f}")
    ax6.set_xlabel('Degree Per Game (Ball Involvement)')
    ax6.set_ylabel('Count')
    ax6.set_title('Ball Involvement Distribution', fontsize=11, fontweight='bold')
    ax6.legend()
    
    plt.suptitle(f"DENI AVDIJA - COMPREHENSIVE NETWORK ANALYSIS\n{deni_metrics['TEAM_ABBREVIATION']} - {deni_metrics['SEASON']}",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_deni_comprehensive_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_deni_comprehensive_network.png")


def plot_allstar_comparison_table(comparison_df):
    """Create detailed comparison table visualization."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Prepare table data
    table_data = comparison_df[['Description', 'Deni', 'AllStar_Mean', 'AllStar_Median', 
                                 'Pct_vs_All', 'Pct_vs_AllStars', 'vs_AllStar_Avg']].copy()
    
    # Format numbers
    table_data['Deni'] = table_data['Deni'].apply(lambda x: f'{x:.2f}' if x < 10 else f'{x:.0f}')
    table_data['AllStar_Mean'] = table_data['AllStar_Mean'].apply(lambda x: f'{x:.2f}' if x < 10 else f'{x:.0f}')
    table_data['AllStar_Median'] = table_data['AllStar_Median'].apply(lambda x: f'{x:.2f}' if x < 10 else f'{x:.0f}')
    table_data['Pct_vs_All'] = table_data['Pct_vs_All'].apply(lambda x: f'{x:.0f}%')
    table_data['Pct_vs_AllStars'] = table_data['Pct_vs_AllStars'].apply(lambda x: f'{x:.0f}%')
    table_data['vs_AllStar_Avg'] = table_data['vs_AllStar_Avg'].apply(lambda x: f'{x:.0f}%')
    
    table_data.columns = ['Metric', 'Deni', 'All-Star\nMean', 'All-Star\nMedian',
                          'Pct vs\nAll', 'Pct vs\nAll-Stars', '% of\nAll-Star Avg']
    
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['lightblue']*7)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Color code cells
    for i in range(len(table_data)):
        # Highlight high percentiles
        pct_all = float(table_data.iloc[i]['Pct vs\nAll'].replace('%', ''))
        if pct_all >= 90:
            table[(i+1, 4)].set_facecolor('lightgreen')
        elif pct_all >= 75:
            table[(i+1, 4)].set_facecolor('lightyellow')
        elif pct_all < 50:
            table[(i+1, 4)].set_facecolor('lightcoral')
    
    plt.title('DENI AVDIJA vs ALL-STAR NETWORK METRICS\nDetailed Comparison Table',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_deni_allstar_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_deni_allstar_comparison_table.png")


def print_full_analysis(deni_metrics, comparison_df, player_df):
    """Print comprehensive analysis."""
    
    stars = player_df[player_df['Is_Network_Star']]
    
    print("\n" + "="*90)
    print("DENI AVDIJA - FULL NETWORK METRICS COMPARISON TO ALL-STARS")
    print("="*90)
    
    print(f"\n[SEASON: {deni_metrics['SEASON']} | Team: {deni_metrics['TEAM_ABBREVIATION']}]")
    print(f"[Comparison Base: {len(stars)} Historical Network All-Stars]")
    
    print("\n" + "="*90)
    print("DETAILED METRICS COMPARISON")
    print("="*90)
    
    print("\n[CORE VOLUME METRICS]")
    print("-" * 85)
    print(f"{'Metric':<30} {'Deni':>12} {'All-Star Avg':>12} {'All-Star Med':>12} {'Pct vs All':>10}")
    print("-" * 85)
    
    volume_metrics = ['Weighted_Degree', 'Weighted_In_Degree', 'Weighted_Out_Degree']
    for _, row in comparison_df[comparison_df['Metric'].isin(volume_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.0f} {row['AllStar_Mean']:>12.0f} "
              f"{row['AllStar_Median']:>12.0f} {row['Pct_vs_All']:>9.0f}%")
    
    print("\n[CENTRALITY METRICS]")
    print("-" * 85)
    centrality_metrics = ['Eigenvector_Centrality', 'Betweenness_Centrality']
    for _, row in comparison_df[comparison_df['Metric'].isin(centrality_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.4f} {row['AllStar_Mean']:>12.4f} "
              f"{row['AllStar_Median']:>12.4f} {row['Pct_vs_All']:>9.0f}%")
    
    print("\n[PER-GAME METRICS]")
    print("-" * 85)
    pg_metrics = ['Degree_Per_Game', 'In_Degree_Per_Game', 'Out_Degree_Per_Game']
    for _, row in comparison_df[comparison_df['Metric'].isin(pg_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.1f} {row['AllStar_Mean']:>12.1f} "
              f"{row['AllStar_Median']:>12.1f} {row['Pct_vs_All']:>9.0f}%")
    
    print("\n[RATIO METRICS]")
    print("-" * 85)
    ratio_metrics = ['In_Out_Ratio', 'Out_In_Ratio', 'Net_Pass_Flow']
    for _, row in comparison_df[comparison_df['Metric'].isin(ratio_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.2f} {row['AllStar_Mean']:>12.2f} "
              f"{row['AllStar_Median']:>12.2f} {row['Pct_vs_All']:>9.0f}%")
    
    print("\n[TEAM CONCENTRATION METRICS]")
    print("-" * 85)
    conc_metrics = ['Degree_Concentration', 'In_Degree_Concentration', 'Out_Degree_Concentration']
    for _, row in comparison_df[comparison_df['Metric'].isin(conc_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.1%} {row['AllStar_Mean']:>12.1%} "
              f"{row['AllStar_Median']:>12.1%} {row['Pct_vs_All']:>9.0f}%")
    
    print("\n[CONNECTION EFFICIENCY]")
    print("-" * 85)
    conn_metrics = ['Avg_Pass_Per_Connection_In', 'Avg_Pass_Per_Connection_Out', 'In_Degree', 'Out_Degree']
    for _, row in comparison_df[comparison_df['Metric'].isin(conn_metrics)].iterrows():
        print(f"{row['Description']:<30} {row['Deni']:>12.1f} {row['AllStar_Mean']:>12.1f} "
              f"{row['AllStar_Median']:>12.1f} {row['Pct_vs_All']:>9.0f}%")
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY: DENI vs ALL-STAR AVERAGES")
    print("="*90)
    
    key_metrics_summary = comparison_df[comparison_df['Metric'].isin([
        'Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
        'Degree_Per_Game', 'In_Out_Ratio', 'Degree_Concentration'
    ])]
    
    print("\n| Metric                        | Deni     | All-Star Avg | % of Avg | Percentile |")
    print("|-------------------------------|----------|--------------|----------|------------|")
    
    for _, row in key_metrics_summary.iterrows():
        deni_str = f"{row['Deni']:.2f}" if row['Deni'] < 10 else f"{row['Deni']:.0f}"
        star_str = f"{row['AllStar_Mean']:.2f}" if row['AllStar_Mean'] < 10 else f"{row['AllStar_Mean']:.0f}"
        print(f"| {row['Description']:<29} | {deni_str:>8} | {star_str:>12} | {row['vs_AllStar_Avg']:>7.0f}% | {row['Pct_vs_All']:>9.0f}% |")
    
    # Verdict
    print("\n" + "="*90)
    print("ALL-STAR VERDICT (Network Metrics)")
    print("="*90)
    
    avg_pct = comparison_df[comparison_df['Metric'].isin([
        'Weighted_Degree', 'Eigenvector_Centrality', 'Degree_Per_Game'
    ])]['Pct_vs_All'].mean()
    
    avg_vs_star = comparison_df[comparison_df['Metric'].isin([
        'Weighted_Degree', 'Eigenvector_Centrality', 'Degree_Per_Game'
    ])]['vs_AllStar_Avg'].mean()
    
    print(f"\nKey Network Metrics:")
    print(f"  - Average Percentile (vs All): {avg_pct:.0f}%")
    print(f"  - Average % of All-Star Avg: {avg_vs_star:.0f}%")
    
    # Strengths
    strengths = comparison_df[comparison_df['Pct_vs_All'] >= 85]['Description'].tolist()
    weaknesses = comparison_df[comparison_df['Pct_vs_All'] < 50]['Description'].tolist()
    
    print(f"\n  STRENGTHS (>= 85th percentile):")
    for s in strengths:
        print(f"    [+] {s}")
    
    if weaknesses:
        print(f"\n  AREAS BELOW MEDIAN:")
        for w in weaknesses:
            print(f"    [-] {w}")
    
    if avg_pct >= 85 and avg_vs_star >= 95:
        verdict = "ELITE Network All-Star Profile"
        symbol = "[ELITE]"
    elif avg_pct >= 75:
        verdict = "STRONG Network All-Star Profile"
        symbol = "[STRONG]"
    elif avg_pct >= 60:
        verdict = "Solid All-Star Candidate"
        symbol = "[SOLID]"
    else:
        verdict = "Developing"
        symbol = "[DEVELOPING]"
    
    print(f"\n{symbol} FINAL VERDICT: {verdict}")
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*60)
    print("DENI AVDIJA - FULL NETWORK ANALYSIS")
    print("="*60)
    
    print("\n[LOADING AND ENHANCING DATA]")
    player_df = load_and_enhance_data()
    
    print("\n[GETTING DENI'S NETWORK METRICS]")
    deni_metrics = get_deni_metrics()
    
    print("\n[COMPARING TO ALL-STARS]")
    comparison_df = compare_to_allstars(deni_metrics, player_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_comprehensive_analysis(deni_metrics, comparison_df, player_df)
    plot_allstar_comparison_table(comparison_df)
    
    print_full_analysis(deni_metrics, comparison_df, player_df)
    
    comparison_df.to_csv(OUTPUT_DIR / 'deni_full_network_comparison.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
