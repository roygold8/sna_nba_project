"""
DENI AVDIJA ALL-STAR ANALYSIS - FIXED VERSION
==============================================
METHODOLOGY DISCLOSURE:
-----------------------
1. We do NOT have actual 2025-26 passing data for Deni Avdija
2. Network metrics are ESTIMATED based on:
   - His current per-game statistics
   - Historical patterns of players with similar stat lines
   - Reasonable assumptions about pass involvement

This analysis compares Deni's ESTIMATED network profile against
REAL NBA All-Stars from seasons 2015-16 to 2023-24.

For a rigorous analysis, actual passing data would need to be fetched
once the 2025-26 season data becomes available via NBA API.

Author: NBA Network Analysis Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

try:
    from unidecode import unidecode
except ImportError:
    unidecode = lambda x: x

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_deni_fixed")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_allstar_data():
    """Load player data with real All-Star designations."""
    
    # Try to load file with Is_Star column
    star_file = Path("output/nba_player_metrics_with_stars.csv")
    if star_file.exists():
        df = pd.read_csv(star_file)
        print(f"[OK] Loaded {len(df)} player-seasons with Is_Star column")
        return df
    
    # Fallback: use regular file and estimate stars based on metrics
    player_file = Path("output/nba_player_metrics.csv")
    if player_file.exists():
        df = pd.read_csv(player_file)
        
        # Define "stars" as top performers (top 10% in weighted degree)
        threshold = df['Weighted_Degree'].quantile(0.90)
        df['Is_Star'] = df['Weighted_Degree'] >= threshold
        
        print(f"[WARNING] Using estimated All-Stars (top 10% by Weighted Degree)")
        print(f"[OK] Loaded {len(df)} player-seasons")
        return df
    
    raise FileNotFoundError("No player metrics file found")


def estimate_deni_network_metrics():
    """
    Estimate Deni's network metrics based on available statistics.
    
    METHODOLOGY:
    ------------
    Based on Deni's 2025-26 per-game stats (as of season progress):
    - Points: ~18.5 PPG
    - Assists: ~7.0 APG  
    - Rebounds: ~8.5 RPG
    
    Network metric estimation approach:
    1. Passes made ≈ Assists × 12 (rough ratio from historical data)
    2. Passes received estimated from usage rate and rebound patterns
    3. Centrality metrics estimated from similar players historically
    
    DISCLAIMER: These are ESTIMATES, not actual measured values.
    """
    
    # Deni's approximate stats (2025-26 season)
    games_played = 40
    ppg = 18.5
    apg = 7.0
    rpg = 8.5
    
    # Estimation methodology (based on historical patterns)
    # Players with ~7 assists typically have 80-90 passes made per game
    # Players with high usage typically receive 90-110 passes per game
    
    passes_made_per_game = 85  # Estimated from assist rate
    passes_received_per_game = 100  # Estimated from usage/touches
    
    # Total weighted degree = (passes made + passes received) × games
    weighted_degree = games_played * (passes_made_per_game + passes_received_per_game)
    
    # Centrality estimates based on similar playmaking forwards
    # Reference: Players like Draymond Green, Kyle Anderson, Julius Randle
    eigenvector_estimate = 0.42  # High for a forward with playmaking duties
    betweenness_estimate = 0.008  # Moderate - not a pure point guard
    closeness_estimate = 0.75  # High - involved in many plays
    
    metrics = {
        'PLAYER_NAME': 'Deni Avdija',
        'TEAM_ABBREVIATION': 'POR',
        'SEASON': '2025-26 (ESTIMATED)',
        'GP': games_played,
        
        # Traditional stats (actual)
        'PTS': ppg * games_played,
        'AST': apg * games_played,
        'REB': rpg * games_played,
        'PTS_PER_GAME': ppg,
        'AST_PER_GAME': apg,
        'REB_PER_GAME': rpg,
        
        # Network metrics (ESTIMATED)
        'Weighted_Degree': weighted_degree,
        'Weighted_In_Degree': games_played * passes_received_per_game,
        'Weighted_Out_Degree': games_played * passes_made_per_game,
        'Degree_Per_Game': passes_made_per_game + passes_received_per_game,
        
        # Centrality (ESTIMATED)
        'Eigenvector_Centrality': eigenvector_estimate,
        'Betweenness_Centrality': betweenness_estimate,
        'Closeness_Centrality': closeness_estimate,
        
        # Derived metrics
        'In_Out_Ratio': passes_received_per_game / passes_made_per_game,
        'Out_In_Ratio': passes_made_per_game / passes_received_per_game,
        
        # Methodology flag
        '_ESTIMATED': True,
    }
    
    return metrics


def compare_to_allstars(deni_metrics, player_df):
    """Compare Deni's estimated metrics to real All-Stars."""
    
    all_stars = player_df[player_df['Is_Star'] == True].copy()
    all_players = player_df.copy()
    
    # Add derived metrics
    for df in [all_stars, all_players]:
        df['Degree_Per_Game'] = df['Weighted_Degree'] / (df['GP'] + 1)
        df['In_Out_Ratio'] = df['Weighted_In_Degree'] / (df['Weighted_Out_Degree'] + 1)
    
    metrics_to_compare = [
        ('Weighted_Degree', 'Total Pass Volume (Season)'),
        ('Degree_Per_Game', 'Pass Volume Per Game'),
        ('Eigenvector_Centrality', 'Eigenvector Centrality'),
        ('Betweenness_Centrality', 'Betweenness Centrality'),
        ('In_Out_Ratio', 'In/Out Ratio (Finisher vs Distributor)'),
    ]
    
    comparison = []
    
    for metric, description in metrics_to_compare:
        if metric not in deni_metrics or metric not in player_df.columns:
            continue
        
        deni_val = deni_metrics[metric]
        
        # All-Star statistics
        star_mean = all_stars[metric].mean()
        star_median = all_stars[metric].median()
        star_std = all_stars[metric].std()
        
        # All player statistics
        all_mean = all_players[metric].mean()
        all_std = all_players[metric].std()
        
        # Percentiles
        pct_vs_all = (all_players[metric] < deni_val).mean() * 100
        pct_vs_stars = (all_stars[metric] < deni_val).mean() * 100
        
        # Z-scores
        z_vs_all = (deni_val - all_mean) / all_std if all_std > 0 else 0
        z_vs_stars = (deni_val - star_mean) / star_std if star_std > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Description': description,
            'Deni_Value': deni_val,
            'AllStar_Mean': star_mean,
            'AllStar_Median': star_median,
            'All_Player_Mean': all_mean,
            'Pct_vs_All': pct_vs_all,
            'Pct_vs_Stars': pct_vs_stars,
            'Z_vs_All': z_vs_all,
            'Z_vs_Stars': z_vs_stars,
            'Pct_of_AllStar_Avg': (deni_val / star_mean * 100) if star_mean != 0 else 0,
        })
    
    return pd.DataFrame(comparison)


def find_similar_players(deni_metrics, player_df):
    """Find historical All-Stars with similar network profiles."""
    
    all_stars = player_df[player_df['Is_Star'] == True].copy()
    
    # Add derived metrics
    all_stars['Degree_Per_Game'] = all_stars['Weighted_Degree'] / (all_stars['GP'] + 1)
    all_stars['In_Out_Ratio'] = all_stars['Weighted_In_Degree'] / (all_stars['Weighted_Out_Degree'] + 1)
    
    # Normalize and calculate distance
    compare_metrics = ['Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality']
    
    for metric in compare_metrics:
        if metric in all_stars.columns:
            mean = all_stars[metric].mean()
            std = all_stars[metric].std()
            all_stars[f'{metric}_norm'] = (all_stars[metric] - mean) / (std + 0.001)
    
    # Deni's normalized values
    deni_norm = {}
    for metric in compare_metrics:
        if metric in deni_metrics:
            mean = all_stars[metric].mean()
            std = all_stars[metric].std()
            deni_norm[f'{metric}_norm'] = (deni_metrics[metric] - mean) / (std + 0.001)
    
    # Calculate Euclidean distance
    def calc_distance(row):
        dist = 0
        for metric in compare_metrics:
            norm_col = f'{metric}_norm'
            if norm_col in deni_norm and norm_col in row.index:
                dist += (row[norm_col] - deni_norm[norm_col])**2
        return np.sqrt(dist)
    
    all_stars['Distance_to_Deni'] = all_stars.apply(calc_distance, axis=1)
    
    similar = all_stars.nsmallest(10, 'Distance_to_Deni')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'Weighted_Degree', 
         'Eigenvector_Centrality', 'Distance_to_Deni']
    ]
    
    return similar


def plot_analysis(deni_metrics, comparison_df, player_df, similar_df):
    """Create comprehensive visualizations."""
    
    all_stars = player_df[player_df['Is_Star'] == True].copy()
    role_players = player_df[player_df['Is_Star'] == False].copy()
    
    # Add derived metrics
    for df in [all_stars, role_players]:
        df['Degree_Per_Game'] = df['Weighted_Degree'] / (df['GP'] + 1)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Percentile Rankings
    ax1 = fig.add_subplot(2, 3, 1)
    
    metrics = comparison_df.sort_values('Pct_vs_All', ascending=True)
    colors = ['green' if p >= 75 else 'orange' if p >= 50 else 'red' for p in metrics['Pct_vs_All']]
    bars = ax1.barh(metrics['Description'], metrics['Pct_vs_All'], color=colors, edgecolor='black')
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Median')
    ax1.axvline(x=75, color='gold', linestyle='--', alpha=0.7, label='75th')
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='Elite')
    ax1.set_xlabel('Percentile (vs All Players)')
    ax1.set_title("Deni's Network Percentile Rankings\n(ESTIMATED VALUES)", fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.legend(loc='lower right')
    
    # 2. Weighted Degree Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(role_players['Weighted_Degree'], bins=50, alpha=0.4, label='Role Players', color='gray')
    ax2.hist(all_stars['Weighted_Degree'], bins=30, alpha=0.7, label='All-Stars', color='gold')
    ax2.axvline(x=deni_metrics['Weighted_Degree'], color='red', linestyle='--', lw=3,
                label=f"Deni (Est.): {deni_metrics['Weighted_Degree']:.0f}")
    ax2.axvline(x=all_stars['Weighted_Degree'].mean(), color='darkorange', linestyle='-', lw=2,
                label=f"All-Star Avg: {all_stars['Weighted_Degree'].mean():.0f}")
    ax2.set_xlabel('Weighted Degree')
    ax2.set_ylabel('Count')
    ax2.set_title('Total Pass Volume Distribution', fontweight='bold')
    ax2.legend()
    
    # 3. Degree Per Game Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(role_players['Degree_Per_Game'], bins=50, alpha=0.4, label='Role Players', color='gray')
    ax3.hist(all_stars['Degree_Per_Game'], bins=30, alpha=0.7, label='All-Stars', color='gold')
    ax3.axvline(x=deni_metrics['Degree_Per_Game'], color='red', linestyle='--', lw=3,
                label=f"Deni (Est.): {deni_metrics['Degree_Per_Game']:.0f}")
    ax3.set_xlabel('Degree Per Game')
    ax3.set_ylabel('Count')
    ax3.set_title('Ball Involvement Per Game', fontweight='bold')
    ax3.legend()
    
    # 4. Eigenvector vs Betweenness
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(role_players['Eigenvector_Centrality'], role_players['Betweenness_Centrality'],
                alpha=0.2, s=15, c='gray', label='Role Players')
    ax4.scatter(all_stars['Eigenvector_Centrality'], all_stars['Betweenness_Centrality'],
                alpha=0.6, s=50, c='gold', edgecolors='black', linewidths=0.5, label='All-Stars')
    ax4.scatter(deni_metrics['Eigenvector_Centrality'], deni_metrics['Betweenness_Centrality'],
                c='red', s=250, marker='*', edgecolors='black', linewidths=2, 
                label='DENI (Est.)', zorder=10)
    ax4.set_xlabel('Eigenvector Centrality')
    ax4.set_ylabel('Betweenness Centrality')
    ax4.set_title('Centrality Comparison\n(Estimated Position)', fontweight='bold')
    ax4.legend(fontsize=8)
    
    # 5. Comparison Table
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    table_data = comparison_df[['Description', 'Deni_Value', 'AllStar_Mean', 'Pct_vs_All', 'Pct_of_AllStar_Avg']].copy()
    table_data['Deni_Value'] = table_data['Deni_Value'].apply(lambda x: f'{x:.2f}' if x < 10 else f'{x:.0f}')
    table_data['AllStar_Mean'] = table_data['AllStar_Mean'].apply(lambda x: f'{x:.2f}' if x < 10 else f'{x:.0f}')
    table_data['Pct_vs_All'] = table_data['Pct_vs_All'].apply(lambda x: f'{x:.0f}%')
    table_data['Pct_of_AllStar_Avg'] = table_data['Pct_of_AllStar_Avg'].apply(lambda x: f'{x:.0f}%')
    table_data.columns = ['Metric', 'Deni (Est.)', 'All-Star Avg', '% vs All', '% of AS Avg']
    
    table = ax5.table(cellText=table_data.values,
                      colLabels=table_data.columns,
                      cellLoc='center',
                      loc='center',
                      colColours=['lightblue']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax5.set_title('Detailed Comparison (ESTIMATED VALUES)', fontweight='bold', pad=20)
    
    # 6. Similar All-Stars
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    similar_table = similar_df[['PLAYER_NAME', 'SEASON', 'TEAM_ABBREVIATION', 'Distance_to_Deni']].copy()
    similar_table['PLAYER_NAME'] = similar_table['PLAYER_NAME'].apply(lambda x: unidecode(x)[:20])
    similar_table['Distance_to_Deni'] = similar_table['Distance_to_Deni'].apply(lambda x: f'{x:.2f}')
    similar_table.columns = ['Player', 'Season', 'Team', 'Distance']
    
    table2 = ax6.table(cellText=similar_table.values,
                       colLabels=similar_table.columns,
                       cellLoc='center',
                       loc='center',
                       colColours=['lightyellow']*4)
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.6)
    ax6.set_title('Most Similar All-Stars\n(by Network Profile)', fontweight='bold', pad=20)
    
    plt.suptitle(f"DENI AVDIJA ALL-STAR ANALYSIS\n"
                 f"*** IMPORTANT: Network metrics are ESTIMATED, not from actual API data ***\n"
                 f"Season: {deni_metrics['SEASON']} | Comparison: {len(all_stars)} All-Star Seasons",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_deni_allstar_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_deni_allstar_analysis.png")


def print_summary(deni_metrics, comparison_df, similar_df, player_df):
    """Print comprehensive summary."""
    
    all_stars = player_df[player_df['Is_Star'] == True]
    
    print("\n" + "="*90)
    print("DENI AVDIJA ALL-STAR ANALYSIS - FIXED VERSION")
    print("="*90)
    
    print("\n[METHODOLOGY DISCLOSURE]")
    print("-"*90)
    print("  *** IMPORTANT: The following network metrics are ESTIMATED ***")
    print("")
    print("  We do NOT have actual 2025-26 passing data for Deni Avdija.")
    print("  Network metrics were estimated using:")
    print("    1. His current per-game statistics (PTS, AST, REB)")
    print("    2. Historical patterns from players with similar stat lines")
    print("    3. Reasonable assumptions about pass involvement")
    print("")
    print("  For rigorous analysis, actual passing data would need to be fetched")
    print("  once the 2025-26 season data becomes available via NBA API.")
    
    print("\n[DENI'S ESTIMATED NETWORK PROFILE]")
    print("-"*90)
    print(f"  Season: {deni_metrics['SEASON']}")
    print(f"  Team: {deni_metrics['TEAM_ABBREVIATION']}")
    print(f"  Games: {deni_metrics['GP']}")
    print(f"\n  Traditional Stats (Actual):")
    print(f"    Points Per Game: {deni_metrics['PTS_PER_GAME']:.1f}")
    print(f"    Assists Per Game: {deni_metrics['AST_PER_GAME']:.1f}")
    print(f"    Rebounds Per Game: {deni_metrics['REB_PER_GAME']:.1f}")
    print(f"\n  Network Metrics (ESTIMATED):")
    print(f"    Weighted Degree: {deni_metrics['Weighted_Degree']:.0f}")
    print(f"    Degree Per Game: {deni_metrics['Degree_Per_Game']:.0f}")
    print(f"    Eigenvector Centrality: {deni_metrics['Eigenvector_Centrality']:.3f}")
    print(f"    Betweenness Centrality: {deni_metrics['Betweenness_Centrality']:.4f}")
    
    print("\n[COMPARISON TO ALL-STARS]")
    print("-"*90)
    print(f"  All-Stars in comparison: {len(all_stars)} player-seasons")
    print(f"\n  {'Metric':<35} {'Deni (Est.)':<12} {'AS Avg':<12} {'% vs All':<10} {'% of AS':<10}")
    print("  " + "-"*80)
    
    for _, row in comparison_df.iterrows():
        deni_str = f"{row['Deni_Value']:.2f}" if row['Deni_Value'] < 10 else f"{row['Deni_Value']:.0f}"
        mean_str = f"{row['AllStar_Mean']:.2f}" if row['AllStar_Mean'] < 10 else f"{row['AllStar_Mean']:.0f}"
        print(f"  {row['Description']:<35} {deni_str:<12} {mean_str:<12} "
              f"{row['Pct_vs_All']:.0f}%{'':<6} {row['Pct_of_AllStar_Avg']:.0f}%")
    
    print("\n[MOST SIMILAR ALL-STARS]")
    print("-"*90)
    for _, row in similar_df.head(5).iterrows():
        name = unidecode(row['PLAYER_NAME'])
        print(f"  {name:<25} ({row['SEASON']}, {row['TEAM_ABBREVIATION']}) - Distance: {row['Distance_to_Deni']:.2f}")
    
    # Verdict
    print("\n[VERDICT]")
    print("-"*90)
    
    avg_pct = comparison_df['Pct_vs_All'].mean()
    avg_vs_star = comparison_df['Pct_of_AllStar_Avg'].mean()
    
    print(f"  Average Percentile (vs All Players): {avg_pct:.0f}%")
    print(f"  Average % of All-Star Average: {avg_vs_star:.0f}%")
    
    if avg_pct >= 85:
        verdict = "STRONG All-Star candidate based on ESTIMATED network metrics"
    elif avg_pct >= 70:
        verdict = "Solid All-Star candidate based on ESTIMATED network metrics"
    else:
        verdict = "Developing player based on ESTIMATED network metrics"
    
    print(f"\n  Verdict: {verdict}")
    print(f"\n  *** CAVEAT: This is based on ESTIMATED network values, not actual data ***")
    
    print("\n[PRESENTATION-READY STATEMENT]")
    print("-"*90)
    print(f"  'Based on estimated network metrics, Deni Avdija's profile ranks in the")
    print(f"   {avg_pct:.0f}th percentile among all NBA players and achieves {avg_vs_star:.0f}% of")
    print(f"   typical All-Star network involvement. His most similar historical All-Star")
    print(f"   profiles include {unidecode(similar_df.iloc[0]['PLAYER_NAME'])} ({similar_df.iloc[0]['SEASON']}).")
    print(f"   NOTE: Network metrics are estimated based on traditional stats, as actual")
    print(f"   2025-26 passing data is not yet available via NBA API.'")
    
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*70)
    print("DENI AVDIJA ALL-STAR ANALYSIS - FIXED VERSION")
    print("="*70)
    
    print("\n[LOADING ALL-STAR DATA]")
    player_df = load_allstar_data()
    
    all_stars = player_df[player_df['Is_Star'] == True]
    print(f"  All-Stars in dataset: {len(all_stars)}")
    
    print("\n[ESTIMATING DENI'S NETWORK METRICS]")
    print("  *** Note: These are ESTIMATES based on traditional stats ***")
    deni_metrics = estimate_deni_network_metrics()
    
    print("\n[COMPARING TO ALL-STARS]")
    comparison_df = compare_to_allstars(deni_metrics, player_df)
    
    print("\n[FINDING SIMILAR PLAYERS]")
    similar_df = find_similar_players(deni_metrics, player_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_analysis(deni_metrics, comparison_df, player_df, similar_df)
    
    print_summary(deni_metrics, comparison_df, similar_df, player_df)
    
    # Save data
    comparison_df.to_csv(OUTPUT_DIR / 'deni_comparison.csv', index=False)
    similar_df.to_csv(OUTPUT_DIR / 'similar_allstars.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
