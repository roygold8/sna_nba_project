"""
Deni Avdija All-Star Analysis - NETWORK METRICS ONLY
=====================================================
Pure SNA analysis without traditional stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_deni_network")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_historical_data():
    """Load historical player metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    
    # Calculate network-only ratios
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    player_df['Net_Pass_Flow'] = player_df['Weighted_In_Degree'] - player_df['Weighted_Out_Degree']
    player_df['Degree_Concentration'] = player_df['Weighted_Degree'] / player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_Degree'].transform('sum')
    
    # Identify Network All-Stars (purely by network metrics)
    team_stats = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std'],
        'Eigenvector_Centrality': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                          'Team_Degree_Mean', 'Team_Degree_Std',
                          'Team_Eigen_Mean', 'Team_Eigen_Std']
    
    player_df = player_df.merge(team_stats, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    player_df['Degree_ZScore'] = (player_df['Weighted_Degree'] - player_df['Team_Degree_Mean']) / (player_df['Team_Degree_Std'] + 1)
    player_df['Eigen_ZScore'] = (player_df['Eigenvector_Centrality'] - player_df['Team_Eigen_Mean']) / (player_df['Team_Eigen_Std'] + 0.01)
    
    # Network Star = top network position
    player_df['Is_Network_Star'] = (player_df['Degree_ZScore'] > 1.5) | (player_df['Eigen_ZScore'] > 1.5)
    
    print(f"Loaded {len(player_df)} player-seasons")
    print(f"Network Stars identified: {player_df['Is_Network_Star'].sum()}")
    
    return player_df


def get_deni_network_metrics():
    """Get Deni's estimated network metrics for 2025-26."""
    
    # Based on his 6.9 APG (elite for forward) and primary ball handler role
    # Estimate network metrics from similar playmaking forwards
    gp = 40
    
    # High-usage playmaking forward estimates:
    # - Ben Simmons (2020-21): ~7380 total degree, similar playmaking profile
    # - LeBron James style: 5500-6000 total degree
    # - Doncic style: 8000-9000 total degree
    # Deni with 6.9 APG and 26 PPG is in upper tier
    
    passes_made_per_game = 85   # High playmaker
    passes_received_per_game = 100  # Primary option receives more
    
    deni_metrics = {
        'PLAYER_NAME': 'Deni Avdija',
        'TEAM_ABBREVIATION': 'POR',
        'SEASON': '2025-26',
        'GP': gp,
        
        # Core Network Metrics
        'Weighted_Degree': gp * (passes_made_per_game + passes_received_per_game),  # 7400
        'Weighted_In_Degree': gp * passes_received_per_game,  # 4000
        'Weighted_Out_Degree': gp * passes_made_per_game,  # 3400
        
        # Degree counts (unique connections)
        'In_Degree': 12,  # Receives from all teammates
        'Out_Degree': 12,  # Passes to all teammates
        'Total_Degree': 24,
        
        # Centrality estimates (based on similar players)
        'Eigenvector_Centrality': 0.42,  # High for playmaker
        'Betweenness_Centrality': 0.008,  # Moderate (hub not bridge)
        
        # Calculated ratios
        'In_Out_Ratio': passes_received_per_game / passes_made_per_game,  # 1.18
        'Degree_Per_Game': passes_made_per_game + passes_received_per_game,  # 185
        'Net_Pass_Flow': gp * (passes_received_per_game - passes_made_per_game),  # +600
        'Degree_Concentration': 0.18,  # ~18% of team's passes involve him
    }
    
    return deni_metrics


def compare_network_metrics(deni_metrics, player_df):
    """Compare Deni's network metrics to All-Stars."""
    
    network_stars = player_df[player_df['Is_Network_Star']]
    all_players = player_df
    
    # Network-only metrics
    metrics = [
        'Weighted_Degree', 'Weighted_In_Degree', 'Weighted_Out_Degree',
        'In_Out_Ratio', 'Degree_Per_Game', 'Net_Pass_Flow',
        'Eigenvector_Centrality', 'Betweenness_Centrality',
        'In_Degree', 'Out_Degree', 'Total_Degree'
    ]
    
    comparison = []
    
    for metric in metrics:
        if metric not in deni_metrics or metric not in player_df.columns:
            continue
        
        deni_val = deni_metrics[metric]
        
        star_mean = network_stars[metric].mean()
        star_std = network_stars[metric].std()
        all_mean = all_players[metric].mean()
        all_std = all_players[metric].std()
        
        pct_all = (all_players[metric] < deni_val).mean() * 100
        pct_stars = (network_stars[metric] < deni_val).mean() * 100
        
        z_all = (deni_val - all_mean) / all_std if all_std > 0 else 0
        z_star = (deni_val - star_mean) / star_std if star_std > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Deni': deni_val,
            'Network_Star_Mean': star_mean,
            'All_Players_Mean': all_mean,
            'Pct_vs_All': pct_all,
            'Pct_vs_Stars': pct_stars,
            'Z_vs_All': z_all,
            'Z_vs_Stars': z_star
        })
    
    return pd.DataFrame(comparison)


def find_similar_network_profiles(deni_metrics, player_df, top_n=15):
    """Find players with similar NETWORK profiles."""
    
    # Network-only metrics for similarity
    metrics = ['Weighted_Degree', 'In_Out_Ratio', 'Degree_Per_Game', 
               'Eigenvector_Centrality', 'Net_Pass_Flow']
    
    player_df_work = player_df.copy()
    deni_norm = {}
    
    for metric in metrics:
        if metric in player_df.columns and metric in deni_metrics:
            mean = player_df[metric].mean()
            std = player_df[metric].std()
            player_df_work[f'{metric}_norm'] = (player_df[metric] - mean) / (std + 1)
            deni_norm[f'{metric}_norm'] = (deni_metrics[metric] - mean) / (std + 1)
    
    distances = []
    for idx, row in player_df_work.iterrows():
        dist = 0
        for metric in metrics:
            if f'{metric}_norm' in deni_norm:
                dist += (row[f'{metric}_norm'] - deni_norm[f'{metric}_norm']) ** 2
        distances.append(np.sqrt(dist))
    
    player_df_work['Distance'] = distances
    
    similar = player_df_work.nsmallest(top_n, 'Distance')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 
         'Weighted_Degree', 'In_Out_Ratio', 'Degree_Per_Game',
         'Eigenvector_Centrality', 'Is_Network_Star', 'Distance']
    ]
    
    return similar


def plot_deni_network_analysis(deni_metrics, comparison_df, player_df):
    """Create network-only visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    stars = player_df[player_df['Is_Network_Star']]
    others = player_df[~player_df['Is_Network_Star']]
    
    # 1. Weighted Degree Distribution
    ax1 = axes[0, 0]
    ax1.hist(others['Weighted_Degree'], bins=50, alpha=0.5, label='Role Players', color='gray')
    ax1.hist(stars['Weighted_Degree'], bins=30, alpha=0.7, label='Network Stars', color='gold')
    ax1.axvline(x=deni_metrics['Weighted_Degree'], color='red', linestyle='--', lw=3, 
                label=f"Deni: {deni_metrics['Weighted_Degree']:.0f}")
    ax1.set_xlabel('Weighted Degree (Total Pass Volume)')
    ax1.set_ylabel('Count')
    ax1.set_title('Weighted Degree Distribution', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # 2. Degree Per Game
    ax2 = axes[0, 1]
    ax2.hist(others['Degree_Per_Game'], bins=50, alpha=0.5, label='Role Players', color='gray')
    ax2.hist(stars['Degree_Per_Game'], bins=30, alpha=0.7, label='Network Stars', color='gold')
    ax2.axvline(x=deni_metrics['Degree_Per_Game'], color='red', linestyle='--', lw=3,
                label=f"Deni: {deni_metrics['Degree_Per_Game']:.0f}")
    ax2.set_xlabel('Degree Per Game')
    ax2.set_ylabel('Count')
    ax2.set_title('Network Involvement Per Game', fontsize=11, fontweight='bold')
    ax2.legend()
    
    # 3. Eigenvector Centrality
    ax3 = axes[0, 2]
    ax3.hist(others['Eigenvector_Centrality'], bins=50, alpha=0.5, label='Role Players', color='gray')
    ax3.hist(stars['Eigenvector_Centrality'], bins=30, alpha=0.7, label='Network Stars', color='gold')
    ax3.axvline(x=deni_metrics['Eigenvector_Centrality'], color='red', linestyle='--', lw=3,
                label=f"Deni: {deni_metrics['Eigenvector_Centrality']:.2f}")
    ax3.set_xlabel('Eigenvector Centrality')
    ax3.set_ylabel('Count')
    ax3.set_title('Eigenvector Centrality (Influence)', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # 4. In-Degree vs Out-Degree scatter
    ax4 = axes[1, 0]
    ax4.scatter(others['Weighted_Out_Degree'], others['Weighted_In_Degree'], 
                alpha=0.3, s=20, c='gray', label='Role Players')
    ax4.scatter(stars['Weighted_Out_Degree'], stars['Weighted_In_Degree'], 
                alpha=0.7, s=40, c='gold', edgecolors='black', linewidths=0.5, label='Network Stars')
    ax4.scatter(deni_metrics['Weighted_Out_Degree'], deni_metrics['Weighted_In_Degree'], 
                c='red', s=200, marker='*', edgecolors='black', linewidths=2, 
                label='DENI AVDIJA', zorder=10)
    
    # 45-degree line
    max_val = max(player_df['Weighted_Out_Degree'].max(), player_df['Weighted_In_Degree'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5)
    ax4.set_xlabel('Weighted Out-Degree (Passes Made)')
    ax4.set_ylabel('Weighted In-Degree (Passes Received)')
    ax4.set_title('Pass Flow: In vs Out', fontsize=11, fontweight='bold')
    ax4.legend()
    
    # 5. Percentile Rankings
    ax5 = axes[1, 1]
    key_metrics = comparison_df[comparison_df['Metric'].isin(
        ['Weighted_Degree', 'Degree_Per_Game', 'Eigenvector_Centrality', 'In_Out_Ratio', 'Weighted_In_Degree']
    )]
    
    colors = ['green' if p >= 75 else 'orange' if p >= 50 else 'red' for p in key_metrics['Pct_vs_All']]
    bars = ax5.barh(key_metrics['Metric'], key_metrics['Pct_vs_All'], color=colors, edgecolor='black')
    ax5.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Median')
    ax5.axvline(x=75, color='gold', linestyle='--', alpha=0.7, label='75th')
    ax5.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='90th (Elite)')
    ax5.set_xlabel('Percentile (vs All Players)')
    ax5.set_title("Deni's Network Percentile Rankings", fontsize=11, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.set_xlim(0, 100)
    
    for bar, pct in zip(bars, key_metrics['Pct_vs_All']):
        ax5.text(pct + 2, bar.get_y() + bar.get_height()/2, f'{pct:.0f}%',
                va='center', fontsize=9, fontweight='bold')
    
    # 6. Comparison to Network Star Average
    ax6 = axes[1, 2]
    compare_metrics = comparison_df[comparison_df['Metric'].isin(
        ['Weighted_Degree', 'Degree_Per_Game', 'Eigenvector_Centrality', 'Weighted_In_Degree', 'Weighted_Out_Degree']
    )]
    
    x = np.arange(len(compare_metrics))
    width = 0.35
    
    deni_pct = (compare_metrics['Deni'].values / compare_metrics['Network_Star_Mean'].values) * 100
    star_pct = np.ones(len(compare_metrics)) * 100
    
    ax6.bar(x - width/2, deni_pct, width, label='Deni', color='red')
    ax6.bar(x + width/2, star_pct, width, label='Network Star Avg', color='gold')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels([m[:15] for m in compare_metrics['Metric']], rotation=30, ha='right')
    ax6.set_ylabel('% of Network Star Average')
    ax6.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax6.set_title('Deni vs Network Star Benchmarks', fontsize=11, fontweight='bold')
    ax6.legend()
    
    plt.suptitle(f"DENI AVDIJA NETWORK ANALYSIS\n{deni_metrics['TEAM_ABBREVIATION']} - {deni_metrics['SEASON']} (Network Metrics Only)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_deni_network_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_deni_network_analysis.png")


def print_analysis(deni_metrics, comparison_df, similar_players):
    """Print network-only analysis."""
    
    print("\n" + "="*80)
    print("DENI AVDIJA - NETWORK METRICS ANALYSIS (No Traditional Stats)")
    print("="*80)
    
    print(f"\n[SEASON: {deni_metrics['SEASON']} | Team: {deni_metrics['TEAM_ABBREVIATION']}]")
    
    print("\n[CORE NETWORK METRICS]")
    print("-" * 50)
    print(f"  Weighted Degree (Total Pass Volume): {deni_metrics['Weighted_Degree']:.0f}")
    print(f"  Weighted In-Degree (Passes Received): {deni_metrics['Weighted_In_Degree']:.0f}")
    print(f"  Weighted Out-Degree (Passes Made): {deni_metrics['Weighted_Out_Degree']:.0f}")
    print(f"  Net Pass Flow: {deni_metrics['Net_Pass_Flow']:+.0f}")
    
    print("\n[NETWORK POSITION METRICS]")
    print("-" * 50)
    print(f"  Degree Per Game: {deni_metrics['Degree_Per_Game']:.1f}")
    print(f"  In/Out Ratio: {deni_metrics['In_Out_Ratio']:.2f}")
    print(f"  Eigenvector Centrality: {deni_metrics['Eigenvector_Centrality']:.3f}")
    print(f"  Betweenness Centrality: {deni_metrics['Betweenness_Centrality']:.4f}")
    
    print("\n[COMPARISON TO NETWORK STARS (Historical All-Stars by Network)]")
    print("-" * 75)
    print(f"{'Metric':<25} {'Deni':>10} {'Star Avg':>12} {'Pct vs All':>12} {'Pct vs Stars':>12}")
    print("-" * 75)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Metric']:<25} {row['Deni']:>10.1f} {row['Network_Star_Mean']:>12.1f} "
              f"{row['Pct_vs_All']:>11.0f}% {row['Pct_vs_Stars']:>11.0f}%")
    
    print("\n[MOST SIMILAR NETWORK PROFILES (Historical)]")
    print("-" * 80)
    
    star_count = 0
    for i, (_, row) in enumerate(similar_players.iterrows(), 1):
        name = unidecode(row['PLAYER_NAME'])
        star = "[NETWORK STAR]" if row['Is_Network_Star'] else ""
        if row['Is_Network_Star']:
            star_count += 1
        print(f"{i:2d}. {name:<22} ({row['TEAM_ABBREVIATION']} {row['SEASON']}) "
              f"Deg: {row['Weighted_Degree']:.0f}, DPG: {row['Degree_Per_Game']:.0f}, "
              f"Eigen: {row['Eigenvector_Centrality']:.2f} {star}")
    
    # Verdict
    print("\n" + "="*80)
    print("ALL-STAR VERDICT (Based on Network Position Only)")
    print("="*80)
    
    degree_pct = comparison_df[comparison_df['Metric'] == 'Weighted_Degree']['Pct_vs_All'].values[0]
    dpg_pct = comparison_df[comparison_df['Metric'] == 'Degree_Per_Game']['Pct_vs_All'].values[0]
    eigen_pct = comparison_df[comparison_df['Metric'] == 'Eigenvector_Centrality']['Pct_vs_All'].values[0]
    
    print(f"\nDeni's Network Percentiles:")
    print(f"  - Weighted Degree: {degree_pct:.0f}th percentile")
    print(f"  - Degree Per Game: {dpg_pct:.0f}th percentile")
    print(f"  - Eigenvector Centrality: {eigen_pct:.0f}th percentile")
    
    avg_pct = (degree_pct + dpg_pct + eigen_pct) / 3
    star_similarity = (star_count / len(similar_players)) * 100
    
    print(f"\n  Average Network Percentile: {avg_pct:.0f}%")
    print(f"  Similar Players who were Network Stars: {star_count}/{len(similar_players)} ({star_similarity:.0f}%)")
    
    # Final verdict
    if avg_pct >= 85 and star_similarity >= 70:
        verdict = "STRONG Network All-Star Profile"
        symbol = "[STRONG]"
    elif avg_pct >= 75 and star_similarity >= 50:
        verdict = "Solid Network All-Star Candidate"
        symbol = "[SOLID]"
    elif avg_pct >= 60:
        verdict = "Borderline All-Star Network Profile"
        symbol = "[BORDERLINE]"
    else:
        verdict = "Below All-Star Network Threshold"
        symbol = "[DEVELOPING]"
    
    print(f"\n{symbol} VERDICT: {verdict}")
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("DENI AVDIJA - NETWORK METRICS ONLY ANALYSIS")
    print("="*60)
    
    print("\n[LOADING HISTORICAL DATA]")
    player_df = load_historical_data()
    
    print("\n[GETTING DENI'S NETWORK METRICS]")
    deni_metrics = get_deni_network_metrics()
    print(f"Estimated Weighted Degree: {deni_metrics['Weighted_Degree']}")
    print(f"Estimated Degree Per Game: {deni_metrics['Degree_Per_Game']}")
    
    print("\n[COMPARING TO NETWORK STARS]")
    comparison_df = compare_network_metrics(deni_metrics, player_df)
    
    print("\n[FINDING SIMILAR NETWORK PROFILES]")
    similar_players = find_similar_network_profiles(deni_metrics, player_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_deni_network_analysis(deni_metrics, comparison_df, player_df)
    
    print_analysis(deni_metrics, comparison_df, similar_players)
    
    comparison_df.to_csv(OUTPUT_DIR / 'deni_network_comparison.csv', index=False)
    similar_players.to_csv(OUTPUT_DIR / 'deni_similar_network_profiles.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
