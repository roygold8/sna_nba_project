"""
Deni Avdija vs REAL NBA All-Stars Network Analysis
===================================================
Uses actual All-Star selections (MVP, All-NBA, All-Star) from Is_Star column
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_deni_vs_allstars")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_real_allstar_data():
    """Load data with actual NBA All-Star designations."""
    # Use the file with Is_Star column (actual All-Stars)
    player_df = pd.read_csv("output/nba_player_metrics_with_stars.csv")
    
    all_stars = player_df[player_df['Is_Star'] == True]
    role_players = player_df[player_df['Is_Star'] == False]
    
    print(f"Total player-seasons: {len(player_df)}")
    print(f"REAL NBA All-Stars: {len(all_stars)} ({len(all_stars)/len(player_df)*100:.1f}%)")
    print(f"Role Players: {len(role_players)} ({len(role_players)/len(player_df)*100:.1f}%)")
    
    # Add derived metrics
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Out_In_Ratio'] = player_df['Weighted_Out_Degree'] / (player_df['Weighted_In_Degree'] + 1)
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    player_df['In_Degree_Per_Game'] = player_df['Weighted_In_Degree'] / (player_df['GP'] + 1)
    player_df['Out_Degree_Per_Game'] = player_df['Weighted_Out_Degree'] / (player_df['GP'] + 1)
    
    # Team concentration
    team_totals = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_Degree'].sum().reset_index()
    team_totals.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Team_Total_Degree']
    player_df = player_df.merge(team_totals, on=['TEAM_ABBREVIATION', 'SEASON'])
    player_df['Degree_Concentration'] = player_df['Weighted_Degree'] / player_df['Team_Total_Degree']
    
    # Unique All-Stars by name
    unique_allstars = all_stars['PLAYER_NAME'].unique()
    print(f"\nUnique All-Star players: {len(unique_allstars)}")
    print("\nSample All-Stars in dataset:")
    sample = all_stars.groupby('PLAYER_NAME').first().reset_index()[['PLAYER_NAME', 'SEASON']].head(15)
    for _, row in sample.iterrows():
        print(f"  - {unidecode(row['PLAYER_NAME'])} ({row['SEASON']})")
    
    return player_df


def get_deni_metrics():
    """Get Deni's estimated network metrics for 2025-26."""
    gp = 40
    passes_made_per_game = 85
    passes_received_per_game = 100
    
    deni_metrics = {
        'PLAYER_NAME': 'Deni Avdija',
        'TEAM_ABBREVIATION': 'POR',
        'SEASON': '2025-26',
        'GP': gp,
        'Weighted_Degree': gp * (passes_made_per_game + passes_received_per_game),
        'Weighted_In_Degree': gp * passes_received_per_game,
        'Weighted_Out_Degree': gp * passes_made_per_game,
        'Eigenvector_Centrality': 0.42,
        'Betweenness_Centrality': 0.008,
        'Degree_Per_Game': passes_made_per_game + passes_received_per_game,
        'In_Degree_Per_Game': passes_received_per_game,
        'Out_Degree_Per_Game': passes_made_per_game,
        'In_Out_Ratio': passes_received_per_game / passes_made_per_game,
        'Out_In_Ratio': passes_made_per_game / passes_received_per_game,
        'Net_Pass_Flow': gp * (passes_received_per_game - passes_made_per_game),
        'Degree_Concentration': 0.18,
        'Black_Hole_Ratio': passes_received_per_game / (passes_made_per_game + 1),
    }
    
    return deni_metrics


def compare_to_real_allstars(deni_metrics, player_df):
    """Compare Deni to REAL NBA All-Stars."""
    
    all_stars = player_df[player_df['Is_Star'] == True]
    all_players = player_df
    
    metrics = [
        ('Weighted_Degree', 'Total Pass Volume'),
        ('Weighted_In_Degree', 'Passes Received'),
        ('Weighted_Out_Degree', 'Passes Made'),
        ('Eigenvector_Centrality', 'Eigenvector Centrality'),
        ('Betweenness_Centrality', 'Betweenness Centrality'),
        ('Degree_Per_Game', 'Degree Per Game'),
        ('In_Degree_Per_Game', 'Received Per Game'),
        ('Out_Degree_Per_Game', 'Made Per Game'),
        ('In_Out_Ratio', 'In/Out Ratio'),
        ('Out_In_Ratio', 'Out/In Ratio'),
        ('Net_Pass_Flow', 'Net Pass Flow'),
        ('Degree_Concentration', 'Team Degree Share'),
        ('Black_Hole_Ratio', 'Black Hole Ratio'),
    ]
    
    comparison = []
    
    for metric, description in metrics:
        if metric not in deni_metrics or metric not in player_df.columns:
            continue
        
        deni_val = deni_metrics[metric]
        
        star_mean = all_stars[metric].mean()
        star_std = all_stars[metric].std()
        star_median = all_stars[metric].median()
        star_min = all_stars[metric].min()
        star_max = all_stars[metric].max()
        all_mean = all_players[metric].mean()
        all_std = all_players[metric].std()
        
        pct_all = (all_players[metric] < deni_val).mean() * 100
        pct_stars = (all_stars[metric] < deni_val).mean() * 100
        
        z_all = (deni_val - all_mean) / all_std if all_std > 0 else 0
        z_star = (deni_val - star_mean) / star_std if star_std > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Description': description,
            'Deni': deni_val,
            'AllStar_Mean': star_mean,
            'AllStar_Median': star_median,
            'AllStar_Std': star_std,
            'AllStar_Min': star_min,
            'AllStar_Max': star_max,
            'All_Mean': all_mean,
            'Pct_vs_All': pct_all,
            'Pct_vs_AllStars': pct_stars,
            'Z_vs_All': z_all,
            'Z_vs_AllStars': z_star,
            'vs_AllStar_Avg': (deni_val / star_mean * 100) if star_mean != 0 else 0
        })
    
    return pd.DataFrame(comparison)


def plot_deni_vs_allstars(deni_metrics, comparison_df, player_df):
    """Create comparison visualizations with REAL All-Stars."""
    
    fig = plt.figure(figsize=(20, 16))
    
    all_stars = player_df[player_df['Is_Star'] == True]
    role_players = player_df[player_df['Is_Star'] == False]
    
    # 1. Radar Chart
    ax1 = fig.add_subplot(2, 3, 1, polar=True)
    
    radar_metrics = ['Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
                     'Degree_Per_Game', 'In_Out_Ratio', 'Degree_Concentration']
    radar_labels = ['Total Degree', 'Eigenvector', 'Betweenness', 
                    'Degree/Game', 'In/Out Ratio', 'Team Share']
    
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
    ax1.set_title('Deni vs REAL All-Star Average\n(1.0 = All-Star Avg)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 2. Percentile Rankings
    ax2 = fig.add_subplot(2, 3, 2)
    
    key_metrics = comparison_df.sort_values('Pct_vs_All', ascending=True)
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
    
    # 3. Eigenvector vs Betweenness (REAL All-Stars)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(role_players['Eigenvector_Centrality'], role_players['Betweenness_Centrality'],
                alpha=0.2, s=15, c='gray', label=f'Role Players (n={len(role_players)})')
    ax3.scatter(all_stars['Eigenvector_Centrality'], all_stars['Betweenness_Centrality'],
                alpha=0.6, s=50, c='gold', edgecolors='black', linewidths=0.5, 
                label=f'NBA All-Stars (n={len(all_stars)})')
    ax3.scatter(deni_metrics['Eigenvector_Centrality'], deni_metrics['Betweenness_Centrality'],
                c='red', s=250, marker='*', edgecolors='black', linewidths=2, 
                label='DENI 2025-26', zorder=10)
    ax3.set_xlabel('Eigenvector Centrality (Influence)')
    ax3.set_ylabel('Betweenness Centrality (Bridge)')
    ax3.set_title('Centrality: Eigenvector vs Betweenness\n(Real NBA All-Stars)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    
    # 4. Weighted Degree Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(role_players['Weighted_Degree'], bins=50, alpha=0.4, label='Role Players', color='gray')
    ax4.hist(all_stars['Weighted_Degree'], bins=30, alpha=0.7, label='NBA All-Stars', color='gold')
    ax4.axvline(x=deni_metrics['Weighted_Degree'], color='red', linestyle='--', lw=3,
                label=f"Deni: {deni_metrics['Weighted_Degree']:.0f}")
    ax4.axvline(x=all_stars['Weighted_Degree'].mean(), color='darkorange', linestyle='-', lw=2,
                label=f"All-Star Avg: {all_stars['Weighted_Degree'].mean():.0f}")
    ax4.set_xlabel('Weighted Degree (Total Pass Volume)')
    ax4.set_ylabel('Count')
    ax4.set_title('Total Pass Volume Distribution\n(Real NBA All-Stars)', fontsize=11, fontweight='bold')
    ax4.legend()
    
    # 5. In-Degree vs Out-Degree
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(role_players['Weighted_Out_Degree'], role_players['Weighted_In_Degree'],
                alpha=0.2, s=15, c='gray', label='Role Players')
    ax5.scatter(all_stars['Weighted_Out_Degree'], all_stars['Weighted_In_Degree'],
                alpha=0.6, s=50, c='gold', edgecolors='black', linewidths=0.5, label='NBA All-Stars')
    ax5.scatter(deni_metrics['Weighted_Out_Degree'], deni_metrics['Weighted_In_Degree'],
                c='red', s=250, marker='*', edgecolors='black', linewidths=2, 
                label='DENI', zorder=10)
    
    max_val = max(player_df['Weighted_Out_Degree'].max(), player_df['Weighted_In_Degree'].max())
    ax5.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5)
    ax5.set_xlabel('Passes Made')
    ax5.set_ylabel('Passes Received')
    ax5.set_title('Pass Flow: Made vs Received\n(Real NBA All-Stars)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    
    # 6. Degree Per Game Distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(role_players['Degree_Per_Game'], bins=50, alpha=0.4, label='Role Players', color='gray')
    ax6.hist(all_stars['Degree_Per_Game'], bins=30, alpha=0.7, label='NBA All-Stars', color='gold')
    ax6.axvline(x=deni_metrics['Degree_Per_Game'], color='red', linestyle='--', lw=3,
                label=f"Deni: {deni_metrics['Degree_Per_Game']:.0f}")
    ax6.axvline(x=all_stars['Degree_Per_Game'].mean(), color='darkorange', linestyle='-', lw=2,
                label=f"All-Star Avg: {all_stars['Degree_Per_Game'].mean():.0f}")
    ax6.set_xlabel('Degree Per Game (Ball Involvement)')
    ax6.set_ylabel('Count')
    ax6.set_title('Ball Involvement Per Game\n(Real NBA All-Stars)', fontsize=11, fontweight='bold')
    ax6.legend()
    
    plt.suptitle(f"DENI AVDIJA vs REAL NBA ALL-STARS\n{deni_metrics['TEAM_ABBREVIATION']} - {deni_metrics['SEASON']} | Comparison: 366 All-Star Seasons",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_deni_vs_real_allstars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_deni_vs_real_allstars.png")


def plot_comparison_table(comparison_df):
    """Create detailed comparison table."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    table_data = comparison_df[['Description', 'Deni', 'AllStar_Mean', 'AllStar_Median', 
                                 'Pct_vs_All', 'Pct_vs_AllStars', 'vs_AllStar_Avg']].copy()
    
    # Format numbers
    table_data['Deni'] = table_data['Deni'].apply(lambda x: f'{x:.3f}' if x < 10 else f'{x:.0f}')
    table_data['AllStar_Mean'] = table_data['AllStar_Mean'].apply(lambda x: f'{x:.3f}' if x < 10 else f'{x:.0f}')
    table_data['AllStar_Median'] = table_data['AllStar_Median'].apply(lambda x: f'{x:.3f}' if x < 10 else f'{x:.0f}')
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
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code cells based on percentile
    for i in range(len(table_data)):
        pct_all = float(table_data.iloc[i]['Pct vs\nAll'].replace('%', ''))
        if pct_all >= 90:
            table[(i+1, 4)].set_facecolor('lightgreen')
        elif pct_all >= 75:
            table[(i+1, 4)].set_facecolor('lightyellow')
        elif pct_all < 50:
            table[(i+1, 4)].set_facecolor('lightcoral')
    
    plt.title('DENI AVDIJA vs REAL NBA ALL-STARS (366 All-Star Seasons)\nNetwork Metrics Comparison',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_deni_allstar_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_deni_allstar_comparison_table.png")


def find_similar_allstars(deni_metrics, player_df):
    """Find All-Stars with similar network profiles."""
    
    all_stars = player_df[player_df['Is_Star'] == True].copy()
    
    # Normalize metrics for comparison
    metrics_to_compare = ['Weighted_Degree', 'Eigenvector_Centrality', 'Betweenness_Centrality',
                          'In_Out_Ratio', 'Degree_Per_Game']
    
    for metric in metrics_to_compare:
        if metric in all_stars.columns:
            mean = all_stars[metric].mean()
            std = all_stars[metric].std()
            all_stars[f'{metric}_norm'] = (all_stars[metric] - mean) / (std + 0.001)
    
    # Calculate Deni's normalized values
    deni_norm = {}
    for metric in metrics_to_compare:
        if metric in deni_metrics:
            mean = all_stars[metric].mean()
            std = all_stars[metric].std()
            deni_norm[f'{metric}_norm'] = (deni_metrics[metric] - mean) / (std + 0.001)
    
    # Calculate distance
    def calc_distance(row):
        dist = 0
        for metric in metrics_to_compare:
            if f'{metric}_norm' in deni_norm and f'{metric}_norm' in row.index:
                dist += (row[f'{metric}_norm'] - deni_norm[f'{metric}_norm'])**2
        return np.sqrt(dist)
    
    all_stars['Distance_to_Deni'] = all_stars.apply(calc_distance, axis=1)
    similar = all_stars.nsmallest(15, 'Distance_to_Deni')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'Weighted_Degree', 
         'Eigenvector_Centrality', 'Betweenness_Centrality', 'Distance_to_Deni']
    ]
    
    return similar


def print_full_analysis(deni_metrics, comparison_df, player_df, similar_allstars):
    """Print comprehensive analysis."""
    
    all_stars = player_df[player_df['Is_Star'] == True]
    
    print("\n" + "="*90)
    print("DENI AVDIJA vs REAL NBA ALL-STARS - NETWORK METRICS COMPARISON")
    print("="*90)
    
    print(f"\n[SEASON: {deni_metrics['SEASON']} | Team: {deni_metrics['TEAM_ABBREVIATION']}]")
    print(f"[Comparison Base: {len(all_stars)} REAL All-Star Seasons (MVP, All-NBA, All-Star)]")
    
    print("\n" + "="*90)
    print("DETAILED METRICS COMPARISON")
    print("="*90)
    
    print("\n{:<30} {:>10} {:>12} {:>12} {:>10} {:>10}".format(
        'Metric', 'Deni', 'AS Mean', 'AS Median', 'Pct All', 'vs AS Avg'
    ))
    print("-" * 85)
    
    for _, row in comparison_df.iterrows():
        deni_str = f"{row['Deni']:.2f}" if row['Deni'] < 10 else f"{row['Deni']:.0f}"
        mean_str = f"{row['AllStar_Mean']:.2f}" if row['AllStar_Mean'] < 10 else f"{row['AllStar_Mean']:.0f}"
        med_str = f"{row['AllStar_Median']:.2f}" if row['AllStar_Median'] < 10 else f"{row['AllStar_Median']:.0f}"
        print(f"{row['Description']:<30} {deni_str:>10} {mean_str:>12} {med_str:>12} "
              f"{row['Pct_vs_All']:>9.0f}% {row['vs_AllStar_Avg']:>9.0f}%")
    
    print("\n" + "="*90)
    print("MOST SIMILAR ALL-STARS TO DENI'S NETWORK PROFILE")
    print("="*90)
    
    print("\n{:<25} {:>6} {:>10} {:>10} {:>12} {:>10}".format(
        'Player', 'Season', 'Team', 'Degree', 'Eigenvector', 'Distance'
    ))
    print("-" * 75)
    for _, row in similar_allstars.iterrows():
        name = unidecode(row['PLAYER_NAME'])[:24]
        print(f"{name:<25} {row['SEASON']:>6} {row['TEAM_ABBREVIATION']:>10} "
              f"{row['Weighted_Degree']:>10.0f} {row['Eigenvector_Centrality']:>12.3f} "
              f"{row['Distance_to_Deni']:>10.2f}")
    
    # Summary
    print("\n" + "="*90)
    print("VERDICT: DENI vs REAL NBA ALL-STARS")
    print("="*90)
    
    key_metrics = ['Weighted_Degree', 'Eigenvector_Centrality', 'Degree_Per_Game']
    avg_pct = comparison_df[comparison_df['Metric'].isin(key_metrics)]['Pct_vs_All'].mean()
    avg_vs_star = comparison_df[comparison_df['Metric'].isin(key_metrics)]['vs_AllStar_Avg'].mean()
    pct_vs_allstars = comparison_df[comparison_df['Metric'].isin(key_metrics)]['Pct_vs_AllStars'].mean()
    
    print(f"\nKey Network Metrics Summary:")
    print(f"  - Average Percentile (vs All Players): {avg_pct:.0f}%")
    print(f"  - Average % of All-Star Average: {avg_vs_star:.0f}%")
    print(f"  - Average Percentile (vs All-Stars): {pct_vs_allstars:.0f}%")
    
    strengths = comparison_df[comparison_df['Pct_vs_All'] >= 85]['Description'].tolist()
    weaknesses = comparison_df[comparison_df['Pct_vs_All'] < 50]['Description'].tolist()
    
    print(f"\n  STRENGTHS (>= 85th percentile of ALL players):")
    for s in strengths:
        print(f"    [+] {s}")
    
    if weaknesses:
        print(f"\n  AREAS BELOW MEDIAN:")
        for w in weaknesses:
            print(f"    [-] {w}")
    
    # Final verdict
    if avg_pct >= 85 and avg_vs_star >= 95:
        verdict = "ELITE All-Star Network Profile"
        symbol = "[ELITE]"
    elif avg_pct >= 75:
        verdict = "STRONG All-Star Candidate"
        symbol = "[STRONG]"
    elif avg_pct >= 60:
        verdict = "Solid All-Star Candidate"
        symbol = "[SOLID]"
    else:
        verdict = "Below All-Star Average"
        symbol = "[DEVELOPING]"
    
    print(f"\n{symbol} FINAL VERDICT: {verdict}")
    
    # Compare to similar players
    print(f"\nMost Similar NBA All-Star: {unidecode(similar_allstars.iloc[0]['PLAYER_NAME'])} "
          f"({similar_allstars.iloc[0]['SEASON']}, {similar_allstars.iloc[0]['TEAM_ABBREVIATION']})")
    
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*60)
    print("DENI AVDIJA vs REAL NBA ALL-STARS")
    print("="*60)
    
    print("\n[LOADING DATA WITH REAL ALL-STAR DESIGNATIONS]")
    player_df = load_real_allstar_data()
    
    print("\n[GETTING DENI'S NETWORK METRICS]")
    deni_metrics = get_deni_metrics()
    
    print("\n[COMPARING TO REAL ALL-STARS]")
    comparison_df = compare_to_real_allstars(deni_metrics, player_df)
    
    print("\n[FINDING SIMILAR ALL-STARS]")
    similar_allstars = find_similar_allstars(deni_metrics, player_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_deni_vs_allstars(deni_metrics, comparison_df, player_df)
    plot_comparison_table(comparison_df)
    
    print_full_analysis(deni_metrics, comparison_df, player_df, similar_allstars)
    
    comparison_df.to_csv(OUTPUT_DIR / 'deni_vs_real_allstars.csv', index=False)
    similar_allstars.to_csv(OUTPUT_DIR / 'similar_allstars.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
