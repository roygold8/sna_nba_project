"""
Density & Std Degree: Playoffs vs Regular Season
=================================================
Compare network structure between playoff and regular season games
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import time

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_playoffs_density")
OUTPUT_DIR.mkdir(exist_ok=True)

# Playoff teams by season
PLAYOFF_TEAMS = {
    '2015-16': ['GSW', 'SAS', 'OKC', 'LAC', 'POR', 'DAL', 'MEM', 'HOU',
                'CLE', 'TOR', 'MIA', 'ATL', 'BOS', 'CHA', 'IND', 'DET'],
    '2016-17': ['GSW', 'SAS', 'HOU', 'LAC', 'UTA', 'OKC', 'MEM', 'POR',
                'CLE', 'BOS', 'TOR', 'WAS', 'ATL', 'MIL', 'IND', 'CHI'],
    '2017-18': ['HOU', 'GSW', 'POR', 'OKC', 'UTA', 'NOP', 'SAS', 'MIN',
                'TOR', 'BOS', 'PHI', 'CLE', 'IND', 'MIA', 'MIL', 'WAS'],
    '2018-19': ['GSW', 'DEN', 'POR', 'HOU', 'UTA', 'OKC', 'SAS', 'LAC',
                'MIL', 'TOR', 'PHI', 'BOS', 'IND', 'BKN', 'ORL', 'DET'],
    '2019-20': ['LAL', 'LAC', 'DEN', 'HOU', 'OKC', 'UTA', 'DAL', 'POR',
                'MIL', 'TOR', 'BOS', 'MIA', 'IND', 'PHI', 'BKN', 'ORL'],
    '2020-21': ['UTA', 'PHX', 'DEN', 'LAC', 'DAL', 'POR', 'LAL', 'MEM',
                'PHI', 'BKN', 'MIL', 'NYK', 'ATL', 'MIA', 'BOS', 'WAS'],
    '2021-22': ['PHX', 'MEM', 'GSW', 'DAL', 'UTA', 'DEN', 'MIN', 'NOP',
                'MIA', 'BOS', 'MIL', 'PHI', 'TOR', 'CHI', 'BKN', 'ATL'],
    '2022-23': ['DEN', 'MEM', 'SAC', 'PHX', 'LAC', 'GSW', 'LAL', 'MIN',
                'MIL', 'BOS', 'PHI', 'CLE', 'NYK', 'BKN', 'MIA', 'ATL'],
    '2023-24': ['OKC', 'DEN', 'MIN', 'LAC', 'DAL', 'PHX', 'NOP', 'LAL',
                'BOS', 'NYK', 'MIL', 'CLE', 'ORL', 'IND', 'PHI', 'MIA']
}

# Deep playoff runs (Conference Finals or better)
DEEP_RUNS = {
    '2015-16': ['GSW', 'OKC', 'CLE', 'TOR'],
    '2016-17': ['GSW', 'SAS', 'CLE', 'BOS'],
    '2017-18': ['GSW', 'HOU', 'CLE', 'BOS'],
    '2018-19': ['GSW', 'POR', 'MIL', 'TOR'],
    '2019-20': ['LAL', 'DEN', 'MIA', 'BOS'],
    '2020-21': ['PHX', 'LAC', 'MIL', 'ATL'],
    '2021-22': ['GSW', 'DAL', 'BOS', 'MIA'],
    '2022-23': ['DEN', 'LAL', 'MIA', 'BOS'],
    '2023-24': ['DAL', 'MIN', 'BOS', 'IND']
}

# Champions
CHAMPIONS = {
    '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW', '2018-19': 'TOR',
    '2019-20': 'LAL', '2020-21': 'MIL', '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
}


def calculate_entropy(degrees):
    """Calculate Shannon Entropy."""
    degrees = np.array(degrees)
    if len(degrees) == 0 or np.sum(degrees) == 0:
        return 0
    probs = degrees / np.sum(degrees)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(degrees))
    return entropy / max_entropy if max_entropy > 0 else 0


def calculate_team_metrics(player_df):
    """Calculate density and std degree for each team-season."""
    
    team_metrics = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        degrees = group['Weighted_Degree'].values
        
        if len(degrees) < 3:
            continue
        
        # Density = Mean² / Mean(x²)
        mean_degree = np.mean(degrees)
        mean_degree_squared = np.mean(degrees ** 2)
        density = (mean_degree ** 2) / mean_degree_squared if mean_degree_squared > 0 else 0
        
        # Std Degree (Hierarchy)
        std_degree = np.std(degrees)
        
        # Other metrics
        max_degree = np.max(degrees)
        entropy = calculate_entropy(degrees)
        gp = group['GP'].max()
        
        # Playoff status
        made_playoffs = team in PLAYOFF_TEAMS.get(season, [])
        deep_run = team in DEEP_RUNS.get(season, [])
        is_champion = team == CHAMPIONS.get(season, '')
        
        # Success level
        if is_champion:
            success_level = 'Champion'
        elif deep_run:
            success_level = 'Deep Run'
        elif made_playoffs:
            success_level = 'First Round'
        else:
            success_level = 'Lottery'
        
        team_metrics.append({
            'Team': team,
            'Season': season,
            'Density': density,
            'Std_Degree': std_degree,
            'Mean_Degree': mean_degree,
            'Max_Degree': max_degree,
            'Entropy': entropy,
            'GP': gp,
            'Made_Playoffs': made_playoffs,
            'Deep_Run': deep_run,
            'Is_Champion': is_champion,
            'Success_Level': success_level,
            'N_Players': len(degrees)
        })
    
    return pd.DataFrame(team_metrics)


def compare_playoff_vs_lottery(team_df):
    """Compare metrics between playoff and lottery teams."""
    
    playoff = team_df[team_df['Made_Playoffs']]
    lottery = team_df[~team_df['Made_Playoffs']]
    
    metrics = ['Density', 'Std_Degree', 'Mean_Degree', 'Max_Degree', 'Entropy']
    
    comparison = []
    
    for metric in metrics:
        play_mean = playoff[metric].mean()
        play_std = playoff[metric].std()
        lot_mean = lottery[metric].mean()
        lot_std = lottery[metric].std()
        
        t_stat, p_val = stats.ttest_ind(playoff[metric], lottery[metric])
        
        # Effect size
        pooled_std = np.sqrt((play_std**2 + lot_std**2) / 2)
        cohens_d = (play_mean - lot_mean) / pooled_std if pooled_std > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Playoff_Mean': play_mean,
            'Playoff_Std': play_std,
            'Lottery_Mean': lot_mean,
            'Lottery_Std': lot_std,
            'Difference': play_mean - lot_mean,
            'Pct_Diff': ((play_mean - lot_mean) / lot_mean * 100) if lot_mean != 0 else 0,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Cohens_d': cohens_d
        })
    
    return pd.DataFrame(comparison)


def plot_comparison(team_df, comparison_df):
    """Create comparison visualizations."""
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Density: Playoff vs Lottery
    ax1 = fig.add_subplot(2, 3, 1)
    
    playoff = team_df[team_df['Made_Playoffs']]
    lottery = team_df[~team_df['Made_Playoffs']]
    
    ax1.hist(lottery['Density'], bins=25, alpha=0.5, label=f'Lottery (n={len(lottery)})', color='gray')
    ax1.hist(playoff['Density'], bins=25, alpha=0.5, label=f'Playoff (n={len(playoff)})', color='green')
    ax1.axvline(x=lottery['Density'].mean(), color='gray', linestyle='--', lw=2)
    ax1.axvline(x=playoff['Density'].mean(), color='darkgreen', linestyle='--', lw=2)
    ax1.set_xlabel('Network Density')
    ax1.set_ylabel('Count')
    ax1.set_title('DENSITY: Playoff vs Lottery', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # Add stats
    row = comparison_df[comparison_df['Metric'] == 'Density'].iloc[0]
    ax1.text(0.05, 0.95, f"Playoff: {row['Playoff_Mean']:.3f}\nLottery: {row['Lottery_Mean']:.3f}\np = {row['P_Value']:.4f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Std Degree: Playoff vs Lottery
    ax2 = fig.add_subplot(2, 3, 2)
    
    ax2.hist(lottery['Std_Degree'], bins=25, alpha=0.5, label=f'Lottery', color='gray')
    ax2.hist(playoff['Std_Degree'], bins=25, alpha=0.5, label=f'Playoff', color='green')
    ax2.axvline(x=lottery['Std_Degree'].mean(), color='gray', linestyle='--', lw=2)
    ax2.axvline(x=playoff['Std_Degree'].mean(), color='darkgreen', linestyle='--', lw=2)
    ax2.set_xlabel('Std of Weighted Degree (Hierarchy)')
    ax2.set_ylabel('Count')
    ax2.set_title('HIERARCHY: Playoff vs Lottery', fontsize=11, fontweight='bold')
    ax2.legend()
    
    row = comparison_df[comparison_df['Metric'] == 'Std_Degree'].iloc[0]
    ax2.text(0.05, 0.95, f"Playoff: {row['Playoff_Mean']:.0f}\nLottery: {row['Lottery_Mean']:.0f}\np = {row['P_Value']:.4f}",
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Boxplot by Success Level - Density
    ax3 = fig.add_subplot(2, 3, 3)
    
    order = ['Lottery', 'First Round', 'Deep Run', 'Champion']
    colors = {'Lottery': 'gray', 'First Round': 'lightblue', 'Deep Run': 'orange', 'Champion': 'gold'}
    
    for i, level in enumerate(order):
        data = team_df[team_df['Success_Level'] == level]['Density']
        bp = ax3.boxplot([data], positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors[level])
    
    ax3.set_xticks(range(len(order)))
    ax3.set_xticklabels(order, rotation=15)
    ax3.set_ylabel('Network Density')
    ax3.set_title('DENSITY by Playoff Success', fontsize=11, fontweight='bold')
    
    # 4. Boxplot by Success Level - Std Degree
    ax4 = fig.add_subplot(2, 3, 4)
    
    for i, level in enumerate(order):
        data = team_df[team_df['Success_Level'] == level]['Std_Degree']
        bp = ax4.boxplot([data], positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors[level])
    
    ax4.set_xticks(range(len(order)))
    ax4.set_xticklabels(order, rotation=15)
    ax4.set_ylabel('Std of Weighted Degree')
    ax4.set_title('HIERARCHY by Playoff Success', fontsize=11, fontweight='bold')
    
    # 5. Bar chart comparison
    ax5 = fig.add_subplot(2, 3, 5)
    
    x = np.arange(2)
    width = 0.35
    
    density_play = playoff['Density'].mean()
    density_lot = lottery['Density'].mean()
    std_play = playoff['Std_Degree'].mean()
    std_lot = lottery['Std_Degree'].mean()
    
    # Normalize for comparison
    ax5.bar(x - width/2, [density_lot, std_lot/3000], width, label='Lottery', color='gray')
    ax5.bar(x + width/2, [density_play, std_play/3000], width, label='Playoff', color='green')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Density', 'Std Degree\n(normalized)'])
    ax5.set_ylabel('Value')
    ax5.set_title('Playoff vs Lottery Comparison', fontsize=11, fontweight='bold')
    ax5.legend()
    
    # 6. Effect sizes
    ax6 = fig.add_subplot(2, 3, 6)
    
    colors_effect = ['green' if d > 0 else 'red' for d in comparison_df['Pct_Diff']]
    bars = ax6.barh(comparison_df['Metric'], comparison_df['Pct_Diff'], color=colors_effect)
    ax6.axvline(x=0, color='black', linewidth=1)
    ax6.set_xlabel('% Difference (Playoff vs Lottery)')
    ax6.set_title('Metric Differences', fontsize=11, fontweight='bold')
    
    for bar, (_, row) in zip(bars, comparison_df.iterrows()):
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        ax6.text(row['Pct_Diff'] + (1 if row['Pct_Diff'] >= 0 else -1), 
                bar.get_y() + bar.get_height()/2, f"{row['Pct_Diff']:+.1f}%{sig}", 
                va='center', fontsize=9)
    
    plt.suptitle('DENSITY & HIERARCHY: Playoff Teams vs Lottery Teams\n(Regular Season Network Structure)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_density_hierarchy_playoff_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_density_hierarchy_playoff_comparison.png")


def plot_champions_profile(team_df):
    """Show champions' density and std degree over years."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get champion data
    champ_data = []
    for season, team in CHAMPIONS.items():
        champ_row = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
        if len(champ_row) > 0:
            champ_data.append({
                'Season': season,
                'Team': team,
                'Density': champ_row['Density'].values[0],
                'Std_Degree': champ_row['Std_Degree'].values[0]
            })
    
    champ_df = pd.DataFrame(champ_data)
    
    # 1. Champions Density
    ax1 = axes[0]
    ax1.bar(champ_df['Season'], champ_df['Density'], color='gold', edgecolor='black')
    ax1.axhline(y=team_df['Density'].mean(), color='red', linestyle='--', lw=2, 
                label=f"League Avg: {team_df['Density'].mean():.3f}")
    ax1.axhline(y=team_df[team_df['Made_Playoffs']]['Density'].mean(), color='green', linestyle='--', lw=2,
                label=f"Playoff Avg: {team_df[team_df['Made_Playoffs']]['Density'].mean():.3f}")
    ax1.set_ylabel('Network Density')
    ax1.set_title('CHAMPIONS: Density Over Years', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right')
    
    for i, row in champ_df.iterrows():
        ax1.text(i, row['Density'] + 0.01, row['Team'], ha='center', fontsize=8)
    
    # 2. Champions Std Degree
    ax2 = axes[1]
    ax2.bar(champ_df['Season'], champ_df['Std_Degree'], color='gold', edgecolor='black')
    ax2.axhline(y=team_df['Std_Degree'].mean(), color='red', linestyle='--', lw=2,
                label=f"League Avg: {team_df['Std_Degree'].mean():.0f}")
    ax2.axhline(y=team_df[team_df['Made_Playoffs']]['Std_Degree'].mean(), color='green', linestyle='--', lw=2,
                label=f"Playoff Avg: {team_df[team_df['Made_Playoffs']]['Std_Degree'].mean():.0f}")
    ax2.set_ylabel('Std of Weighted Degree (Hierarchy)')
    ax2.set_title('CHAMPIONS: Hierarchy Over Years', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right')
    
    for i, row in champ_df.iterrows():
        ax2.text(i, row['Std_Degree'] + 50, row['Team'], ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_champions_density_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_champions_density_hierarchy.png")


def print_analysis(team_df, comparison_df):
    """Print detailed analysis."""
    
    playoff = team_df[team_df['Made_Playoffs']]
    lottery = team_df[~team_df['Made_Playoffs']]
    
    print("\n" + "="*80)
    print("DENSITY & STD DEGREE: PLAYOFF vs LOTTERY TEAMS")
    print("="*80)
    
    print(f"\nData Summary:")
    print(f"  Playoff Teams: {len(playoff)} team-seasons")
    print(f"  Lottery Teams: {len(lottery)} team-seasons")
    
    print("\n" + "-"*80)
    print("METRIC COMPARISON")
    print("-"*80)
    
    print(f"\n{'Metric':<20} {'Playoff':>12} {'Lottery':>12} {'Diff':>10} {'p-value':>12}")
    print("-"*70)
    
    for _, row in comparison_df.iterrows():
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        print(f"{row['Metric']:<20} {row['Playoff_Mean']:>12.2f} {row['Lottery_Mean']:>12.2f} "
              f"{row['Pct_Diff']:>+9.1f}% {row['P_Value']:>11.4f}{sig}")
    
    print("\n" + "-"*80)
    print("BY PLAYOFF SUCCESS LEVEL")
    print("-"*80)
    
    for level in ['Lottery', 'First Round', 'Deep Run', 'Champion']:
        subset = team_df[team_df['Success_Level'] == level]
        print(f"\n{level} ({len(subset)} teams):")
        print(f"  Density: {subset['Density'].mean():.3f} (+/- {subset['Density'].std():.3f})")
        print(f"  Std Degree: {subset['Std_Degree'].mean():.0f} (+/- {subset['Std_Degree'].std():.0f})")
    
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80)
    
    density_row = comparison_df[comparison_df['Metric'] == 'Density'].iloc[0]
    std_row = comparison_df[comparison_df['Metric'] == 'Std_Degree'].iloc[0]
    
    print(f"\n1. DENSITY:")
    print(f"   Playoff: {density_row['Playoff_Mean']:.3f}")
    print(f"   Lottery: {density_row['Lottery_Mean']:.3f}")
    print(f"   Difference: {density_row['Pct_Diff']:+.1f}% (p = {density_row['P_Value']:.4f})")
    if density_row['Playoff_Mean'] < density_row['Lottery_Mean']:
        print(f"   --> Playoff teams have LOWER density (more star-heavy)")
    else:
        print(f"   --> Playoff teams have HIGHER density (more equal)")
    
    print(f"\n2. HIERARCHY (Std Degree):")
    print(f"   Playoff: {std_row['Playoff_Mean']:.0f}")
    print(f"   Lottery: {std_row['Lottery_Mean']:.0f}")
    print(f"   Difference: {std_row['Pct_Diff']:+.1f}% (p = {std_row['P_Value']:.4f})")
    if std_row['Playoff_Mean'] > std_row['Lottery_Mean']:
        print(f"   --> Playoff teams have HIGHER hierarchy (clearer star structure)")
    else:
        print(f"   --> Playoff teams have LOWER hierarchy (more equal)")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*70)
    print("DENSITY & STD DEGREE: PLAYOFF vs REGULAR SEASON ANALYSIS")
    print("="*70)
    
    print("\n[LOADING PLAYER DATA]")
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    print(f"Loaded {len(player_df)} player-seasons")
    
    print("\n[CALCULATING TEAM METRICS]")
    team_df = calculate_team_metrics(player_df)
    print(f"Calculated for {len(team_df)} team-seasons")
    print(f"  Playoff teams: {team_df['Made_Playoffs'].sum()}")
    print(f"  Lottery teams: {(~team_df['Made_Playoffs']).sum()}")
    
    print("\n[COMPARING PLAYOFF vs LOTTERY]")
    comparison_df = compare_playoff_vs_lottery(team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_comparison(team_df, comparison_df)
    plot_champions_profile(team_df)
    
    print_analysis(team_df, comparison_df)
    
    # Save data
    team_df.to_csv(OUTPUT_DIR / 'team_metrics_playoff_status.csv', index=False)
    comparison_df.to_csv(OUTPUT_DIR / 'playoff_vs_lottery_comparison.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
