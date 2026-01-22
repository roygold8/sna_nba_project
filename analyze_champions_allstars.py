"""
Champions vs Others & All-Stars vs Others Analysis
===================================================
1. Team Level: What makes Champions different from other teams?
2. Player Level: What makes All-Stars different from role players?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_champions_allstars")
OUTPUT_DIR.mkdir(exist_ok=True)

# Championship teams
CHAMPIONS = {
    '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW',
    '2018-19': 'TOR', '2019-20': 'LAL', '2020-21': 'MIL',
    '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
}


def load_data():
    """Load all data."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    # Calculate additional team metrics
    team_agg = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'min'],
        'Betweenness_Centrality': ['mean', 'max'],
        'Eigenvector_Centrality': ['mean', 'max'],
        'PTS': ['sum', 'mean', 'max'],
        'AST': ['sum', 'mean'],
        'PLAYER_ID': 'count'
    }).reset_index()
    
    team_agg.columns = ['TEAM_ABBREVIATION', 'SEASON',
                        'Avg_Degree', 'Std_Degree', 'Max_Degree', 'Min_Degree',
                        'Avg_Betweenness', 'Max_Betweenness',
                        'Avg_Eigenvector', 'Max_Eigenvector',
                        'Total_PTS', 'Avg_PTS', 'Max_PTS',
                        'Total_AST', 'Avg_AST',
                        'Roster_Size']
    
    team_df = team_df.merge(team_agg, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    
    # Mark champions
    team_df['Is_Champion'] = team_df.apply(
        lambda r: CHAMPIONS.get(r['SEASON']) == r['TEAM_ABBREVIATION'], axis=1
    )
    
    # Calculate player ratios
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    player_df['PTS_Per_Degree'] = player_df['PTS'] / (player_df['Weighted_Degree'] + 1)
    
    print(f"Loaded {len(player_df)} player records, {len(team_df)} team records")
    print(f"Champions: {team_df['Is_Champion'].sum()}")
    
    return player_df, team_df


def identify_allstars(player_df):
    """Identify All-Star caliber players based on network and performance metrics."""
    
    # Method 1: Top performers by weighted degree within each team-season
    # (players with Z-score > 1.5 within their team)
    team_stats = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std'],
        'PTS': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                          'Team_Degree_Mean', 'Team_Degree_Std',
                          'Team_PTS_Mean', 'Team_PTS_Std']
    
    player_df = player_df.merge(team_stats, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    player_df['Degree_ZScore'] = (player_df['Weighted_Degree'] - player_df['Team_Degree_Mean']) / (player_df['Team_Degree_Std'] + 1)
    player_df['PTS_ZScore'] = (player_df['PTS'] - player_df['Team_PTS_Mean']) / (player_df['Team_PTS_Std'] + 1)
    
    # All-Star criteria: High degree OR high scoring (top performers)
    player_df['Is_AllStar_Degree'] = player_df['Degree_ZScore'] > 1.5
    player_df['Is_AllStar_PTS'] = player_df['PTS_ZScore'] > 1.5
    player_df['Is_AllStar'] = player_df['Is_AllStar_Degree'] | player_df['Is_AllStar_PTS']
    
    # Method 2: Absolute thresholds (league-wide top performers)
    pts_threshold = player_df['PTS'].quantile(0.90)  # Top 10% scorers
    degree_threshold = player_df['Weighted_Degree'].quantile(0.90)  # Top 10% network players
    
    player_df['Is_Elite'] = (player_df['PTS'] > pts_threshold) | (player_df['Weighted_Degree'] > degree_threshold)
    
    print(f"\nAll-Stars identified (Z-score > 1.5): {player_df['Is_AllStar'].sum()}")
    print(f"Elite players (Top 10%): {player_df['Is_Elite'].sum()}")
    
    return player_df


# ========== CHAMPIONS ANALYSIS ==========

def analyze_champions_metrics(team_df):
    """Analyze which metrics differentiate champions from others."""
    
    champs = team_df[team_df['Is_Champion']]
    others = team_df[~team_df['Is_Champion']]
    
    # Metrics to compare
    metrics = [
        'W_PCT', 'WINS',
        'Density', 'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration',
        'Star_Weighted_Degree', 'Avg_Degree', 'Std_Degree', 'Max_Degree',
        'Avg_Betweenness', 'Max_Betweenness', 'Avg_Eigenvector', 'Max_Eigenvector',
        'Total_PTS', 'Avg_PTS', 'Total_AST'
    ]
    
    results = []
    
    for metric in metrics:
        if metric not in team_df.columns:
            continue
        
        champ_vals = champs[metric].dropna()
        other_vals = others[metric].dropna()
        
        if len(champ_vals) < 3 or len(other_vals) < 10:
            continue
        
        # Statistics
        champ_mean = champ_vals.mean()
        other_mean = other_vals.mean()
        diff = champ_mean - other_mean
        pct_diff = (diff / other_mean) * 100 if other_mean != 0 else 0
        
        # T-test
        t_stat, t_pval = stats.ttest_ind(champ_vals, other_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((champ_vals.std()**2 + other_vals.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Metric': metric,
            'Champions_Mean': champ_mean,
            'Others_Mean': other_mean,
            'Difference': diff,
            'Pct_Difference': pct_diff,
            'T_Statistic': t_stat,
            'P_Value': t_pval,
            'Cohens_D': cohens_d,
            'Significant': t_pval < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    
    return results_df


def plot_champions_analysis(team_df, results_df):
    """Create visualizations for champions analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Select top 6 most significant metrics
    top_metrics = results_df.head(6)['Metric'].tolist()
    
    for ax, metric in zip(axes.flatten(), top_metrics):
        champs = team_df[team_df['Is_Champion']][metric]
        others = team_df[~team_df['Is_Champion']][metric]
        
        # Box plot
        data = pd.DataFrame({
            'Value': pd.concat([champs, others]),
            'Group': ['Champions'] * len(champs) + ['Others'] * len(others)
        })
        
        sns.boxplot(data=data, x='Group', y='Value', ax=ax,
                    hue='Group', palette=['gold', 'gray'], legend=False)
        
        # Add individual points
        sns.stripplot(data=data, x='Group', y='Value', ax=ax,
                      color='black', alpha=0.5, size=5)
        
        # Get stats
        row = results_df[results_df['Metric'] == metric].iloc[0]
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else ""
        
        ax.set_title(f"{metric}\nChamps: {row['Champions_Mean']:.2f} vs Others: {row['Others_Mean']:.2f}\n"
                     f"p = {row['P_Value']:.4f} {sig}, d = {row['Cohens_D']:.2f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
    
    plt.suptitle('CHAMPIONS vs OTHERS: Top Differentiating Metrics',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_champions_vs_others_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_champions_vs_others_boxplots.png")


def plot_champions_effect_sizes(results_df):
    """Plot effect sizes for all metrics."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by effect size
    results_sorted = results_df.sort_values('Cohens_D', ascending=True)
    
    # Color by significance
    colors = ['green' if sig else 'gray' for sig in results_sorted['Significant']]
    
    # Bar plot
    y_pos = range(len(results_sorted))
    bars = ax.barh(y_pos, results_sorted['Cohens_D'], color=colors, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['Metric'])
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    ax.axvline(x=0.2, color='blue', linestyle='--', lw=1, alpha=0.5, label='Small Effect')
    ax.axvline(x=0.5, color='orange', linestyle='--', lw=1, alpha=0.5, label='Medium Effect')
    ax.axvline(x=0.8, color='red', linestyle='--', lw=1, alpha=0.5, label='Large Effect')
    ax.axvline(x=-0.2, color='blue', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(x=-0.5, color='orange', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(x=-0.8, color='red', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_xlabel("Cohen's d Effect Size\n(Positive = Champions Higher)", fontsize=11)
    ax.set_title("Champions vs Others: Effect Sizes\n(Green = p < 0.05, Gray = Not Significant)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Add values
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        offset = 0.02 if row['Cohens_D'] >= 0 else -0.02
        ha = 'left' if row['Cohens_D'] >= 0 else 'right'
        ax.text(row['Cohens_D'] + offset, i, f"{row['Cohens_D']:.2f}", va='center', ha=ha, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_champions_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_champions_effect_sizes.png")


# ========== ALL-STARS ANALYSIS ==========

def analyze_allstar_metrics(player_df):
    """Analyze which metrics differentiate All-Stars from others."""
    
    allstars = player_df[player_df['Is_AllStar']]
    others = player_df[~player_df['Is_AllStar']]
    
    # Metrics to compare
    metrics = [
        'Weighted_Degree', 'Weighted_In_Degree', 'Weighted_Out_Degree',
        'In_Degree', 'Out_Degree', 'Total_Degree',
        'Betweenness_Centrality', 'Eigenvector_Centrality',
        'In_Out_Ratio', 'Degree_Per_Game', 'PTS_Per_Degree',
        'PTS', 'AST', 'REB', 'MIN', 'GP'
    ]
    
    results = []
    
    for metric in metrics:
        if metric not in player_df.columns:
            continue
        
        allstar_vals = allstars[metric].dropna()
        other_vals = others[metric].dropna()
        
        if len(allstar_vals) < 10 or len(other_vals) < 100:
            continue
        
        # Statistics
        allstar_mean = allstar_vals.mean()
        other_mean = other_vals.mean()
        diff = allstar_mean - other_mean
        pct_diff = (diff / other_mean) * 100 if other_mean != 0 else 0
        
        # Mann-Whitney U test (better for non-normal distributions)
        u_stat, u_pval = stats.mannwhitneyu(allstar_vals, other_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((allstar_vals.std()**2 + other_vals.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Metric': metric,
            'AllStar_Mean': allstar_mean,
            'Others_Mean': other_mean,
            'Difference': diff,
            'Pct_Difference': pct_diff,
            'U_Statistic': u_stat,
            'P_Value': u_pval,
            'Cohens_D': cohens_d,
            'Significant': u_pval < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    
    return results_df


def plot_allstar_analysis(player_df, results_df):
    """Create visualizations for All-Star analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Select top 6 most significant metrics
    top_metrics = results_df.head(6)['Metric'].tolist()
    
    for ax, metric in zip(axes.flatten(), top_metrics):
        allstars = player_df[player_df['Is_AllStar']][metric]
        others = player_df[~player_df['Is_AllStar']][metric]
        
        # Violin plot (better for distributions)
        data = pd.DataFrame({
            'Value': pd.concat([allstars, others]),
            'Group': ['All-Stars'] * len(allstars) + ['Role Players'] * len(others)
        })
        
        sns.violinplot(data=data, x='Group', y='Value', ax=ax,
                       hue='Group', palette=['gold', 'lightblue'], legend=False)
        
        # Get stats
        row = results_df[results_df['Metric'] == metric].iloc[0]
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else ""
        
        ax.set_title(f"{metric}\nAll-Stars: {row['AllStar_Mean']:.1f} vs Others: {row['Others_Mean']:.1f}\n"
                     f"p = {row['P_Value']:.2e} {sig}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
    
    plt.suptitle('ALL-STARS vs ROLE PLAYERS: Top Differentiating Metrics',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_allstars_vs_others_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_allstars_vs_others_violin.png")


def plot_allstar_effect_sizes(results_df):
    """Plot effect sizes for All-Star metrics."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by effect size
    results_sorted = results_df.sort_values('Cohens_D', ascending=True)
    
    # Color by significance
    colors = ['green' if sig else 'gray' for sig in results_sorted['Significant']]
    
    y_pos = range(len(results_sorted))
    bars = ax.barh(y_pos, results_sorted['Cohens_D'], color=colors, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['Metric'])
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    ax.axvline(x=0.2, color='blue', linestyle='--', lw=1, alpha=0.5, label='Small')
    ax.axvline(x=0.5, color='orange', linestyle='--', lw=1, alpha=0.5, label='Medium')
    ax.axvline(x=0.8, color='red', linestyle='--', lw=1, alpha=0.5, label='Large')
    
    ax.set_xlabel("Cohen's d Effect Size\n(Positive = All-Stars Higher)", fontsize=11)
    ax.set_title("All-Stars vs Role Players: Effect Sizes\n(Green = p < 0.05)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        ax.text(row['Cohens_D'] + 0.05, i, f"{row['Cohens_D']:.2f}", va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_allstars_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_allstars_effect_sizes.png")


def plot_allstar_scatter(player_df):
    """Scatter plot showing All-Stars in network space."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Weighted Degree vs PTS
    ax1 = axes[0]
    allstars = player_df[player_df['Is_AllStar']]
    others = player_df[~player_df['Is_AllStar']]
    
    ax1.scatter(others['Weighted_Degree'], others['PTS'], 
                c='lightgray', alpha=0.3, s=20, label='Role Players')
    ax1.scatter(allstars['Weighted_Degree'], allstars['PTS'],
                c='gold', alpha=0.8, s=60, edgecolors='black', linewidths=0.5, label='All-Stars')
    
    ax1.set_xlabel('Weighted Degree (Network Centrality)', fontsize=11)
    ax1.set_ylabel('Points', fontsize=11)
    ax1.set_title('Network Position vs Scoring', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Add correlation
    corr, _ = stats.pearsonr(player_df['Weighted_Degree'], player_df['PTS'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. In-Degree vs Out-Degree
    ax2 = axes[1]
    ax2.scatter(others['Weighted_Out_Degree'], others['Weighted_In_Degree'],
                c='lightgray', alpha=0.3, s=20, label='Role Players')
    ax2.scatter(allstars['Weighted_Out_Degree'], allstars['Weighted_In_Degree'],
                c='gold', alpha=0.8, s=60, edgecolors='black', linewidths=0.5, label='All-Stars')
    
    # 45-degree line
    max_val = max(player_df['Weighted_Out_Degree'].max(), player_df['Weighted_In_Degree'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='Equal In/Out')
    
    ax2.set_xlabel('Weighted Out-Degree (Passes Made)', fontsize=11)
    ax2.set_ylabel('Weighted In-Degree (Passes Received)', fontsize=11)
    ax2.set_title('Pass Flow: All-Stars vs Role Players', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_allstars_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 05_allstars_scatter.png")


def print_summary(champ_results, allstar_results, player_df, team_df):
    """Print comprehensive summary."""
    
    print("\n" + "="*80)
    print("CHAMPIONS vs OTHERS - KEY METRICS")
    print("="*80)
    
    print("\n[TOP 10 MOST SIGNIFICANT DIFFERENCES]")
    print("-" * 70)
    print(f"{'Metric':<25} {'Champs':>10} {'Others':>10} {'Diff%':>8} {'p-value':>10} {'Effect':>8}")
    print("-" * 70)
    
    for _, row in champ_results.head(10).iterrows():
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else "*" if row['P_Value'] < 0.1 else ""
        print(f"{row['Metric']:<25} {row['Champions_Mean']:>10.2f} {row['Others_Mean']:>10.2f} "
              f"{row['Pct_Difference']:>+7.1f}% {row['P_Value']:>10.4f} {row['Cohens_D']:>+7.2f} {sig}")
    
    print("\n" + "="*80)
    print("ALL-STARS vs ROLE PLAYERS - KEY METRICS")
    print("="*80)
    
    print("\n[TOP 10 MOST SIGNIFICANT DIFFERENCES]")
    print("-" * 70)
    print(f"{'Metric':<25} {'All-Stars':>10} {'Others':>10} {'Diff%':>8} {'p-value':>12} {'Effect':>8}")
    print("-" * 70)
    
    for _, row in allstar_results.head(10).iterrows():
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else ""
        print(f"{row['Metric']:<25} {row['AllStar_Mean']:>10.2f} {row['Others_Mean']:>10.2f} "
              f"{row['Pct_Difference']:>+7.1f}% {row['P_Value']:>12.2e} {row['Cohens_D']:>+7.2f} {sig}")
    
    # Summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    champs = team_df[team_df['Is_Champion']]
    allstars = player_df[player_df['Is_AllStar']]
    
    print(f"\nChampions: {len(champs)} teams")
    print(f"  - Average Win%: {champs['W_PCT'].mean():.3f}")
    print(f"  - Average Star Degree: {champs['Star_Weighted_Degree'].mean():.0f}")
    
    print(f"\nAll-Stars: {len(allstars)} player-seasons")
    print(f"  - Average Weighted Degree: {allstars['Weighted_Degree'].mean():.0f}")
    print(f"  - Average PTS: {allstars['PTS'].mean():.0f}")
    print(f"  - Average Betweenness: {allstars['Betweenness_Centrality'].mean():.3f}")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("CHAMPIONS & ALL-STARS ANALYSIS")
    print("="*60)
    
    # Load data
    player_df, team_df = load_data()
    
    # Identify All-Stars
    player_df = identify_allstars(player_df)
    
    # ===== CHAMPIONS ANALYSIS =====
    print("\n[ANALYZING CHAMPIONS vs OTHERS]")
    print("-" * 40)
    champ_results = analyze_champions_metrics(team_df)
    plot_champions_analysis(team_df, champ_results)
    plot_champions_effect_sizes(champ_results)
    
    # ===== ALL-STARS ANALYSIS =====
    print("\n[ANALYZING ALL-STARS vs ROLE PLAYERS]")
    print("-" * 40)
    allstar_results = analyze_allstar_metrics(player_df)
    plot_allstar_analysis(player_df, allstar_results)
    plot_allstar_effect_sizes(allstar_results)
    plot_allstar_scatter(player_df)
    
    # Print summary
    print_summary(champ_results, allstar_results, player_df, team_df)
    
    # Save results
    champ_results.to_csv(OUTPUT_DIR / 'champions_metrics_comparison.csv', index=False)
    allstar_results.to_csv(OUTPUT_DIR / 'allstars_metrics_comparison.csv', index=False)
    
    print(f"\n[OK] Saved all results to {OUTPUT_DIR}/")
    print(f"\n[COMPLETE] Analysis finished!")


if __name__ == "__main__":
    main()
