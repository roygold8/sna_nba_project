"""
Champions vs Others - NETWORK METRICS ONLY
==========================================
Pure SNA analysis without traditional stats (PTS, AST, REB)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_champions_network")
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
    
    # Calculate network-only team metrics
    team_agg = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'min'],
        'Weighted_In_Degree': ['mean', 'std', 'max'],
        'Weighted_Out_Degree': ['mean', 'std', 'max'],
        'Betweenness_Centrality': ['mean', 'max', 'std'],
        'Eigenvector_Centrality': ['mean', 'max', 'std'],
        'In_Degree': ['mean'],
        'Out_Degree': ['mean'],
        'PLAYER_ID': 'count'
    }).reset_index()
    
    team_agg.columns = ['TEAM_ABBREVIATION', 'SEASON',
                        'Avg_Degree', 'Std_Degree', 'Max_Degree', 'Min_Degree',
                        'Avg_In_Degree', 'Std_In_Degree', 'Max_In_Degree',
                        'Avg_Out_Degree', 'Std_Out_Degree', 'Max_Out_Degree',
                        'Avg_Betweenness', 'Max_Betweenness', 'Std_Betweenness',
                        'Avg_Eigenvector', 'Max_Eigenvector', 'Std_Eigenvector',
                        'Avg_In_Connections', 'Avg_Out_Connections',
                        'Roster_Size']
    
    team_df = team_df.merge(team_agg, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    
    # Calculate additional network metrics
    team_df['Degree_Range'] = team_df['Max_Degree'] - team_df['Min_Degree']
    team_df['Degree_CV'] = team_df['Std_Degree'] / (team_df['Avg_Degree'] + 1)
    team_df['Top1_Pct'] = team_df['Max_Degree'] / (team_df['Avg_Degree'] * team_df['Roster_Size'] + 1)
    team_df['Betweenness_Range'] = team_df['Max_Betweenness'] - team_df['Avg_Betweenness']
    team_df['Eigenvector_Range'] = team_df['Max_Eigenvector'] - team_df['Avg_Eigenvector']
    
    # Mark champions
    team_df['Is_Champion'] = team_df.apply(
        lambda r: CHAMPIONS.get(r['SEASON']) == r['TEAM_ABBREVIATION'], axis=1
    )
    
    # Player network metrics
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    player_df['Net_Pass_Flow'] = player_df['Weighted_In_Degree'] - player_df['Weighted_Out_Degree']
    
    print(f"Loaded {len(player_df)} player records, {len(team_df)} team records")
    print(f"Champions: {team_df['Is_Champion'].sum()}")
    
    return player_df, team_df


def identify_allstars_network(player_df):
    """Identify All-Stars based purely on NETWORK metrics."""
    
    # Calculate Z-scores within team for network metrics only
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
    
    # All-Star based on network position only
    player_df['Is_Network_Star'] = (player_df['Degree_ZScore'] > 1.5) | (player_df['Eigen_ZScore'] > 1.5)
    
    # League-wide top 10% in network metrics
    degree_threshold = player_df['Weighted_Degree'].quantile(0.90)
    eigen_threshold = player_df['Eigenvector_Centrality'].quantile(0.90)
    player_df['Is_Network_Elite'] = (player_df['Weighted_Degree'] > degree_threshold) | \
                                     (player_df['Eigenvector_Centrality'] > eigen_threshold)
    
    print(f"\nNetwork Stars (Z > 1.5): {player_df['Is_Network_Star'].sum()}")
    print(f"Network Elite (Top 10%): {player_df['Is_Network_Elite'].sum()}")
    
    return player_df


def analyze_champions_network_metrics(team_df):
    """Analyze which NETWORK metrics differentiate champions."""
    
    champs = team_df[team_df['Is_Champion']]
    others = team_df[~team_df['Is_Champion']]
    
    # ONLY network metrics
    network_metrics = [
        'Density', 'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration',
        'Star_Weighted_Degree', 'Avg_Degree', 'Std_Degree', 'Max_Degree', 'Min_Degree',
        'Degree_Range', 'Degree_CV', 'Top1_Pct',
        'Avg_In_Degree', 'Avg_Out_Degree', 'Max_In_Degree', 'Max_Out_Degree',
        'Avg_Betweenness', 'Max_Betweenness', 'Std_Betweenness', 'Betweenness_Range',
        'Avg_Eigenvector', 'Max_Eigenvector', 'Std_Eigenvector', 'Eigenvector_Range'
    ]
    
    results = []
    
    for metric in network_metrics:
        if metric not in team_df.columns:
            continue
        
        champ_vals = champs[metric].dropna()
        other_vals = others[metric].dropna()
        
        if len(champ_vals) < 3 or len(other_vals) < 10:
            continue
        
        champ_mean = champ_vals.mean()
        other_mean = other_vals.mean()
        diff = champ_mean - other_mean
        pct_diff = (diff / other_mean) * 100 if other_mean != 0 else 0
        
        t_stat, t_pval = stats.ttest_ind(champ_vals, other_vals)
        
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


def plot_champions_network_analysis(team_df, results_df):
    """Visualize champion network differences."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Top 6 network metrics
    top_metrics = results_df.head(6)['Metric'].tolist()
    
    for ax, metric in zip(axes.flatten(), top_metrics):
        champs = team_df[team_df['Is_Champion']][metric]
        others = team_df[~team_df['Is_Champion']][metric]
        
        data = pd.DataFrame({
            'Value': pd.concat([champs, others]),
            'Group': ['Champions'] * len(champs) + ['Others'] * len(others)
        })
        
        sns.boxplot(data=data, x='Group', y='Value', ax=ax,
                    hue='Group', palette=['gold', 'gray'], legend=False)
        sns.stripplot(data=data, x='Group', y='Value', ax=ax,
                      color='black', alpha=0.5, size=5)
        
        row = results_df[results_df['Metric'] == metric].iloc[0]
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else ""
        
        ax.set_title(f"{metric}\nChamps: {row['Champions_Mean']:.3f} vs Others: {row['Others_Mean']:.3f}\n"
                     f"p = {row['P_Value']:.4f} {sig}, d = {row['Cohens_D']:.2f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('')
    
    plt.suptitle('CHAMPIONS vs OTHERS: Network Metrics Only',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_champions_network_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_champions_network_boxplots.png")


def plot_network_effect_sizes(results_df):
    """Plot effect sizes for network metrics."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    results_sorted = results_df.sort_values('Cohens_D', ascending=True)
    colors = ['green' if sig else 'gray' for sig in results_sorted['Significant']]
    
    y_pos = range(len(results_sorted))
    ax.barh(y_pos, results_sorted['Cohens_D'], color=colors, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['Metric'])
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    ax.axvline(x=0.2, color='blue', linestyle='--', lw=1, alpha=0.5, label='Small')
    ax.axvline(x=0.5, color='orange', linestyle='--', lw=1, alpha=0.5, label='Medium')
    ax.axvline(x=0.8, color='red', linestyle='--', lw=1, alpha=0.5, label='Large')
    ax.axvline(x=-0.2, color='blue', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(x=-0.5, color='orange', linestyle='--', lw=1, alpha=0.5)
    
    ax.set_xlabel("Cohen's d Effect Size\n(Positive = Champions Higher)", fontsize=11)
    ax.set_title("Champions vs Others: Network Metrics Effect Sizes\n(Green = p < 0.05)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        offset = 0.02 if row['Cohens_D'] >= 0 else -0.02
        ha = 'left' if row['Cohens_D'] >= 0 else 'right'
        ax.text(row['Cohens_D'] + offset, i, f"{row['Cohens_D']:.2f}", va='center', ha=ha, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_champions_network_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_champions_network_effect_sizes.png")


def plot_network_correlations_with_winning(team_df):
    """Plot network metrics correlation with winning."""
    
    network_metrics = [
        'Density', 'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration',
        'Star_Weighted_Degree', 'Avg_Degree', 'Std_Degree', 'Max_Degree',
        'Degree_CV', 'Avg_Betweenness', 'Max_Betweenness', 'Avg_Eigenvector'
    ]
    
    correlations = []
    for metric in network_metrics:
        if metric in team_df.columns:
            valid = team_df[[metric, 'W_PCT']].dropna()
            if len(valid) > 10:
                corr, pval = stats.pearsonr(valid[metric], valid['W_PCT'])
                correlations.append({'Metric': metric, 'Correlation': corr, 'P_Value': pval})
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green' if p < 0.05 else 'gray' for p in corr_df['P_Value']]
    ax.barh(corr_df['Metric'], corr_df['Correlation'], color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', lw=1)
    
    ax.set_xlabel('Correlation with Win %', fontsize=11)
    ax.set_title('Network Metrics Correlation with Team Success\n(Green = p < 0.05)',
                 fontsize=12, fontweight='bold')
    
    for i, (_, row) in enumerate(corr_df.iterrows()):
        ax.text(row['Correlation'] + 0.01, i, f"r={row['Correlation']:.3f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_network_correlations_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_network_correlations_winning.png")
    
    return corr_df


def print_summary(results_df, corr_df, team_df):
    """Print summary."""
    
    print("\n" + "="*80)
    print("CHAMPIONS vs OTHERS - NETWORK METRICS ONLY")
    print("="*80)
    
    print("\n[TOP DIFFERENTIATING NETWORK METRICS]")
    print("-" * 75)
    print(f"{'Metric':<25} {'Champs':>10} {'Others':>10} {'Diff%':>8} {'p-value':>10} {'Effect':>8}")
    print("-" * 75)
    
    for _, row in results_df.iterrows():
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else "*" if row['P_Value'] < 0.1 else ""
        print(f"{row['Metric']:<25} {row['Champions_Mean']:>10.3f} {row['Others_Mean']:>10.3f} "
              f"{row['Pct_Difference']:>+7.1f}% {row['P_Value']:>10.4f} {row['Cohens_D']:>+7.2f} {sig}")
    
    print("\n[NETWORK METRICS CORRELATION WITH WIN %]")
    print("-" * 50)
    
    corr_sorted = corr_df.sort_values('Correlation', ascending=False)
    for _, row in corr_sorted.iterrows():
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else ""
        print(f"{row['Metric']:<25} r = {row['Correlation']:>+.3f} {sig}")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("CHAMPIONS ANALYSIS - NETWORK METRICS ONLY")
    print("="*60)
    
    player_df, team_df = load_data()
    player_df = identify_allstars_network(player_df)
    
    print("\n[ANALYZING CHAMPIONS - NETWORK METRICS]")
    results_df = analyze_champions_network_metrics(team_df)
    
    plot_champions_network_analysis(team_df, results_df)
    plot_network_effect_sizes(results_df)
    corr_df = plot_network_correlations_with_winning(team_df)
    
    print_summary(results_df, corr_df, team_df)
    
    results_df.to_csv(OUTPUT_DIR / 'champions_network_comparison.csv', index=False)
    corr_df.to_csv(OUTPUT_DIR / 'network_correlations.csv', index=False)
    
    print(f"\n[OK] Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
