"""
SNA Metrics vs Winning Analysis
===============================
Comprehensive analysis of Social Network Analysis metrics and their relationship to team success.

Team-Level Analysis:
1. Density vs Winning
2. Average Degree vs Winning
3. Std of Degree vs Winning
4. Number of Stars (0, 1, 2+) vs Winning
5. Additional metrics (Gini, Centralization, etc.)

Player-Level Analysis:
- Star identification and characteristics
- Network position vs performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("output_sna_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load player and team metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    print(f"Loaded {len(player_df)} player-season records")
    print(f"Loaded {len(team_df)} team-season records")
    
    return player_df, team_df


def define_stars(player_df, method='zscore', threshold=1.5):
    """
    Define star players based on their network centrality lift above team average.
    
    Methods:
    - 'zscore': Players with weighted degree Z-score > threshold within their team
    - 'percentile': Players in top X percentile within their team
    - 'absolute': Players with weighted degree > threshold * team average
    """
    player_df = player_df.copy()
    
    # Calculate team statistics for each player's team-season
    team_stats = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Team_Mean_Degree', 'Team_Std_Degree']
    
    # Merge back to player data
    player_df = player_df.merge(team_stats, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    # Calculate Z-score for each player within their team
    player_df['Degree_ZScore'] = (player_df['Weighted_Degree'] - player_df['Team_Mean_Degree']) / (player_df['Team_Std_Degree'] + 1)
    
    # Define stars based on method
    if method == 'zscore':
        player_df['Is_Star'] = player_df['Degree_ZScore'] > threshold
    elif method == 'percentile':
        # Top 20% within each team
        player_df['Is_Star'] = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON'])['Weighted_Degree'].transform(
            lambda x: x >= x.quantile(0.8)
        )
    elif method == 'absolute':
        player_df['Is_Star'] = player_df['Weighted_Degree'] > (threshold * player_df['Team_Mean_Degree'])
    
    # Count stars per team
    star_counts = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Is_Star': 'sum',
        'PLAYER_NAME': lambda x: list(x[player_df.loc[x.index, 'Is_Star']])
    }).reset_index()
    star_counts.columns = ['TEAM_ABBREVIATION', 'SEASON', 'Num_Stars', 'Star_Names']
    
    print(f"\n[STAR DEFINITION] Method: {method}, Threshold: {threshold}")
    print(f"Total stars identified: {player_df['Is_Star'].sum()}")
    print(f"Average stars per team: {star_counts['Num_Stars'].mean():.2f}")
    
    return player_df, star_counts


def calculate_team_network_metrics(player_df, team_df):
    """Calculate additional team-level network metrics from player data."""
    
    # Aggregate player metrics to team level
    team_agg = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'min'],
        'Betweenness_Centrality': ['mean', 'std', 'max'],
        'Eigenvector_Centrality': ['mean', 'std', 'max'],
        'In_Degree': ['mean', 'std'],
        'Out_Degree': ['mean', 'std'],
        'PLAYER_ID': 'count'
    }).reset_index()
    
    # Flatten column names
    team_agg.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                        'Avg_Weighted_Degree', 'Std_Weighted_Degree', 'Max_Weighted_Degree', 'Min_Weighted_Degree',
                        'Avg_Betweenness', 'Std_Betweenness', 'Max_Betweenness',
                        'Avg_Eigenvector', 'Std_Eigenvector', 'Max_Eigenvector',
                        'Avg_In_Degree', 'Std_In_Degree',
                        'Avg_Out_Degree', 'Std_Out_Degree',
                        'Roster_Size']
    
    # Calculate degree range (hierarchy indicator)
    team_agg['Degree_Range'] = team_agg['Max_Weighted_Degree'] - team_agg['Min_Weighted_Degree']
    team_agg['Degree_CV'] = team_agg['Std_Weighted_Degree'] / (team_agg['Avg_Weighted_Degree'] + 1)  # Coefficient of Variation
    
    # Merge with existing team data
    team_enhanced = team_df.merge(team_agg, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    
    return team_enhanced


def plot_density_vs_winning(team_df):
    """1. Density vs Winning"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot with regression
    ax1 = axes[0]
    sns.regplot(data=team_df, x='Density', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red', 'lw': 2})
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(team_df['Density'].dropna(), 
                                    team_df.loc[team_df['Density'].notna(), 'W_PCT'])
    ax1.set_title(f'Network Density vs Win %\nr = {corr:.3f}, p = {p_value:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Network Density', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # Density distribution by winning teams
    ax2 = axes[1]
    team_df['Win_Category'] = pd.cut(team_df['W_PCT'], bins=[0, 0.4, 0.5, 0.6, 1.0], 
                                      labels=['Losing (<40%)', 'Below Avg (40-50%)', 
                                              'Above Avg (50-60%)', 'Elite (>60%)'])
    sns.boxplot(data=team_df, x='Win_Category', y='Density', ax=ax2, palette='RdYlGn')
    ax2.set_title('Network Density by Team Performance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Win Category', fontsize=11)
    ax2.set_ylabel('Network Density', fontsize=11)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_density_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_density_vs_winning.png")
    
    return corr, p_value


def plot_avg_degree_vs_winning(team_df):
    """2. Average Degree vs Winning"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    sns.regplot(data=team_df, x='Avg_Weighted_Degree', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red', 'lw': 2})
    
    corr, p_value = stats.pearsonr(team_df['Avg_Weighted_Degree'].dropna(), 
                                    team_df.loc[team_df['Avg_Weighted_Degree'].notna(), 'W_PCT'])
    ax1.set_title(f'Average Weighted Degree vs Win %\nr = {corr:.3f}, p = {p_value:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Average Weighted Degree (Team)', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # By season trend
    ax2 = axes[1]
    season_avg = team_df.groupby('SEASON').agg({
        'Avg_Weighted_Degree': 'mean',
        'W_PCT': 'mean'
    }).reset_index()
    ax2.plot(range(len(season_avg)), season_avg['Avg_Weighted_Degree'], 'bo-', label='Avg Degree', lw=2)
    ax2.set_xticks(range(len(season_avg)))
    ax2.set_xticklabels(season_avg['SEASON'], rotation=45, ha='right')
    ax2.set_xlabel('Season', fontsize=11)
    ax2.set_ylabel('Average Weighted Degree', fontsize=11)
    ax2.set_title('Average Weighted Degree Over Seasons', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_avg_degree_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_avg_degree_vs_winning.png")
    
    return corr, p_value


def plot_std_degree_vs_winning(team_df):
    """3. Standard Deviation of Degree vs Winning (Hierarchy)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    sns.regplot(data=team_df, x='Std_Weighted_Degree', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'red', 'lw': 2})
    
    corr, p_value = stats.pearsonr(team_df['Std_Weighted_Degree'].dropna(), 
                                    team_df.loc[team_df['Std_Weighted_Degree'].notna(), 'W_PCT'])
    ax1.set_title(f'Degree Std (Hierarchy) vs Win %\nr = {corr:.3f}, p = {p_value:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Std of Weighted Degree', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # Coefficient of Variation
    ax2 = axes[1]
    sns.regplot(data=team_df, x='Degree_CV', y='W_PCT', ax=ax2,
                scatter_kws={'alpha': 0.6, 's': 50, 'color': 'green'}, 
                line_kws={'color': 'darkgreen', 'lw': 2})
    
    corr_cv, p_cv = stats.pearsonr(team_df['Degree_CV'].dropna(), 
                                    team_df.loc[team_df['Degree_CV'].notna(), 'W_PCT'])
    ax2.set_title(f'Degree CV (Relative Hierarchy) vs Win %\nr = {corr_cv:.3f}, p = {p_cv:.4f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coefficient of Variation of Degree', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_std_degree_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_std_degree_vs_winning.png")
    
    return corr, p_value


def plot_num_stars_vs_winning(team_df, star_counts):
    """4 & 5. Number of Stars vs Winning"""
    # Merge star counts with team data
    team_stars = team_df.merge(star_counts, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    team_stars['Num_Stars'] = team_stars['Num_Stars'].fillna(0)
    
    # Create star categories
    team_stars['Star_Category'] = team_stars['Num_Stars'].apply(
        lambda x: '0 Stars' if x == 0 else ('1 Star' if x == 1 else '2+ Stars')
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Boxplot: Stars vs Win%
    ax1 = axes[0, 0]
    order = ['0 Stars', '1 Star', '2+ Stars']
    sns.boxplot(data=team_stars, x='Star_Category', y='W_PCT', ax=ax1, 
                order=order, palette=['#ff6b6b', '#ffd93d', '#6bcb77'])
    ax1.set_title('Win % by Number of Stars', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Stars (Degree Z-score > 1.5)', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # Add mean values
    for i, cat in enumerate(order):
        mean_val = team_stars[team_stars['Star_Category'] == cat]['W_PCT'].mean()
        ax1.annotate(f'Mean: {mean_val:.3f}', xy=(i, mean_val), xytext=(i, mean_val + 0.05),
                     ha='center', fontsize=9, fontweight='bold')
    
    # Scatter: Num Stars vs Win% with jitter
    ax2 = axes[0, 1]
    team_stars['Num_Stars_Jitter'] = team_stars['Num_Stars'] + np.random.uniform(-0.15, 0.15, len(team_stars))
    ax2.scatter(team_stars['Num_Stars_Jitter'], team_stars['W_PCT'], alpha=0.5, s=40)
    
    # Add trend line
    z = np.polyfit(team_stars['Num_Stars'], team_stars['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, team_stars['Num_Stars'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r-', lw=2, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    ax2.legend()
    ax2.set_xlabel('Number of Stars', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    ax2.set_title('Win % vs Number of Stars (Scatter)', fontsize=12, fontweight='bold')
    
    # Distribution of star counts
    ax3 = axes[1, 0]
    star_dist = team_stars['Star_Category'].value_counts().reindex(order)
    bars = ax3.bar(order, star_dist.values, color=['#ff6b6b', '#ffd93d', '#6bcb77'], edgecolor='black')
    ax3.set_title('Distribution of Team Star Counts', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Star Category', fontsize=11)
    ax3.set_ylabel('Number of Team-Seasons', fontsize=11)
    for bar, val in zip(bars, star_dist.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), 
                 ha='center', fontweight='bold')
    
    # ANOVA test
    ax4 = axes[1, 1]
    groups = [team_stars[team_stars['Star_Category'] == cat]['W_PCT'].values for cat in order]
    f_stat, anova_p = stats.f_oneway(*groups)
    
    # Show stats
    stats_text = f"""Statistical Analysis:
    
ANOVA F-statistic: {f_stat:.3f}
ANOVA p-value: {anova_p:.4f}

Mean Win% by Category:
  0 Stars: {team_stars[team_stars['Star_Category'] == '0 Stars']['W_PCT'].mean():.3f}
  1 Star:  {team_stars[team_stars['Star_Category'] == '1 Star']['W_PCT'].mean():.3f}
  2+ Stars: {team_stars[team_stars['Star_Category'] == '2+ Stars']['W_PCT'].mean():.3f}

Sample Sizes:
  0 Stars: {len(team_stars[team_stars['Star_Category'] == '0 Stars'])}
  1 Star:  {len(team_stars[team_stars['Star_Category'] == '1 Star'])}
  2+ Stars: {len(team_stars[team_stars['Star_Category'] == '2+ Stars'])}
"""
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Statistical Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_num_stars_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_num_stars_vs_winning.png")
    
    return team_stars


def plot_additional_metrics(team_df):
    """6. Additional SNA Metrics vs Winning"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = [
        ('Gini_Coefficient', 'Gini Coefficient (Inequality)'),
        ('Degree_Centralization', 'Degree Centralization'),
        ('Top2_Concentration', 'Top 2 Player Concentration'),
        ('Degree_Range', 'Degree Range (Max - Min)'),
        ('Max_Betweenness', 'Max Betweenness Centrality'),
        ('Max_Eigenvector', 'Max Eigenvector Centrality')
    ]
    
    correlations = {}
    
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        if metric not in team_df.columns:
            ax.text(0.5, 0.5, f'{metric}\nNot Available', ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
            
        valid_data = team_df[[metric, 'W_PCT']].dropna()
        if len(valid_data) < 10:
            ax.text(0.5, 0.5, f'{metric}\nInsufficient Data', ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        sns.regplot(data=valid_data, x=metric, y='W_PCT', ax=ax,
                    scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': 'red', 'lw': 2})
        
        corr, p_value = stats.pearsonr(valid_data[metric], valid_data['W_PCT'])
        correlations[metric] = (corr, p_value)
        
        # Color based on significance
        sig_color = 'green' if p_value < 0.05 else 'gray'
        ax.set_title(f'{title}\nr = {corr:.3f}, p = {p_value:.4f}', 
                     fontsize=10, fontweight='bold', color=sig_color)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylabel('Win %', fontsize=9)
    
    plt.suptitle('Additional SNA Metrics vs Winning', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_additional_metrics_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 05_additional_metrics_vs_winning.png")
    
    return correlations


def plot_player_analysis(player_df):
    """Player-Level Analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Star vs Non-Star: Weighted Degree Distribution
    ax1 = axes[0, 0]
    sns.boxplot(data=player_df, x='Is_Star', y='Weighted_Degree', ax=ax1,
                palette=['lightcoral', 'lightgreen'])
    ax1.set_xticklabels(['Role Players', 'Stars'])
    ax1.set_title('Weighted Degree: Stars vs Role Players', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Weighted Degree')
    
    # 2. Star vs Non-Star: Points
    ax2 = axes[0, 1]
    sns.boxplot(data=player_df, x='Is_Star', y='PTS', ax=ax2,
                palette=['lightcoral', 'lightgreen'])
    ax2.set_xticklabels(['Role Players', 'Stars'])
    ax2.set_title('Points: Stars vs Role Players', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Points Per Season')
    
    # 3. Weighted Degree vs Points (colored by star status)
    ax3 = axes[0, 2]
    colors = ['red' if s else 'blue' for s in player_df['Is_Star']]
    ax3.scatter(player_df['Weighted_Degree'], player_df['PTS'], c=colors, alpha=0.3, s=20)
    ax3.set_xlabel('Weighted Degree')
    ax3.set_ylabel('Points')
    ax3.set_title('Weighted Degree vs Points\n(Red=Stars, Blue=Role Players)', fontsize=11, fontweight='bold')
    
    # Add correlation
    corr, p = stats.pearsonr(player_df['Weighted_Degree'], player_df['PTS'])
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Betweenness Centrality Distribution
    ax4 = axes[1, 0]
    sns.boxplot(data=player_df, x='Is_Star', y='Betweenness_Centrality', ax=ax4,
                palette=['lightcoral', 'lightgreen'])
    ax4.set_xticklabels(['Role Players', 'Stars'])
    ax4.set_title('Betweenness Centrality: Stars vs Role Players', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Betweenness Centrality')
    
    # 5. Top Stars by Weighted Degree
    ax5 = axes[1, 1]
    top_stars = player_df[player_df['Is_Star']].nlargest(15, 'Weighted_Degree')
    top_stars['Label'] = top_stars.apply(lambda r: f"{unidecode(r['PLAYER_NAME'].split(',')[0] if ',' in r['PLAYER_NAME'] else r['PLAYER_NAME'].split()[-1])} ({r['SEASON'][-5:]})", axis=1)
    
    bars = ax5.barh(range(len(top_stars)), top_stars['Weighted_Degree'], color='steelblue', edgecolor='black')
    ax5.set_yticks(range(len(top_stars)))
    ax5.set_yticklabels(top_stars['Label'], fontsize=8)
    ax5.set_xlabel('Weighted Degree')
    ax5.set_title('Top 15 Stars by Weighted Degree', fontsize=11, fontweight='bold')
    ax5.invert_yaxis()
    
    # 6. Z-Score Distribution
    ax6 = axes[1, 2]
    sns.histplot(data=player_df, x='Degree_ZScore', hue='Is_Star', ax=ax6, bins=50, alpha=0.6)
    ax6.axvline(x=1.5, color='red', linestyle='--', lw=2, label='Star Threshold (1.5)')
    ax6.set_xlabel('Degree Z-Score (within team)')
    ax6.set_title('Distribution of Degree Z-Scores', fontsize=11, fontweight='bold')
    ax6.legend()
    
    plt.suptitle('Player-Level Network Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_player_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 06_player_analysis.png")


def plot_star_threshold_sensitivity(player_df, team_df):
    """Analyze different star definition thresholds"""
    thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    results = []
    
    for thresh in thresholds:
        _, star_counts = define_stars(player_df.copy(), method='zscore', threshold=thresh)
        team_stars = team_df.merge(star_counts, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
        team_stars['Num_Stars'] = team_stars['Num_Stars'].fillna(0)
        
        # Correlation between num stars and winning
        corr, p = stats.pearsonr(team_stars['Num_Stars'], team_stars['W_PCT'])
        
        results.append({
            'Threshold': thresh,
            'Avg_Stars_Per_Team': team_stars['Num_Stars'].mean(),
            'Correlation': corr,
            'P_Value': p
        })
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Avg stars per threshold
    ax1 = axes[0]
    ax1.bar(range(len(thresholds)), results_df['Avg_Stars_Per_Team'], color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(thresholds)))
    ax1.set_xticklabels([f'{t}' for t in thresholds])
    ax1.set_xlabel('Z-Score Threshold')
    ax1.set_ylabel('Average Stars Per Team')
    ax1.set_title('Star Count by Threshold Definition', fontsize=12, fontweight='bold')
    
    # Plot 2: Correlation by threshold
    ax2 = axes[1]
    colors = ['green' if p < 0.05 else 'gray' for p in results_df['P_Value']]
    ax2.bar(range(len(thresholds)), results_df['Correlation'], color=colors, edgecolor='black')
    ax2.set_xticks(range(len(thresholds)))
    ax2.set_xticklabels([f'{t}' for t in thresholds])
    ax2.set_xlabel('Z-Score Threshold')
    ax2.set_ylabel('Correlation with Win%')
    ax2.set_title('Stars vs Winning Correlation by Threshold\n(Green = p < 0.05)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', lw=1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_star_threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 07_star_threshold_sensitivity.png")
    
    return results_df


def create_summary_heatmap(team_df):
    """Create correlation heatmap of all SNA metrics vs success metrics"""
    
    sna_metrics = ['Density', 'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration',
                   'Avg_Weighted_Degree', 'Std_Weighted_Degree', 'Degree_CV', 'Degree_Range',
                   'Max_Betweenness', 'Max_Eigenvector', 'Avg_Betweenness', 'Avg_Eigenvector']
    
    success_metrics = ['W_PCT', 'WINS']
    
    # Filter to available columns
    sna_available = [m for m in sna_metrics if m in team_df.columns]
    success_available = [m for m in success_metrics if m in team_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = []
    for sna in sna_available:
        row = []
        for success in success_available:
            valid = team_df[[sna, success]].dropna()
            if len(valid) > 10:
                corr, _ = stats.pearsonr(valid[sna], valid[success])
                row.append(corr)
            else:
                row.append(np.nan)
        corr_matrix.append(row)
    
    corr_df = pd.DataFrame(corr_matrix, index=sna_available, columns=success_available)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='RdYlGn', center=0, fmt='.3f',
                linewidths=0.5, ax=ax, vmin=-0.3, vmax=0.3,
                cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('SNA Metrics vs Team Success\nCorrelation Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Success Metrics', fontsize=11)
    ax.set_ylabel('SNA Metrics', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 08_correlation_heatmap.png")
    
    return corr_df


def print_summary_report(team_df, player_df, star_counts, correlations):
    """Print comprehensive summary report"""
    
    print("\n" + "="*80)
    print("SNA METRICS VS WINNING - SUMMARY REPORT")
    print("="*80)
    
    # Team-level correlations
    print("\n[TEAM-LEVEL CORRELATIONS WITH WIN%]")
    print("-" * 50)
    
    key_metrics = ['Density', 'Avg_Weighted_Degree', 'Std_Weighted_Degree', 
                   'Gini_Coefficient', 'Degree_Centralization', 'Top2_Concentration']
    
    for metric in key_metrics:
        if metric in team_df.columns:
            valid = team_df[[metric, 'W_PCT']].dropna()
            if len(valid) > 10:
                corr, p = stats.pearsonr(valid[metric], valid['W_PCT'])
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                print(f"  {metric:30s}: r = {corr:+.3f} (p = {p:.4f}) {sig}")
    
    # Star analysis
    print("\n[STAR ANALYSIS]")
    print("-" * 50)
    team_stars = team_df.merge(star_counts, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    team_stars['Num_Stars'] = team_stars['Num_Stars'].fillna(0)
    
    for n_stars in [0, 1, 2]:
        mask = team_stars['Num_Stars'] == n_stars if n_stars < 2 else team_stars['Num_Stars'] >= 2
        label = f'{n_stars} Stars' if n_stars < 2 else '2+ Stars'
        subset = team_stars[mask]
        print(f"  {label:15s}: n = {len(subset):3d}, Mean Win% = {subset['W_PCT'].mean():.3f}")
    
    # Player-level stats
    print("\n[PLAYER-LEVEL ANALYSIS]")
    print("-" * 50)
    stars = player_df[player_df['Is_Star']]
    non_stars = player_df[~player_df['Is_Star']]
    
    print(f"  Total Stars Identified: {len(stars)}")
    print(f"  Total Role Players: {len(non_stars)}")
    print(f"\n  Stars avg Weighted Degree: {stars['Weighted_Degree'].mean():.1f}")
    print(f"  Role Players avg Weighted Degree: {non_stars['Weighted_Degree'].mean():.1f}")
    print(f"\n  Stars avg Points: {stars['PTS'].mean():.1f}")
    print(f"  Role Players avg Points: {non_stars['PTS'].mean():.1f}")
    
    # Top stars
    print("\n[TOP 10 STARS BY WEIGHTED DEGREE]")
    print("-" * 50)
    top_stars = player_df[player_df['Is_Star']].nlargest(10, 'Weighted_Degree')
    for i, (_, row) in enumerate(top_stars.iterrows(), 1):
        name = unidecode(row['PLAYER_NAME'])
        print(f"  {i:2d}. {name:25s} ({row['TEAM_ABBREVIATION']}, {row['SEASON']}): {row['Weighted_Degree']:.0f}")
    
    print("\n" + "="*80)


def main():
    """Main execution"""
    print("="*60)
    print("SNA METRICS VS WINNING ANALYSIS")
    print("="*60)
    
    # Load data
    player_df, team_df = load_data()
    
    # Define stars
    player_df, star_counts = define_stars(player_df, method='zscore', threshold=1.5)
    
    # Calculate team network metrics
    team_df = calculate_team_network_metrics(player_df, team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    print("-" * 40)
    
    # 1. Density vs Winning
    corr1, p1 = plot_density_vs_winning(team_df)
    
    # 2. Avg Degree vs Winning
    corr2, p2 = plot_avg_degree_vs_winning(team_df)
    
    # 3. Std Degree vs Winning
    corr3, p3 = plot_std_degree_vs_winning(team_df)
    
    # 4 & 5. Number of Stars vs Winning
    team_stars = plot_num_stars_vs_winning(team_df, star_counts)
    
    # 6. Additional metrics
    add_corrs = plot_additional_metrics(team_df)
    
    # Player analysis
    plot_player_analysis(player_df)
    
    # Star threshold sensitivity
    threshold_results = plot_star_threshold_sensitivity(player_df.drop(columns=['Is_Star', 'Team_Mean_Degree', 'Team_Std_Degree', 'Degree_ZScore']), team_df)
    
    # Correlation heatmap
    corr_heatmap = create_summary_heatmap(team_df)
    
    # Print summary
    print_summary_report(team_df, player_df, star_counts, add_corrs)
    
    # Save enhanced data
    player_df.to_csv(OUTPUT_DIR / 'player_metrics_with_stars.csv', index=False)
    team_df.to_csv(OUTPUT_DIR / 'team_metrics_enhanced.csv', index=False)
    print(f"\n[OK] Saved enhanced CSVs to {OUTPUT_DIR}/")
    
    print(f"\n[COMPLETE] All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
