"""
Ball Flow Analysis: Ball Huggers vs Elite Passers
==================================================
Analyzes team success based on having different types of players:
- Ball Huggers: High IN, Low OUT (receive but don't distribute)
- Elite Passers: High OUT, Low IN (distribute more than receive)
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
sns.set_palette("husl")

OUTPUT_DIR = Path("output_ball_flow_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load player and team metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    print(f"Loaded {len(player_df)} player-season records")
    print(f"Loaded {len(team_df)} team-season records")
    
    return player_df, team_df


def calculate_flow_ratios(player_df):
    """Calculate in/out flow ratios for each player."""
    player_df = player_df.copy()
    
    # Calculate ratios (add 1 to avoid division by zero)
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Out_In_Ratio'] = player_df['Weighted_Out_Degree'] / (player_df['Weighted_In_Degree'] + 1)
    
    # Net flow: positive = receives more, negative = passes more
    player_df['Net_Flow'] = player_df['Weighted_In_Degree'] - player_df['Weighted_Out_Degree']
    player_df['Net_Flow_Ratio'] = player_df['Net_Flow'] / (player_df['Weighted_Degree'] + 1)
    
    print(f"\nFlow Ratio Statistics:")
    print(f"  In/Out Ratio - Mean: {player_df['In_Out_Ratio'].mean():.2f}, Std: {player_df['In_Out_Ratio'].std():.2f}")
    print(f"  Out/In Ratio - Mean: {player_df['Out_In_Ratio'].mean():.2f}, Std: {player_df['Out_In_Ratio'].std():.2f}")
    print(f"  Net Flow - Mean: {player_df['Net_Flow'].mean():.1f}, Range: [{player_df['Net_Flow'].min():.0f}, {player_df['Net_Flow'].max():.0f}]")
    
    return player_df


def classify_player_types(player_df, method='zscore', threshold=1.0):
    """
    Classify players into Ball Huggers, Elite Passers, or Balanced.
    
    Ball Hugger: High In/Out ratio (receives much more than passes)
    Elite Passer: High Out/In ratio (passes much more than receives)
    """
    player_df = player_df.copy()
    
    # Calculate team-level statistics for relative comparison
    team_stats = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'In_Out_Ratio': ['mean', 'std'],
        'Out_In_Ratio': ['mean', 'std'],
        'Net_Flow': ['mean', 'std']
    }).reset_index()
    team_stats.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                          'Team_InOut_Mean', 'Team_InOut_Std',
                          'Team_OutIn_Mean', 'Team_OutIn_Std',
                          'Team_NetFlow_Mean', 'Team_NetFlow_Std']
    
    player_df = player_df.merge(team_stats, on=['TEAM_ABBREVIATION', 'SEASON'])
    
    # Z-scores within team
    player_df['InOut_ZScore'] = (player_df['In_Out_Ratio'] - player_df['Team_InOut_Mean']) / (player_df['Team_InOut_Std'] + 0.01)
    player_df['OutIn_ZScore'] = (player_df['Out_In_Ratio'] - player_df['Team_OutIn_Mean']) / (player_df['Team_OutIn_Std'] + 0.01)
    
    # Classify based on Z-scores
    # Ball Hugger: High In/Out Z-score (significantly above team average)
    # Elite Passer: High Out/In Z-score (significantly above team average)
    player_df['Is_Ball_Hugger'] = player_df['InOut_ZScore'] > threshold
    player_df['Is_Elite_Passer'] = player_df['OutIn_ZScore'] > threshold
    
    # Also use absolute thresholds for comparison
    # Ball Hugger: In/Out ratio > 1.3 (receives 30%+ more than passes)
    # Elite Passer: Out/In ratio > 1.3 (passes 30%+ more than receives)
    player_df['Is_Ball_Hugger_Abs'] = player_df['In_Out_Ratio'] > 1.3
    player_df['Is_Elite_Passer_Abs'] = player_df['Out_In_Ratio'] > 1.3
    
    # Extreme versions (for centric analysis)
    player_df['Is_Extreme_Hugger'] = player_df['In_Out_Ratio'] > 1.5
    player_df['Is_Extreme_Passer'] = player_df['Out_In_Ratio'] > 1.5
    
    print(f"\n[PLAYER CLASSIFICATION] (Z-score threshold: {threshold})")
    print(f"  Ball Huggers (Z-score): {player_df['Is_Ball_Hugger'].sum()}")
    print(f"  Elite Passers (Z-score): {player_df['Is_Elite_Passer'].sum()}")
    print(f"  Ball Huggers (Absolute >1.3): {player_df['Is_Ball_Hugger_Abs'].sum()}")
    print(f"  Elite Passers (Absolute >1.3): {player_df['Is_Elite_Passer_Abs'].sum()}")
    print(f"  Extreme Huggers (>1.5): {player_df['Is_Extreme_Hugger'].sum()}")
    print(f"  Extreme Passers (>1.5): {player_df['Is_Extreme_Passer'].sum()}")
    
    return player_df


def aggregate_team_types(player_df):
    """Count ball huggers and elite passers per team."""
    
    team_counts = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Is_Ball_Hugger': 'sum',
        'Is_Elite_Passer': 'sum',
        'Is_Ball_Hugger_Abs': 'sum',
        'Is_Elite_Passer_Abs': 'sum',
        'Is_Extreme_Hugger': 'sum',
        'Is_Extreme_Passer': 'sum',
        'PLAYER_ID': 'count'
    }).reset_index()
    
    team_counts.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                           'Num_Ball_Huggers', 'Num_Elite_Passers',
                           'Num_Ball_Huggers_Abs', 'Num_Elite_Passers_Abs',
                           'Num_Extreme_Huggers', 'Num_Extreme_Passers',
                           'Roster_Size']
    
    # Create categories
    def categorize(n):
        if n == 0: return '0'
        elif n == 1: return '1'
        else: return '2+'
    
    team_counts['Hugger_Category'] = team_counts['Num_Ball_Huggers_Abs'].apply(categorize)
    team_counts['Passer_Category'] = team_counts['Num_Elite_Passers_Abs'].apply(categorize)
    team_counts['Extreme_Hugger_Cat'] = team_counts['Num_Extreme_Huggers'].apply(categorize)
    team_counts['Extreme_Passer_Cat'] = team_counts['Num_Extreme_Passers'].apply(categorize)
    
    return team_counts


def merge_with_success(team_counts, team_df):
    """Merge team type counts with success metrics."""
    team_merged = team_counts.merge(
        team_df[['TEAM_ABBREVIATION', 'SEASON', 'W_PCT', 'WINS', 'LOSSES']],
        on=['TEAM_ABBREVIATION', 'SEASON'],
        how='left'
    )
    return team_merged


def plot_ball_huggers_vs_winning(team_df):
    """Analyze ball huggers (high in/out) vs team success."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Boxplot: Num Ball Huggers vs Win%
    ax1 = axes[0, 0]
    order = ['0', '1', '2+']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    sns.boxplot(data=team_df, x='Hugger_Category', y='W_PCT', ax=ax1, 
                order=order, hue='Hugger_Category', palette=colors, legend=False)
    
    # Add means
    for i, cat in enumerate(order):
        subset = team_df[team_df['Hugger_Category'] == cat]['W_PCT']
        if len(subset) > 0:
            mean_val = subset.mean()
            ax1.annotate(f'n={len(subset)}\nμ={mean_val:.3f}', 
                        xy=(i, mean_val), xytext=(i, mean_val + 0.08),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_title('Ball Huggers (In/Out > 1.3) vs Win %', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Ball Huggers on Team', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # 2. Scatter: Num Ball Huggers vs Win%
    ax2 = axes[0, 1]
    jitter = np.random.uniform(-0.15, 0.15, len(team_df))
    ax2.scatter(team_df['Num_Ball_Huggers_Abs'] + jitter, team_df['W_PCT'], 
                alpha=0.5, s=40, c='#e74c3c', edgecolors='black', linewidths=0.5)
    
    # Trend line
    z = np.polyfit(team_df['Num_Ball_Huggers_Abs'], team_df['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, team_df['Num_Ball_Huggers_Abs'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r-', lw=2, label=f'Trend: {z[0]:+.3f}x + {z[1]:.3f}')
    
    corr, pval = stats.pearsonr(team_df['Num_Ball_Huggers_Abs'], team_df['W_PCT'])
    ax2.set_title(f'Ball Huggers vs Win %\nr = {corr:.3f}, p = {pval:.4f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Ball Huggers', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    ax2.legend()
    
    # 3. Extreme Ball Huggers
    ax3 = axes[1, 0]
    order = ['0', '1', '2+']
    sns.boxplot(data=team_df, x='Extreme_Hugger_Cat', y='W_PCT', ax=ax3,
                order=order, hue='Extreme_Hugger_Cat', palette=colors, legend=False)
    
    for i, cat in enumerate(order):
        subset = team_df[team_df['Extreme_Hugger_Cat'] == cat]['W_PCT']
        if len(subset) > 0:
            ax3.annotate(f'n={len(subset)}\nμ={subset.mean():.3f}', 
                        xy=(i, subset.mean()), xytext=(i, subset.mean() + 0.08),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_title('EXTREME Ball Huggers (In/Out > 1.5) vs Win %', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Extreme Ball Huggers', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 4. Distribution
    ax4 = axes[1, 1]
    dist = team_df['Num_Ball_Huggers_Abs'].value_counts().sort_index()
    bars = ax4.bar(dist.index, dist.values, color='#e74c3c', edgecolor='black')
    ax4.set_xlabel('Number of Ball Huggers per Team', fontsize=11)
    ax4.set_ylabel('Number of Team-Seasons', fontsize=11)
    ax4.set_title('Distribution of Ball Huggers per Team', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, dist.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha='center', fontweight='bold')
    
    plt.suptitle('BALL HUGGERS ANALYSIS\n(Players who receive passes but don\'t distribute)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_ball_huggers_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_ball_huggers_vs_winning.png")


def plot_elite_passers_vs_winning(team_df):
    """Analyze elite passers (high out/in) vs team success."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Boxplot: Num Elite Passers vs Win%
    ax1 = axes[0, 0]
    order = ['0', '1', '2+']
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']  # Red, Yellow, Green (opposite of huggers)
    sns.boxplot(data=team_df, x='Passer_Category', y='W_PCT', ax=ax1,
                order=order, hue='Passer_Category', palette=colors, legend=False)
    
    for i, cat in enumerate(order):
        subset = team_df[team_df['Passer_Category'] == cat]['W_PCT']
        if len(subset) > 0:
            mean_val = subset.mean()
            ax1.annotate(f'n={len(subset)}\nμ={mean_val:.3f}', 
                        xy=(i, mean_val), xytext=(i, mean_val + 0.08),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_title('Elite Passers (Out/In > 1.3) vs Win %', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Elite Passers on Team', fontsize=11)
    ax1.set_ylabel('Win Percentage', fontsize=11)
    
    # 2. Scatter
    ax2 = axes[0, 1]
    jitter = np.random.uniform(-0.15, 0.15, len(team_df))
    ax2.scatter(team_df['Num_Elite_Passers_Abs'] + jitter, team_df['W_PCT'],
                alpha=0.5, s=40, c='#2ecc71', edgecolors='black', linewidths=0.5)
    
    z = np.polyfit(team_df['Num_Elite_Passers_Abs'], team_df['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, team_df['Num_Elite_Passers_Abs'].max(), 100)
    ax2.plot(x_line, p(x_line), 'g-', lw=2, label=f'Trend: {z[0]:+.3f}x + {z[1]:.3f}')
    
    corr, pval = stats.pearsonr(team_df['Num_Elite_Passers_Abs'], team_df['W_PCT'])
    ax2.set_title(f'Elite Passers vs Win %\nr = {corr:.3f}, p = {pval:.4f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Elite Passers', fontsize=11)
    ax2.set_ylabel('Win Percentage', fontsize=11)
    ax2.legend()
    
    # 3. Extreme Passers
    ax3 = axes[1, 0]
    sns.boxplot(data=team_df, x='Extreme_Passer_Cat', y='W_PCT', ax=ax3,
                order=order, hue='Extreme_Passer_Cat', palette=colors, legend=False)
    
    for i, cat in enumerate(order):
        subset = team_df[team_df['Extreme_Passer_Cat'] == cat]['W_PCT']
        if len(subset) > 0:
            ax3.annotate(f'n={len(subset)}\nμ={subset.mean():.3f}',
                        xy=(i, subset.mean()), xytext=(i, subset.mean() + 0.08),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_title('EXTREME Elite Passers (Out/In > 1.5) vs Win %', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Extreme Elite Passers', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 4. Distribution
    ax4 = axes[1, 1]
    dist = team_df['Num_Elite_Passers_Abs'].value_counts().sort_index()
    bars = ax4.bar(dist.index, dist.values, color='#2ecc71', edgecolor='black')
    ax4.set_xlabel('Number of Elite Passers per Team', fontsize=11)
    ax4.set_ylabel('Number of Team-Seasons', fontsize=11)
    ax4.set_title('Distribution of Elite Passers per Team', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, dist.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha='center', fontweight='bold')
    
    plt.suptitle('ELITE PASSERS ANALYSIS\n(Players who distribute more than they receive)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_elite_passers_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_elite_passers_vs_winning.png")


def plot_combined_analysis(team_df):
    """Combined analysis: Huggers vs Passers matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heatmap: Num Huggers x Num Passers -> Win%
    ax1 = axes[0, 0]
    pivot = team_df.pivot_table(values='W_PCT', 
                                 index='Hugger_Category',
                                 columns='Passer_Category',
                                 aggfunc='mean')
    # Reorder
    pivot = pivot.reindex(index=['0', '1', '2+'], columns=['0', '1', '2+'])
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                ax=ax1, cbar_kws={'label': 'Win %'}, vmin=0.35, vmax=0.65)
    ax1.set_title('Win % by Team Composition\n(Huggers vs Passers)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Elite Passers', fontsize=11)
    ax1.set_ylabel('Number of Ball Huggers', fontsize=11)
    
    # 2. Sample sizes heatmap
    ax2 = axes[0, 1]
    pivot_n = team_df.pivot_table(values='W_PCT',
                                   index='Hugger_Category',
                                   columns='Passer_Category',
                                   aggfunc='count')
    pivot_n = pivot_n.reindex(index=['0', '1', '2+'], columns=['0', '1', '2+'])
    
    sns.heatmap(pivot_n, annot=True, fmt='.0f', cmap='Blues',
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title('Sample Sizes by Category', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Elite Passers', fontsize=11)
    ax2.set_ylabel('Number of Ball Huggers', fontsize=11)
    
    # 3. Net comparison: Passers - Huggers
    ax3 = axes[1, 0]
    team_df['Net_Passers_Huggers'] = team_df['Num_Elite_Passers_Abs'] - team_df['Num_Ball_Huggers_Abs']
    
    def net_cat(n):
        if n < 0: return 'More Huggers'
        elif n == 0: return 'Equal'
        else: return 'More Passers'
    
    team_df['Net_Category'] = team_df['Net_Passers_Huggers'].apply(net_cat)
    order = ['More Huggers', 'Equal', 'More Passers']
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    
    sns.boxplot(data=team_df, x='Net_Category', y='W_PCT', ax=ax3,
                order=order, hue='Net_Category', palette=colors, legend=False)
    
    for i, cat in enumerate(order):
        subset = team_df[team_df['Net_Category'] == cat]['W_PCT']
        if len(subset) > 0:
            ax3.annotate(f'n={len(subset)}\nμ={subset.mean():.3f}',
                        xy=(i, subset.mean()), xytext=(i, subset.mean() + 0.08),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_title('Team Balance: Passers vs Huggers', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Team Composition', fontsize=11)
    ax3.set_ylabel('Win Percentage', fontsize=11)
    
    # 4. Scatter: Net balance vs Win%
    ax4 = axes[1, 1]
    jitter = np.random.uniform(-0.2, 0.2, len(team_df))
    colors_scatter = ['#e74c3c' if x < 0 else '#2ecc71' if x > 0 else '#f1c40f' 
                      for x in team_df['Net_Passers_Huggers']]
    ax4.scatter(team_df['Net_Passers_Huggers'] + jitter, team_df['W_PCT'],
                c=colors_scatter, alpha=0.5, s=40, edgecolors='black', linewidths=0.5)
    
    corr, pval = stats.pearsonr(team_df['Net_Passers_Huggers'], team_df['W_PCT'])
    z = np.polyfit(team_df['Net_Passers_Huggers'], team_df['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(team_df['Net_Passers_Huggers'].min(), team_df['Net_Passers_Huggers'].max(), 100)
    ax4.plot(x_line, p(x_line), 'b-', lw=2)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_title(f'Net (Passers - Huggers) vs Win %\nr = {corr:.3f}, p = {pval:.4f}', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Net: Elite Passers - Ball Huggers', fontsize=11)
    ax4.set_ylabel('Win Percentage', fontsize=11)
    
    plt.suptitle('COMBINED ANALYSIS: Ball Huggers vs Elite Passers',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_combined_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_combined_analysis.png")
    
    return team_df


def plot_player_scatter(player_df):
    """Scatter plot of all players: In vs Out degree."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Main scatter: In vs Out with quadrants
    ax1 = axes[0]
    
    # Color by type
    colors = []
    for _, row in player_df.iterrows():
        if row['Is_Ball_Hugger_Abs'] and row['Is_Elite_Passer_Abs']:
            colors.append('purple')  # Both (unlikely)
        elif row['Is_Ball_Hugger_Abs']:
            colors.append('#e74c3c')  # Ball Hugger - Red
        elif row['Is_Elite_Passer_Abs']:
            colors.append('#2ecc71')  # Elite Passer - Green
        else:
            colors.append('#3498db')  # Balanced - Blue
    
    ax1.scatter(player_df['Weighted_Out_Degree'], player_df['Weighted_In_Degree'],
                c=colors, alpha=0.4, s=20)
    
    # 45-degree line
    max_val = max(player_df['Weighted_Out_Degree'].max(), player_df['Weighted_In_Degree'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Equal In/Out')
    
    # Threshold lines
    ax1.plot([0, max_val], [0, max_val * 1.3], 'r--', lw=1, alpha=0.5, label='In/Out = 1.3 (Hugger)')
    ax1.plot([0, max_val * 1.3], [0, max_val], 'g--', lw=1, alpha=0.5, label='Out/In = 1.3 (Passer)')
    
    ax1.set_xlabel('Weighted OUT Degree (Passes Made)', fontsize=11)
    ax1.set_ylabel('Weighted IN Degree (Passes Received)', fontsize=11)
    ax1.set_title('Player Pass Flow: In vs Out\n(Red=Huggers, Green=Passers, Blue=Balanced)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # 2. Annotate extreme players
    ax2 = axes[1]
    ax2.scatter(player_df['Weighted_Out_Degree'], player_df['Weighted_In_Degree'],
                c=colors, alpha=0.4, s=20)
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=2)
    
    # Find extreme huggers and passers
    extreme_huggers = player_df.nlargest(5, 'In_Out_Ratio')
    extreme_passers = player_df.nlargest(5, 'Out_In_Ratio')
    
    for _, row in extreme_huggers.iterrows():
        name = unidecode(row['PLAYER_NAME'].split(',')[0] if ',' in row['PLAYER_NAME'] else row['PLAYER_NAME'].split()[-1])
        ax2.annotate(f"{name}\n({row['SEASON'][-5:]})",
                    xy=(row['Weighted_Out_Degree'], row['Weighted_In_Degree']),
                    fontsize=7, color='#e74c3c', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    for _, row in extreme_passers.iterrows():
        name = unidecode(row['PLAYER_NAME'].split(',')[0] if ',' in row['PLAYER_NAME'] else row['PLAYER_NAME'].split()[-1])
        ax2.annotate(f"{name}\n({row['SEASON'][-5:]})",
                    xy=(row['Weighted_Out_Degree'], row['Weighted_In_Degree']),
                    fontsize=7, color='#2ecc71', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax2.set_xlabel('Weighted OUT Degree (Passes Made)', fontsize=11)
    ax2.set_ylabel('Weighted IN Degree (Passes Received)', fontsize=11)
    ax2.set_title('Extreme Players Annotated', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_player_flow_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_player_flow_scatter.png")


def print_top_players(player_df):
    """Print top ball huggers and elite passers."""
    
    print("\n" + "="*80)
    print("TOP BALL HUGGERS (High In/Out Ratio)")
    print("="*80)
    
    top_huggers = player_df.nlargest(15, 'In_Out_Ratio')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 
         'Weighted_In_Degree', 'Weighted_Out_Degree', 'In_Out_Ratio', 'PTS']
    ]
    
    print(f"{'Player':<25} {'Team':<5} {'Season':<8} {'In':<7} {'Out':<7} {'Ratio':<6} {'PTS':<6}")
    print("-" * 80)
    for _, row in top_huggers.iterrows():
        name = unidecode(row['PLAYER_NAME'])[:24]
        print(f"{name:<25} {row['TEAM_ABBREVIATION']:<5} {row['SEASON']:<8} "
              f"{row['Weighted_In_Degree']:<7.0f} {row['Weighted_Out_Degree']:<7.0f} "
              f"{row['In_Out_Ratio']:<6.2f} {row['PTS']:<6.0f}")
    
    print("\n" + "="*80)
    print("TOP ELITE PASSERS (High Out/In Ratio)")
    print("="*80)
    
    top_passers = player_df.nlargest(15, 'Out_In_Ratio')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON',
         'Weighted_In_Degree', 'Weighted_Out_Degree', 'Out_In_Ratio', 'AST']
    ]
    
    print(f"{'Player':<25} {'Team':<5} {'Season':<8} {'In':<7} {'Out':<7} {'Ratio':<6} {'AST':<6}")
    print("-" * 80)
    for _, row in top_passers.iterrows():
        name = unidecode(row['PLAYER_NAME'])[:24]
        print(f"{name:<25} {row['TEAM_ABBREVIATION']:<5} {row['SEASON']:<8} "
              f"{row['Weighted_In_Degree']:<7.0f} {row['Weighted_Out_Degree']:<7.0f} "
              f"{row['Out_In_Ratio']:<6.2f} {row['AST']:<6.0f}")


def print_summary_stats(team_df):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Ball Huggers
    print("\n[BALL HUGGERS IMPACT ON WINNING]")
    for cat in ['0', '1', '2+']:
        subset = team_df[team_df['Hugger_Category'] == cat]
        print(f"  {cat} Ball Huggers: n={len(subset):3d}, Mean Win%={subset['W_PCT'].mean():.3f}, "
              f"Std={subset['W_PCT'].std():.3f}")
    
    corr_h, p_h = stats.pearsonr(team_df['Num_Ball_Huggers_Abs'], team_df['W_PCT'])
    print(f"\n  Correlation (Num Huggers vs Win%): r = {corr_h:.3f}, p = {p_h:.4f}")
    
    # Elite Passers
    print("\n[ELITE PASSERS IMPACT ON WINNING]")
    for cat in ['0', '1', '2+']:
        subset = team_df[team_df['Passer_Category'] == cat]
        print(f"  {cat} Elite Passers: n={len(subset):3d}, Mean Win%={subset['W_PCT'].mean():.3f}, "
              f"Std={subset['W_PCT'].std():.3f}")
    
    corr_p, p_p = stats.pearsonr(team_df['Num_Elite_Passers_Abs'], team_df['W_PCT'])
    print(f"\n  Correlation (Num Passers vs Win%): r = {corr_p:.3f}, p = {p_p:.4f}")
    
    # Net comparison
    print("\n[NET BALANCE: PASSERS - HUGGERS]")
    for cat in ['More Huggers', 'Equal', 'More Passers']:
        subset = team_df[team_df['Net_Category'] == cat]
        print(f"  {cat:<15}: n={len(subset):3d}, Mean Win%={subset['W_PCT'].mean():.3f}")
    
    corr_net, p_net = stats.pearsonr(team_df['Net_Passers_Huggers'], team_df['W_PCT'])
    print(f"\n  Correlation (Net Balance vs Win%): r = {corr_net:.3f}, p = {p_net:.4f}")
    
    # ANOVA tests
    print("\n[STATISTICAL TESTS]")
    groups_h = [team_df[team_df['Hugger_Category'] == cat]['W_PCT'].values for cat in ['0', '1', '2+']]
    f_h, p_anova_h = stats.f_oneway(*groups_h)
    print(f"  Ball Huggers ANOVA: F={f_h:.3f}, p={p_anova_h:.4f}")
    
    groups_p = [team_df[team_df['Passer_Category'] == cat]['W_PCT'].values for cat in ['0', '1', '2+']]
    f_p, p_anova_p = stats.f_oneway(*groups_p)
    print(f"  Elite Passers ANOVA: F={f_p:.3f}, p={p_anova_p:.4f}")


def main():
    """Main execution."""
    print("="*60)
    print("BALL FLOW ANALYSIS: Huggers vs Passers")
    print("="*60)
    
    # Load data
    player_df, team_df = load_data()
    
    # Calculate flow ratios
    player_df = calculate_flow_ratios(player_df)
    
    # Classify players
    player_df = classify_player_types(player_df, threshold=1.0)
    
    # Aggregate to team level
    team_counts = aggregate_team_types(player_df)
    
    # Merge with success
    team_merged = merge_with_success(team_counts, team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    print("-" * 40)
    
    # Generate plots
    plot_ball_huggers_vs_winning(team_merged)
    plot_elite_passers_vs_winning(team_merged)
    team_merged = plot_combined_analysis(team_merged)
    plot_player_scatter(player_df)
    
    # Print analysis
    print_top_players(player_df)
    print_summary_stats(team_merged)
    
    # Save enhanced data
    player_df.to_csv(OUTPUT_DIR / 'players_with_flow_types.csv', index=False)
    team_merged.to_csv(OUTPUT_DIR / 'teams_with_flow_counts.csv', index=False)
    print(f"\n[OK] Saved enhanced CSVs to {OUTPUT_DIR}/")
    
    print(f"\n[COMPLETE] All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
