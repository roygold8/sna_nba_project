"""
Entropy and Team Success Analysis
==================================
Calculate Shannon Entropy of pass distribution and correlate with winning.

High Entropy = More unpredictable/equal pass distribution (Spurs style)
Low Entropy = Concentrated/predictable passing through few players (Heliocentric)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_entropy_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_shannon_entropy(values):
    """Calculate Shannon entropy from a distribution of values."""
    values = np.array(values)
    values = values[values > 0]  # Remove zeros
    
    if len(values) == 0:
        return 0
    
    # Normalize to get probabilities
    probs = values / values.sum()
    
    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return entropy


def calculate_normalized_entropy(values):
    """Calculate normalized entropy (0-1 scale)."""
    values = np.array(values)
    values = values[values > 0]
    
    if len(values) <= 1:
        return 0
    
    entropy = calculate_shannon_entropy(values)
    max_entropy = np.log2(len(values))  # Max entropy when all equal
    
    return entropy / max_entropy if max_entropy > 0 else 0


def load_and_calculate_entropy(player_df):
    """Calculate entropy metrics for each team from player data."""
    
    results = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        # Get weighted degree distribution
        degrees = group['Weighted_Degree'].values
        in_degrees = group['Weighted_In_Degree'].values
        out_degrees = group['Weighted_Out_Degree'].values
        
        # Calculate various entropy measures
        degree_entropy = calculate_shannon_entropy(degrees)
        degree_entropy_norm = calculate_normalized_entropy(degrees)
        
        in_entropy = calculate_shannon_entropy(in_degrees)
        out_entropy = calculate_shannon_entropy(out_degrees)
        
        # Additional metrics
        n_players = len(degrees)
        max_possible_entropy = np.log2(n_players) if n_players > 0 else 0
        
        # Gini from player data (for comparison)
        sorted_degrees = np.sort(degrees)
        n = len(sorted_degrees)
        if n > 0 and sorted_degrees.sum() > 0:
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_degrees) - (n + 1) * sorted_degrees.sum()) / (n * sorted_degrees.sum())
        else:
            gini = 0
        
        results.append({
            'TEAM_ABBREVIATION': team,
            'SEASON': season,
            'Degree_Entropy': degree_entropy,
            'Degree_Entropy_Norm': degree_entropy_norm,
            'In_Degree_Entropy': in_entropy,
            'Out_Degree_Entropy': out_entropy,
            'Max_Entropy': max_possible_entropy,
            'Entropy_Ratio': degree_entropy / max_possible_entropy if max_possible_entropy > 0 else 0,
            'Num_Players': n_players,
            'Gini_Calc': gini
        })
    
    return pd.DataFrame(results)


def merge_with_success(entropy_df, team_df):
    """Merge entropy metrics with team success."""
    merged = entropy_df.merge(
        team_df[['TEAM_ABBREVIATION', 'SEASON', 'W_PCT', 'WINS', 'LOSSES', 
                 'Density', 'Gini_Coefficient', 'Star_Weighted_Degree', 'Star_Player_Name']],
        on=['TEAM_ABBREVIATION', 'SEASON'],
        how='left'
    )
    return merged


def plot_entropy_vs_winning(df):
    """Main entropy vs winning analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Raw Entropy vs Win%
    ax1 = axes[0, 0]
    sns.regplot(data=df, x='Degree_Entropy', y='W_PCT', ax=ax1,
                scatter_kws={'alpha': 0.5, 's': 40}, line_kws={'color': 'red', 'lw': 2})
    
    corr1, p1 = stats.pearsonr(df['Degree_Entropy'].dropna(), 
                                df.loc[df['Degree_Entropy'].notna(), 'W_PCT'])
    ax1.set_title(f'Pass Entropy vs Win %\nr = {corr1:.3f}, p = {p1:.4f}',
                  fontsize=11, fontweight='bold',
                  color='green' if p1 < 0.05 else 'black')
    ax1.set_xlabel('Shannon Entropy of Pass Distribution', fontsize=10)
    ax1.set_ylabel('Win Percentage', fontsize=10)
    
    # 2. Normalized Entropy vs Win%
    ax2 = axes[0, 1]
    sns.regplot(data=df, x='Degree_Entropy_Norm', y='W_PCT', ax=ax2,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'green'},
                line_kws={'color': 'darkgreen', 'lw': 2})
    
    corr2, p2 = stats.pearsonr(df['Degree_Entropy_Norm'].dropna(),
                                df.loc[df['Degree_Entropy_Norm'].notna(), 'W_PCT'])
    ax2.set_title(f'Normalized Entropy vs Win %\nr = {corr2:.3f}, p = {p2:.4f}',
                  fontsize=11, fontweight='bold',
                  color='green' if p2 < 0.05 else 'black')
    ax2.set_xlabel('Normalized Entropy (0-1 scale)', fontsize=10)
    ax2.set_ylabel('Win Percentage', fontsize=10)
    
    # 3. Entropy vs Gini (should be inversely related)
    ax3 = axes[0, 2]
    sns.regplot(data=df, x='Degree_Entropy_Norm', y='Gini_Coefficient', ax=ax3,
                scatter_kws={'alpha': 0.5, 's': 40, 'color': 'purple'},
                line_kws={'color': 'darkviolet', 'lw': 2})
    
    corr3, p3 = stats.pearsonr(df['Degree_Entropy_Norm'].dropna(),
                                df.loc[df['Degree_Entropy_Norm'].notna(), 'Gini_Coefficient'])
    ax3.set_title(f'Entropy vs Gini (Should be Inverse)\nr = {corr3:.3f}, p = {p3:.4f}',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Normalized Entropy', fontsize=10)
    ax3.set_ylabel('Gini Coefficient', fontsize=10)
    
    # 4. Entropy Quartiles vs Win%
    ax4 = axes[1, 0]
    df['Entropy_Quartile'] = pd.qcut(df['Degree_Entropy_Norm'], q=4, 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
    sns.boxplot(data=df, x='Entropy_Quartile', y='W_PCT', ax=ax4,
                order=['Low', 'Medium', 'High', 'Very High'],
                hue='Entropy_Quartile', palette=colors, legend=False)
    
    for i, level in enumerate(['Low', 'Medium', 'High', 'Very High']):
        subset = df[df['Entropy_Quartile'] == level]['W_PCT']
        ax4.annotate(f'n={len(subset)}\nμ={subset.mean():.3f}',
                    xy=(i, subset.mean()), xytext=(i, subset.mean() + 0.08),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_title('Win % by Entropy Quartile', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Entropy Level', fontsize=10)
    ax4.set_ylabel('Win Percentage', fontsize=10)
    
    # 5. In vs Out Entropy
    ax5 = axes[1, 1]
    ax5.scatter(df['In_Degree_Entropy'], df['Out_Degree_Entropy'], 
                c=df['W_PCT'], cmap='RdYlGn', s=40, alpha=0.7)
    ax5.plot([df['In_Degree_Entropy'].min(), df['In_Degree_Entropy'].max()],
             [df['In_Degree_Entropy'].min(), df['In_Degree_Entropy'].max()],
             'k--', lw=1, label='Equal Line')
    ax5.set_xlabel('In-Degree Entropy (Receiving)', fontsize=10)
    ax5.set_ylabel('Out-Degree Entropy (Passing)', fontsize=10)
    ax5.set_title('In vs Out Entropy\n(Color = Win%)', fontsize=11, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=df['W_PCT'].min(), vmax=df['W_PCT'].max()))
    plt.colorbar(sm, ax=ax5, label='Win %')
    
    # 6. Entropy by Season trend
    ax6 = axes[1, 2]
    season_avg = df.groupby('SEASON').agg({
        'Degree_Entropy_Norm': 'mean',
        'W_PCT': 'mean'
    }).reset_index()
    
    ax6.plot(range(len(season_avg)), season_avg['Degree_Entropy_Norm'], 'bo-', lw=2, markersize=8)
    ax6.set_xticks(range(len(season_avg)))
    ax6.set_xticklabels(season_avg['SEASON'], rotation=45, ha='right')
    ax6.set_xlabel('Season', fontsize=10)
    ax6.set_ylabel('Average Normalized Entropy', fontsize=10)
    ax6.set_title('League Entropy Trend Over Time', fontsize=11, fontweight='bold')
    
    plt.suptitle('ENTROPY ANALYSIS: Pass Distribution Unpredictability vs Team Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_entropy_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_entropy_vs_winning.png")
    
    return {'Entropy': (corr1, p1), 'Entropy_Norm': (corr2, p2)}


def plot_entropy_extremes(df):
    """Analyze teams with extreme entropy values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 highest entropy teams
    ax1 = axes[0]
    top_entropy = df.nlargest(15, 'Degree_Entropy_Norm')
    
    labels = [f"{row['TEAM_ABBREVIATION']} {row['SEASON'][-5:]}" for _, row in top_entropy.iterrows()]
    colors = ['green' if w > 0.5 else 'red' for w in top_entropy['W_PCT']]
    
    bars = ax1.barh(range(len(top_entropy)), top_entropy['Degree_Entropy_Norm'], color=colors, edgecolor='black')
    ax1.set_yticks(range(len(top_entropy)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Normalized Entropy')
    ax1.set_title('Top 15 Highest Entropy Teams\n(Green = Winning, Red = Losing)', fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add win % labels
    for i, (_, row) in enumerate(top_entropy.iterrows()):
        ax1.text(row['Degree_Entropy_Norm'] + 0.01, i, f"{row['W_PCT']:.3f}", 
                va='center', fontsize=8)
    
    # Top 10 lowest entropy teams (heliocentric)
    ax2 = axes[1]
    low_entropy = df.nsmallest(15, 'Degree_Entropy_Norm')
    
    labels = [f"{row['TEAM_ABBREVIATION']} {row['SEASON'][-5:]}" for _, row in low_entropy.iterrows()]
    colors = ['green' if w > 0.5 else 'red' for w in low_entropy['W_PCT']]
    
    bars = ax2.barh(range(len(low_entropy)), low_entropy['Degree_Entropy_Norm'], color=colors, edgecolor='black')
    ax2.set_yticks(range(len(low_entropy)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Normalized Entropy')
    ax2.set_title('Top 15 Lowest Entropy Teams (Most Heliocentric)\n(Green = Winning, Red = Losing)', 
                  fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add win % and star labels
    for i, (_, row) in enumerate(low_entropy.iterrows()):
        star = unidecode(str(row['Star_Player_Name']).split(',')[0] if ',' in str(row['Star_Player_Name']) else str(row['Star_Player_Name']).split()[-1])
        ax2.text(row['Degree_Entropy_Norm'] + 0.01, i, f"{row['W_PCT']:.3f} ({star})", 
                va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_entropy_extremes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_entropy_extremes.png")


def plot_entropy_champions(df):
    """Compare entropy of championship teams."""
    
    # Define champions
    champions = {
        '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW',
        '2018-19': 'TOR', '2019-20': 'LAL', '2020-21': 'MIL',
        '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
    }
    
    df['Is_Champion'] = df.apply(
        lambda r: champions.get(r['SEASON']) == r['TEAM_ABBREVIATION'], axis=1
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Champion entropy comparison
    ax1 = axes[0]
    champ_df = df[df['Is_Champion']].sort_values('SEASON')
    
    colors = plt.cm.RdYlGn(champ_df['W_PCT'] / champ_df['W_PCT'].max())
    
    bars = ax1.bar(range(len(champ_df)), champ_df['Degree_Entropy_Norm'], color=colors, edgecolor='black')
    ax1.set_xticks(range(len(champ_df)))
    ax1.set_xticklabels([f"{row['TEAM_ABBREVIATION']}\n{row['SEASON'][-5:]}" for _, row in champ_df.iterrows()], 
                        fontsize=8)
    ax1.set_ylabel('Normalized Entropy')
    ax1.set_title('Champions: Pass Distribution Entropy', fontsize=11, fontweight='bold')
    ax1.axhline(y=df['Degree_Entropy_Norm'].mean(), color='red', linestyle='--', 
                label=f'League Avg: {df["Degree_Entropy_Norm"].mean():.3f}')
    ax1.legend()
    
    # Add star player labels
    for i, (_, row) in enumerate(champ_df.iterrows()):
        star = unidecode(str(row['Star_Player_Name']).split(',')[0] if ',' in str(row['Star_Player_Name']) else str(row['Star_Player_Name']).split()[-1])[:10]
        ax1.text(i, row['Degree_Entropy_Norm'] + 0.01, star, ha='center', fontsize=7, rotation=45)
    
    # 2. Box plot: Champions vs Others
    ax2 = axes[1]
    df['Category'] = df['Is_Champion'].map({True: 'Champions', False: 'Other Teams'})
    
    sns.boxplot(data=df, x='Category', y='Degree_Entropy_Norm', ax=ax2,
                hue='Category', palette=['gold', 'gray'], legend=False)
    
    # Add stats
    champ_mean = df[df['Is_Champion']]['Degree_Entropy_Norm'].mean()
    other_mean = df[~df['Is_Champion']]['Degree_Entropy_Norm'].mean()
    
    t_stat, t_pval = stats.ttest_ind(
        df[df['Is_Champion']]['Degree_Entropy_Norm'],
        df[~df['Is_Champion']]['Degree_Entropy_Norm']
    )
    
    ax2.set_title(f'Champions vs Others Entropy\nChamps: {champ_mean:.3f}, Others: {other_mean:.3f}\nt-test p = {t_pval:.4f}',
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('Normalized Entropy')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_entropy_champions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_entropy_champions.png")
    
    return df


def print_summary(df, correlations):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("ENTROPY AND TEAM SUCCESS - SUMMARY")
    print("="*80)
    
    print("\n[CORRELATIONS WITH WIN%]")
    print("-" * 50)
    for name, (corr, pval) in correlations.items():
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name:25s}: r = {corr:+.3f}, p = {pval:.4f} {sig}")
    
    # Entropy quartiles
    print("\n[WIN% BY ENTROPY QUARTILE]")
    print("-" * 50)
    for level in ['Low', 'Medium', 'High', 'Very High']:
        subset = df[df['Entropy_Quartile'] == level]
        print(f"  {level:15s}: n = {len(subset):3d}, Mean Win% = {subset['W_PCT'].mean():.3f}")
    
    # Champions
    print("\n[CHAMPION TEAMS ENTROPY]")
    print("-" * 50)
    champs = df[df['Is_Champion']].sort_values('SEASON')
    league_mean = df['Degree_Entropy_Norm'].mean()
    
    for _, row in champs.iterrows():
        star = unidecode(str(row['Star_Player_Name']))[:20]
        diff = row['Degree_Entropy_Norm'] - league_mean
        direction = "ABOVE" if diff > 0 else "BELOW"
        print(f"  {row['SEASON']} {row['TEAM_ABBREVIATION']}: {row['Degree_Entropy_Norm']:.3f} "
              f"({abs(diff):.3f} {direction} avg) - {star}")
    
    # Highest and lowest entropy successful teams
    print("\n[HIGHEST ENTROPY WINNING TEAMS (>50% Win)]")
    print("-" * 50)
    winners = df[df['W_PCT'] > 0.5].nlargest(5, 'Degree_Entropy_Norm')
    for _, row in winners.iterrows():
        print(f"  {row['TEAM_ABBREVIATION']} {row['SEASON']}: Entropy={row['Degree_Entropy_Norm']:.3f}, Win%={row['W_PCT']:.3f}")
    
    print("\n[LOWEST ENTROPY WINNING TEAMS (Heliocentric)]")
    print("-" * 50)
    winners_low = df[df['W_PCT'] > 0.5].nsmallest(5, 'Degree_Entropy_Norm')
    for _, row in winners_low.iterrows():
        star = unidecode(str(row['Star_Player_Name']))[:15]
        print(f"  {row['TEAM_ABBREVIATION']} {row['SEASON']}: Entropy={row['Degree_Entropy_Norm']:.3f}, Win%={row['W_PCT']:.3f} ({star})")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("ENTROPY AND TEAM SUCCESS ANALYSIS")
    print("="*60)
    
    # Load data
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    print(f"Loaded {len(player_df)} player-season records")
    print(f"Loaded {len(team_df)} team-season records")
    
    # Calculate entropy
    print("\n[Calculating entropy metrics...]")
    entropy_df = load_and_calculate_entropy(player_df)
    print(f"  Calculated entropy for {len(entropy_df)} team-seasons")
    
    # Merge with success
    merged_df = merge_with_success(entropy_df, team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    print("-" * 40)
    
    # Generate plots
    correlations = plot_entropy_vs_winning(merged_df)
    plot_entropy_extremes(merged_df)
    merged_df = plot_entropy_champions(merged_df)
    
    # Print summary
    print_summary(merged_df, correlations)
    
    # Save data
    merged_df.to_csv(OUTPUT_DIR / 'team_entropy_metrics.csv', index=False)
    print(f"\n[OK] Saved data to {OUTPUT_DIR}/")
    
    print(f"\n[COMPLETE] All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
