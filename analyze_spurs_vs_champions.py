"""
Spurs "Beautiful Game" Analysis
================================
Compare Spurs' team-oriented style to other championship teams.
Show how Spurs are outliers with:
- Lower star concentration (low max degree)
- Higher density (everyone connects)
- More equal distribution (lower Gini)
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

OUTPUT_DIR = Path("output_spurs_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Championship teams by season
CHAMPIONS = {
    '2015-16': 'CLE',  # Cleveland Cavaliers
    '2016-17': 'GSW',  # Golden State Warriors
    '2017-18': 'GSW',  # Golden State Warriors
    '2018-19': 'TOR',  # Toronto Raptors
    '2019-20': 'LAL',  # Los Angeles Lakers
    '2020-21': 'MIL',  # Milwaukee Bucks
    '2021-22': 'GSW',  # Golden State Warriors
    '2022-23': 'DEN',  # Denver Nuggets
    '2023-24': 'BOS',  # Boston Celtics
}

# Top teams (60+ wins or conference finals+)
ELITE_TEAMS = {
    '2015-16': ['GSW', 'SAS', 'CLE', 'OKC'],  # GSW 73-9, SAS 67-15
    '2016-17': ['GSW', 'SAS', 'CLE', 'HOU'],
    '2017-18': ['HOU', 'GSW', 'TOR', 'BOS'],
    '2018-19': ['MIL', 'TOR', 'GSW', 'DEN'],
    '2019-20': ['MIL', 'LAL', 'TOR', 'LAC'],
    '2020-21': ['UTA', 'PHX', 'MIL', 'PHI'],
    '2021-22': ['PHX', 'MEM', 'MIA', 'GSW'],
    '2022-23': ['MIL', 'BOS', 'DEN', 'PHX'],
    '2023-24': ['BOS', 'OKC', 'DEN', 'MIN'],
}


def load_data():
    """Load player and team metrics."""
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    team_df = pd.read_csv("output/nba_team_metrics.csv")
    
    # Calculate additional metrics from player data
    team_agg = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'min'],
        'PLAYER_ID': 'count'
    }).reset_index()
    team_agg.columns = ['TEAM_ABBREVIATION', 'SEASON', 
                        'Avg_Degree', 'Std_Degree', 'Max_Degree', 'Min_Degree', 'Roster_Size']
    
    team_df = team_df.merge(team_agg, on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    
    # Calculate degree range and CV
    team_df['Degree_Range'] = team_df['Max_Degree'] - team_df['Min_Degree']
    team_df['Degree_CV'] = team_df['Std_Degree'] / (team_df['Avg_Degree'] + 1)
    
    # Mark champions
    team_df['Is_Champion'] = team_df.apply(
        lambda r: CHAMPIONS.get(r['SEASON']) == r['TEAM_ABBREVIATION'], axis=1
    )
    
    # Mark Spurs
    team_df['Is_Spurs'] = team_df['TEAM_ABBREVIATION'] == 'SAS'
    
    # Mark elite teams
    team_df['Is_Elite'] = team_df.apply(
        lambda r: r['TEAM_ABBREVIATION'] in ELITE_TEAMS.get(r['SEASON'], []), axis=1
    )
    
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Champions: {team_df['Is_Champion'].sum()}")
    print(f"Spurs seasons: {team_df['Is_Spurs'].sum()}")
    
    return player_df, team_df


def get_spurs_metrics(team_df, player_df):
    """Extract detailed Spurs metrics."""
    spurs = team_df[team_df['Is_Spurs']].copy()
    
    print("\n" + "="*80)
    print("SAN ANTONIO SPURS - DETAILED METRICS")
    print("="*80)
    
    for _, row in spurs.iterrows():
        season = row['SEASON']
        print(f"\n--- {season} Spurs (Win%: {row['W_PCT']:.3f}, {row['WINS']:.0f} wins) ---")
        print(f"  Density:              {row['Density']:.4f}")
        print(f"  Star Weighted Degree: {row['Star_Weighted_Degree']:.0f}")
        print(f"  Gini Coefficient:     {row['Gini_Coefficient']:.4f}")
        print(f"  Degree Centralization:{row['Degree_Centralization']:.4f}")
        print(f"  Top2 Concentration:   {row['Top2_Concentration']:.4f}")
        print(f"  Avg Degree:           {row['Avg_Degree']:.0f}")
        print(f"  Std Degree:           {row['Std_Degree']:.0f}")
        print(f"  Degree CV:            {row['Degree_CV']:.4f}")
        
        # Top players for this season
        spurs_players = player_df[(player_df['TEAM_ABBREVIATION'] == 'SAS') & 
                                   (player_df['SEASON'] == season)].nlargest(5, 'Weighted_Degree')
        print(f"  Top 5 Players:")
        for i, (_, p) in enumerate(spurs_players.iterrows(), 1):
            name = unidecode(p['PLAYER_NAME'])
            print(f"    {i}. {name}: {p['Weighted_Degree']:.0f}")
    
    return spurs


def get_champions_metrics(team_df, player_df):
    """Extract champion team metrics."""
    champions = team_df[team_df['Is_Champion']].copy()
    
    print("\n" + "="*80)
    print("NBA CHAMPIONS - METRICS")
    print("="*80)
    
    for _, row in champions.iterrows():
        season = row['SEASON']
        team = row['TEAM_ABBREVIATION']
        print(f"\n--- {season} {team} (Win%: {row['W_PCT']:.3f}) ---")
        print(f"  Density:              {row['Density']:.4f}")
        print(f"  Star Weighted Degree: {row['Star_Weighted_Degree']:.0f}")
        print(f"  Gini Coefficient:     {row['Gini_Coefficient']:.4f}")
        print(f"  Top2 Concentration:   {row['Top2_Concentration']:.4f}")
        print(f"  Avg Degree:           {row['Avg_Degree']:.0f}")
        
        # Star player
        star_name = unidecode(str(row['Star_Player_Name']))
        print(f"  Star Player:          {star_name}")
    
    return champions


def compare_spurs_to_champions(team_df):
    """Compare Spurs to Champions statistically."""
    
    spurs = team_df[team_df['Is_Spurs']]
    champions = team_df[team_df['Is_Champion']]
    all_teams = team_df
    
    metrics = ['Density', 'Star_Weighted_Degree', 'Gini_Coefficient', 
               'Degree_Centralization', 'Top2_Concentration', 'Avg_Degree', 
               'Std_Degree', 'Degree_CV']
    
    print("\n" + "="*80)
    print("COMPARISON: SPURS vs CHAMPIONS vs ALL TEAMS")
    print("="*80)
    
    comparison_data = []
    
    print(f"\n{'Metric':<25} {'Spurs Avg':>12} {'Champs Avg':>12} {'All Avg':>12} {'Spurs Rank':>12}")
    print("-" * 75)
    
    for metric in metrics:
        if metric not in team_df.columns:
            continue
        
        spurs_avg = spurs[metric].mean()
        champs_avg = champions[metric].mean()
        all_avg = all_teams[metric].mean()
        
        # Calculate Spurs percentile
        spurs_pct = (all_teams[metric] < spurs_avg).mean() * 100
        
        print(f"{metric:<25} {spurs_avg:>12.3f} {champs_avg:>12.3f} {all_avg:>12.3f} {spurs_pct:>10.1f}%ile")
        
        comparison_data.append({
            'Metric': metric,
            'Spurs_Avg': spurs_avg,
            'Champions_Avg': champs_avg,
            'All_Teams_Avg': all_avg,
            'Spurs_Percentile': spurs_pct
        })
    
    return pd.DataFrame(comparison_data)


def plot_spurs_vs_champions(team_df):
    """Create visualization comparing Spurs to Champions."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Define colors
    colors = []
    for _, row in team_df.iterrows():
        if row['Is_Spurs']:
            colors.append('#000000')  # Black for Spurs
        elif row['Is_Champion']:
            colors.append('#FFD700')  # Gold for Champions
        elif row['Is_Elite']:
            colors.append('#1E90FF')  # Blue for Elite
        else:
            colors.append('#D3D3D3')  # Gray for others
    
    team_df = team_df.copy()
    team_df['Color'] = colors
    
    # 1. Star Weighted Degree vs Win%
    ax1 = axes[0, 0]
    for _, row in team_df.iterrows():
        ax1.scatter(row['Star_Weighted_Degree'], row['W_PCT'], 
                   c=row['Color'], s=80 if row['Is_Spurs'] or row['Is_Champion'] else 30,
                   edgecolors='black' if row['Is_Spurs'] else 'none',
                   linewidths=2 if row['Is_Spurs'] else 0,
                   alpha=0.8 if row['Is_Spurs'] or row['Is_Champion'] else 0.4,
                   zorder=10 if row['Is_Spurs'] else 5 if row['Is_Champion'] else 1)
    
    # Annotate Spurs
    spurs = team_df[team_df['Is_Spurs']]
    for _, row in spurs.iterrows():
        ax1.annotate(f"SAS\n{row['SEASON'][-5:]}", 
                    xy=(row['Star_Weighted_Degree'], row['W_PCT']),
                    fontsize=7, fontweight='bold')
    
    ax1.set_xlabel('Star Player Weighted Degree', fontsize=10)
    ax1.set_ylabel('Win %', fontsize=10)
    ax1.set_title('Star Degree vs Win %\n(Black=Spurs, Gold=Champions)', fontsize=11, fontweight='bold')
    
    # 2. Density vs Win%
    ax2 = axes[0, 1]
    for _, row in team_df.iterrows():
        ax2.scatter(row['Density'], row['W_PCT'],
                   c=row['Color'], s=80 if row['Is_Spurs'] or row['Is_Champion'] else 30,
                   edgecolors='black' if row['Is_Spurs'] else 'none',
                   linewidths=2 if row['Is_Spurs'] else 0,
                   alpha=0.8 if row['Is_Spurs'] or row['Is_Champion'] else 0.4,
                   zorder=10 if row['Is_Spurs'] else 5 if row['Is_Champion'] else 1)
    
    for _, row in spurs.iterrows():
        ax2.annotate(f"SAS\n{row['SEASON'][-5:]}", 
                    xy=(row['Density'], row['W_PCT']),
                    fontsize=7, fontweight='bold')
    
    ax2.set_xlabel('Network Density', fontsize=10)
    ax2.set_ylabel('Win %', fontsize=10)
    ax2.set_title('Density vs Win %', fontsize=11, fontweight='bold')
    
    # 3. Gini (Inequality) vs Win%
    ax3 = axes[0, 2]
    for _, row in team_df.iterrows():
        ax3.scatter(row['Gini_Coefficient'], row['W_PCT'],
                   c=row['Color'], s=80 if row['Is_Spurs'] or row['Is_Champion'] else 30,
                   edgecolors='black' if row['Is_Spurs'] else 'none',
                   linewidths=2 if row['Is_Spurs'] else 0,
                   alpha=0.8 if row['Is_Spurs'] or row['Is_Champion'] else 0.4,
                   zorder=10 if row['Is_Spurs'] else 5 if row['Is_Champion'] else 1)
    
    for _, row in spurs.iterrows():
        ax3.annotate(f"SAS\n{row['SEASON'][-5:]}", 
                    xy=(row['Gini_Coefficient'], row['W_PCT']),
                    fontsize=7, fontweight='bold')
    
    ax3.set_xlabel('Gini Coefficient (Inequality)', fontsize=10)
    ax3.set_ylabel('Win %', fontsize=10)
    ax3.set_title('Pass Inequality vs Win %', fontsize=11, fontweight='bold')
    
    # 4. Top2 Concentration vs Win%
    ax4 = axes[1, 0]
    for _, row in team_df.iterrows():
        ax4.scatter(row['Top2_Concentration'], row['W_PCT'],
                   c=row['Color'], s=80 if row['Is_Spurs'] or row['Is_Champion'] else 30,
                   edgecolors='black' if row['Is_Spurs'] else 'none',
                   linewidths=2 if row['Is_Spurs'] else 0,
                   alpha=0.8 if row['Is_Spurs'] or row['Is_Champion'] else 0.4,
                   zorder=10 if row['Is_Spurs'] else 5 if row['Is_Champion'] else 1)
    
    for _, row in spurs.iterrows():
        ax4.annotate(f"SAS\n{row['SEASON'][-5:]}", 
                    xy=(row['Top2_Concentration'], row['W_PCT']),
                    fontsize=7, fontweight='bold')
    
    ax4.set_xlabel('Top 2 Players Concentration', fontsize=10)
    ax4.set_ylabel('Win %', fontsize=10)
    ax4.set_title('Star Concentration vs Win %', fontsize=11, fontweight='bold')
    
    # 5. Box plot comparison
    ax5 = axes[1, 1]
    plot_data = []
    for _, row in team_df.iterrows():
        if row['Is_Spurs']:
            cat = 'Spurs'
        elif row['Is_Champion']:
            cat = 'Champions'
        else:
            cat = 'Other'
        plot_data.append({'Category': cat, 'Star_Degree': row['Star_Weighted_Degree']})
    
    plot_df = pd.DataFrame(plot_data)
    order = ['Other', 'Spurs', 'Champions']
    sns.boxplot(data=plot_df, x='Category', y='Star_Degree', ax=ax5, order=order,
                hue='Category', palette=['gray', 'black', 'gold'], legend=False)
    ax5.set_title('Star Degree Distribution', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Star Weighted Degree')
    
    # 6. Comparison bar chart
    ax6 = axes[1, 2]
    metrics = ['Density', 'Gini_Coefficient', 'Top2_Concentration']
    metric_labels = ['Density', 'Gini\n(Inequality)', 'Top2\nConcentration']
    
    spurs_vals = [spurs[m].mean() for m in metrics]
    champs_vals = [team_df[team_df['Is_Champion']][m].mean() for m in metrics]
    all_vals = [team_df[m].mean() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax6.bar(x - width, spurs_vals, width, label='Spurs', color='black')
    ax6.bar(x, champs_vals, width, label='Champions', color='gold', edgecolor='black')
    ax6.bar(x + width, all_vals, width, label='All Teams', color='gray')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(metric_labels)
    ax6.set_ylabel('Value')
    ax6.set_title('Spurs vs Champions vs All', fontsize=11, fontweight='bold')
    ax6.legend()
    
    plt.suptitle('SAN ANTONIO SPURS: "Beautiful Game" Analysis\nComparing to NBA Champions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_spurs_vs_champions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: 01_spurs_vs_champions.png")


def plot_spurs_outlier_analysis(team_df):
    """Show Spurs as outliers in network metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate Z-scores for Spurs
    spurs = team_df[team_df['Is_Spurs']].copy()
    
    metrics = ['Star_Weighted_Degree', 'Gini_Coefficient', 'Top2_Concentration', 
               'Degree_Centralization', 'Density', 'Avg_Degree']
    
    z_scores = []
    for metric in metrics:
        if metric in team_df.columns:
            mean = team_df[metric].mean()
            std = team_df[metric].std()
            for _, row in spurs.iterrows():
                z = (row[metric] - mean) / std if std > 0 else 0
                z_scores.append({
                    'Season': row['SEASON'],
                    'Metric': metric,
                    'Z_Score': z,
                    'Value': row[metric]
                })
    
    z_df = pd.DataFrame(z_scores)
    
    # 1. Z-score heatmap for all Spurs seasons
    ax1 = axes[0]
    pivot = z_df.pivot(index='Metric', columns='Season', values='Z_Score')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax1,
                cbar_kws={'label': 'Z-Score (Std Dev from Mean)'})
    ax1.set_title('Spurs Z-Scores Across Seasons\n(Negative = Below Average)', 
                  fontsize=11, fontweight='bold')
    
    # 2. Highlight 2015-16 and 2016-17 Spurs (their best years in our data)
    ax2 = axes[1]
    best_spurs = spurs[spurs['SEASON'].isin(['2015-16', '2016-17'])]
    champions = team_df[team_df['Is_Champion']]
    
    # Compare on key metrics
    metrics_compare = ['Star_Weighted_Degree', 'Top2_Concentration', 'Gini_Coefficient']
    
    data = []
    for _, row in best_spurs.iterrows():
        for m in metrics_compare:
            data.append({'Team': f"SAS {row['SEASON'][-5:]}", 'Metric': m, 'Value': row[m]})
    
    for _, row in champions.iterrows():
        for m in metrics_compare:
            data.append({'Team': f"{row['TEAM_ABBREVIATION']} {row['SEASON'][-5:]}", 
                        'Metric': m, 'Value': row[m]})
    
    compare_df = pd.DataFrame(data)
    
    # Highlight Spurs
    colors = ['black' if 'SAS' in t else 'gold' for t in compare_df['Team'].unique()]
    
    pivot2 = compare_df.pivot(index='Team', columns='Metric', values='Value')
    
    # Normalize for comparison
    pivot2_norm = (pivot2 - pivot2.mean()) / pivot2.std()
    
    sns.heatmap(pivot2_norm, annot=pivot2.round(3), fmt='', cmap='RdYlGn_r', center=0, ax=ax2)
    ax2.set_title('Best Spurs vs Champions\n(Normalized, Lower Star Metrics = Spurs Style)', 
                  fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_spurs_outlier_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_spurs_outlier_analysis.png")


def print_key_findings(team_df, comparison_df):
    """Print key findings about Spurs."""
    
    print("\n" + "="*80)
    print("KEY FINDINGS: SPURS 'BEAUTIFUL GAME' STYLE")
    print("="*80)
    
    spurs = team_df[team_df['Is_Spurs']]
    champions = team_df[team_df['Is_Champion']]
    
    # Best Spurs season (2016-17: 61-21)
    best_spurs = spurs[spurs['SEASON'] == '2016-17'].iloc[0]
    
    print(f"""
SPURS 2016-17 (61-21, 73% Win Rate) - Peak of Pop's System:
- Star Weighted Degree: {best_spurs['Star_Weighted_Degree']:.0f} (vs Champions avg: {champions['Star_Weighted_Degree'].mean():.0f})
- Gini Coefficient:     {best_spurs['Gini_Coefficient']:.3f} (vs Champions avg: {champions['Gini_Coefficient'].mean():.3f})
- Top2 Concentration:   {best_spurs['Top2_Concentration']:.3f} (vs Champions avg: {champions['Top2_Concentration'].mean():.3f})
- Degree Centralization:{best_spurs['Degree_Centralization']:.3f} (vs Champions avg: {champions['Degree_Centralization'].mean():.3f})

KEY DIFFERENCES FROM CHAMPIONS:
""")
    
    for _, row in comparison_df.iterrows():
        diff = row['Spurs_Avg'] - row['Champions_Avg']
        pct_diff = (diff / row['Champions_Avg']) * 100 if row['Champions_Avg'] != 0 else 0
        direction = "LOWER" if diff < 0 else "HIGHER"
        print(f"  {row['Metric']:<25}: Spurs are {abs(pct_diff):.1f}% {direction} than Champions")
    
    print("""
INTERPRETATION:
- LOWER Star Degree: Spurs don't rely on one dominant passer/hub
- LOWER Top2 Concentration: Ball doesn't just go through 2 stars
- LOWER Gini: More equal distribution of passes among all players
- LOWER Centralization: No single player controls the network

This is the "Beautiful Game" philosophy:
- Motion offense with constant ball/player movement
- Multiple playmakers instead of one star dominating
- Any player can be the hub on any possession
""")


def main():
    """Main execution."""
    print("="*60)
    print("SPURS 'BEAUTIFUL GAME' ANALYSIS")
    print("="*60)
    
    # Load data
    player_df, team_df = load_data()
    
    # Get detailed metrics
    spurs = get_spurs_metrics(team_df, player_df)
    champions = get_champions_metrics(team_df, player_df)
    
    # Compare
    comparison_df = compare_spurs_to_champions(team_df)
    
    # Generate plots
    print("\n[GENERATING VISUALIZATIONS]")
    print("-" * 40)
    plot_spurs_vs_champions(team_df)
    plot_spurs_outlier_analysis(team_df)
    
    # Print findings
    print_key_findings(team_df, comparison_df)
    
    # Save data
    comparison_df.to_csv(OUTPUT_DIR / 'spurs_vs_champions_comparison.csv', index=False)
    team_df[team_df['Is_Spurs'] | team_df['Is_Champion']].to_csv(
        OUTPUT_DIR / 'spurs_and_champions_metrics.csv', index=False
    )
    
    print(f"\n[OK] Saved data to {OUTPUT_DIR}/")
    print(f"\n[COMPLETE] Analysis saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
