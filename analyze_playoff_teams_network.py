"""
Playoff Teams vs Non-Playoff Teams Network Analysis
====================================================
Compares network metrics between teams that made playoffs vs those that didn't
Uses team success metrics to identify playoff teams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from unidecode import unidecode

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_playoff_teams_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Champions by season
CHAMPIONS = {
    '2015-16': 'CLE',
    '2016-17': 'GSW',
    '2017-18': 'GSW',
    '2018-19': 'TOR',
    '2019-20': 'LAL',
    '2020-21': 'MIL',
    '2021-22': 'GSW',
    '2022-23': 'DEN',
    '2023-24': 'BOS'
}

# Playoff teams by season (top 16 teams)
PLAYOFF_TEAMS = {
    '2015-16': ['GSW', 'SAS', 'OKC', 'LAC', 'POR', 'DAL', 'MEM', 'HOU',  # West
                'CLE', 'TOR', 'MIA', 'ATL', 'BOS', 'CHA', 'IND', 'DET'],  # East
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


def load_and_enrich_data():
    """Load player data and add playoff/champion indicators."""
    
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    
    # Add derived metrics
    player_df['Degree_Per_Game'] = player_df['Weighted_Degree'] / (player_df['GP'] + 1)
    player_df['In_Out_Ratio'] = player_df['Weighted_In_Degree'] / (player_df['Weighted_Out_Degree'] + 1)
    player_df['Net_Pass_Flow'] = player_df['Weighted_In_Degree'] - player_df['Weighted_Out_Degree']
    
    # Add playoff indicator
    player_df['Made_Playoffs'] = player_df.apply(
        lambda x: x['TEAM_ABBREVIATION'] in PLAYOFF_TEAMS.get(x['SEASON'], []), axis=1
    )
    
    # Add champion indicator
    player_df['Is_Champion'] = player_df.apply(
        lambda x: x['TEAM_ABBREVIATION'] == CHAMPIONS.get(x['SEASON'], ''), axis=1
    )
    
    # Deep playoff run (conference finals or better)
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
    
    player_df['Deep_Run'] = player_df.apply(
        lambda x: x['TEAM_ABBREVIATION'] in DEEP_RUNS.get(x['SEASON'], []), axis=1
    )
    
    print(f"Loaded {len(player_df)} player-seasons")
    print(f"Playoff Teams: {player_df['Made_Playoffs'].sum()} ({player_df['Made_Playoffs'].mean()*100:.1f}%)")
    print(f"Champions: {player_df['Is_Champion'].sum()}")
    print(f"Deep Run Teams: {player_df['Deep_Run'].sum()}")
    
    return player_df


def aggregate_team_metrics(player_df):
    """Aggregate player metrics to team level."""
    
    team_agg = player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'sum'],
        'Degree_Per_Game': ['mean', 'std', 'max'],
        'Eigenvector_Centrality': ['mean', 'max'],
        'Betweenness_Centrality': ['mean', 'max'],
        'In_Out_Ratio': ['mean', 'std'],
        'Made_Playoffs': 'first',
        'Is_Champion': 'first',
        'Deep_Run': 'first',
        'GP': 'max'
    }).reset_index()
    
    team_agg.columns = ['Team', 'Season', 
                        'Avg_Degree', 'Std_Degree', 'Max_Degree', 'Total_Degree',
                        'Avg_Degree_PG', 'Std_Degree_PG', 'Max_Degree_PG',
                        'Avg_Eigen', 'Max_Eigen',
                        'Avg_Between', 'Max_Between',
                        'Avg_InOut', 'Std_InOut',
                        'Made_Playoffs', 'Is_Champion', 'Deep_Run', 'GP']
    
    # Top player concentration
    for _, row in team_agg.iterrows():
        team_agg.loc[_, 'Top_Concentration'] = row['Max_Degree'] / row['Total_Degree'] * 100
    
    return team_agg


def compare_playoff_vs_lottery(player_df, team_df):
    """Compare metrics between playoff and non-playoff teams."""
    
    playoff_players = player_df[player_df['Made_Playoffs']]
    lottery_players = player_df[~player_df['Made_Playoffs']]
    
    playoff_teams = team_df[team_df['Made_Playoffs']]
    lottery_teams = team_df[~team_df['Made_Playoffs']]
    
    # Player-level comparison
    player_metrics = ['Weighted_Degree', 'Degree_Per_Game', 'Eigenvector_Centrality', 
                      'Betweenness_Centrality', 'In_Out_Ratio']
    
    player_comparison = []
    for metric in player_metrics:
        play_mean = playoff_players[metric].mean()
        lot_mean = lottery_players[metric].mean()
        t_stat, p_val = stats.ttest_ind(playoff_players[metric].dropna(), 
                                         lottery_players[metric].dropna())
        
        # Effect size
        pooled_std = np.sqrt((playoff_players[metric].std()**2 + lottery_players[metric].std()**2) / 2)
        cohens_d = (play_mean - lot_mean) / pooled_std if pooled_std > 0 else 0
        
        player_comparison.append({
            'Metric': metric,
            'Level': 'Player',
            'Playoff_Mean': play_mean,
            'Lottery_Mean': lot_mean,
            'Diff': play_mean - lot_mean,
            'Pct_Diff': ((play_mean - lot_mean) / lot_mean * 100) if lot_mean != 0 else 0,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Cohens_d': cohens_d
        })
    
    # Team-level comparison
    team_metrics = ['Avg_Degree', 'Std_Degree', 'Max_Degree', 'Top_Concentration',
                    'Avg_Degree_PG', 'Avg_Eigen', 'Avg_Between']
    
    team_comparison = []
    for metric in team_metrics:
        play_mean = playoff_teams[metric].mean()
        lot_mean = lottery_teams[metric].mean()
        t_stat, p_val = stats.ttest_ind(playoff_teams[metric].dropna(), 
                                         lottery_teams[metric].dropna())
        
        pooled_std = np.sqrt((playoff_teams[metric].std()**2 + lottery_teams[metric].std()**2) / 2)
        cohens_d = (play_mean - lot_mean) / pooled_std if pooled_std > 0 else 0
        
        team_comparison.append({
            'Metric': metric,
            'Level': 'Team',
            'Playoff_Mean': play_mean,
            'Lottery_Mean': lot_mean,
            'Diff': play_mean - lot_mean,
            'Pct_Diff': ((play_mean - lot_mean) / lot_mean * 100) if lot_mean != 0 else 0,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Cohens_d': cohens_d
        })
    
    return pd.DataFrame(player_comparison + team_comparison)


def compare_by_playoff_success(team_df):
    """Compare metrics across playoff success levels."""
    
    # Create success levels
    team_df = team_df.copy()
    team_df['Success_Level'] = 'Lottery'
    team_df.loc[team_df['Made_Playoffs'], 'Success_Level'] = 'Playoff'
    team_df.loc[team_df['Deep_Run'], 'Success_Level'] = 'Deep Run'
    team_df.loc[team_df['Is_Champion'], 'Success_Level'] = 'Champion'
    
    return team_df


def plot_comparison(comparison_df, player_df, team_df):
    """Create comparison visualizations."""
    
    fig = plt.figure(figsize=(20, 16))
    
    playoff_players = player_df[player_df['Made_Playoffs']]
    lottery_players = player_df[~player_df['Made_Playoffs']]
    
    # 1. Player-level metrics comparison
    ax1 = fig.add_subplot(2, 3, 1)
    
    player_comp = comparison_df[comparison_df['Level'] == 'Player']
    x = np.arange(len(player_comp))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, player_comp['Lottery_Mean'], width, label='Lottery', color='gray')
    bars2 = ax1.bar(x + width/2, player_comp['Playoff_Mean'], width, label='Playoff', color='green')
    
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Player Metrics: Playoff vs Lottery Teams', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in player_comp['Metric']], rotation=0, fontsize=8)
    ax1.legend()
    
    # Add significance markers
    for i, row in enumerate(player_comp.itertuples()):
        if row.P_Value < 0.001:
            ax1.text(i, max(row.Lottery_Mean, row.Playoff_Mean) * 1.05, '***', ha='center')
        elif row.P_Value < 0.01:
            ax1.text(i, max(row.Lottery_Mean, row.Playoff_Mean) * 1.05, '**', ha='center')
        elif row.P_Value < 0.05:
            ax1.text(i, max(row.Lottery_Mean, row.Playoff_Mean) * 1.05, '*', ha='center')
    
    # 2. Team-level metrics comparison
    ax2 = fig.add_subplot(2, 3, 2)
    
    team_comp = comparison_df[comparison_df['Level'] == 'Team']
    
    colors = ['green' if d > 0 else 'red' for d in team_comp['Pct_Diff']]
    bars = ax2.barh(team_comp['Metric'], team_comp['Pct_Diff'], color=colors)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('% Difference (Playoff vs Lottery)')
    ax2.set_title('Team-Level Metric Differences', fontsize=11, fontweight='bold')
    
    for bar, pct, p in zip(bars, team_comp['Pct_Diff'], team_comp['P_Value']):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax2.text(pct + (2 if pct >= 0 else -2), bar.get_y() + bar.get_height()/2,
                f'{pct:+.1f}%{sig}', va='center', fontsize=9)
    
    # 3. Degree distribution by playoff status
    ax3 = fig.add_subplot(2, 3, 3)
    
    ax3.hist(lottery_players['Degree_Per_Game'], bins=40, alpha=0.5, label='Lottery Teams', color='gray')
    ax3.hist(playoff_players['Degree_Per_Game'], bins=40, alpha=0.5, label='Playoff Teams', color='green')
    ax3.axvline(x=lottery_players['Degree_Per_Game'].mean(), color='gray', linestyle='--', lw=2)
    ax3.axvline(x=playoff_players['Degree_Per_Game'].mean(), color='darkgreen', linestyle='--', lw=2)
    ax3.set_xlabel('Degree Per Game')
    ax3.set_ylabel('Count')
    ax3.set_title('Ball Involvement: Playoff vs Lottery', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # 4. Success levels boxplot
    ax4 = fig.add_subplot(2, 3, 4)
    
    team_df_levels = compare_by_playoff_success(team_df)
    order = ['Lottery', 'Playoff', 'Deep Run', 'Champion']
    
    sns.boxplot(data=team_df_levels, x='Success_Level', y='Avg_Degree_PG', order=order, ax=ax4, palette='RdYlGn')
    ax4.set_xlabel('Playoff Success Level')
    ax4.set_ylabel('Avg Degree Per Game')
    ax4.set_title('Ball Involvement by Success Level', fontsize=11, fontweight='bold')
    
    # 5. Top Concentration by success
    ax5 = fig.add_subplot(2, 3, 5)
    
    sns.boxplot(data=team_df_levels, x='Success_Level', y='Top_Concentration', order=order, ax=ax5, palette='RdYlGn')
    ax5.set_xlabel('Playoff Success Level')
    ax5.set_ylabel('Top Player Concentration (%)')
    ax5.set_title('Star Concentration by Success Level', fontsize=11, fontweight='bold')
    
    # 6. Std Degree (hierarchy) by success
    ax6 = fig.add_subplot(2, 3, 6)
    
    sns.boxplot(data=team_df_levels, x='Success_Level', y='Std_Degree', order=order, ax=ax6, palette='RdYlGn')
    ax6.set_xlabel('Playoff Success Level')
    ax6.set_ylabel('Std of Weighted Degree')
    ax6.set_title('Team Hierarchy by Success Level', fontsize=11, fontweight='bold')
    
    plt.suptitle('PLAYOFF TEAMS vs LOTTERY TEAMS - NETWORK METRICS COMPARISON',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_playoff_vs_lottery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_playoff_vs_lottery.png")


def plot_champions_analysis(team_df, player_df):
    """Analyze champion teams network characteristics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    team_df_levels = compare_by_playoff_success(team_df)
    
    # 1. Champions vs others: key metrics
    ax1 = axes[0, 0]
    
    champ_teams = team_df[team_df['Is_Champion']]
    other_teams = team_df[~team_df['Is_Champion']]
    
    metrics = ['Avg_Degree_PG', 'Max_Degree_PG', 'Top_Concentration', 'Std_Degree']
    labels = ['Avg Degree/G', 'Star Degree/G', 'Star Conc %', 'Std Degree']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    champ_vals = [champ_teams[m].mean() for m in metrics]
    other_vals = [other_teams[m].mean() for m in metrics]
    
    # Normalize for comparison
    champ_norm = [c / o * 100 if o != 0 else 100 for c, o in zip(champ_vals, other_vals)]
    
    ax1.bar(x, champ_norm, color='gold', edgecolor='black')
    ax1.axhline(y=100, color='red', linestyle='--', label='League Average')
    ax1.set_ylabel('% of League Average')
    ax1.set_title('Champions vs League Average', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    for i, val in enumerate(champ_norm):
        ax1.text(i, val + 2, f'{val:.0f}%', ha='center', fontweight='bold')
    
    # 2. Scatter: Star Degree vs Team Success
    ax2 = axes[0, 1]
    
    colors = team_df_levels['Success_Level'].map({
        'Lottery': 'gray', 'Playoff': 'blue', 'Deep Run': 'orange', 'Champion': 'gold'
    })
    
    ax2.scatter(team_df_levels['Max_Degree_PG'], team_df_levels['Avg_Degree_PG'],
                c=colors, s=50, alpha=0.6)
    
    # Highlight champions
    champs = team_df_levels[team_df_levels['Is_Champion']]
    ax2.scatter(champs['Max_Degree_PG'], champs['Avg_Degree_PG'],
                c='gold', s=150, edgecolors='black', linewidths=2, marker='*', label='Champions')
    
    ax2.set_xlabel('Star Degree Per Game')
    ax2.set_ylabel('Team Avg Degree Per Game')
    ax2.set_title('Star vs Team Ball Movement', fontweight='bold')
    ax2.legend()
    
    # 3. Champions profile over years
    ax3 = axes[1, 0]
    
    champ_data = []
    for season in sorted(CHAMPIONS.keys()):
        team = CHAMPIONS[season]
        team_season = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
        if len(team_season) > 0:
            champ_data.append({
                'Season': season,
                'Team': team,
                'Max_Degree_PG': team_season['Max_Degree_PG'].values[0],
                'Avg_Degree_PG': team_season['Avg_Degree_PG'].values[0],
                'Top_Concentration': team_season['Top_Concentration'].values[0]
            })
    
    champ_df = pd.DataFrame(champ_data)
    
    ax3.bar(champ_df['Season'], champ_df['Max_Degree_PG'], color='gold', edgecolor='black')
    ax3.set_ylabel('Star Degree Per Game')
    ax3.set_title('Champion Star Ball Involvement', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add team labels
    for i, row in champ_df.iterrows():
        ax3.text(i, row['Max_Degree_PG'] + 2, row['Team'], ha='center', fontsize=9)
    
    # 4. Champion concentration
    ax4 = axes[1, 1]
    
    ax4.bar(champ_df['Season'], champ_df['Top_Concentration'], color='orange', edgecolor='black')
    ax4.axhline(y=team_df['Top_Concentration'].mean(), color='red', linestyle='--', 
                label=f"League Avg: {team_df['Top_Concentration'].mean():.1f}%")
    ax4.set_ylabel('Top Player Concentration (%)')
    ax4.set_title('Champion Star Concentration', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    
    plt.suptitle('CHAMPIONS NETWORK PROFILE',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_champions_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_champions_network.png")


def print_analysis(comparison_df, team_df, player_df):
    """Print detailed analysis."""
    
    print("\n" + "="*80)
    print("PLAYOFF TEAMS vs LOTTERY TEAMS - NETWORK COMPARISON")
    print("="*80)
    
    playoff_teams = team_df[team_df['Made_Playoffs']]
    lottery_teams = team_df[~team_df['Made_Playoffs']]
    
    print(f"\nData Summary:")
    print(f"  Playoff Teams: {len(playoff_teams)} team-seasons")
    print(f"  Lottery Teams: {len(lottery_teams)} team-seasons")
    
    print("\n" + "-"*80)
    print("SIGNIFICANT DIFFERENCES (p < 0.05)")
    print("-"*80)
    
    sig = comparison_df[comparison_df['P_Value'] < 0.05].sort_values('Cohens_d', ascending=False)
    
    for _, row in sig.iterrows():
        direction = "HIGHER" if row['Diff'] > 0 else "LOWER"
        stars = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*'
        print(f"  {row['Metric']:<25} Playoff teams are {row['Pct_Diff']:+.1f}% {direction} {stars}")
        print(f"    (Playoff: {row['Playoff_Mean']:.2f}, Lottery: {row['Lottery_Mean']:.2f}, d={row['Cohens_d']:.2f})")
    
    print("\n" + "-"*80)
    print("CHAMPIONS PROFILE")
    print("-"*80)
    
    champ_teams = team_df[team_df['Is_Champion']]
    other_teams = team_df[~team_df['Is_Champion']]
    
    metrics = ['Avg_Degree_PG', 'Max_Degree_PG', 'Top_Concentration', 'Std_Degree', 'Avg_Eigen']
    
    print(f"{'Metric':<25} {'Champions':>12} {'Others':>12} {'Diff':>10}")
    print("-"*60)
    for metric in metrics:
        champ_mean = champ_teams[metric].mean()
        other_mean = other_teams[metric].mean()
        pct_diff = (champ_mean - other_mean) / other_mean * 100 if other_mean != 0 else 0
        print(f"{metric:<25} {champ_mean:>12.2f} {other_mean:>12.2f} {pct_diff:>+9.1f}%")
    
    print("\n" + "-"*80)
    print("CHAMPIONS BY YEAR")
    print("-"*80)
    
    for season in sorted(CHAMPIONS.keys()):
        team = CHAMPIONS[season]
        team_data = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
        if len(team_data) > 0:
            row = team_data.iloc[0]
            print(f"  {season} {team}: Star={row['Max_Degree_PG']:.0f}/g, Avg={row['Avg_Degree_PG']:.0f}/g, Conc={row['Top_Concentration']:.1f}%")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Key insights
    insights = []
    
    # Check if playoff teams have more ball movement
    degree_row = comparison_df[comparison_df['Metric'] == 'Degree_Per_Game']
    if len(degree_row) > 0 and degree_row.iloc[0]['P_Value'] < 0.05:
        direction = "MORE" if degree_row.iloc[0]['Diff'] > 0 else "LESS"
        insights.append(f"1. Playoff teams have {direction} ball involvement per game ({degree_row.iloc[0]['Pct_Diff']:+.1f}%)")
    
    # Check star concentration
    conc_mean_play = playoff_teams['Top_Concentration'].mean()
    conc_mean_lot = lottery_teams['Top_Concentration'].mean()
    if conc_mean_play > conc_mean_lot:
        insights.append(f"2. Playoff teams have HIGHER star concentration ({conc_mean_play:.1f}% vs {conc_mean_lot:.1f}%)")
    else:
        insights.append(f"2. Playoff teams have LOWER star concentration ({conc_mean_play:.1f}% vs {conc_mean_lot:.1f}%)")
    
    # Check hierarchy
    std_mean_play = playoff_teams['Std_Degree'].mean()
    std_mean_lot = lottery_teams['Std_Degree'].mean()
    if std_mean_play > std_mean_lot:
        insights.append(f"3. Playoff teams have MORE hierarchical networks (higher variance)")
    else:
        insights.append(f"3. Playoff teams have MORE egalitarian networks (lower variance)")
    
    for insight in insights:
        print(f"  {insight}")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("PLAYOFF TEAMS vs LOTTERY TEAMS NETWORK ANALYSIS")
    print("="*60)
    
    print("\n[LOADING DATA]")
    player_df = load_and_enrich_data()
    
    print("\n[AGGREGATING TEAM METRICS]")
    team_df = aggregate_team_metrics(player_df)
    print(f"Created {len(team_df)} team-seasons")
    
    print("\n[COMPARING PLAYOFF vs LOTTERY]")
    comparison_df = compare_playoff_vs_lottery(player_df, team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_comparison(comparison_df, player_df, team_df)
    plot_champions_analysis(team_df, player_df)
    
    print_analysis(comparison_df, team_df, player_df)
    
    comparison_df.to_csv(OUTPUT_DIR / 'playoff_vs_lottery_comparison.csv', index=False)
    team_df.to_csv(OUTPUT_DIR / 'team_metrics_with_playoff.csv', index=False)
    
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
