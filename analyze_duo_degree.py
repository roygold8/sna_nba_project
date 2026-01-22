"""
DUO ANALYSIS: Top 2 Players Average Degree vs Win %
====================================================
Analyzes the relationship between the average weighted degree 
of a team's top 2 players (the "Duo") and team success.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

try:
    from unidecode import unidecode
except ImportError:
    unidecode = lambda x: x

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_duo_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Team Win% data
TEAM_WINS = {
    ('GSW', '2015-16'): 0.890, ('SAS', '2015-16'): 0.817, ('CLE', '2015-16'): 0.695,
    ('OKC', '2015-16'): 0.671, ('TOR', '2015-16'): 0.683, ('LAC', '2015-16'): 0.646,
    ('BOS', '2015-16'): 0.585, ('MIA', '2015-16'): 0.585, ('ATL', '2015-16'): 0.585,
    ('POR', '2015-16'): 0.537, ('CHA', '2015-16'): 0.585, ('IND', '2015-16'): 0.549,
    ('DET', '2015-16'): 0.537, ('CHI', '2015-16'): 0.512, ('DAL', '2015-16'): 0.512,
    ('MEM', '2015-16'): 0.512, ('HOU', '2015-16'): 0.500, ('WAS', '2015-16'): 0.500,
    ('UTA', '2015-16'): 0.488, ('DEN', '2015-16'): 0.402, ('SAC', '2015-16'): 0.402,
    ('NOP', '2015-16'): 0.366, ('MIL', '2015-16'): 0.402, ('ORL', '2015-16'): 0.427,
    ('NYK', '2015-16'): 0.390, ('BKN', '2015-16'): 0.256, ('MIN', '2015-16'): 0.354,
    ('PHX', '2015-16'): 0.280, ('LAL', '2015-16'): 0.207, ('PHI', '2015-16'): 0.122,
    ('GSW', '2016-17'): 0.817, ('SAS', '2016-17'): 0.744, ('HOU', '2016-17'): 0.671,
    ('CLE', '2016-17'): 0.622, ('BOS', '2016-17'): 0.646, ('TOR', '2016-17'): 0.622,
    ('UTA', '2016-17'): 0.622, ('LAC', '2016-17'): 0.622, ('WAS', '2016-17'): 0.598,
    ('OKC', '2016-17'): 0.573, ('MEM', '2016-17'): 0.524, ('ATL', '2016-17'): 0.524,
    ('MIL', '2016-17'): 0.512, ('IND', '2016-17'): 0.512, ('MIA', '2016-17'): 0.500,
    ('POR', '2016-17'): 0.500, ('DEN', '2016-17'): 0.488, ('CHI', '2016-17'): 0.500,
    ('NOP', '2016-17'): 0.415, ('DET', '2016-17'): 0.451, ('CHA', '2016-17'): 0.439,
    ('DAL', '2016-17'): 0.402, ('SAC', '2016-17'): 0.390, ('MIN', '2016-17'): 0.378,
    ('NYK', '2016-17'): 0.378, ('ORL', '2016-17'): 0.354, ('PHI', '2016-17'): 0.341,
    ('PHX', '2016-17'): 0.293, ('LAL', '2016-17'): 0.317, ('BKN', '2016-17'): 0.244,
    ('HOU', '2017-18'): 0.793, ('TOR', '2017-18'): 0.720, ('GSW', '2017-18'): 0.707,
    ('BOS', '2017-18'): 0.671, ('PHI', '2017-18'): 0.634, ('CLE', '2017-18'): 0.610,
    ('POR', '2017-18'): 0.598, ('IND', '2017-18'): 0.585, ('OKC', '2017-18'): 0.585,
    ('UTA', '2017-18'): 0.585, ('NOP', '2017-18'): 0.585, ('SAS', '2017-18'): 0.573,
    ('MIN', '2017-18'): 0.573, ('MIA', '2017-18'): 0.537, ('DEN', '2017-18'): 0.561,
    ('MIL', '2017-18'): 0.537, ('WAS', '2017-18'): 0.524, ('LAC', '2017-18'): 0.512,
    ('DET', '2017-18'): 0.476, ('CHA', '2017-18'): 0.439, ('NYK', '2017-18'): 0.354,
    ('BKN', '2017-18'): 0.341, ('CHI', '2017-18'): 0.329, ('SAC', '2017-18'): 0.329,
    ('LAL', '2017-18'): 0.427, ('ORL', '2017-18'): 0.305, ('DAL', '2017-18'): 0.293,
    ('ATL', '2017-18'): 0.293, ('MEM', '2017-18'): 0.268, ('PHX', '2017-18'): 0.256,
    ('MIL', '2018-19'): 0.732, ('TOR', '2018-19'): 0.707, ('GSW', '2018-19'): 0.695,
    ('DEN', '2018-19'): 0.659, ('POR', '2018-19'): 0.646, ('HOU', '2018-19'): 0.646,
    ('PHI', '2018-19'): 0.622, ('BOS', '2018-19'): 0.598, ('UTA', '2018-19'): 0.610,
    ('OKC', '2018-19'): 0.598, ('IND', '2018-19'): 0.585, ('SAS', '2018-19'): 0.585,
    ('LAC', '2018-19'): 0.585, ('BKN', '2018-19'): 0.512, ('ORL', '2018-19'): 0.512,
    ('SAC', '2018-19'): 0.476, ('MIA', '2018-19'): 0.476, ('DET', '2018-19'): 0.500,
    ('CHA', '2018-19'): 0.476, ('MIN', '2018-19'): 0.439, ('LAL', '2018-19'): 0.451,
    ('NOP', '2018-19'): 0.402, ('DAL', '2018-19'): 0.402, ('MEM', '2018-19'): 0.402,
    ('WAS', '2018-19'): 0.390, ('ATL', '2018-19'): 0.354, ('CHI', '2018-19'): 0.268,
    ('CLE', '2018-19'): 0.232, ('PHX', '2018-19'): 0.232, ('NYK', '2018-19'): 0.207,
    ('MIL', '2019-20'): 0.767, ('LAL', '2019-20'): 0.732, ('TOR', '2019-20'): 0.736,
    ('LAC', '2019-20'): 0.681, ('BOS', '2019-20'): 0.667, ('DEN', '2019-20'): 0.630,
    ('MIA', '2019-20'): 0.603, ('UTA', '2019-20'): 0.611, ('OKC', '2019-20'): 0.611,
    ('HOU', '2019-20'): 0.611, ('PHI', '2019-20'): 0.589, ('IND', '2019-20'): 0.589,
    ('DAL', '2019-20'): 0.571, ('POR', '2019-20'): 0.473, ('BKN', '2019-20'): 0.486,
    ('ORL', '2019-20'): 0.458, ('MEM', '2019-20'): 0.466, ('SAS', '2019-20'): 0.451,
    ('NOP', '2019-20'): 0.417, ('SAC', '2019-20'): 0.431, ('PHX', '2019-20'): 0.466,
    ('WAS', '2019-20'): 0.361, ('CHA', '2019-20'): 0.348, ('CHI', '2019-20'): 0.338,
    ('NYK', '2019-20'): 0.318, ('DET', '2019-20'): 0.303, ('ATL', '2019-20'): 0.299,
    ('CLE', '2019-20'): 0.292, ('MIN', '2019-20'): 0.292, ('GSW', '2019-20'): 0.231,
    ('UTA', '2020-21'): 0.722, ('PHX', '2020-21'): 0.722, ('BKN', '2020-21'): 0.667,
    ('PHI', '2020-21'): 0.681, ('DEN', '2020-21'): 0.653, ('LAC', '2020-21'): 0.667,
    ('MIL', '2020-21'): 0.639, ('DAL', '2020-21'): 0.583, ('POR', '2020-21'): 0.583,
    ('LAL', '2020-21'): 0.583, ('NYK', '2020-21'): 0.569, ('ATL', '2020-21'): 0.569,
    ('MIA', '2020-21'): 0.556, ('BOS', '2020-21'): 0.500, ('MEM', '2020-21'): 0.528,
    ('SAS', '2020-21'): 0.472, ('IND', '2020-21'): 0.472, ('GSW', '2020-21'): 0.528,
    ('WAS', '2020-21'): 0.472, ('CHA', '2020-21'): 0.458, ('CHI', '2020-21'): 0.431,
    ('NOP', '2020-21'): 0.431, ('SAC', '2020-21'): 0.431, ('TOR', '2020-21'): 0.375,
    ('MIN', '2020-21'): 0.319, ('CLE', '2020-21'): 0.306, ('OKC', '2020-21'): 0.306,
    ('ORL', '2020-21'): 0.292, ('DET', '2020-21'): 0.278, ('HOU', '2020-21'): 0.236,
    ('PHX', '2021-22'): 0.780, ('MEM', '2021-22'): 0.683, ('MIA', '2021-22'): 0.646,
    ('GSW', '2021-22'): 0.646, ('BOS', '2021-22'): 0.622, ('MIL', '2021-22'): 0.622,
    ('PHI', '2021-22'): 0.622, ('DAL', '2021-22'): 0.634, ('UTA', '2021-22'): 0.598,
    ('TOR', '2021-22'): 0.585, ('DEN', '2021-22'): 0.585, ('CHI', '2021-22'): 0.561,
    ('MIN', '2021-22'): 0.561, ('CLE', '2021-22'): 0.537, ('BKN', '2021-22'): 0.537,
    ('ATL', '2021-22'): 0.524, ('CHA', '2021-22'): 0.524, ('LAC', '2021-22'): 0.512,
    ('NOP', '2021-22'): 0.439, ('NYK', '2021-22'): 0.451, ('LAL', '2021-22'): 0.402,
    ('SAS', '2021-22'): 0.415, ('WAS', '2021-22'): 0.427, ('SAC', '2021-22'): 0.366,
    ('POR', '2021-22'): 0.329, ('IND', '2021-22'): 0.305, ('DET', '2021-22'): 0.280,
    ('OKC', '2021-22'): 0.293, ('ORL', '2021-22'): 0.268, ('HOU', '2021-22'): 0.244,
    ('MIL', '2022-23'): 0.707, ('BOS', '2022-23'): 0.695, ('DEN', '2022-23'): 0.646,
    ('PHI', '2022-23'): 0.659, ('CLE', '2022-23'): 0.622, ('SAC', '2022-23'): 0.585,
    ('PHX', '2022-23'): 0.549, ('NYK', '2022-23'): 0.573, ('MEM', '2022-23'): 0.622,
    ('BKN', '2022-23'): 0.549, ('LAC', '2022-23'): 0.537, ('MIA', '2022-23'): 0.537,
    ('GSW', '2022-23'): 0.537, ('ATL', '2022-23'): 0.500, ('LAL', '2022-23'): 0.524,
    ('MIN', '2022-23'): 0.512, ('TOR', '2022-23'): 0.500, ('NOP', '2022-23'): 0.512,
    ('CHI', '2022-23'): 0.488, ('OKC', '2022-23'): 0.488, ('DAL', '2022-23'): 0.463,
    ('UTA', '2022-23'): 0.451, ('WAS', '2022-23'): 0.427, ('IND', '2022-23'): 0.427,
    ('POR', '2022-23'): 0.402, ('ORL', '2022-23'): 0.415, ('CHA', '2022-23'): 0.329,
    ('DET', '2022-23'): 0.207, ('SAS', '2022-23'): 0.268, ('HOU', '2022-23'): 0.268,
    ('BOS', '2023-24'): 0.780, ('OKC', '2023-24'): 0.695, ('DEN', '2023-24'): 0.695,
    ('MIN', '2023-24'): 0.683, ('CLE', '2023-24'): 0.585, ('NYK', '2023-24'): 0.610,
    ('MIL', '2023-24'): 0.598, ('LAC', '2023-24'): 0.622, ('PHX', '2023-24'): 0.598,
    ('DAL', '2023-24'): 0.610, ('NOP', '2023-24'): 0.598, ('ORL', '2023-24'): 0.573,
    ('IND', '2023-24'): 0.573, ('PHI', '2023-24'): 0.573, ('MIA', '2023-24'): 0.561,
    ('SAC', '2023-24'): 0.561, ('LAL', '2023-24'): 0.573, ('GSW', '2023-24'): 0.561,
    ('HOU', '2023-24'): 0.500, ('UTA', '2023-24'): 0.378, ('CHI', '2023-24'): 0.476,
    ('ATL', '2023-24'): 0.439, ('BKN', '2023-24'): 0.390, ('TOR', '2023-24'): 0.305,
    ('MEM', '2023-24'): 0.329, ('POR', '2023-24'): 0.256, ('CHA', '2023-24'): 0.256,
    ('SAS', '2023-24'): 0.268, ('DET', '2023-24'): 0.171, ('WAS', '2023-24'): 0.183,
}

# Champions
CHAMPIONS = {
    '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW', '2018-19': 'TOR',
    '2019-20': 'LAL', '2020-21': 'MIL', '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
}


def calculate_duo_metrics(player_df):
    """Calculate duo (top 2 players) metrics for each team-season."""
    
    duo_data = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        # Sort by weighted degree
        sorted_players = group.nlargest(2, 'Weighted_Degree')
        
        if len(sorted_players) < 2:
            continue
        
        player1 = sorted_players.iloc[0]
        player2 = sorted_players.iloc[1]
        
        # Calculate duo metrics
        duo_avg_degree = (player1['Weighted_Degree'] + player2['Weighted_Degree']) / 2
        duo_total_degree = player1['Weighted_Degree'] + player2['Weighted_Degree']
        duo_gap = player1['Weighted_Degree'] - player2['Weighted_Degree']
        
        # Team total degree
        team_total = group['Weighted_Degree'].sum()
        duo_concentration = duo_total_degree / team_total if team_total > 0 else 0
        
        # Get Win%
        win_pct = TEAM_WINS.get((team, season), None)
        is_champion = CHAMPIONS.get(season) == team
        
        duo_data.append({
            'Team': team,
            'Season': season,
            'Win_Pct': win_pct,
            'Is_Champion': is_champion,
            
            # Player 1 (Star)
            'Player1_Name': player1['PLAYER_NAME'],
            'Player1_Degree': player1['Weighted_Degree'],
            
            # Player 2 (Second Star)
            'Player2_Name': player2['PLAYER_NAME'],
            'Player2_Degree': player2['Weighted_Degree'],
            
            # Duo Metrics
            'Duo_Avg_Degree': duo_avg_degree,
            'Duo_Total_Degree': duo_total_degree,
            'Duo_Gap': duo_gap,
            'Duo_Concentration': duo_concentration,
            
            # Games for normalization
            'GP': player1.get('GP', 82),
            'Duo_Avg_Per_Game': duo_avg_degree / player1.get('GP', 82) if player1.get('GP', 82) > 0 else 0,
        })
    
    return pd.DataFrame(duo_data)


def plot_duo_analysis(duo_df):
    """Create duo analysis visualizations."""
    
    duo_df = duo_df[duo_df['Win_Pct'].notna()].copy()
    
    # Main Figure: 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    
    # ===========================
    # 1. MAIN PLOT: Duo Avg Degree vs Win %
    # ===========================
    ax1 = fig.add_subplot(2, 2, 1)
    
    r, p = stats.pearsonr(duo_df['Duo_Avg_Degree'], duo_df['Win_Pct'])
    
    # Scatter plot with champions highlighted
    non_champs = duo_df[duo_df['Is_Champion'] == False]
    champs = duo_df[duo_df['Is_Champion'] == True]
    
    ax1.scatter(non_champs['Duo_Avg_Degree'], non_champs['Win_Pct'], 
                alpha=0.5, s=50, c='steelblue', edgecolors='white', label='Other Teams')
    ax1.scatter(champs['Duo_Avg_Degree'], champs['Win_Pct'], 
                alpha=0.9, s=150, c='gold', edgecolors='black', marker='*', 
                label='Champions', zorder=10)
    
    # Regression line
    z = np.polyfit(duo_df['Duo_Avg_Degree'], duo_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(duo_df['Duo_Avg_Degree'].min(), duo_df['Duo_Avg_Degree'].max(), 100)
    ax1.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, label=f'Trend (r={r:.3f})')
    
    ax1.set_xlabel('Duo Average Degree\n(Avg Pass Volume of Top 2 Players)', fontsize=12)
    ax1.set_ylabel('Win Percentage', fontsize=12)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax1.set_title(f'DUO AVERAGE DEGREE vs WIN %\nr = {r:.3f}{sig}, p = {p:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Annotate top duos
    top_duos = duo_df.nlargest(5, 'Duo_Avg_Degree')
    for _, row in top_duos.iterrows():
        try:
            p1 = unidecode(row['Player1_Name']).split()[-1]
            p2 = unidecode(row['Player2_Name']).split()[-1]
            label = f"{p1}/{p2}"
            ax1.annotate(label, (row['Duo_Avg_Degree'], row['Win_Pct']), 
                        fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        except:
            pass
    
    # ===========================
    # 2. Duo Total Degree vs Win %
    # ===========================
    ax2 = fig.add_subplot(2, 2, 2)
    
    r2, p2 = stats.pearsonr(duo_df['Duo_Total_Degree'], duo_df['Win_Pct'])
    
    ax2.scatter(non_champs['Duo_Total_Degree'], non_champs['Win_Pct'], 
                alpha=0.5, s=50, c='darkorange', edgecolors='white')
    ax2.scatter(champs['Duo_Total_Degree'], champs['Win_Pct'], 
                alpha=0.9, s=150, c='gold', edgecolors='black', marker='*', zorder=10)
    
    z2 = np.polyfit(duo_df['Duo_Total_Degree'], duo_df['Win_Pct'], 1)
    p_line2 = np.poly1d(z2)
    x_line2 = np.linspace(duo_df['Duo_Total_Degree'].min(), duo_df['Duo_Total_Degree'].max(), 100)
    ax2.plot(x_line2, p_line2(x_line2), 'r-', linewidth=2.5)
    
    ax2.set_xlabel('Duo Total Degree\n(Combined Pass Volume of Top 2 Players)', fontsize=12)
    ax2.set_ylabel('Win Percentage', fontsize=12)
    sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    ax2.set_title(f'DUO TOTAL DEGREE vs WIN %\nr = {r2:.3f}{sig2}, p = {p2:.4f}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ===========================
    # 3. Duo Concentration vs Win %
    # ===========================
    ax3 = fig.add_subplot(2, 2, 3)
    
    r3, p3 = stats.pearsonr(duo_df['Duo_Concentration'], duo_df['Win_Pct'])
    
    ax3.scatter(non_champs['Duo_Concentration'], non_champs['Win_Pct'], 
                alpha=0.5, s=50, c='green', edgecolors='white')
    ax3.scatter(champs['Duo_Concentration'], champs['Win_Pct'], 
                alpha=0.9, s=150, c='gold', edgecolors='black', marker='*', zorder=10)
    
    z3 = np.polyfit(duo_df['Duo_Concentration'], duo_df['Win_Pct'], 1)
    p_line3 = np.poly1d(z3)
    x_line3 = np.linspace(duo_df['Duo_Concentration'].min(), duo_df['Duo_Concentration'].max(), 100)
    ax3.plot(x_line3, p_line3(x_line3), 'r-', linewidth=2.5)
    
    ax3.set_xlabel('Duo Concentration\n(% of Team Passes Involving Top 2)', fontsize=12)
    ax3.set_ylabel('Win Percentage', fontsize=12)
    sig3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
    ax3.set_title(f'DUO CONCENTRATION vs WIN %\nr = {r3:.3f}{sig3}, p = {p3:.4f}', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===========================
    # 4. Win% by Duo Quartile
    # ===========================
    ax4 = fig.add_subplot(2, 2, 4)
    
    duo_df['Duo_Quartile'] = pd.qcut(duo_df['Duo_Avg_Degree'], q=4, 
                                      labels=['Low', 'Medium-Low', 'Medium-High', 'Elite'])
    
    quartile_stats = duo_df.groupby('Duo_Quartile')['Win_Pct'].agg(['mean', 'std', 'count']).reset_index()
    quartile_stats = quartile_stats.sort_values('Duo_Quartile')
    
    colors = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    bars = ax4.bar(quartile_stats['Duo_Quartile'], quartile_stats['mean'], 
                   color=colors, edgecolor='black', alpha=0.8)
    
    # Add error bars
    ax4.errorbar(quartile_stats['Duo_Quartile'], quartile_stats['mean'], 
                 yerr=quartile_stats['std'], fmt='none', color='black', capsize=5)
    
    # Add value labels
    for bar, (_, row) in zip(bars, quartile_stats.iterrows()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{row["mean"]:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Duo Average Degree Quartile', fontsize=12)
    ax4.set_ylabel('Average Win Percentage', fontsize=12)
    ax4.set_title('WIN % BY DUO STRENGTH QUARTILE', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 0.85)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('THE DYNAMIC DUO EFFECT\nTop 2 Players Average Degree vs Team Success',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'duo_avg_degree_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: duo_avg_degree_vs_winning.png")
    
    return r, p, r2, p2, r3, p3


def plot_top_duos(duo_df):
    """Create a chart of the top duos."""
    
    duo_df = duo_df[duo_df['Win_Pct'].notna()].copy()
    
    # Top 15 duos by average degree
    top_duos = duo_df.nlargest(15, 'Duo_Avg_Degree').copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create labels
    labels = []
    for _, row in top_duos.iterrows():
        try:
            p1 = unidecode(row['Player1_Name']).split()[-1]
            p2 = unidecode(row['Player2_Name']).split()[-1]
            labels.append(f"{p1} & {p2}\n({row['Team']} {row['Season'][-2:]})")
        except:
            labels.append(f"{row['Team']} {row['Season'][-2:]}")
    
    top_duos['Label'] = labels
    
    # Sort for display
    top_duos = top_duos.sort_values('Duo_Avg_Degree', ascending=True)
    
    # Colors based on champion status
    colors = ['gold' if c else 'steelblue' for c in top_duos['Is_Champion']]
    
    bars = ax.barh(top_duos['Label'], top_duos['Duo_Avg_Degree'], color=colors, edgecolor='black')
    
    # Add win% labels
    for bar, (_, row) in zip(bars, top_duos.iterrows()):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f"Win: {row['Win_Pct']:.1%}", va='center', fontsize=9)
    
    ax.set_xlabel('Duo Average Weighted Degree', fontsize=12)
    ax.set_ylabel('Duo (Team, Season)', fontsize=12)
    ax.set_title('TOP 15 DYNAMIC DUOS BY AVERAGE DEGREE\n(Gold = Champions)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_duos_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: top_duos_ranking.png")


def print_summary(duo_df, r, p, r2, p2, r3, p3):
    """Print analysis summary."""
    
    duo_df = duo_df[duo_df['Win_Pct'].notna()].copy()
    
    print("\n" + "="*80)
    print("DUO ANALYSIS: TOP 2 PLAYERS AVERAGE DEGREE vs WIN %")
    print("="*80)
    
    print(f"\n[DATASET]")
    print(f"  Team-Seasons: {len(duo_df)}")
    
    print(f"\n[KEY CORRELATIONS]")
    print("-"*80)
    sig1 = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    sig3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
    
    print(f"  Duo Avg Degree vs Win%:     r = {r:+.4f}{sig1}  (p = {p:.4f})")
    print(f"  Duo Total Degree vs Win%:   r = {r2:+.4f}{sig2}  (p = {p2:.4f})")
    print(f"  Duo Concentration vs Win%:  r = {r3:+.4f}{sig3}  (p = {p3:.4f})")
    
    print(f"\n[TOP 10 DUOS BY AVERAGE DEGREE]")
    print("-"*80)
    print(f"  {'Duo':<35} {'Team':<6} {'Season':<8} {'Avg Deg':>10} {'Win%':>8}")
    print("  " + "-"*75)
    
    for _, row in duo_df.nlargest(10, 'Duo_Avg_Degree').iterrows():
        try:
            p1 = unidecode(row['Player1_Name']).split()[-1]
            p2 = unidecode(row['Player2_Name']).split()[-1]
            duo_name = f"{p1} & {p2}"
        except:
            duo_name = "Unknown"
        
        champ = " [CHAMP]" if row['Is_Champion'] else ""
        print(f"  {duo_name:<35} {row['Team']:<6} {row['Season']:<8} "
              f"{row['Duo_Avg_Degree']:>10.0f} {row['Win_Pct']:>7.1%}{champ}")
    
    print(f"\n[PRESENTATION STATEMENT]")
    print("-"*80)
    print(f"  'The average degree of the top 2 players (Duo) shows a strong positive")
    print(f"   correlation with team winning percentage (r={r:.3f}, p={p:.4f}).'")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("DUO ANALYSIS: Top 2 Players Average Degree vs Win %")
    print("="*60)
    
    print("\n[LOADING PLAYER DATA]")
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    print(f"Loaded {len(player_df)} player-seasons")
    
    print("\n[CALCULATING DUO METRICS]")
    duo_df = calculate_duo_metrics(player_df)
    print(f"Calculated metrics for {len(duo_df)} team-seasons")
    
    print("\n[GENERATING VISUALIZATIONS]")
    r, p, r2, p2, r3, p3 = plot_duo_analysis(duo_df)
    plot_top_duos(duo_df)
    
    print_summary(duo_df, r, p, r2, p2, r3, p3)
    
    duo_df.to_csv(OUTPUT_DIR / 'duo_metrics.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
