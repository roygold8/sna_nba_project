"""
Network Density vs Winning Analysis
===================================
Correct Density Formula: (Mean Degree)² / Mean(Degree²)
This measures equality of ball distribution (1 = perfectly equal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_density_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Team success data
TEAM_WINS = {
    # 2015-16
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
    
    # 2016-17
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
    
    # 2017-18
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
    
    # 2018-19
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
    
    # 2019-20
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
    
    # 2020-21
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
    
    # 2021-22
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
    
    # 2022-23
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
    
    # 2023-24
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


def calculate_correct_density(player_df):
    """
    Calculate correct network density for each team-season.
    
    Density = (Mean Degree)² / Mean(Degree²)
    
    This measures how evenly distributed the degrees are:
    - 1.0 = perfectly equal (everyone has same degree)
    - Lower = more concentrated (star-heavy)
    """
    
    team_density = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        degrees = group['Weighted_Degree'].values
        
        if len(degrees) < 2:
            continue
        
        mean_degree = np.mean(degrees)
        mean_degree_squared = np.mean(degrees ** 2)
        
        # Correct density formula
        if mean_degree_squared > 0:
            density = (mean_degree ** 2) / mean_degree_squared
        else:
            density = 0
        
        # Also calculate other metrics for context
        std_degree = np.std(degrees)
        max_degree = np.max(degrees)
        gini = calculate_gini(degrees)
        
        # Get Win%
        win_pct = TEAM_WINS.get((team, season), None)
        
        team_density.append({
            'Team': team,
            'Season': season,
            'Density': density,
            'Mean_Degree': mean_degree,
            'Std_Degree': std_degree,
            'Max_Degree': max_degree,
            'Gini': gini,
            'Win_Pct': win_pct,
            'N_Players': len(degrees)
        })
    
    return pd.DataFrame(team_density)


def calculate_gini(values):
    """Calculate Gini coefficient for inequality."""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def plot_density_analysis(team_df):
    """Create density vs winning visualizations."""
    
    # Filter out missing win data
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Main: Density vs Win%
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Calculate correlation
    r, p = stats.pearsonr(team_df['Density'], team_df['Win_Pct'])
    
    ax1.scatter(team_df['Density'], team_df['Win_Pct'], alpha=0.6, s=50, c='steelblue', edgecolors='white')
    
    # Add regression line
    z = np.polyfit(team_df['Density'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Density'].min(), team_df['Density'].max(), 100)
    ax1.plot(x_line, p_line(x_line), 'r-', linewidth=2, label=f'r = {r:.3f}, p = {p:.4f}')
    
    ax1.set_xlabel('Network Density\n(1 = equal distribution, lower = star-heavy)', fontsize=10)
    ax1.set_ylabel('Win Percentage')
    ax1.set_title(f'Network Density vs Win %\n(Density = Mean²/Mean(x²))', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0.3, 1.0)
    
    # 2. Gini vs Win% (inverse of density concept)
    ax2 = fig.add_subplot(2, 3, 2)
    
    r_gini, p_gini = stats.pearsonr(team_df['Gini'], team_df['Win_Pct'])
    
    ax2.scatter(team_df['Gini'], team_df['Win_Pct'], alpha=0.6, s=50, c='darkorange', edgecolors='white')
    
    z = np.polyfit(team_df['Gini'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Gini'].min(), team_df['Gini'].max(), 100)
    ax2.plot(x_line, p_line(x_line), 'r-', linewidth=2, label=f'r = {r_gini:.3f}, p = {p_gini:.4f}')
    
    ax2.set_xlabel('Gini Coefficient\n(0 = equal, 1 = one player dominates)', fontsize=10)
    ax2.set_ylabel('Win Percentage')
    ax2.set_title('Inequality (Gini) vs Win %', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    
    # 3. Density Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    
    ax3.hist(team_df['Density'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax3.axvline(x=team_df['Density'].mean(), color='red', linestyle='--', lw=2, 
                label=f'Mean: {team_df["Density"].mean():.3f}')
    ax3.axvline(x=team_df['Density'].median(), color='orange', linestyle='--', lw=2,
                label=f'Median: {team_df["Density"].median():.3f}')
    ax3.set_xlabel('Network Density')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Network Density', fontsize=11, fontweight='bold')
    ax3.legend()
    
    # 4. Boxplot by Win% quartile
    ax4 = fig.add_subplot(2, 3, 4)
    
    team_df['Win_Quartile'] = pd.qcut(team_df['Win_Pct'], q=4, labels=['Bottom 25%', '25-50%', '50-75%', 'Top 25%'])
    
    colors = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    bp = ax4.boxplot([team_df[team_df['Win_Quartile'] == q]['Density'] for q in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']],
                     labels=['Bottom 25%', '25-50%', '50-75%', 'Top 25%'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Win % Quartile')
    ax4.set_ylabel('Network Density')
    ax4.set_title('Network Density by Win % Quartile', fontsize=11, fontweight='bold')
    
    # 5. Density vs Std Degree scatter (showing relationship)
    ax5 = fig.add_subplot(2, 3, 5)
    
    ax5.scatter(team_df['Std_Degree'], team_df['Density'], alpha=0.6, s=50, c='purple', edgecolors='white')
    ax5.set_xlabel('Std Deviation of Degree (Hierarchy)')
    ax5.set_ylabel('Network Density')
    ax5.set_title('Density vs Hierarchy\n(Lower density = higher hierarchy)', fontsize=11, fontweight='bold')
    
    r_hier, p_hier = stats.pearsonr(team_df['Std_Degree'], team_df['Density'])
    ax5.text(0.05, 0.95, f'r = {r_hier:.3f}', transform=ax5.transAxes, fontsize=10)
    
    # 6. Density over seasons
    ax6 = fig.add_subplot(2, 3, 6)
    
    season_avg = team_df.groupby('Season').agg({
        'Density': 'mean',
        'Win_Pct': 'mean'
    }).reset_index()
    
    ax6.bar(season_avg['Season'], season_avg['Density'], color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axhline(y=team_df['Density'].mean(), color='red', linestyle='--', lw=2, label='Overall Mean')
    ax6.set_xlabel('Season')
    ax6.set_ylabel('Average Network Density')
    ax6.set_title('Network Density Over Seasons', fontsize=11, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    
    plt.suptitle('NETWORK DENSITY vs WINNING\nDensity = (Mean Degree)² / Mean(Degree²)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_density_vs_winning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_density_vs_winning.png")


def plot_density_interpretation(team_df):
    """Create interpretation plot showing what density means."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Example: High vs Low Density teams
    ax1 = axes[0]
    
    high_density = team_df.nlargest(10, 'Density')[['Team', 'Season', 'Density', 'Win_Pct']]
    low_density = team_df.nsmallest(10, 'Density')[['Team', 'Season', 'Density', 'Win_Pct']]
    
    x = np.arange(2)
    ax1.bar(x, [high_density['Win_Pct'].mean(), low_density['Win_Pct'].mean()], 
            color=['steelblue', 'darkorange'], edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['High Density\n(Equal Dist.)', 'Low Density\n(Star-Heavy)'])
    ax1.set_ylabel('Average Win %')
    ax1.set_title('Top 10 Highest vs Lowest Density Teams', fontweight='bold')
    
    for i, val in enumerate([high_density['Win_Pct'].mean(), low_density['Win_Pct'].mean()]):
        ax1.text(i, val + 0.02, f'{val:.1%}', ha='center', fontweight='bold')
    
    # 2. Density components
    ax2 = axes[1]
    
    # Show relationship between Mean and Mean² components
    ax2.scatter(team_df['Mean_Degree'], team_df['Std_Degree'], 
                c=team_df['Density'], cmap='RdYlGn', s=50, alpha=0.7)
    plt.colorbar(ax2.scatter(team_df['Mean_Degree'], team_df['Std_Degree'], 
                             c=team_df['Density'], cmap='RdYlGn', s=50, alpha=0.7), 
                 ax=ax2, label='Density')
    ax2.set_xlabel('Mean Degree')
    ax2.set_ylabel('Std Degree')
    ax2.set_title('Density = f(Mean, Std)\nGreen = High Density (Equal)', fontweight='bold')
    
    # 3. Champions density
    ax3 = axes[2]
    
    champions = {
        '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW', '2018-19': 'TOR',
        '2019-20': 'LAL', '2020-21': 'MIL', '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
    }
    
    champ_densities = []
    for season, team in champions.items():
        champ_row = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
        if len(champ_row) > 0:
            champ_densities.append({
                'Team': team,
                'Season': season,
                'Density': champ_row['Density'].values[0]
            })
    
    champ_df = pd.DataFrame(champ_densities)
    
    ax3.bar(champ_df['Season'], champ_df['Density'], color='gold', edgecolor='black')
    ax3.axhline(y=team_df['Density'].mean(), color='red', linestyle='--', lw=2, 
                label=f'League Avg: {team_df["Density"].mean():.3f}')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Network Density')
    ax3.set_title('Champions Network Density', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    # Add team labels
    for i, row in champ_df.iterrows():
        ax3.text(i, row['Density'] + 0.01, row['Team'], ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_density_interpretation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_density_interpretation.png")


def print_analysis(team_df):
    """Print detailed analysis."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    print("\n" + "="*80)
    print("NETWORK DENSITY vs WINNING ANALYSIS")
    print("="*80)
    
    print(f"\nFormula: Density = (Mean Degree)² / Mean(Degree²)")
    print(f"Interpretation:")
    print(f"  - 1.0 = Perfectly equal ball distribution")
    print(f"  - Lower = More concentrated on star players")
    
    print(f"\nData Summary:")
    print(f"  Team-Seasons: {len(team_df)}")
    print(f"  Density Range: {team_df['Density'].min():.3f} - {team_df['Density'].max():.3f}")
    print(f"  Density Mean: {team_df['Density'].mean():.3f}")
    
    # Correlation
    r, p = stats.pearsonr(team_df['Density'], team_df['Win_Pct'])
    r_gini, p_gini = stats.pearsonr(team_df['Gini'], team_df['Win_Pct'])
    
    print("\n" + "-"*80)
    print("CORRELATIONS WITH WINNING")
    print("-"*80)
    print(f"  Density vs Win%:  r = {r:+.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
    print(f"  Gini vs Win%:     r = {r_gini:+.3f}, p = {p_gini:.4f} {'***' if p_gini < 0.001 else '**' if p_gini < 0.01 else '*' if p_gini < 0.05 else ''}")
    
    # Top/Bottom teams
    print("\n" + "-"*80)
    print("HIGHEST DENSITY TEAMS (Most Equal Ball Distribution)")
    print("-"*80)
    high_density = team_df.nlargest(10, 'Density')[['Team', 'Season', 'Density', 'Win_Pct']]
    for _, row in high_density.iterrows():
        print(f"  {row['Team']} {row['Season']}: Density={row['Density']:.3f}, Win%={row['Win_Pct']:.1%}")
    
    print("\n" + "-"*80)
    print("LOWEST DENSITY TEAMS (Most Star-Heavy)")
    print("-"*80)
    low_density = team_df.nsmallest(10, 'Density')[['Team', 'Season', 'Density', 'Win_Pct']]
    for _, row in low_density.iterrows():
        print(f"  {row['Team']} {row['Season']}: Density={row['Density']:.3f}, Win%={row['Win_Pct']:.1%}")
    
    # Champions
    print("\n" + "-"*80)
    print("CHAMPIONS DENSITY")
    print("-"*80)
    
    champions = {
        '2015-16': 'CLE', '2016-17': 'GSW', '2017-18': 'GSW', '2018-19': 'TOR',
        '2019-20': 'LAL', '2020-21': 'MIL', '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS'
    }
    
    for season, team in sorted(champions.items()):
        champ_row = team_df[(team_df['Team'] == team) & (team_df['Season'] == season)]
        if len(champ_row) > 0:
            density = champ_row['Density'].values[0]
            percentile = (team_df['Density'] < density).mean() * 100
            print(f"  {season} {team}: Density={density:.3f} ({percentile:.0f}th percentile)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if r < 0:
        print(f"\n  1. LOWER density (more star-heavy) correlates with MORE winning (r={r:.3f})")
    else:
        print(f"\n  1. HIGHER density (more equal) correlates with MORE winning (r={r:.3f})")
    
    if r_gini > 0:
        print(f"  2. HIGHER inequality (Gini) correlates with MORE winning (r={r_gini:.3f})")
    else:
        print(f"  2. LOWER inequality (Gini) correlates with MORE winning (r={r_gini:.3f})")
    
    # Compare high/low density win rates
    median_density = team_df['Density'].median()
    high_win = team_df[team_df['Density'] > median_density]['Win_Pct'].mean()
    low_win = team_df[team_df['Density'] <= median_density]['Win_Pct'].mean()
    
    print(f"\n  3. High Density teams (equal): {high_win:.1%} avg Win%")
    print(f"     Low Density teams (star-heavy): {low_win:.1%} avg Win%")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*60)
    print("NETWORK DENSITY vs WINNING ANALYSIS")
    print("="*60)
    
    print("\n[LOADING DATA]")
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    print(f"Loaded {len(player_df)} player-seasons")
    
    print("\n[CALCULATING CORRECT DENSITY]")
    print("Formula: Density = (Mean Degree)² / Mean(Degree²)")
    team_df = calculate_correct_density(player_df)
    print(f"Calculated density for {len(team_df)} team-seasons")
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_density_analysis(team_df)
    plot_density_interpretation(team_df)
    
    print_analysis(team_df)
    
    team_df.to_csv(OUTPUT_DIR / 'team_density_metrics.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
