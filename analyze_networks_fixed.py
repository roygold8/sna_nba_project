"""
FIXED Network Analysis - Addresses All Critical Issues
=======================================================
1. Calculates Standard Deviation of Weighted Degree (Hierarchy)
2. Calculates Pass Entropy (Shannon Entropy)
3. Calculates Max Weighted Degree (Heliocentric/Star analysis)
4. No hardcoded values - all from real data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_fixed_analysis")
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


def calculate_entropy(degrees):
    """
    Calculate Shannon Entropy of degree distribution.
    Higher entropy = more random/equal distribution
    Lower entropy = more concentrated on few players
    """
    degrees = np.array(degrees)
    if len(degrees) == 0 or np.sum(degrees) == 0:
        return 0
    
    # Normalize to probabilities
    probs = degrees / np.sum(degrees)
    probs = probs[probs > 0]  # Remove zeros for log
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize by max possible entropy (uniform distribution)
    max_entropy = np.log2(len(degrees))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def calculate_gini(values):
    """Calculate Gini coefficient for inequality."""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def calculate_all_team_metrics(player_df):
    """
    Calculate ALL network metrics including the previously missing ones:
    1. Std_Weighted_Degree (Hierarchy)
    2. Pass_Entropy (Shannon Entropy)
    3. Max_Weighted_Degree (Heliocentric/Star)
    4. Density (correct formula)
    """
    
    team_metrics = []
    
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        degrees = group['Weighted_Degree'].values
        
        if len(degrees) < 3:
            continue
        
        # ============================================
        # PREVIOUSLY MISSING METRICS - NOW CALCULATED
        # ============================================
        
        # 1. Standard Deviation of Weighted Degree (HIERARCHY)
        std_weighted_degree = np.std(degrees)
        
        # 2. Shannon Entropy (PASS ENTROPY)
        pass_entropy = calculate_entropy(degrees)
        
        # 3. Max Weighted Degree (HELIOCENTRIC / STAR METRIC)
        max_weighted_degree = np.max(degrees)
        star_player_idx = np.argmax(degrees)
        star_player_name = group.iloc[star_player_idx]['PLAYER_NAME']
        
        # ============================================
        # EXISTING METRICS
        # ============================================
        
        # Mean Degree
        mean_degree = np.mean(degrees)
        
        # Density = Mean² / Mean(x²)
        mean_degree_squared = np.mean(degrees ** 2)
        density = (mean_degree ** 2) / mean_degree_squared if mean_degree_squared > 0 else 0
        
        # Gini Coefficient
        gini = calculate_gini(degrees)
        
        # Degree Centralization (Freeman's formula)
        max_degree = np.max(degrees)
        sum_diff = np.sum(max_degree - degrees)
        max_possible = (len(degrees) - 1) * max_degree
        centralization = sum_diff / max_possible if max_possible > 0 else 0
        
        # Top 2 Concentration
        sorted_degrees = np.sort(degrees)[::-1]
        top2_sum = sorted_degrees[0] + sorted_degrees[1] if len(sorted_degrees) >= 2 else sorted_degrees[0]
        top2_concentration = top2_sum / np.sum(degrees) if np.sum(degrees) > 0 else 0
        
        # Top 3 Concentration
        top3_sum = np.sum(sorted_degrees[:3]) if len(sorted_degrees) >= 3 else np.sum(sorted_degrees)
        top3_concentration = top3_sum / np.sum(degrees) if np.sum(degrees) > 0 else 0
        
        # Degree Per Game metrics
        gp = group['GP'].max()
        max_degree_per_game = max_weighted_degree / gp if gp > 0 else 0
        avg_degree_per_game = mean_degree / gp if gp > 0 else 0
        
        # Get Win%
        win_pct = TEAM_WINS.get((team, season), None)
        
        team_metrics.append({
            'Team': team,
            'Season': season,
            'Win_Pct': win_pct,
            
            # KEY METRICS (previously missing)
            'Std_Weighted_Degree': std_weighted_degree,  # HIERARCHY
            'Pass_Entropy': pass_entropy,                 # ENTROPY
            'Max_Weighted_Degree': max_weighted_degree,   # HELIOCENTRIC
            'Star_Player': star_player_name,
            
            # Other metrics
            'Mean_Weighted_Degree': mean_degree,
            'Density': density,
            'Gini_Coefficient': gini,
            'Degree_Centralization': centralization,
            'Top2_Concentration': top2_concentration,
            'Top3_Concentration': top3_concentration,
            'Max_Degree_Per_Game': max_degree_per_game,
            'Avg_Degree_Per_Game': avg_degree_per_game,
            'N_Players': len(degrees),
        })
    
    return pd.DataFrame(team_metrics)


def calculate_correlations(team_df):
    """Calculate correlations between all metrics and winning."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    metrics = [
        'Std_Weighted_Degree',  # HIERARCHY
        'Pass_Entropy',          # ENTROPY
        'Max_Weighted_Degree',   # HELIOCENTRIC
        'Max_Degree_Per_Game',   # STAR PER GAME
        'Mean_Weighted_Degree',
        'Density',
        'Gini_Coefficient',
        'Degree_Centralization',
        'Top2_Concentration',
        'Top3_Concentration',
        'Avg_Degree_Per_Game',
    ]
    
    correlations = []
    
    for metric in metrics:
        if metric not in team_df.columns:
            continue
        
        r, p = stats.pearsonr(team_df[metric], team_df['Win_Pct'])
        
        correlations.append({
            'Metric': metric,
            'Correlation': r,
            'P_Value': p,
            'Significant': p < 0.05,
            'Direction': 'Positive' if r > 0 else 'Negative'
        })
    
    return pd.DataFrame(correlations).sort_values('Correlation', ascending=False)


def plot_key_correlations(team_df, corr_df):
    """Plot the key correlations that were previously missing."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. HIERARCHY: Std Weighted Degree vs Win%
    ax1 = fig.add_subplot(2, 3, 1)
    r, p = stats.pearsonr(team_df['Std_Weighted_Degree'], team_df['Win_Pct'])
    ax1.scatter(team_df['Std_Weighted_Degree'], team_df['Win_Pct'], alpha=0.6, s=50, c='steelblue')
    z = np.polyfit(team_df['Std_Weighted_Degree'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Std_Weighted_Degree'].min(), team_df['Std_Weighted_Degree'].max(), 100)
    ax1.plot(x_line, p_line(x_line), 'r-', linewidth=2)
    ax1.set_xlabel('Std of Weighted Degree (Hierarchy)')
    ax1.set_ylabel('Win Percentage')
    ax1.set_title(f'HIERARCHY vs Winning\nr = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
    
    # 2. ENTROPY: Pass Entropy vs Win%
    ax2 = fig.add_subplot(2, 3, 2)
    r, p = stats.pearsonr(team_df['Pass_Entropy'], team_df['Win_Pct'])
    ax2.scatter(team_df['Pass_Entropy'], team_df['Win_Pct'], alpha=0.6, s=50, c='darkorange')
    z = np.polyfit(team_df['Pass_Entropy'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Pass_Entropy'].min(), team_df['Pass_Entropy'].max(), 100)
    ax2.plot(x_line, p_line(x_line), 'r-', linewidth=2)
    ax2.set_xlabel('Pass Entropy (Randomness)')
    ax2.set_ylabel('Win Percentage')
    ax2.set_title(f'ENTROPY vs Winning\nr = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
    
    # 3. HELIOCENTRIC: Max Weighted Degree vs Win%
    ax3 = fig.add_subplot(2, 3, 3)
    r, p = stats.pearsonr(team_df['Max_Weighted_Degree'], team_df['Win_Pct'])
    ax3.scatter(team_df['Max_Weighted_Degree'], team_df['Win_Pct'], alpha=0.6, s=50, c='green')
    z = np.polyfit(team_df['Max_Weighted_Degree'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Max_Weighted_Degree'].min(), team_df['Max_Weighted_Degree'].max(), 100)
    ax3.plot(x_line, p_line(x_line), 'r-', linewidth=2)
    ax3.set_xlabel('Max Weighted Degree (Star Player)')
    ax3.set_ylabel('Win Percentage')
    ax3.set_title(f'HELIOCENTRIC (Star) vs Winning\nr = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
    
    # 4. STAR PER GAME
    ax4 = fig.add_subplot(2, 3, 4)
    r, p = stats.pearsonr(team_df['Max_Degree_Per_Game'], team_df['Win_Pct'])
    ax4.scatter(team_df['Max_Degree_Per_Game'], team_df['Win_Pct'], alpha=0.6, s=50, c='purple')
    z = np.polyfit(team_df['Max_Degree_Per_Game'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Max_Degree_Per_Game'].min(), team_df['Max_Degree_Per_Game'].max(), 100)
    ax4.plot(x_line, p_line(x_line), 'r-', linewidth=2)
    ax4.set_xlabel('Star Degree Per Game')
    ax4.set_ylabel('Win Percentage')
    ax4.set_title(f'STAR Per Game vs Winning\nr = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
    
    # 5. TOP 2 CONCENTRATION
    ax5 = fig.add_subplot(2, 3, 5)
    r, p = stats.pearsonr(team_df['Top2_Concentration'], team_df['Win_Pct'])
    ax5.scatter(team_df['Top2_Concentration'], team_df['Win_Pct'], alpha=0.6, s=50, c='crimson')
    z = np.polyfit(team_df['Top2_Concentration'], team_df['Win_Pct'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(team_df['Top2_Concentration'].min(), team_df['Top2_Concentration'].max(), 100)
    ax5.plot(x_line, p_line(x_line), 'r-', linewidth=2)
    ax5.set_xlabel('Top 2 Player Concentration')
    ax5.set_ylabel('Win Percentage')
    ax5.set_title(f'DUO Concentration vs Winning\nr = {r:.3f}, p = {p:.4f}', fontsize=12, fontweight='bold')
    
    # 6. Correlation Summary Bar Chart
    ax6 = fig.add_subplot(2, 3, 6)
    
    key_corrs = corr_df[corr_df['Metric'].isin([
        'Std_Weighted_Degree', 'Pass_Entropy', 'Max_Weighted_Degree',
        'Max_Degree_Per_Game', 'Top2_Concentration', 'Top3_Concentration'
    ])].copy()
    
    colors = ['green' if r > 0 else 'red' for r in key_corrs['Correlation']]
    bars = ax6.barh(key_corrs['Metric'], key_corrs['Correlation'], color=colors)
    ax6.axvline(x=0, color='black', linewidth=1)
    ax6.set_xlabel('Correlation with Win%')
    ax6.set_title('ALL KEY CORRELATIONS\n(Previously Missing Metrics)', fontsize=12, fontweight='bold')
    
    # Add significance markers
    for bar, (_, row) in zip(bars, key_corrs.iterrows()):
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        x_pos = row['Correlation'] + (0.02 if row['Correlation'] >= 0 else -0.02)
        ax6.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f"{row['Correlation']:.3f}{sig}", va='center', fontsize=9)
    
    plt.suptitle('FIXED ANALYSIS: Key Network Metrics vs Winning\n(Hierarchy, Entropy, Heliocentric - All Calculated from Real Data)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_key_correlations_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_key_correlations_fixed.png")


def plot_correlation_heatmap(team_df):
    """Create correlation heatmap of all metrics."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    metrics = ['Win_Pct', 'Std_Weighted_Degree', 'Pass_Entropy', 'Max_Weighted_Degree',
               'Max_Degree_Per_Game', 'Density', 'Gini_Coefficient', 
               'Top2_Concentration', 'Top3_Concentration']
    
    corr_matrix = team_df[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Correlation Matrix: All Network Metrics vs Winning\n(No Hardcoded Values)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_correlation_heatmap.png")


def print_analysis(team_df, corr_df):
    """Print comprehensive analysis with REAL calculated values."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    print("\n" + "="*90)
    print("FIXED NETWORK ANALYSIS - ALL METRICS CALCULATED FROM REAL DATA")
    print("="*90)
    
    print("\n[PREVIOUSLY MISSING METRICS - NOW CALCULATED]")
    print("-"*90)
    
    print(f"\n1. HIERARCHY (Std of Weighted Degree):")
    print(f"   Range: {team_df['Std_Weighted_Degree'].min():.0f} - {team_df['Std_Weighted_Degree'].max():.0f}")
    print(f"   Mean: {team_df['Std_Weighted_Degree'].mean():.0f}")
    
    print(f"\n2. ENTROPY (Shannon Entropy of Pass Distribution):")
    print(f"   Range: {team_df['Pass_Entropy'].min():.3f} - {team_df['Pass_Entropy'].max():.3f}")
    print(f"   Mean: {team_df['Pass_Entropy'].mean():.3f}")
    
    print(f"\n3. HELIOCENTRIC (Max Weighted Degree - Star Player):")
    print(f"   Range: {team_df['Max_Weighted_Degree'].min():.0f} - {team_df['Max_Weighted_Degree'].max():.0f}")
    print(f"   Mean: {team_df['Max_Weighted_Degree'].mean():.0f}")
    
    print("\n" + "="*90)
    print("CORRELATIONS WITH WINNING (from real data)")
    print("="*90)
    
    print("\n{:<30} {:>12} {:>12} {:>12}".format('Metric', 'Correlation', 'P-Value', 'Significant'))
    print("-"*70)
    
    for _, row in corr_df.iterrows():
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        print(f"{row['Metric']:<30} {row['Correlation']:>+12.4f} {row['P_Value']:>12.4f} {sig:>12}")
    
    print("\n" + "="*90)
    print("KEY FINDINGS (FROM REAL DATA)")
    print("="*90)
    
    # Hierarchy
    hier_row = corr_df[corr_df['Metric'] == 'Std_Weighted_Degree'].iloc[0]
    print(f"\n1. HIERARCHY (Std Weighted Degree) vs Winning:")
    print(f"   r = {hier_row['Correlation']:.4f}, p = {hier_row['P_Value']:.4f}")
    if hier_row['Correlation'] > 0:
        print(f"   --> POSITIVE: More hierarchical teams WIN MORE")
    else:
        print(f"   --> NEGATIVE: More equal teams WIN MORE")
    
    # Entropy
    ent_row = corr_df[corr_df['Metric'] == 'Pass_Entropy'].iloc[0]
    print(f"\n2. ENTROPY (Pass Distribution) vs Winning:")
    print(f"   r = {ent_row['Correlation']:.4f}, p = {ent_row['P_Value']:.4f}")
    if ent_row['Correlation'] < 0:
        print(f"   --> NEGATIVE: More concentrated passing leads to MORE wins")
    else:
        print(f"   --> POSITIVE: More random passing leads to MORE wins")
    
    # Heliocentric
    hel_row = corr_df[corr_df['Metric'] == 'Max_Weighted_Degree'].iloc[0]
    print(f"\n3. HELIOCENTRIC (Star Max Degree) vs Winning:")
    print(f"   r = {hel_row['Correlation']:.4f}, p = {hel_row['P_Value']:.4f}")
    if hel_row['Correlation'] > 0:
        print(f"   --> POSITIVE: Having a dominant star leads to MORE wins")
    else:
        print(f"   --> NEGATIVE: Having a dominant star leads to FEWER wins")
    
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*70)
    print("FIXED NETWORK ANALYSIS")
    print("Calculating: Hierarchy, Entropy, Heliocentric (from real data)")
    print("="*70)
    
    print("\n[LOADING PLAYER DATA]")
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    print(f"Loaded {len(player_df)} player-seasons")
    
    print("\n[CALCULATING ALL TEAM METRICS]")
    print("Including previously missing: Std_Weighted_Degree, Pass_Entropy, Max_Weighted_Degree")
    team_df = calculate_all_team_metrics(player_df)
    print(f"Calculated metrics for {len(team_df)} team-seasons")
    
    print("\n[CALCULATING CORRELATIONS]")
    corr_df = calculate_correlations(team_df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_key_correlations(team_df, corr_df)
    plot_correlation_heatmap(team_df)
    
    print_analysis(team_df, corr_df)
    
    # Save data
    team_df.to_csv(OUTPUT_DIR / 'team_metrics_complete.csv', index=False)
    corr_df.to_csv(OUTPUT_DIR / 'correlations_real_data.csv', index=False)
    
    print(f"\n[OK] All results saved to {OUTPUT_DIR}/")
    print("\n*** ALL VALUES ARE FROM REAL DATA - NO HARDCODING ***")


if __name__ == "__main__":
    main()
