"""
analyze_stars_and_structure.py - Stars & Team Structure Analysis

Tests 5 specific theses about team structure and player accolades:
1. Star Network Signature: Stars vs Role Players network metrics
2. Team Connectivity: Mean Weighted Degree vs Win%
3. Team Hierarchy: Degree Spread vs Win%
4. Playoff Factor: Network metrics vs Playoff success
5. Championship DNA: Profile of top vs bottom teams

Usage:
    python analyze_stars_and_structure.py

Author: NBA Network Analysis Project
"""

import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# NBA API imports
try:
    from nba_api.stats.endpoints import playerawards, commonteamroster
    from nba_api.stats.static import teams as nba_teams
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("[WARNING] nba_api not available. Using fallback star detection.")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("output")
ANALYSIS_DIR = Path("output_analysis")
FIGURE_DPI = 150

SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24'
]

# Award types that define a "Star"
STAR_AWARDS = [
    'All-NBA',
    'All-Star',
    'NBA Most Valuable Player',
    'NBA MVP',
    'All Star',
    'All-Defensive',
]

# API rate limiting
API_SLEEP = 0.6


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def format_player_name(name: str) -> str:
    """Convert 'Last, First' to 'First Last' format."""
    if not name or not isinstance(name, str):
        return str(name) if name else "Unknown"
    name = str(name).strip()
    if ',' in name:
        parts = name.split(', ', 1)
        if len(parts) == 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    return name


def season_to_year(season: str) -> int:
    """Convert '2023-24' to 2024 (end year)."""
    return int('20' + season.split('-')[1])


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load player and team metrics from CSV files."""
    # Try enhanced versions first
    player_paths = [
        OUTPUT_DIR / "nba_player_metrics_enhanced.csv",
        OUTPUT_DIR / "nba_player_metrics.csv"
    ]
    team_paths = [
        OUTPUT_DIR / "nba_team_metrics_enhanced.csv",
        OUTPUT_DIR / "nba_team_metrics.csv"
    ]
    
    player_df = None
    team_df = None
    
    for path in player_paths:
        if path.exists():
            player_df = pd.read_csv(path)
            print(f"[OK] Loaded player metrics: {player_df.shape} from {path.name}")
            break
    
    for path in team_paths:
        if path.exists():
            team_df = pd.read_csv(path)
            print(f"[OK] Loaded team metrics: {team_df.shape} from {path.name}")
            break
    
    if player_df is None or team_df is None:
        raise FileNotFoundError("Required CSV files not found. Run 02_build_metrics.py first.")
    
    return player_df, team_df


# =============================================================================
# PHASE 1: DATA ENRICHMENT - FETCHING AWARDS
# =============================================================================
def fetch_player_awards_safe(player_id: int) -> List[dict]:
    """Safely fetch player awards with error handling."""
    try:
        awards = playerawards.PlayerAwards(player_id=player_id)
        time.sleep(API_SLEEP)
        df = awards.get_data_frames()[0]
        return df.to_dict('records')
    except Exception as e:
        return []


def get_star_players_from_api(player_df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """
    Fetch awards data from NBA API and identify star players.
    
    Returns:
        Dictionary mapping (player_id, season) -> is_star
    """
    print("\n[INFO] Fetching player awards from NBA API...")
    
    star_dict = {}
    unique_players = player_df['PLAYER_ID'].unique()
    
    # Sample a subset if too many players (API rate limits)
    # Focus on players with high weighted degree (likely stars)
    top_players = player_df.groupby('PLAYER_ID')['Weighted_Degree'].max().nlargest(200).index.tolist()
    
    print(f"[INFO] Checking awards for {len(top_players)} top players...")
    
    for i, player_id in enumerate(top_players):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(top_players)}")
        
        awards_list = fetch_player_awards_safe(player_id)
        
        for award in awards_list:
            award_type = str(award.get('DESCRIPTION', ''))
            season = award.get('SEASON', '')
            
            # Check if this is a star-level award
            is_star_award = any(star in award_type for star in STAR_AWARDS)
            
            if is_star_award and season:
                # Convert season format if needed (e.g., "2023-24" or "2024")
                if len(season) == 4:
                    # Convert "2024" to "2023-24"
                    year = int(season)
                    season = f"{year-1}-{str(year)[2:]}"
                
                star_dict[(player_id, season)] = True
    
    print(f"[OK] Found {len(star_dict)} star player-seasons")
    return star_dict


def get_star_players_fallback(player_df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """
    Fallback method: Identify stars based on performance metrics.
    Top 3 players per team by Weighted_Degree + top scorers league-wide.
    """
    print("\n[INFO] Using fallback star detection (top performers)...")
    
    star_dict = {}
    
    # Method 1: Top 30 players by weighted degree each season
    for season in player_df['SEASON'].unique():
        season_df = player_df[player_df['SEASON'] == season]
        top_by_degree = season_df.nlargest(30, 'Weighted_Degree')['PLAYER_ID'].tolist()
        
        for pid in top_by_degree:
            star_dict[(pid, season)] = True
    
    # Method 2: Top scorers each season (>20 PPG proxy)
    if 'PTS' in player_df.columns:
        for season in player_df['SEASON'].unique():
            season_df = player_df[player_df['SEASON'] == season]
            top_scorers = season_df[season_df['PTS'] >= 1500]['PLAYER_ID'].tolist()  # ~20 PPG * 75 games
            
            for pid in top_scorers:
                star_dict[(pid, season)] = True
    
    print(f"[OK] Identified {len(star_dict)} star player-seasons (fallback method)")
    return star_dict


def enrich_with_star_status(player_df: pd.DataFrame) -> pd.DataFrame:
    """Add Is_Star column to player dataframe."""
    df = player_df.copy()
    
    # Try API first, fallback if not available
    if HAS_NBA_API:
        try:
            star_dict = get_star_players_from_api(df)
        except Exception as e:
            print(f"[WARNING] API fetch failed: {e}")
            star_dict = get_star_players_fallback(df)
    else:
        star_dict = get_star_players_fallback(df)
    
    # Map to dataframe
    df['Is_Star'] = df.apply(
        lambda row: star_dict.get((row['PLAYER_ID'], row['SEASON']), False),
        axis=1
    )
    
    star_count = df['Is_Star'].sum()
    print(f"[OK] Marked {star_count} player-seasons as Stars ({star_count/len(df)*100:.1f}%)")
    
    return df


def add_playoff_proxy(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add playoff success proxy metrics.
    Since we don't have actual playoff data, we'll use:
    - PlayoffTeam: W_PCT >= 0.5 (roughly playoff threshold)
    - EliteTeam: W_PCT >= 0.65 (Conference Finals caliber)
    - PlayoffScore: Scaled metric based on wins
    """
    df = team_df.copy()
    
    # Playoff proxy based on win percentage
    df['Is_Playoff_Team'] = df['W_PCT'] >= 0.488  # ~40 wins = playoff in most years
    df['Is_Elite_Team'] = df['W_PCT'] >= 0.634  # ~52 wins = elite
    df['Is_Championship_Caliber'] = df['W_PCT'] >= 0.720  # ~59 wins = championship caliber
    
    # Playoff score (0-100 scale based on wins)
    # Assuming 82 games, scale wins to expected playoff success
    df['Playoff_Score'] = np.clip(df['W_PCT'] * 100, 0, 100)
    
    # Bonus for elite teams (simulating playoff wins)
    df['Playoff_Score'] = df['Playoff_Score'] + df['Is_Elite_Team'].astype(int) * 10
    df['Playoff_Score'] = df['Playoff_Score'] + df['Is_Championship_Caliber'].astype(int) * 15
    
    print(f"[OK] Added playoff proxy metrics")
    print(f"  Playoff teams: {df['Is_Playoff_Team'].sum()}")
    print(f"  Elite teams: {df['Is_Elite_Team'].sum()}")
    
    return df


def calculate_team_aggregates(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate team-level aggregates from player data."""
    df = team_df.copy()
    
    # Calculate per-team statistics from player data
    team_agg = player_df.groupby(['TEAM_ID', 'SEASON']).agg({
        'Weighted_Degree': ['mean', 'std', 'max', 'min'],
        'Betweenness_Centrality': ['mean', 'std', 'max'],
        'Is_Star': 'sum'
    }).reset_index()
    
    # Flatten column names
    team_agg.columns = [
        'TEAM_ID', 'SEASON',
        'Mean_Weighted_Degree', 'Std_Weighted_Degree', 'Max_Weighted_Degree', 'Min_Weighted_Degree',
        'Mean_Betweenness', 'Std_Betweenness', 'Max_Betweenness',
        'Num_Stars'
    ]
    
    # Calculate degree spread (max - min)
    team_agg['Degree_Spread'] = team_agg['Max_Weighted_Degree'] - team_agg['Min_Weighted_Degree']
    
    # Merge with team data
    df = df.merge(team_agg, on=['TEAM_ID', 'SEASON'], how='left')
    
    # Fill NaN values
    for col in ['Mean_Weighted_Degree', 'Std_Weighted_Degree', 'Mean_Betweenness']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"[OK] Calculated team aggregates")
    
    return df


# =============================================================================
# THESIS 1: STAR NETWORK SIGNATURE
# =============================================================================
def analyze_star_network_signature(player_df: pd.DataFrame) -> plt.Figure:
    """
    Compare network metrics of Stars vs Role Players.
    Creates boxplots for Weighted_Degree and Betweenness.
    """
    print("\n" + "="*60)
    print("THESIS 1: STAR NETWORK SIGNATURE")
    print("="*60)
    
    df = player_df[player_df['Weighted_Degree'] > 0].copy()
    df['Player_Type'] = df['Is_Star'].map({True: 'Stars', False: 'Role Players'})
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Weighted Degree
    ax1 = axes[0]
    sns.boxplot(
        data=df,
        x='Player_Type',
        y='Weighted_Degree',
        palette={'Stars': '#E74C3C', 'Role Players': '#3498DB'},
        ax=ax1
    )
    
    # Statistical test
    stars = df[df['Is_Star']]['Weighted_Degree']
    role_players = df[~df['Is_Star']]['Weighted_Degree']
    stat, p_val = stats.mannwhitneyu(stars, role_players, alternative='greater')
    
    ax1.set_title(f'Weighted Degree\n(Mann-Whitney p = {p_val:.2e})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Weighted Degree (Pass Involvement)', fontsize=11)
    
    # Plot 2: Betweenness Centrality
    ax2 = axes[1]
    sns.boxplot(
        data=df,
        x='Player_Type',
        y='Betweenness_Centrality',
        palette={'Stars': '#E74C3C', 'Role Players': '#3498DB'},
        ax=ax2
    )
    
    # Statistical test
    stars_bc = df[df['Is_Star']]['Betweenness_Centrality']
    role_bc = df[~df['Is_Star']]['Betweenness_Centrality']
    stat2, p_val2 = stats.mannwhitneyu(stars_bc, role_bc, alternative='greater')
    
    ax2.set_title(f'Betweenness Centrality\n(Mann-Whitney p = {p_val2:.2e})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Betweenness Centrality', fontsize=11)
    
    fig.suptitle(
        'Star Network Signature: Do Accolades Correlate with Central Network Roles?',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nWeighted Degree:")
    print(f"  Stars median: {stars.median():.0f}")
    print(f"  Role Players median: {role_players.median():.0f}")
    print(f"  Difference: {((stars.median() / role_players.median()) - 1) * 100:.1f}% higher for Stars")
    print(f"  p-value: {p_val:.2e} ({'Significant' if p_val < 0.05 else 'Not significant'})")
    
    print(f"\nBetweenness Centrality:")
    print(f"  Stars median: {stars_bc.median():.4f}")
    print(f"  Role Players median: {role_bc.median():.4f}")
    print(f"  p-value: {p_val2:.2e} ({'Significant' if p_val2 < 0.05 else 'Not significant'})")
    
    # Identify "isolated stars" (high scoring but low betweenness)
    if 'PTS' in df.columns:
        high_scorers = df[df['PTS'] >= 1200]  # ~15 PPG
        isolated = high_scorers[high_scorers['Betweenness_Centrality'] < high_scorers['Betweenness_Centrality'].quantile(0.25)]
        print(f"\n'Isolated Stars' (high scoring, low betweenness): {len(isolated)}")
        if len(isolated) > 0:
            top_isolated = isolated.nlargest(5, 'PTS')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'PTS', 'Betweenness_Centrality']]
            print(top_isolated.to_string(index=False))
    
    return fig


# =============================================================================
# THESIS 2: TEAM CONNECTIVITY
# =============================================================================
def analyze_team_connectivity(team_df: pd.DataFrame) -> plt.Figure:
    """
    Check if teams with higher average pass involvement win more.
    Scatter plot of Mean_Weighted_Degree vs Win%.
    """
    print("\n" + "="*60)
    print("THESIS 2: TEAM CONNECTIVITY (High Average Degree)")
    print("="*60)
    
    df = team_df[['Mean_Weighted_Degree', 'W_PCT', 'TEAM_ABBREVIATION', 'SEASON']].dropna()
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df['Mean_Weighted_Degree'], df['W_PCT'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.regplot(
        data=df,
        x='Mean_Weighted_Degree',
        y='W_PCT',
        scatter_kws={'alpha': 0.6, 's': 60},
        line_kws={'color': '#E74C3C', 'linewidth': 2},
        ax=ax
    )
    
    ax.set_xlabel('Mean Weighted Degree (Avg Pass Involvement)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Percentage', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Team Connectivity: Do Higher-Passing Teams Win More?\n'
        f'Pearson r = {corr:.3f}, p = {p_val:.4f}',
        fontsize=14, fontweight='bold'
    )
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nCorrelation: r = {corr:.3f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Result: {'Significant' if p_val < 0.05 else 'Not significant'} relationship")
    
    if corr > 0:
        print("=> Teams with higher average pass involvement tend to win MORE")
    else:
        print("=> Teams with higher average pass involvement tend to win LESS")
    
    return fig


# =============================================================================
# THESIS 3: TEAM HIERARCHY
# =============================================================================
def analyze_team_hierarchy(team_df: pd.DataFrame) -> plt.Figure:
    """
    Check if having hierarchy (wide degree gap) is better than equality.
    Scatter plot of Std_Weighted_Degree vs Win%.
    """
    print("\n" + "="*60)
    print("THESIS 3: TEAM HIERARCHY (Degree Distribution Spread)")
    print("="*60)
    
    df = team_df[['Std_Weighted_Degree', 'W_PCT', 'TEAM_ABBREVIATION', 'SEASON', 'Gini_Coefficient']].dropna()
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df['Std_Weighted_Degree'], df['W_PCT'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by Gini coefficient
    scatter = ax.scatter(
        df['Std_Weighted_Degree'],
        df['W_PCT'],
        c=df['Gini_Coefficient'],
        cmap='RdYlBu_r',
        s=60,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add regression line
    z = np.polyfit(df['Std_Weighted_Degree'], df['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Std_Weighted_Degree'].min(), df['Std_Weighted_Degree'].max(), 100)
    ax.plot(x_line, p(x_line), color='#E74C3C', linewidth=2, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Gini Coefficient', fontsize=11)
    
    ax.set_xlabel('Std Dev of Weighted Degree (Hierarchy)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Percentage', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Team Hierarchy: Is Inequality Better?\n'
        f'Pearson r = {corr:.3f}, p = {p_val:.4f}',
        fontsize=14, fontweight='bold'
    )
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nCorrelation (Std Degree vs Win%): r = {corr:.3f}, p = {p_val:.4f}")
    
    # Also check Gini correlation
    gini_corr, gini_p = stats.pearsonr(df['Gini_Coefficient'], df['W_PCT'])
    print(f"Correlation (Gini vs Win%): r = {gini_corr:.3f}, p = {gini_p:.4f}")
    
    if corr > 0:
        print("=> More hierarchical teams (higher spread) tend to win MORE")
    else:
        print("=> More egalitarian teams (lower spread) tend to win MORE")
    
    return fig


# =============================================================================
# THESIS 4: PLAYOFF FACTOR
# =============================================================================
def analyze_playoff_factor(team_df: pd.DataFrame) -> plt.Figure:
    """
    Repeat connectivity and hierarchy analysis with playoff success as Y-axis.
    """
    print("\n" + "="*60)
    print("THESIS 4: THE PLAYOFF FACTOR")
    print("="*60)
    
    df = team_df[['Mean_Weighted_Degree', 'Std_Weighted_Degree', 'Playoff_Score', 
                  'Is_Elite_Team', 'Is_Championship_Caliber', 'TEAM_ABBREVIATION', 'SEASON']].dropna()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for elite teams
    colors = df['Is_Championship_Caliber'].map({True: '#FFD700', False: '#3498DB'})
    sizes = df['Is_Elite_Team'].map({True: 100, False: 40})
    
    # Plot 1: Mean Degree vs Playoff Score
    ax1 = axes[0]
    ax1.scatter(
        df['Mean_Weighted_Degree'],
        df['Playoff_Score'],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    corr1, p1 = stats.pearsonr(df['Mean_Weighted_Degree'], df['Playoff_Score'])
    
    # Add regression
    z1 = np.polyfit(df['Mean_Weighted_Degree'], df['Playoff_Score'], 1)
    p_line1 = np.poly1d(z1)
    x_line = np.linspace(df['Mean_Weighted_Degree'].min(), df['Mean_Weighted_Degree'].max(), 100)
    ax1.plot(x_line, p_line1(x_line), color='#E74C3C', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Mean Weighted Degree', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Playoff Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'Connectivity vs Playoff Success\nr = {corr1:.3f}, p = {p1:.4f}', fontsize=12, fontweight='bold')
    
    # Plot 2: Std Degree vs Playoff Score
    ax2 = axes[1]
    ax2.scatter(
        df['Std_Weighted_Degree'],
        df['Playoff_Score'],
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    corr2, p2 = stats.pearsonr(df['Std_Weighted_Degree'], df['Playoff_Score'])
    
    z2 = np.polyfit(df['Std_Weighted_Degree'], df['Playoff_Score'], 1)
    p_line2 = np.poly1d(z2)
    x_line2 = np.linspace(df['Std_Weighted_Degree'].min(), df['Std_Weighted_Degree'].max(), 100)
    ax2.plot(x_line2, p_line2(x_line2), color='#E74C3C', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Std Dev Weighted Degree (Hierarchy)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Playoff Score', fontsize=11, fontweight='bold')
    ax2.set_title(f'Hierarchy vs Playoff Success\nr = {corr2:.3f}, p = {p2:.4f}', fontsize=12, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markersize=12, label='Championship Caliber'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=8, label='Other Teams'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    fig.suptitle(
        'The Playoff Factor: Do Network Metrics Predict Playoff Success?',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nConnectivity vs Playoffs: r = {corr1:.3f}, p = {p1:.4f}")
    print(f"Hierarchy vs Playoffs: r = {corr2:.3f}, p = {p2:.4f}")
    
    # Check if relationships are stronger for playoffs than regular season
    reg_corr1, _ = stats.pearsonr(df['Mean_Weighted_Degree'], team_df.loc[df.index, 'W_PCT'])
    reg_corr2, _ = stats.pearsonr(df['Std_Weighted_Degree'], team_df.loc[df.index, 'W_PCT'])
    
    print(f"\nComparison (Regular Season vs Playoff proxy):")
    print(f"  Connectivity: {reg_corr1:.3f} vs {corr1:.3f}")
    print(f"  Hierarchy: {reg_corr2:.3f} vs {corr2:.3f}")
    
    return fig


# =============================================================================
# THESIS 5: CHAMPIONSHIP DNA
# =============================================================================
def analyze_championship_dna(team_df: pd.DataFrame) -> None:
    """
    Profile top 10 vs bottom 10 teams by Win%.
    """
    print("\n" + "="*60)
    print("THESIS 5: CHAMPIONSHIP DNA - Profiling Top Teams")
    print("="*60)
    
    # Get top 10 and bottom 10 teams
    top_10 = team_df.nlargest(10, 'W_PCT')
    bottom_10 = team_df.nsmallest(10, 'W_PCT')
    
    # Metrics to compare
    metrics = [
        ('W_PCT', 'Win %'),
        ('Clustering_Coefficient', 'Clustering Coef'),
        ('Pass_Entropy', 'Pass Entropy'),
        ('Degree_Centralization', 'Degree Centralization'),
        ('Gini_Coefficient', 'Gini Coefficient'),
        ('Std_Weighted_Degree', 'Weighted Degree Std'),
        ('Mean_Weighted_Degree', 'Mean Weighted Degree'),
        ('Top2_Concentration', 'Top 2 Concentration'),
        ('Density', 'Network Density'),
    ]
    
    # Calculate averages
    print("\n" + "-"*70)
    print(f"{'Metric':<25} {'Top 10 Teams':<15} {'Bottom 10 Teams':<15} {'Difference':<15}")
    print("-"*70)
    
    dna_results = []
    
    for col, name in metrics:
        if col in team_df.columns:
            top_avg = top_10[col].mean()
            bot_avg = bottom_10[col].mean()
            diff_pct = ((top_avg - bot_avg) / (bot_avg + 0.001)) * 100
            
            print(f"{name:<25} {top_avg:<15.3f} {bot_avg:<15.3f} {diff_pct:>+14.1f}%")
            
            dna_results.append({
                'Metric': name,
                'Top_10_Avg': top_avg,
                'Bottom_10_Avg': bot_avg,
                'Diff_Pct': diff_pct
            })
    
    print("-"*70)
    
    # Print top 10 teams
    print("\nTOP 10 TEAMS (by Win %):")
    print("-"*50)
    top_display = top_10[['TEAM_ABBREVIATION', 'SEASON', 'W_PCT', 'WINS']].copy()
    top_display['W_PCT'] = top_display['W_PCT'].apply(lambda x: f"{x:.3f}")
    print(top_display.to_string(index=False))
    
    # Print bottom 10 teams
    print("\nBOTTOM 10 TEAMS (by Win %):")
    print("-"*50)
    bot_display = bottom_10[['TEAM_ABBREVIATION', 'SEASON', 'W_PCT', 'WINS']].copy()
    bot_display['W_PCT'] = bot_display['W_PCT'].apply(lambda x: f"{x:.3f}")
    print(bot_display.to_string(index=False))
    
    # Key findings
    print("\n" + "="*60)
    print("KEY CHAMPIONSHIP DNA FINDINGS:")
    print("="*60)
    
    dna_df = pd.DataFrame(dna_results)
    
    # Find biggest differences
    biggest_positive = dna_df.nlargest(3, 'Diff_Pct')
    biggest_negative = dna_df.nsmallest(3, 'Diff_Pct')
    
    print("\nMetrics where TOP teams are HIGHER:")
    for _, row in biggest_positive.iterrows():
        print(f"  - {row['Metric']}: +{row['Diff_Pct']:.1f}%")
    
    print("\nMetrics where TOP teams are LOWER:")
    for _, row in biggest_negative.iterrows():
        print(f"  - {row['Metric']}: {row['Diff_Pct']:.1f}%")


def create_championship_dna_visual(team_df: pd.DataFrame) -> plt.Figure:
    """Create a visual comparison of top vs bottom teams."""
    
    top_10 = team_df.nlargest(10, 'W_PCT')
    bottom_10 = team_df.nsmallest(10, 'W_PCT')
    
    metrics = ['Clustering_Coefficient', 'Pass_Entropy', 'Degree_Centralization', 
               'Gini_Coefficient', 'Density']
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in team_df.columns]
    
    # Normalize metrics for radar chart
    top_values = []
    bot_values = []
    
    for metric in metrics:
        all_vals = team_df[metric].dropna()
        min_val, max_val = all_vals.min(), all_vals.max()
        
        top_norm = (top_10[metric].mean() - min_val) / (max_val - min_val + 0.001)
        bot_norm = (bottom_10[metric].mean() - min_val) / (max_val - min_val + 0.001)
        
        top_values.append(top_norm)
        bot_values.append(bot_norm)
    
    # Create figure with bar comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, top_values, width, label='Top 10 Teams', color='#27AE60', alpha=0.8)
    bars2 = ax.bar(x + width/2, bot_values, width, label='Bottom 10 Teams', color='#E74C3C', alpha=0.8)
    
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Championship DNA: Network Profile Comparison\nTop 10 vs Bottom 10 Teams by Win%',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars1, top_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, bot_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("NBA STARS & STRUCTURE ANALYSIS")
    print("="*70)
    
    ensure_directory(ANALYSIS_DIR)
    
    # Load data
    player_df, team_df = load_data()
    
    # Phase 1: Data Enrichment
    print("\n" + "="*60)
    print("PHASE 1: DATA ENRICHMENT")
    print("="*60)
    
    player_df = enrich_with_star_status(player_df)
    team_df = add_playoff_proxy(team_df)
    team_df = calculate_team_aggregates(player_df, team_df)
    
    # Save enriched data
    player_df.to_csv(OUTPUT_DIR / "nba_player_metrics_with_stars.csv", index=False)
    team_df.to_csv(OUTPUT_DIR / "nba_team_metrics_with_aggregates.csv", index=False)
    print(f"[OK] Saved enriched CSVs")
    
    # Phase 2: Analyze Theses
    print("\n" + "="*60)
    print("PHASE 2: ANALYZING THESES")
    print("="*60)
    
    figures = []
    
    # Thesis 1: Star Network Signature
    try:
        fig1 = analyze_star_network_signature(player_df)
        fig1.savefig(ANALYSIS_DIR / "thesis1_star_network_signature.png", 
                     dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: thesis1_star_network_signature.png")
        figures.append(fig1)
    except Exception as e:
        print(f"\n[ERROR] Thesis 1: {e}")
    
    # Thesis 2: Team Connectivity
    try:
        fig2 = analyze_team_connectivity(team_df)
        fig2.savefig(ANALYSIS_DIR / "thesis2_team_connectivity.png",
                     dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: thesis2_team_connectivity.png")
        figures.append(fig2)
    except Exception as e:
        print(f"\n[ERROR] Thesis 2: {e}")
    
    # Thesis 3: Team Hierarchy
    try:
        fig3 = analyze_team_hierarchy(team_df)
        fig3.savefig(ANALYSIS_DIR / "thesis3_team_hierarchy.png",
                     dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: thesis3_team_hierarchy.png")
        figures.append(fig3)
    except Exception as e:
        print(f"\n[ERROR] Thesis 3: {e}")
    
    # Thesis 4: Playoff Factor
    try:
        fig4 = analyze_playoff_factor(team_df)
        fig4.savefig(ANALYSIS_DIR / "thesis4_playoff_factor.png",
                     dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: thesis4_playoff_factor.png")
        figures.append(fig4)
    except Exception as e:
        print(f"\n[ERROR] Thesis 4: {e}")
    
    # Thesis 5: Championship DNA
    try:
        analyze_championship_dna(team_df)
        fig5 = create_championship_dna_visual(team_df)
        fig5.savefig(ANALYSIS_DIR / "thesis5_championship_dna.png",
                     dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: thesis5_championship_dna.png")
        figures.append(fig5)
    except Exception as e:
        print(f"\n[ERROR] Thesis 5: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {ANALYSIS_DIR.absolute()}")
    print(f"Generated {len(figures)} visualizations")
    
    # Show figures
    if figures:
        plt.show()


if __name__ == "__main__":
    main()
