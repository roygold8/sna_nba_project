"""
03_analyze.py - NBA Network Analysis Visualization

Creates visualizations to analyze the relationship between team network structure
(heliocentric vs distributed offenses) and winning percentage.

Visualizations:
1. Correlation Heatmap: Network Metrics vs Team Success
2. Heliocentric Plot: Star Player Dominance vs Win %
3. Metrics Over Seasons: Trend analysis
4. Winners vs Losers: Statistical comparison

Usage:
    python 03_analyze.py

Author: NBA Network Analysis Project
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Optional, Tuple, List

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Figure settings
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'

# Color scheme
COLORS = {
    'primary': '#3498DB',
    'secondary': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'info': '#9B59B6',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def format_player_name(name: str) -> str:
    """
    Convert "Last, First" to "First Last" format.
    
    Args:
        name: Name in any format
        
    Returns:
        Name in "First Last" format
    """
    if not name or not isinstance(name, str):
        return str(name) if name else "Unknown"
    
    name = str(name).strip()
    
    if ',' in name:
        parts = name.split(', ', 1)
        if len(parts) == 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    
    return name


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load player and team metrics from CSV files.
    
    Returns:
        Tuple of (player_df, team_df), None for missing files
    """
    player_path = OUTPUT_DIR / "nba_player_metrics.csv"
    team_path = OUTPUT_DIR / "nba_team_metrics.csv"
    
    player_df = None
    team_df = None
    
    if player_path.exists():
        try:
            player_df = pd.read_csv(player_path)
            print(f"[OK] Loaded player metrics: {player_df.shape}")
        except Exception as e:
            print(f"[ERROR] Error loading player metrics: {e}")
    else:
        print(f"[ERROR] Player metrics not found: {player_path}")
    
    if team_path.exists():
        try:
            team_df = pd.read_csv(team_path)
            print(f"[OK] Loaded team metrics: {team_df.shape}")
        except Exception as e:
            print(f"[ERROR] Error loading team metrics: {e}")
    else:
        print(f"[ERROR] Team metrics not found: {team_path}")
    
    return player_df, team_df


def validate_team_data(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean team data for analysis.
    
    Args:
        team_df: Raw team metrics DataFrame
        
    Returns:
        Cleaned DataFrame with valid data
    """
    if team_df is None or team_df.empty:
        return pd.DataFrame()
    
    df = team_df.copy()
    
    # Handle missing W_PCT
    if 'W_PCT' not in df.columns:
        if 'WINS' in df.columns and 'LOSSES' in df.columns:
            total_games = df['WINS'] + df['LOSSES']
            df['W_PCT'] = df['WINS'] / total_games.replace(0, np.nan)
        else:
            df['W_PCT'] = 0.0
    
    # Fill NaN values
    df['W_PCT'] = df['W_PCT'].fillna(0.0)
    
    # Format star player names
    if 'Star_Player_Name' in df.columns:
        df['Star_Player_Name'] = df['Star_Player_Name'].apply(format_player_name)
    
    return df


# =============================================================================
# ANALYSIS 1: CORRELATION HEATMAP
# =============================================================================
def create_correlation_heatmap(team_df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a correlation matrix heatmap showing relationships between
    network metrics and team success.
    
    Args:
        team_df: DataFrame with team metrics
        
    Returns:
        Matplotlib figure or None on error
    """
    print("\n--- Creating Correlation Heatmap ---")
    
    # Select metrics for correlation analysis
    metric_columns = [
        'Density',
        'Gini_Coefficient', 
        'Degree_Centralization',
        'Top2_Concentration',
        'Star_Weighted_Degree',
        'Total_Passes',
        'W_PCT'
    ]
    
    # Filter to available columns
    available_cols = [c for c in metric_columns if c in team_df.columns]
    
    if len(available_cols) < 3:
        print("Warning: Not enough columns for correlation analysis")
        return None
    
    # Create correlation matrix (drop rows with NaN)
    corr_df = team_df[available_cols].dropna()
    
    if len(corr_df) < 10:
        print(f"Warning: Only {len(corr_df)} valid rows for correlation analysis")
        return None
    
    corr_matrix = corr_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create triangular mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Custom colormap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    
    # Draw heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    # Title and labels
    ax.set_title(
        'Correlation Between Network Metrics and Team Success\n(NBA 2015-16 to 2023-24)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Clean up axis labels
    labels = [col.replace('_', ' ').title() for col in available_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    # Print key correlations with Win %
    print("\nKey Correlations with Win %:")
    if 'W_PCT' in corr_matrix.columns:
        for col in corr_matrix.columns:
            if col != 'W_PCT':
                corr_val = corr_matrix.loc[col, 'W_PCT']
                strength = "strong" if abs(corr_val) > 0.3 else "moderate" if abs(corr_val) > 0.15 else "weak"
                print(f"  {col}: {corr_val:+.3f} ({strength})")
    
    return fig


# =============================================================================
# ANALYSIS 2: HELIOCENTRIC PLOT
# =============================================================================
def create_heliocentric_plot(team_df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a scatter plot with regression line showing the relationship
    between star player's weighted degree (heliocentricity) and team win %.
    
    Annotates the top 5 most heliocentric teams on the plot.
    
    Args:
        team_df: DataFrame with team metrics
        
    Returns:
        Matplotlib figure or None on error
    """
    print("\n--- Creating Heliocentric Analysis Plot ---")
    
    # Check required columns
    required_cols = ['Star_Weighted_Degree', 'W_PCT', 'Star_Player_Name', 
                     'TEAM_ABBREVIATION', 'SEASON']
    missing = [c for c in required_cols if c not in team_df.columns]
    
    if missing:
        print(f"Warning: Missing columns for heliocentric plot: {missing}")
        return None
    
    # Prepare data
    plot_cols = required_cols.copy()
    if 'Gini_Coefficient' in team_df.columns:
        plot_cols.append('Gini_Coefficient')
    
    plot_df = team_df[plot_cols].dropna().copy()
    
    if len(plot_df) < 10:
        print(f"Warning: Only {len(plot_df)} valid data points for heliocentric plot")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Determine color variable
    if 'Gini_Coefficient' in plot_df.columns:
        color_var = plot_df['Gini_Coefficient']
        cmap = 'RdYlBu_r'
        cbar_label = 'Gini Coefficient\n(Pass Inequality)'
    else:
        color_var = plot_df['W_PCT']
        cmap = 'viridis'
        cbar_label = 'Win %'
    
    # Scatter plot
    scatter = ax.scatter(
        plot_df['Star_Weighted_Degree'],
        plot_df['W_PCT'],
        c=color_var,
        cmap=cmap,
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Regression analysis
    x = plot_df['Star_Weighted_Degree'].values
    y = plot_df['W_PCT'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    ax.plot(
        x_line, 
        y_line, 
        color=COLORS['secondary'], 
        linewidth=2.5, 
        linestyle='--',
        label=f'Regression: r={r_value:.3f}, p={p_value:.4f}'
    )
    
    # Confidence interval
    n = len(x)
    se = std_err * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
    ci = 1.96 * se
    
    ax.fill_between(
        x_line,
        y_line - ci,
        y_line + ci,
        color=COLORS['secondary'],
        alpha=0.15,
        label='95% CI'
    )
    
    # Annotate top 5 most heliocentric teams
    top_helio = plot_df.nlargest(5, 'Star_Weighted_Degree')
    
    print("\nTop 5 Most Heliocentric Teams:")
    for idx, row in top_helio.iterrows():
        star_name = format_player_name(row['Star_Player_Name'])
        team = row['TEAM_ABBREVIATION']
        season = row['SEASON']
        wd = row['Star_Weighted_Degree']
        win_pct = row['W_PCT']
        
        label = f"{star_name}\n({team} {season})"
        print(f"  {star_name} ({team} {season}): WD={wd:.0f}, Win%={win_pct:.3f}")
        
        # Add annotation
        ax.annotate(
            label,
            xy=(wd, win_pct),
            xytext=(15, 15),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.3', 
                facecolor='white', 
                edgecolor='gray', 
                alpha=0.9
            ),
            arrowprops=dict(
                arrowstyle='->', 
                connectionstyle='arc3,rad=0.2', 
                color='gray'
            )
        )
    
    # Labels and title
    ax.set_xlabel('Star Player Weighted Degree (Total Pass Involvement)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Team Win Percentage', fontsize=14, fontweight='bold')
    ax.set_title(
        'Do Heliocentric Offenses Win More Games?\n'
        'Star Player Ball Dominance vs Team Success (NBA 2015-24)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Reference line at .500
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(x.min(), 0.505, '.500 Win %', fontsize=9, color='gray', alpha=0.7)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Axis limits
    ax.set_xlim(x.min() * 0.95, x.max() * 1.05)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Print regression summary
    print(f"\nRegression Analysis:")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        direction = "POSITIVE" if slope > 0 else "NEGATIVE"
        print(f"  => Statistically significant {direction} relationship")
    else:
        print("  => No statistically significant relationship")
    
    return fig


# =============================================================================
# ANALYSIS 3: METRICS OVER SEASONS
# =============================================================================
def create_metrics_by_season_plot(team_df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a multi-panel plot showing network metrics trends over seasons.
    
    Args:
        team_df: DataFrame with team metrics
        
    Returns:
        Matplotlib figure or None on error
    """
    print("\n--- Creating Metrics Over Seasons Plot ---")
    
    metrics = ['Gini_Coefficient', 'Top2_Concentration', 'Density', 'Degree_Centralization']
    available_metrics = [m for m in metrics if m in team_df.columns]
    
    if len(available_metrics) < 2:
        print("Warning: Not enough metrics for trends plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info']]
    
    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        
        # Aggregate by season
        season_data = team_df.groupby('SEASON')[metric].agg(['mean', 'std']).reset_index()
        
        # Plot with error bars
        ax.errorbar(
            range(len(season_data)),
            season_data['mean'],
            yerr=season_data['std'],
            marker='o',
            markersize=8,
            capsize=4,
            capthick=2,
            linewidth=2,
            color=colors[i],
            ecolor='gray',
            alpha=0.8
        )
        
        ax.set_xticks(range(len(season_data)))
        ax.set_xticklabels(season_data['SEASON'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} by Season', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(len(available_metrics), 4):
        axes[i].set_visible(False)
    
    fig.suptitle(
        'NBA Network Metrics Trends (2015-24)\nMean ± Standard Deviation Across Teams',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    return fig


# =============================================================================
# ANALYSIS 4: WINNERS VS LOSERS
# =============================================================================
def create_winners_vs_losers_plot(team_df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a comparison plot of network metrics between winning and losing teams.
    
    Args:
        team_df: DataFrame with team metrics
        
    Returns:
        Matplotlib figure or None on error
    """
    print("\n--- Creating Winners vs Losers Comparison ---")
    
    if 'W_PCT' not in team_df.columns:
        print("Warning: W_PCT column not found")
        return None
    
    # Classify teams
    df = team_df.copy()
    df['Team_Class'] = df['W_PCT'].apply(
        lambda x: 'Winners (>=.500)' if x >= 0.5 else 'Losers (<.500)'
    )
    
    metrics = ['Gini_Coefficient', 'Top2_Concentration', 'Degree_Centralization', 
               'Star_Weighted_Degree']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        print("Warning: Not enough metrics for winners vs losers plot")
        return None
    
    fig, axes = plt.subplots(1, len(available_metrics), 
                             figsize=(4*len(available_metrics), 6))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    colors = {'Winners (>=.500)': COLORS['success'], 'Losers (<.500)': COLORS['secondary']}
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Box plot
        sns.boxplot(
            data=df,
            x='Team_Class',
            y=metric,
            ax=ax,
            palette=colors,
            width=0.6
        )
        
        # Statistical test (Mann-Whitney U)
        winners = df[df['Team_Class'] == 'Winners (>=.500)'][metric].dropna()
        losers = df[df['Team_Class'] == 'Losers (<.500)'][metric].dropna()
        
        if len(winners) > 0 and len(losers) > 0:
            stat, p_val = stats.mannwhitneyu(winners, losers, alternative='two-sided')
            
            # Significance stars
            if p_val < 0.001:
                significance = '***'
            elif p_val < 0.01:
                significance = '**'
            elif p_val < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            ax.text(
                0.5, 1.02,
                f'p={p_val:.4f} {significance}',
                transform=ax.transAxes,
                ha='center',
                fontsize=10,
                style='italic'
            )
        
        ax.set_xlabel('')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    fig.suptitle(
        'Network Metrics: Winning vs Losing Teams\n'
        '(NBA 2015-24, Mann-Whitney U Test, * p<.05, ** p<.01, *** p<.001)',
        fontsize=14,
        fontweight='bold',
        y=1.05
    )
    
    plt.tight_layout()
    return fig


# =============================================================================
# ANALYSIS 5: TOP HELIOCENTRIC PLAYERS TABLE
# =============================================================================
def print_heliocentric_rankings(team_df: pd.DataFrame, n: int = 15) -> None:
    """
    Print a ranking of most heliocentric team-seasons.
    
    Args:
        team_df: DataFrame with team metrics
        n: Number of top entries to show
    """
    print("\n" + "="*70)
    print("TOP HELIOCENTRIC TEAM-SEASONS (BY STAR WEIGHTED DEGREE)")
    print("="*70)
    
    required = ['Star_Player_Name', 'TEAM_ABBREVIATION', 'SEASON', 
                'Star_Weighted_Degree', 'W_PCT', 'Gini_Coefficient']
    
    if not all(col in team_df.columns for col in required):
        print("Missing required columns for ranking")
        return
    
    ranking_df = team_df[required].dropna().copy()
    ranking_df = ranking_df.nlargest(n, 'Star_Weighted_Degree')
    ranking_df['Star_Player_Name'] = ranking_df['Star_Player_Name'].apply(format_player_name)
    
    print(f"\n{'Rank':<5} {'Player':<22} {'Team':<5} {'Season':<10} {'W.Degree':<10} {'Win%':<8} {'Gini':<6}")
    print("-" * 70)
    
    for rank, (_, row) in enumerate(ranking_df.iterrows(), 1):
        # Encode player name to ASCII to handle special characters
        player_name = row['Star_Player_Name']
        try:
            # Try to encode to ASCII, replacing special chars
            player_name = player_name.encode('ascii', 'replace').decode('ascii')
        except Exception:
            player_name = str(player_name)
        
        print(f"{rank:<5} {player_name:<22} {row['TEAM_ABBREVIATION']:<5} "
              f"{row['SEASON']:<10} {row['Star_Weighted_Degree']:<10.0f} "
              f"{row['W_PCT']:<8.3f} {row['Gini_Coefficient']:<6.3f}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main entry point for analysis and visualization."""
    print("\n" + "="*60)
    print("NBA NETWORK ANALYSIS - VISUALIZATION")
    print("="*60)
    
    # Ensure directories exist
    ensure_directory(OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)
    
    # Load data
    player_df, team_df = load_data()
    
    if team_df is None or team_df.empty:
        print("\n[ERROR] No team data available. Run 02_build_metrics.py first.")
        return
    
    # Validate and clean data
    team_df = validate_team_data(team_df)
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"  Seasons: {team_df['SEASON'].nunique()}")
    print(f"  Teams: {team_df['TEAM_ABBREVIATION'].nunique()}")
    print(f"  Total Team-Seasons: {len(team_df)}")
    
    if 'W_PCT' in team_df.columns:
        avg_w_pct = team_df['W_PCT'].mean()
        print(f"  Average Win %: {avg_w_pct:.3f}")
    
    figures = []
    
    # Analysis 1: Correlation Heatmap
    try:
        fig = create_correlation_heatmap(team_df)
        if fig:
            filepath = FIGURES_DIR / "correlation_heatmap.png"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(('Correlation Heatmap', fig))
    except Exception as e:
        print(f"\n[ERROR] Error creating correlation heatmap: {e}")
    
    # Analysis 2: Heliocentric Plot
    try:
        fig = create_heliocentric_plot(team_df)
        if fig:
            filepath = FIGURES_DIR / "heliocentric_analysis.png"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(('Heliocentric Analysis', fig))
    except Exception as e:
        print(f"\n[ERROR] Error creating heliocentric plot: {e}")
    
    # Analysis 3: Metrics Over Seasons
    try:
        fig = create_metrics_by_season_plot(team_df)
        if fig:
            filepath = FIGURES_DIR / "metrics_by_season.png"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(('Metrics by Season', fig))
    except Exception as e:
        print(f"\n[ERROR] Error creating metrics by season plot: {e}")
    
    # Analysis 4: Winners vs Losers
    try:
        fig = create_winners_vs_losers_plot(team_df)
        if fig:
            filepath = FIGURES_DIR / "winners_vs_losers.png"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(('Winners vs Losers', fig))
    except Exception as e:
        print(f"\n[ERROR] Error creating winners vs losers plot: {e}")
    
    # Print heliocentric rankings
    print_heliocentric_rankings(team_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nGenerated {len(figures)} visualizations in: {FIGURES_DIR}")
    
    # Display all figures
    if figures:
        print("\nDisplaying figures...")
        plt.show()


if __name__ == "__main__":
    main()
