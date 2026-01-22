"""
PLAYOFF vs REGULAR SEASON ANALYSIS
===================================
IMPORTANT METHODOLOGY NOTE:
--------------------------
The NBA API does not provide play-by-play passing data for playoff games 
in the same format as regular season data. Therefore, this analysis 
compares REGULAR SEASON network metrics between:
- Teams that made the playoffs
- Teams that missed the playoffs (lottery teams)
- Teams that had deep playoff runs
- Championship teams

This is a PROXY analysis that examines whether teams with certain 
network structures during the regular season go on to succeed in playoffs.

Author: NBA Network Analysis Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path("output_playoff_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

# Championship teams by season
CHAMPIONS = {
    '2015-16': 'CLE',
    '2016-17': 'GSW',
    '2017-18': 'GSW',
    '2018-19': 'TOR',
    '2019-20': 'LAL',
    '2020-21': 'MIL',
    '2021-22': 'GSW',
    '2022-23': 'DEN',
    '2023-24': 'BOS',
}

# Finals teams (losers)
FINALS_LOSERS = {
    '2015-16': 'GSW',
    '2016-17': 'CLE',
    '2017-18': 'CLE',
    '2018-19': 'GSW',
    '2019-20': 'MIA',
    '2020-21': 'PHX',
    '2021-22': 'BOS',
    '2022-23': 'MIA',
    '2023-24': 'DAL',
}

# Conference Finals teams (approximate - losing CF teams)
CONF_FINALS = {
    '2015-16': ['OKC', 'TOR'],
    '2016-17': ['SAS', 'BOS'],
    '2017-18': ['HOU', 'BOS'],
    '2018-19': ['MIL', 'POR'],
    '2019-20': ['DEN', 'BOS'],
    '2020-21': ['LAC', 'ATL'],
    '2021-22': ['MIA', 'DAL'],
    '2022-23': ['LAL', 'BOS'],
    '2023-24': ['MIN', 'IND'],
}


def load_complete_metrics():
    """Load the complete metrics file."""
    metrics_file = Path("output_complete_sna/complete_team_metrics.csv")
    
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    
    # Fallback to original metrics
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    
    # Aggregate to team level
    team_metrics = []
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        degrees = group['Weighted_Degree'].values
        if len(degrees) < 3:
            continue
        
        team_metrics.append({
            'Team': team,
            'Season': season,
            'Mean_Weighted_Degree': np.mean(degrees),
            'Std_Weighted_Degree': np.std(degrees),
            'Max_Weighted_Degree': np.max(degrees),
            'Num_Nodes': len(degrees),
        })
    
    return pd.DataFrame(team_metrics)


def categorize_playoff_status(df):
    """Categorize teams by playoff success."""
    
    def get_status(row):
        team = row['Team']
        season = row['Season']
        
        if CHAMPIONS.get(season) == team:
            return 'Champion'
        elif FINALS_LOSERS.get(season) == team:
            return 'Finals'
        elif team in CONF_FINALS.get(season, []):
            return 'Conf_Finals'
        elif row.get('Win_Pct', 0) >= 0.500:
            return 'Playoff'
        else:
            return 'Lottery'
    
    df['Playoff_Status'] = df.apply(get_status, axis=1)
    
    # Numeric ordering
    status_order = {'Champion': 5, 'Finals': 4, 'Conf_Finals': 3, 'Playoff': 2, 'Lottery': 1}
    df['Playoff_Depth'] = df['Playoff_Status'].map(status_order)
    
    return df


def analyze_and_plot(df):
    """Create comprehensive playoff analysis."""
    df = df[df['Win_Pct'].notna()].copy()
    df = categorize_playoff_status(df)
    
    # ===========================
    # FIGURE 1: Metrics by Playoff Status
    # ===========================
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = [
        ('Std_Weighted_Degree', 'Hierarchy (Std Degree)'),
        ('Max_Weighted_Degree', 'Star Player Degree'),
        ('Pass_Entropy', 'Pass Entropy'),
        ('Graph_Density', 'Graph Density'),
        ('Mean_Closeness', 'Mean Closeness'),
        ('Gini_Coefficient', 'Gini Coefficient'),
    ]
    
    order = ['Lottery', 'Playoff', 'Conf_Finals', 'Finals', 'Champion']
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in df.columns:
            # Boxplot
            plot_data = [df[df['Playoff_Status'] == status][metric].dropna().values 
                         for status in order if status in df['Playoff_Status'].values]
            labels = [s for s in order if s in df['Playoff_Status'].values]
            
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
            for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label)
            ax.set_title(label, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Statistical test: Champion vs Lottery
            champ = df[df['Playoff_Status'] == 'Champion'][metric].dropna()
            lottery = df[df['Playoff_Status'] == 'Lottery'][metric].dropna()
            if len(champ) > 2 and len(lottery) > 10:
                stat, p = stats.mannwhitneyu(champ, lottery, alternative='two-sided')
                diff = champ.mean() - lottery.mean()
                ax.set_xlabel(f'Champion vs Lottery: diff={diff:+.1f}, p={p:.3f}')
    
    plt.suptitle('NETWORK METRICS BY PLAYOFF SUCCESS\n(Regular Season Metrics → Playoff Outcome)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_metrics_by_playoff_status.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_metrics_by_playoff_status.png")
    
    # ===========================
    # FIGURE 2: Champions Profile
    # ===========================
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    
    champ_df = df[df['Playoff_Status'] == 'Champion'].copy()
    other_df = df[df['Playoff_Status'] != 'Champion'].copy()
    
    # Metrics comparison
    compare_metrics = ['Std_Weighted_Degree', 'Max_Weighted_Degree', 'Pass_Entropy']
    
    for idx, metric in enumerate(compare_metrics):
        ax = axes2[idx]
        
        if metric in df.columns:
            champ_vals = champ_df[metric].dropna()
            other_vals = other_df[metric].dropna()
            
            positions = [1, 2]
            bp = ax.boxplot([other_vals, champ_vals], positions=positions, patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('lightgray')
            bp['boxes'][1].set_facecolor('gold')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(['All Other Teams', 'Champions'])
            ax.set_ylabel(metric.replace('_', ' '))
            
            # Effect size (Cohen's d)
            if len(champ_vals) > 0 and len(other_vals) > 0:
                pooled_std = np.sqrt(((len(champ_vals)-1)*champ_vals.std()**2 + 
                                       (len(other_vals)-1)*other_vals.std()**2) / 
                                      (len(champ_vals) + len(other_vals) - 2))
                cohens_d = (champ_vals.mean() - other_vals.mean()) / pooled_std if pooled_std > 0 else 0
                ax.set_title(f'{metric.replace("_", " ")}\nCohen\'s d = {cohens_d:.2f}', fontweight='bold')
    
    plt.suptitle('CHAMPIONS vs ALL OTHER TEAMS\n(What Makes Championship Networks Different?)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_champions_vs_others.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_champions_vs_others.png")
    
    # ===========================
    # FIGURE 3: Playoff Depth Correlations
    # ===========================
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    
    depth_metrics = ['Std_Weighted_Degree', 'Max_Weighted_Degree', 'Pass_Entropy']
    
    for idx, metric in enumerate(depth_metrics):
        ax = axes3[idx]
        
        if metric in df.columns:
            valid = df[[metric, 'Playoff_Depth']].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[metric], valid['Playoff_Depth'])
                
                # Jitter for visibility
                jitter = np.random.normal(0, 0.1, len(valid))
                ax.scatter(valid[metric], valid['Playoff_Depth'] + jitter, alpha=0.4, s=30)
                
                # Regression line
                z = np.polyfit(valid[metric], valid['Playoff_Depth'], 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
                ax.plot(x_line, p_line(x_line), 'r-', linewidth=2)
                
                ax.set_xlabel(metric.replace('_', ' '))
                ax.set_ylabel('Playoff Depth (1=Lottery, 5=Champion)')
                ax.set_title(f'{metric.replace("_", " ")}\nr = {r:.3f}, p = {p:.4f}', fontweight='bold')
    
    plt.suptitle('REGULAR SEASON METRICS vs PLAYOFF SUCCESS\n(Correlation with Playoff Depth)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_playoff_depth_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_playoff_depth_correlations.png")
    
    return df


def print_summary(df):
    """Print analysis summary."""
    df = df[df['Win_Pct'].notna()].copy()
    
    print("\n" + "="*90)
    print("PLAYOFF vs REGULAR SEASON ANALYSIS - SUMMARY")
    print("="*90)
    
    print("\n[METHODOLOGY NOTE]")
    print("-"*90)
    print("  The NBA API does not provide playoff-specific passing data.")
    print("  This analysis compares REGULAR SEASON network metrics by PLAYOFF OUTCOME.")
    print("  We examine: Do teams with certain network structures during the regular season")
    print("             tend to have more playoff success?")
    
    print("\n[SAMPLE SIZES]")
    print("-"*90)
    for status in ['Champion', 'Finals', 'Conf_Finals', 'Playoff', 'Lottery']:
        count = len(df[df['Playoff_Status'] == status])
        print(f"  {status:<15}: {count} team-seasons")
    
    print("\n[KEY FINDINGS: Champions vs Lottery Teams]")
    print("-"*90)
    
    champ = df[df['Playoff_Status'] == 'Champion']
    lottery = df[df['Playoff_Status'] == 'Lottery']
    
    metrics = ['Std_Weighted_Degree', 'Max_Weighted_Degree', 'Pass_Entropy', 'Graph_Density']
    
    for metric in metrics:
        if metric in df.columns:
            c_mean = champ[metric].mean()
            l_mean = lottery[metric].mean()
            diff = c_mean - l_mean
            pct_diff = (c_mean - l_mean) / l_mean * 100 if l_mean != 0 else 0
            
            # Statistical test
            if len(champ[metric].dropna()) > 2 and len(lottery[metric].dropna()) > 10:
                stat, p = stats.mannwhitneyu(champ[metric].dropna(), lottery[metric].dropna())
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            else:
                sig = 'n/a'
            
            print(f"\n  {metric}:")
            print(f"    Champions:  {c_mean:.2f}")
            print(f"    Lottery:    {l_mean:.2f}")
            print(f"    Difference: {diff:+.2f} ({pct_diff:+.1f}%) {sig}")
    
    print("\n[PLAYOFF DEPTH CORRELATIONS]")
    print("-"*90)
    
    for metric in metrics:
        if metric in df.columns:
            valid = df[[metric, 'Playoff_Depth']].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[metric], valid['Playoff_Depth'])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"  {metric:<30} r = {r:+.3f} {sig}")
    
    print("\n[PRESENTATION-READY STATEMENTS]")
    print("-"*90)
    
    # Calculate actual values
    hier_champ = champ['Std_Weighted_Degree'].mean() if 'Std_Weighted_Degree' in champ.columns else 0
    hier_lottery = lottery['Std_Weighted_Degree'].mean() if 'Std_Weighted_Degree' in lottery.columns else 0
    
    if hier_champ > hier_lottery:
        print(f"\n  'Championship teams show {((hier_champ/hier_lottery)-1)*100:.0f}% higher hierarchy")
        print(f"   (Std of Weighted Degree) compared to lottery teams.'")
    
    print("\n  NOTE: This is a PROXY analysis using regular season data.")
    print("        Actual playoff passing data is not available via NBA API.")
    
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*70)
    print("PLAYOFF vs REGULAR SEASON ANALYSIS")
    print("(Using Regular Season Metrics to Predict Playoff Success)")
    print("="*70)
    
    print("\n[LOADING DATA]")
    df = load_complete_metrics()
    print(f"Loaded {len(df)} team-seasons")
    
    print("\n[CATEGORIZING PLAYOFF STATUS]")
    df = categorize_playoff_status(df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    df = analyze_and_plot(df)
    
    print_summary(df)
    
    df.to_csv(OUTPUT_DIR / 'playoff_comparison_data.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
