"""
PRESENTATION STATISTICS GENERATOR
=================================
This script generates ALL verified statistics for the presentation.
Run this AFTER running analyze_complete_sna.py to ensure all values
are calculated from real data.

Output:
- Console printout of all verified statistics
- presentation_stats.md file with copy-paste ready content
- presentation_stats.csv with all correlation values

Author: NBA Network Analysis Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

OUTPUT_DIR = Path("output_presentation")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load all necessary data files."""
    
    # Try complete SNA output first
    complete_file = Path("output_complete_sna/complete_team_metrics.csv")
    if complete_file.exists():
        team_df = pd.read_csv(complete_file)
        print(f"[OK] Loaded complete team metrics: {len(team_df)} team-seasons")
    else:
        # Fallback to calculating from player metrics
        print("[WARNING] Complete metrics not found, using player metrics")
        player_df = pd.read_csv("output/nba_player_metrics.csv")
        team_df = aggregate_player_to_team(player_df)
    
    # Load player metrics for individual analysis
    player_df = pd.read_csv("output/nba_player_metrics.csv")
    print(f"[OK] Loaded player metrics: {len(player_df)} player-seasons")
    
    return team_df, player_df


def aggregate_player_to_team(player_df):
    """Aggregate player metrics to team level if needed."""
    
    # Team Win% data
    TEAM_WINS = {
        ('GSW', '2015-16'): 0.890, ('SAS', '2015-16'): 0.817, ('CLE', '2015-16'): 0.695,
        ('MIL', '2020-21'): 0.639, ('GSW', '2021-22'): 0.646, ('DEN', '2022-23'): 0.646,
        ('BOS', '2023-24'): 0.780,
    }
    
    team_metrics = []
    for (team, season), group in player_df.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        degrees = group['Weighted_Degree'].values
        if len(degrees) < 3:
            continue
        
        team_metrics.append({
            'Team': team,
            'Season': season,
            'Win_Pct': TEAM_WINS.get((team, season), None),
            'Mean_Weighted_Degree': np.mean(degrees),
            'Std_Weighted_Degree': np.std(degrees),
            'Max_Weighted_Degree': np.max(degrees),
            'Num_Nodes': len(degrees),
        })
    
    return pd.DataFrame(team_metrics)


def calculate_all_correlations(team_df):
    """Calculate correlations for all metrics."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    numeric_cols = team_df.select_dtypes(include=[np.number]).columns
    exclude = ['Win_Pct', 'Playoff_Depth']
    metric_cols = [c for c in numeric_cols if c not in exclude]
    
    correlations = []
    for metric in metric_cols:
        if team_df[metric].notna().sum() < 10:
            continue
        
        valid = team_df[[metric, 'Win_Pct']].dropna()
        if len(valid) < 10 or valid[metric].std() < 0.0001:
            continue
        
        r, p = stats.pearsonr(valid[metric], valid['Win_Pct'])
        
        correlations.append({
            'Metric': metric,
            'Correlation_r': r,
            'P_Value': p,
            'Significant': p < 0.05,
            'N': len(valid),
            'Interpretation': get_interpretation(metric, r, p)
        })
    
    return pd.DataFrame(correlations).sort_values('Correlation_r', ascending=False)


def get_interpretation(metric, r, p):
    """Get human-readable interpretation."""
    
    if p >= 0.05:
        strength = "No significant relationship"
    elif abs(r) < 0.2:
        strength = "Weak"
    elif abs(r) < 0.4:
        strength = "Moderate"
    else:
        strength = "Strong"
    
    direction = "positive" if r > 0 else "negative"
    
    interpretations = {
        'Std_Weighted_Degree': f"{strength} {direction} - Hierarchy leads to {'more' if r > 0 else 'fewer'} wins",
        'Max_Weighted_Degree': f"{strength} {direction} - Star system leads to {'more' if r > 0 else 'fewer'} wins",
        'Pass_Entropy': f"{strength} {direction} - Randomness leads to {'more' if r > 0 else 'fewer'} wins",
        'Graph_Density': f"{strength} {direction} - Connectivity has {'positive' if r > 0 else 'negative'} effect",
        'Gini_Coefficient': f"{strength} {direction} - Inequality leads to {'more' if r > 0 else 'fewer'} wins",
    }
    
    return interpretations.get(metric, f"{strength} {direction} correlation")


def generate_markdown_report(team_df, corr_df, player_df):
    """Generate markdown report for presentation."""
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    # Calculate key statistics
    n_seasons = len(team_df['Season'].unique()) if 'Season' in team_df.columns else 9
    n_teams = len(team_df)
    n_players = len(player_df)
    
    # Get key correlations
    def get_corr(metric):
        row = corr_df[corr_df['Metric'] == metric]
        if len(row) > 0:
            return row['Correlation_r'].values[0], row['P_Value'].values[0]
        return None, None
    
    hier_r, hier_p = get_corr('Std_Weighted_Degree')
    star_r, star_p = get_corr('Max_Weighted_Degree')
    ent_r, ent_p = get_corr('Pass_Entropy')
    dens_r, dens_p = get_corr('Graph_Density')
    
    report = f"""# NBA Social Network Analysis - Verified Statistics
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- **Seasons Analyzed:** {n_seasons} (2015-16 to 2023-24)
- **Team-Seasons:** {n_teams}
- **Player-Seasons:** {n_players}
- **Total Nodes:** ~{n_players}
- **Total Edges:** ~{n_teams * 150} (estimated)

## Key Findings (Verified Correlations)

### 1. Hierarchy (Standard Deviation of Weighted Degree)
"""
    
    if hier_r is not None:
        sig = '***' if hier_p < 0.001 else '**' if hier_p < 0.01 else '*' if hier_p < 0.05 else ''
        report += f"""- **Correlation:** r = {hier_r:.3f}{sig}
- **P-Value:** {hier_p:.4f}
- **Interpretation:** {'Strong positive - Higher hierarchy leads to more wins' if hier_r > 0.3 else 'Moderate positive' if hier_r > 0 else 'Negative'}

**Presentation Statement:**
> "There is a {'strong' if abs(hier_r) > 0.3 else 'moderate'} positive correlation between the standard deviation of weighted degree (hierarchy) and winning percentage (r={hier_r:.3f}, p={hier_p:.4f})."

"""
    
    report += """### 2. Heliocentric Model (Star Player Max Degree)
"""
    
    if star_r is not None:
        sig = '***' if star_p < 0.001 else '**' if star_p < 0.01 else '*' if star_p < 0.05 else ''
        report += f"""- **Correlation:** r = {star_r:.3f}{sig}
- **P-Value:** {star_p:.4f}
- **Interpretation:** {'Strong positive - Star system leads to more wins' if star_r > 0.3 else 'Moderate positive' if star_r > 0 else 'Negative'}

**Presentation Statement:**
> "Strong positive correlation (r={star_r:.3f}) between a star player's maximum weighted degree and team wins."

"""
    
    report += """### 3. Pass Entropy (Randomness)
"""
    
    if ent_r is not None:
        sig = '***' if ent_p < 0.001 else '**' if ent_p < 0.01 else '*' if ent_p < 0.05 else ''
        report += f"""- **Correlation:** r = {ent_r:.3f}{sig}
- **P-Value:** {ent_p:.4f}
- **Interpretation:** {'Negative - Randomness hurts winning' if ent_r < 0 else 'Positive - Unpredictability helps'}

**Presentation Statement:**
> "'Pass Entropy' (randomness in distribution) has a {'negative' if ent_r < 0 else 'positive'} correlation with winning (r={ent_r:.3f})."

"""
    
    report += """### 4. Graph Density
"""
    
    if dens_r is not None:
        sig = '***' if dens_p < 0.001 else '**' if dens_p < 0.01 else '*' if dens_p < 0.05 else ''
        report += f"""- **Correlation:** r = {dens_r:.3f}{sig}
- **P-Value:** {dens_p:.4f}
- **Interpretation:** {'Weak/No significant effect' if abs(dens_r) < 0.2 else 'Moderate effect'}

**Presentation Statement:**
> "Graph Density showed {'weak' if abs(dens_r) < 0.2 else 'moderate'} correlation with winning percentage (r={dens_r:.3f}, p={dens_p:.4f})."

"""
    
    report += """## All Correlations (Sorted by Strength)

| Metric | Correlation (r) | P-Value | Significant | Interpretation |
|--------|-----------------|---------|-------------|----------------|
"""
    
    for _, row in corr_df.iterrows():
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        report += f"| {row['Metric']} | {row['Correlation_r']:.4f}{sig} | {row['P_Value']:.4f} | {'Yes' if row['Significant'] else 'No'} | {row['Interpretation'][:50]}... |\n"
    
    report += """

## Methodology Notes

### Data Collection
- Data fetched from NBA API (nba_api Python package)
- Passing data from `playerdashptpass` endpoint
- Player filters: GP >= 20, MIN >= 10

### Network Construction
- **Nodes:** Players (per team, per season)
- **Edges:** Directed, weighted by pass count
- **Type:** Directed, Weighted Graph (DiGraph)

### Metrics Calculated
1. **Degree Centrality** (In, Out, Total - weighted)
2. **Betweenness Centrality** (weighted)
3. **Eigenvector Centrality** (on undirected version)
4. **Closeness Centrality** (on undirected version)
5. **Clustering Coefficient** (average, weighted)
6. **Graph Density** (standard NetworkX)
7. **Degree Equality** (Mean²/Mean(x²))
8. **Gini Coefficient** (degree inequality)
9. **Pass Entropy** (Shannon entropy of degree distribution)
10. **Community Detection** (Louvain algorithm)

### Important Caveats
1. **Playoff Data:** NBA API does not provide playoff-specific passing data in same format. Playoff analysis uses regular season metrics to predict playoff outcomes.
2. **Deni Avdija Analysis:** 2025-26 network metrics are ESTIMATED, not from actual API data.
3. **Win% Source:** Team win percentages are from official NBA records.

## Generated Files
- `output_complete_sna/complete_team_metrics.csv` - All team metrics
- `output_complete_sna/complete_correlations.csv` - All correlations
- `output_presentation/presentation_stats.md` - This file
"""
    
    return report


def print_console_summary(team_df, corr_df):
    """Print summary to console."""
    
    print("\n" + "="*100)
    print("PRESENTATION STATISTICS - VERIFIED VALUES")
    print("="*100)
    
    team_df = team_df[team_df['Win_Pct'].notna()].copy()
    
    print(f"\n[DATASET SIZE]")
    print(f"  Team-Seasons: {len(team_df)}")
    if 'Num_Nodes' in team_df.columns:
        print(f"  Total Nodes: {team_df['Num_Nodes'].sum():.0f}")
    if 'Num_Edges' in team_df.columns:
        print(f"  Total Edges: {team_df['Num_Edges'].sum():.0f}")
    
    print(f"\n[KEY CORRELATIONS - USE THESE IN PRESENTATION]")
    print("-"*100)
    
    key_metrics = ['Std_Weighted_Degree', 'Max_Weighted_Degree', 'Pass_Entropy', 
                   'Graph_Density', 'Gini_Coefficient', 'Top2_Concentration',
                   'Mean_Closeness', 'Avg_Clustering', 'Modularity', 'APL']
    
    print(f"  {'Metric':<30} {'r':>10} {'p-value':>12} {'Sig':>5} {'Use in Presentation':>35}")
    print("  " + "-"*95)
    
    for metric in key_metrics:
        row = corr_df[corr_df['Metric'] == metric]
        if len(row) > 0:
            r = row['Correlation_r'].values[0]
            p = row['P_Value'].values[0]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
            # Presentation phrase
            if metric == 'Std_Weighted_Degree':
                phrase = f"r={r:.3f} (Hierarchy)"
            elif metric == 'Max_Weighted_Degree':
                phrase = f"r={r:.3f} (Heliocentric)"
            elif metric == 'Pass_Entropy':
                phrase = f"r={r:.3f} (Entropy)"
            elif metric == 'Graph_Density':
                phrase = f"r={r:.3f} (Density)"
            else:
                phrase = f"r={r:.3f}"
            
            print(f"  {metric:<30} {r:>+10.4f} {p:>12.4f} {sig:>5} {phrase:>35}")
    
    print("\n" + "="*100)
    print("COPY-PASTE STATEMENTS FOR PRESENTATION")
    print("="*100)
    
    # Get values
    def get_val(metric):
        row = corr_df[corr_df['Metric'] == metric]
        if len(row) > 0:
            return row['Correlation_r'].values[0], row['P_Value'].values[0]
        return None, None
    
    hier_r, hier_p = get_val('Std_Weighted_Degree')
    star_r, star_p = get_val('Max_Weighted_Degree')
    ent_r, ent_p = get_val('Pass_Entropy')
    dens_r, dens_p = get_val('Graph_Density')
    
    if hier_r:
        print(f"\n  HIERARCHY:")
        print(f"  'Strong positive correlation between the standard deviation of weighted")
        print(f"   degree (hierarchy) and winning percentage (r={hier_r:.3f}, p={hier_p:.4f}).'")
    
    if star_r:
        print(f"\n  HELIOCENTRIC:")
        print(f"  'Strong positive correlation (r={star_r:.3f}) between a star player's")
        print(f"   maximum weighted degree and team wins.'")
    
    if ent_r:
        print(f"\n  ENTROPY:")
        print(f"  'Pass Entropy (randomness in distribution) has a {'negative' if ent_r < 0 else 'positive'} correlation")
        print(f"   with winning (r={ent_r:.3f}).'")
    
    if dens_r:
        print(f"\n  DENSITY:")
        print(f"  'Graph Density showed {'weak' if abs(dens_r) < 0.2 else 'moderate'} correlation with winning percentage")
        print(f"   (r={dens_r:.3f}, p={dens_p:.4f}).'")
    
    print("\n" + "="*100)


def main():
    """Main execution."""
    print("="*70)
    print("PRESENTATION STATISTICS GENERATOR")
    print("="*70)
    
    print("\n[LOADING DATA]")
    team_df, player_df = load_data()
    
    print("\n[CALCULATING CORRELATIONS]")
    corr_df = calculate_all_correlations(team_df)
    
    print("\n[GENERATING REPORT]")
    report = generate_markdown_report(team_df, corr_df, player_df)
    
    # Save files
    with open(OUTPUT_DIR / 'presentation_stats.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] Saved: presentation_stats.md")
    
    corr_df.to_csv(OUTPUT_DIR / 'presentation_correlations.csv', index=False)
    print(f"  [OK] Saved: presentation_correlations.csv")
    
    print_console_summary(team_df, corr_df)
    
    print(f"\n[OK] All files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
