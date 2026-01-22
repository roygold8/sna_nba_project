"""
2025-26 NBA CHAMPIONSHIP PREDICTION - REAL DATA
=================================================
Uses actual 2025-26 season data fetched from NBA API.
Data as of: January 21, 2026

Key Metrics:
- Pass Entropy (LOW = ordered offense)
- Hierarchy (Std of Weighted Degree)
- Top 3-4 Player Concentration
- Average Degree (Ball Movement)
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

try:
    from unidecode import unidecode
except ImportError:
    unidecode = lambda x: x

plt.style.use('seaborn-v0_8-whitegrid')

# Directories
DATA_DIR = Path("data/2025-26")
OUTPUT_DIR = Path("output_2025_26_prediction")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_standings():
    """Load team standings."""
    standings_path = DATA_DIR / 'standings.csv'
    if standings_path.exists():
        df = pd.read_csv(standings_path)
        print(f"[OK] Loaded standings for {len(df)} teams")
        return df
    else:
        print("[ERROR] Standings file not found")
        return None


def load_players():
    """Load filtered players."""
    players_path = DATA_DIR / 'filtered_players.csv'
    if players_path.exists():
        df = pd.read_csv(players_path)
        print(f"[OK] Loaded {len(df)} players")
        return df
    else:
        print("[ERROR] Players file not found")
        return None


def build_team_graphs(players_df):
    """Build passing network graphs for each team."""
    print("\n[BUILDING TEAM NETWORKS]")
    
    team_graphs = {}
    team_players = defaultdict(list)
    
    # Group players by team
    for _, player in players_df.iterrows():
        team_id = player['TEAM_ID']
        team_abbr = player['TEAM_ABBREVIATION']
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER_NAME']
        
        team_players[team_abbr].append({
            'id': player_id,
            'name': player_name,
            'team_id': team_id
        })
    
    # Build graph for each team
    for team_abbr, players in team_players.items():
        G = nx.DiGraph()
        
        # Add nodes from filtered players
        player_ids = set()
        for p in players:
            G.add_node(p['id'], name=p['name'])
            player_ids.add(p['id'])
        
        # Load passing data for each player and add ALL teammates as nodes
        for p in players:
            passing_file = DATA_DIR / f"passing_{p['id']}.json"
            
            if not passing_file.exists():
                continue
            
            try:
                with open(passing_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get passes made data (first result set)
                if 'resultSets' in data and len(data['resultSets']) > 0:
                    passes_made = data['resultSets'][0]
                    headers = passes_made.get('headers', [])
                    rows = passes_made.get('rowSet', [])
                    
                    if 'PASS' in headers and 'PASS_TEAMMATE_PLAYER_ID' in headers:
                        pass_idx = headers.index('PASS')
                        pass_to_id_idx = headers.index('PASS_TEAMMATE_PLAYER_ID')
                        pass_to_name_idx = headers.index('PASS_TO') if 'PASS_TO' in headers else None
                        
                        for row in rows:
                            passes = row[pass_idx]
                            receiver_id = row[pass_to_id_idx]
                            receiver_name = row[pass_to_name_idx] if pass_to_name_idx is not None else 'Unknown'
                            
                            # Add receiver as node if not exists
                            if receiver_id not in G.nodes():
                                G.add_node(receiver_id, name=receiver_name)
                            
                            # Add or update edge
                            if G.has_edge(p['id'], receiver_id):
                                G[p['id']][receiver_id]['weight'] += passes
                            else:
                                G.add_edge(p['id'], receiver_id, weight=passes)
            
            except Exception as e:
                pass
        
        team_graphs[team_abbr] = G
    
    print(f"  [OK] Built networks for {len(team_graphs)} teams")
    return team_graphs


def calculate_team_metrics(team_graphs, standings_df, players_df):
    """Calculate network metrics for each team."""
    print("\n[CALCULATING TEAM METRICS]")
    
    # Build team abbreviation to ID mapping
    team_abbr_to_id = players_df.drop_duplicates('TEAM_ID')[['TEAM_ID', 'TEAM_ABBREVIATION']].set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()
    
    team_metrics = []
    
    for team_abbr, G in team_graphs.items():
        if G.number_of_nodes() < 3 or G.number_of_edges() < 3:
            continue
        
        # Get standings info by matching TeamID
        team_id = team_abbr_to_id.get(team_abbr)
        if team_id:
            team_standings = standings_df[standings_df['TeamID'] == team_id]
        else:
            team_standings = pd.DataFrame()
        
        if team_standings.empty:
            wins = 0
            losses = 0
            win_pct = 0.5
        else:
            wins = team_standings.iloc[0].get('WINS', 0)
            losses = team_standings.iloc[0].get('LOSSES', 0)
            win_pct = team_standings.iloc[0].get('WinPCT', 0.5)
        
        # Basic metrics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Weighted degree calculations
        weighted_degrees = []
        for node in G.nodes():
            in_deg = sum(d.get('weight', 1) for u, v, d in G.in_edges(node, data=True))
            out_deg = sum(d.get('weight', 1) for u, v, d in G.out_edges(node, data=True))
            weighted_degrees.append(in_deg + out_deg)
        
        weighted_degrees = np.array(weighted_degrees)
        
        if len(weighted_degrees) == 0:
            continue
        
        # Key metrics
        avg_degree = np.mean(weighted_degrees)
        std_degree = np.std(weighted_degrees)
        max_degree = np.max(weighted_degrees)
        
        # Top 3-4 concentration
        sorted_degrees = np.sort(weighted_degrees)[::-1]
        total_degree = np.sum(weighted_degrees)
        
        if total_degree > 0:
            top3_conc = np.sum(sorted_degrees[:3]) / total_degree if len(sorted_degrees) >= 3 else 0
            top4_conc = np.sum(sorted_degrees[:4]) / total_degree if len(sorted_degrees) >= 4 else 0
        else:
            top3_conc = 0
            top4_conc = 0
        
        # Pass Entropy (Shannon entropy of degree distribution)
        if total_degree > 0:
            probs = weighted_degrees / total_degree
            probs = probs[probs > 0]  # Remove zeros
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
            # Normalize entropy by max possible entropy
            max_entropy = np.log2(len(weighted_degrees)) if len(weighted_degrees) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 1.0
        
        # Density
        density = nx.density(G)
        
        # Get star player
        node_degrees = {node: deg for node, deg in zip(G.nodes(), weighted_degrees)}
        star_node = max(node_degrees, key=node_degrees.get) if node_degrees else None
        star_name = G.nodes[star_node].get('name', 'Unknown') if star_node else 'Unknown'
        
        # Second star
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        second_star_name = G.nodes[sorted_nodes[1][0]].get('name', 'Unknown') if len(sorted_nodes) > 1 else 'Unknown'
        
        # Duo avg degree
        duo_avg = np.mean(sorted_degrees[:2]) if len(sorted_degrees) >= 2 else 0
        
        team_metrics.append({
            'Team': team_abbr,
            'Wins': wins,
            'Losses': losses,
            'Win_Pct': win_pct,
            'Nodes': n_nodes,
            'Edges': n_edges,
            'Avg_Degree': avg_degree,
            'Std_Degree': std_degree,
            'Max_Degree': max_degree,
            'Top3_Concentration': top3_conc,
            'Top4_Concentration': top4_conc,
            'Pass_Entropy': normalized_entropy,
            'Density': density,
            'Star_Player': star_name,
            'Second_Star': second_star_name,
            'Duo_Avg_Degree': duo_avg,
        })
    
    df = pd.DataFrame(team_metrics)
    print(f"  [OK] Calculated metrics for {len(df)} teams")
    return df


def calculate_championship_score(df):
    """Calculate championship prediction scores."""
    print("\n[CALCULATING CHAMPIONSHIP SCORES]")
    
    df = df.copy()
    
    # Normalize to 0-100 scale
    def normalize(col, invert=False):
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            normalized = (df[col] - min_val) / (max_val - min_val) * 100
            if invert:
                normalized = 100 - normalized
            return normalized
        return 50
    
    # Score components based on research findings
    df['Entropy_Score'] = normalize('Pass_Entropy', invert=True)  # Lower = better
    df['Hierarchy_Score'] = normalize('Std_Degree')  # Higher = better
    df['Concentration_Score'] = normalize('Top4_Concentration')  # Higher = better
    df['Degree_Score'] = normalize('Avg_Degree')  # Higher = better
    df['Star_Score'] = normalize('Max_Degree')  # Higher = better
    df['Win_Score'] = normalize('Win_Pct')  # Current performance
    
    # Championship Score (weighted combination)
    # Updated weights - heavier on current Win% since that reflects actual success
    # Network metrics serve as structural indicators but Win% is the ultimate measure
    # - Current Win%: 40% (strongest indicator of championship potential)
    # - Hierarchy (Std Degree): 15%
    # - Star Max Degree: 15%
    # - Core Concentration: 15%
    # - Low Entropy: 10%
    # - Ball Movement: 5%
    
    df['Championship_Score'] = (
        0.40 * df['Win_Score'] +          # Win% is the strongest indicator
        0.15 * df['Hierarchy_Score'] +
        0.15 * df['Star_Score'] +
        0.15 * df['Concentration_Score'] +
        0.10 * df['Entropy_Score'] +
        0.05 * df['Degree_Score']
    )
    
    return df.sort_values('Championship_Score', ascending=False)


def plot_championship_prediction(df):
    """Create championship prediction visualizations."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # ===========================
    # 1. TOP 10 CHAMPIONSHIP FAVORITES
    # ===========================
    ax1 = fig.add_subplot(2, 2, 1)
    
    top_10 = df.head(10).sort_values('Championship_Score', ascending=True)
    
    # Color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.95, len(top_10)))[::-1]
    
    bars = ax1.barh(top_10['Team'], top_10['Championship_Score'], color=colors, edgecolor='black', linewidth=1.5)
    
    # Add score and win% labels
    for bar, (_, row) in zip(bars, top_10.iterrows()):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{row["Championship_Score"]:.1f}', va='center', fontsize=11, fontweight='bold')
        ax1.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2, 
                f'{row["Win_Pct"]:.1%}', va='center', ha='right', fontsize=9, color='white', fontweight='bold')
    
    ax1.set_xlabel('Championship Prediction Score', fontsize=12, fontweight='bold')
    ax1.set_title('2025-26 NBA CHAMPIONSHIP FAVORITES\nBased on Real Network Analysis (as of Jan 21, 2026)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 105)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ===========================
    # 2. KEY METRICS BREAKDOWN (Top 5)
    # ===========================
    ax2 = fig.add_subplot(2, 2, 2)
    
    top_5 = df.head(5)
    
    metrics = ['Entropy_Score', 'Hierarchy_Score', 'Concentration_Score', 'Star_Score']
    labels = ['Order\n(Low Entropy)', 'Hierarchy\n(Std Degree)', 'Core Focus\n(Top 4 Conc.)', 'Star Power\n(Max Degree)']
    
    x = np.arange(len(top_5))
    width = 0.2
    colors_metrics = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors_metrics)):
        values = top_5[metric].values
        ax2.bar(x + i*width, values, width, label=label, color=color, alpha=0.85, edgecolor='black')
    
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(top_5['Team'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score (0-100)', fontsize=12)
    ax2.set_title('TOP 5 CONTENDERS: Key Metric Breakdown', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===========================
    # 3. ENTROPY vs HIERARCHY SCATTER
    # ===========================
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Scatter plot with size = score
    scatter = ax3.scatter(df['Pass_Entropy'], df['Std_Degree'], 
                          s=df['Championship_Score'] * 3, c=df['Championship_Score'], 
                          cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidths=1)
    
    # Annotate top 8 teams
    for _, row in df.head(8).iterrows():
        ax3.annotate(row['Team'], (row['Pass_Entropy'], row['Std_Degree']),
                    fontsize=10, fontweight='bold', xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Pass Entropy (Lower = More Ordered)', fontsize=12)
    ax3.set_ylabel('Std of Weighted Degree (Higher = More Hierarchical)', fontsize=12)
    ax3.set_title('THE WINNING ZONE: Low Entropy + High Hierarchy', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Championship Score')
    
    # ===========================
    # 4. WIN% vs CHAMPIONSHIP SCORE
    # ===========================
    ax4 = fig.add_subplot(2, 2, 4)
    
    scatter4 = ax4.scatter(df['Win_Pct'], df['Championship_Score'], 
                           s=100, c=df['Hierarchy_Score'], cmap='viridis', 
                           alpha=0.7, edgecolors='black', linewidths=1)
    
    # Annotate all teams
    for _, row in df.iterrows():
        ax4.annotate(row['Team'], (row['Win_Pct'], row['Championship_Score']),
                    fontsize=8, xytext=(3, 3), textcoords='offset points')
    
    # Correlation line
    z = np.polyfit(df['Win_Pct'], df['Championship_Score'], 1)
    p = np.poly1d(z)
    ax4.plot(df['Win_Pct'].sort_values(), p(df['Win_Pct'].sort_values()), 
             "r--", alpha=0.5, label=f'Trend')
    
    # Correlation
    corr, pval = stats.pearsonr(df['Win_Pct'], df['Championship_Score'])
    ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top')
    
    ax4.set_xlabel('Current Win Percentage', fontsize=12)
    ax4.set_ylabel('Championship Prediction Score', fontsize=12)
    ax4.set_title('Network Structure vs Current Performance', fontsize=13, fontweight='bold')
    
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Hierarchy Score')
    
    plt.suptitle('2025-26 NBA CHAMPIONSHIP PREDICTION\nBased on Real Passing Network Analysis\n'
                 'Key Factors: Entropy (Order) | Hierarchy (Std) | Core Concentration | Star Power',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_championship_prediction_real.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_championship_prediction_real.png")


def plot_top_5_analysis(df):
    """Create detailed analysis of top 5 teams."""
    
    fig = plt.figure(figsize=(18, 10))
    
    top_5 = df.head(5)
    
    # ===========================
    # 1. Radar Chart Comparison
    # ===========================
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    
    metrics = ['Entropy_Score', 'Hierarchy_Score', 'Concentration_Score', 'Star_Score', 'Win_Score']
    labels = ['Order\n(Low Entropy)', 'Hierarchy', 'Core Focus', 'Star Power', 'Current Win%']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, color=colors_radar[i], label=row['Team'])
        ax1.fill(angles, values, alpha=0.1, color=colors_radar[i])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title('TOP 5 CONTENDERS COMPARISON', fontsize=13, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # ===========================
    # 2. Championship DNA Table
    # ===========================
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Build table data
    table_data = []
    for _, row in top_5.iterrows():
        try:
            star1 = unidecode(row['Star_Player'].split()[-1]) if pd.notna(row['Star_Player']) else 'N/A'
            star2 = unidecode(row['Second_Star'].split()[-1]) if pd.notna(row['Second_Star']) else 'N/A'
            duo = f"{star1} & {star2}"
        except:
            duo = "N/A"
        
        table_data.append([
            row['Team'],
            f"{row['Championship_Score']:.1f}",
            f"{row['Pass_Entropy']:.3f}",
            f"{row['Std_Degree']:.0f}",
            f"{row['Top4_Concentration']:.1%}",
            duo
        ])
    
    columns = ['Team', 'Score', 'Entropy\n(Low=Good)', 'Hierarchy\n(High=Good)', 
               'Core Conc.\n(High=Good)', 'Dynamic Duo']
    
    table = ax2.table(cellText=table_data,
                      colLabels=columns,
                      cellLoc='center',
                      loc='center',
                      colColours=['#3498db']*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.2)
    
    # Color rows
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if i == 0:  # Top team
                table[(i+1, j)].set_facecolor('#d4edda')
            elif i < 3:
                table[(i+1, j)].set_facecolor('#fff3cd')
    
    ax2.set_title('2025-26 CHAMPIONSHIP DNA\nTop 5 Teams Profile', fontsize=14, fontweight='bold', pad=40)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_top5_analysis_real.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_top5_analysis_real.png")


def print_prediction(df):
    """Print championship prediction summary."""
    
    print("\n" + "="*100)
    print("[CHAMPIONSHIP] 2025-26 NBA CHAMPIONSHIP PREDICTION - REAL DATA")
    print("Based on Actual Passing Networks (as of January 21, 2026)")
    print("="*100)
    
    print("\n[MODEL FACTORS]")
    print("-"*100)
    print("  1. Hierarchy (Std Degree): 25% - Structured, star-driven systems win")
    print("  2. Star Power (Max Degree): 20% - Heliocentric star dominance")
    print("  3. Core Concentration (Top 4): 20% - Core-focused teams win")
    print("  4. Order (Low Entropy): 15% - Predictable, efficient offense")
    print("  5. Ball Movement (Avg Degree): 10% - Passing volume")
    print("  6. Current Win%: 10% - Reality check")
    
    print("\n[TOP 10 CHAMPIONSHIP CONTENDERS]")
    print("-"*100)
    print(f"  {'Rank':<5} {'Team':<6} {'Score':<8} {'Win%':<8} {'Record':<12} {'Star Duo':<40}")
    print("  " + "-"*95)
    
    for rank, (_, row) in enumerate(df.head(10).iterrows(), 1):
        try:
            star1 = unidecode(str(row['Star_Player']).split()[-1]) if pd.notna(row['Star_Player']) else 'N/A'
            star2 = unidecode(str(row['Second_Star']).split()[-1]) if pd.notna(row['Second_Star']) else 'N/A'
            duo = f"{star1} & {star2}"
        except:
            duo = "N/A"
        
        record = f"{int(row['Wins'])}-{int(row['Losses'])}"
        print(f"  {rank:<5} {row['Team']:<6} {row['Championship_Score']:<8.1f} {row['Win_Pct']:<8.1%} {record:<12} {duo:<40}")
    
    # Top team deep dive
    top = df.iloc[0]
    print(f"\n[#1 CHAMPIONSHIP FAVORITE: {top['Team']}]")
    print("-"*100)
    print(f"  Championship Score: {top['Championship_Score']:.1f}")
    print(f"  Current Record: {int(top['Wins'])}-{int(top['Losses'])} ({top['Win_Pct']:.1%})")
    print(f"\n  Network Profile:")
    print(f"    - Entropy: {top['Pass_Entropy']:.3f} (Lower = More Ordered)")
    print(f"    - Hierarchy (Std): {top['Std_Degree']:.0f} (Higher = More Star-Driven)")
    print(f"    - Core Concentration: {top['Top4_Concentration']:.1%}")
    print(f"    - Star Max Degree: {top['Max_Degree']:.0f}")
    print(f"    - Avg Ball Movement: {top['Avg_Degree']:.0f}")
    print(f"\n  Championship Duo:")
    try:
        print(f"    * {unidecode(str(top['Star_Player']))}")
        print(f"    * {unidecode(str(top['Second_Star']))}")
    except:
        print(f"    * {top['Star_Player']}")
        print(f"    * {top['Second_Star']}")
    
    # Key insight
    print("\n[KEY INSIGHT]")
    print("-"*100)
    print(f"  {top['Team']} leads the championship projection because their network shows:")
    if top['Entropy_Score'] > 60:
        print(f"    [+] LOW Entropy ({top['Pass_Entropy']:.3f}) - Highly ordered offense")
    if top['Hierarchy_Score'] > 60:
        print(f"    [+] HIGH Hierarchy ({top['Std_Degree']:.0f}) - Clear star-driven structure")
    if top['Concentration_Score'] > 60:
        print(f"    [+] HIGH Core Concentration ({top['Top4_Concentration']:.1%}) - Core-focused team")
    if top['Star_Score'] > 60:
        print(f"    [+] STAR Power ({top['Max_Degree']:.0f}) - Dominant offensive hub")
    
    print("\n" + "="*100)


def main():
    """Main execution."""
    print("="*70)
    print("2025-26 NBA CHAMPIONSHIP PREDICTION")
    print("Using Real Network Data (as of January 21, 2026)")
    print("="*70)
    
    # Load data
    print("\n[LOADING DATA]")
    standings_df = load_standings()
    players_df = load_players()
    
    if standings_df is None or players_df is None:
        print("[ERROR] Could not load required data files")
        return
    
    # Build team graphs
    team_graphs = build_team_graphs(players_df)
    
    # Calculate metrics
    metrics_df = calculate_team_metrics(team_graphs, standings_df, players_df)
    
    if len(metrics_df) == 0:
        print("[ERROR] No team metrics calculated")
        return
    
    # Calculate championship scores
    results_df = calculate_championship_score(metrics_df)
    
    # Generate visualizations
    print("\n[GENERATING VISUALIZATIONS]")
    plot_championship_prediction(results_df)
    plot_top_5_analysis(results_df)
    
    # Print prediction
    print_prediction(results_df)
    
    # Save data
    results_df.to_csv(OUTPUT_DIR / 'championship_prediction_2025_26_real.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
