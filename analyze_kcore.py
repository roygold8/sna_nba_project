"""
K-CORE ANALYSIS - Finding the Core Structure of NBA Teams
==========================================================
Uses k-core decomposition to identify:
1. The "inner circle" of each team's offense
2. Which players are most deeply embedded
3. How core structure relates to success
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

try:
    from unidecode import unidecode
except ImportError:
    unidecode = lambda x: x

plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path("data/2025-26")
OUTPUT_DIR = Path("output_kcore_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_team_graphs():
    """Build passing network graphs for each team."""
    print("\n[BUILDING TEAM NETWORKS]")
    
    players_df = pd.read_csv(DATA_DIR / 'filtered_players.csv')
    
    team_graphs = {}
    team_players = defaultdict(dict)
    
    # Group players by team
    for _, player in players_df.iterrows():
        team_abbr = player['TEAM_ABBREVIATION']
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER_NAME']
        team_players[team_abbr][player_id] = player_name
    
    # Build graph for each team
    for team_abbr, players in team_players.items():
        G = nx.Graph()  # Undirected for k-core
        
        # Add nodes
        for pid, pname in players.items():
            G.add_node(pid, name=pname)
        
        # Load passing data and add edges
        for pid in players.keys():
            passing_file = DATA_DIR / f"passing_{pid}.json"
            
            if not passing_file.exists():
                continue
            
            try:
                with open(passing_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
                            receiver_name = row[pass_to_name_idx] if pass_to_name_idx else 'Unknown'
                            
                            if receiver_id not in G.nodes():
                                G.add_node(receiver_id, name=receiver_name)
                            
                            # Add/update edge (undirected, sum weights)
                            if G.has_edge(pid, receiver_id):
                                G[pid][receiver_id]['weight'] += passes
                            else:
                                G.add_edge(pid, receiver_id, weight=passes)
            except:
                pass
        
        team_graphs[team_abbr] = G
    
    print(f"  [OK] Built {len(team_graphs)} team networks")
    return team_graphs


def analyze_kcore(team_graphs):
    """Perform k-core analysis on each team."""
    print("\n[K-CORE ANALYSIS]")
    
    results = []
    core_players = {}
    
    for team, G in team_graphs.items():
        if G.number_of_nodes() < 3:
            continue
        
        # Remove self-loops (required for k-core)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Get coreness of each node
        coreness = nx.core_number(G)
        
        if not coreness:
            continue
        
        # Find max k-core value
        max_k = max(coreness.values())
        
        # Get the main core (highest k)
        main_core = nx.k_core(G)
        main_core_size = main_core.number_of_nodes()
        
        # Get players in the main core
        main_core_players = []
        for node in main_core.nodes():
            name = G.nodes[node].get('name', 'Unknown')
            main_core_players.append(name)
        
        # Calculate average coreness
        avg_coreness = np.mean(list(coreness.values()))
        
        # Get k-core at different levels
        k2_core = nx.k_core(G, k=2) if max_k >= 2 else None
        k3_core = nx.k_core(G, k=3) if max_k >= 3 else None
        k4_core = nx.k_core(G, k=4) if max_k >= 4 else None
        
        results.append({
            'Team': team,
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Max_K_Core': max_k,
            'Main_Core_Size': main_core_size,
            'Avg_Coreness': avg_coreness,
            'K2_Core_Size': k2_core.number_of_nodes() if k2_core else 0,
            'K3_Core_Size': k3_core.number_of_nodes() if k3_core else 0,
            'K4_Core_Size': k4_core.number_of_nodes() if k4_core else 0,
        })
        
        core_players[team] = {
            'max_k': max_k,
            'players': main_core_players[:5],  # Top 5 core players
            'coreness': {G.nodes[n].get('name', 'Unknown'): c for n, c in coreness.items()}
        }
    
    df = pd.DataFrame(results)
    print(f"  [OK] Analyzed {len(df)} teams")
    
    return df, core_players


def merge_with_standings(kcore_df):
    """Merge k-core results with team standings."""
    standings_df = pd.read_csv(DATA_DIR / 'standings.csv')
    players_df = pd.read_csv(DATA_DIR / 'filtered_players.csv')
    
    # Build team abbreviation to ID mapping
    team_map = players_df.drop_duplicates('TEAM_ID')[['TEAM_ID', 'TEAM_ABBREVIATION']].set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()
    
    # Add standings data
    win_pcts = {}
    for team, team_id in team_map.items():
        standing = standings_df[standings_df['TeamID'] == team_id]
        if not standing.empty:
            win_pcts[team] = standing.iloc[0]['WinPCT']
    
    kcore_df['Win_Pct'] = kcore_df['Team'].map(win_pcts)
    kcore_df = kcore_df.dropna(subset=['Win_Pct'])
    
    return kcore_df


def plot_kcore_analysis(df, core_players):
    """Create k-core visualizations."""
    
    fig = plt.figure(figsize=(20, 14))
    
    # ===== 1. Teams by Max K-Core =====
    ax1 = fig.add_subplot(2, 2, 1)
    
    df_sorted = df.sort_values('Max_K_Core', ascending=False).head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_sorted)))
    
    bars = ax1.barh(df_sorted['Team'], df_sorted['Max_K_Core'], color=colors, edgecolor='black')
    
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'k={int(row["Max_K_Core"])}', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Maximum K-Core Value', fontsize=14, fontweight='bold')
    ax1.set_title('TEAMS BY NETWORK CORE DEPTH\n(Higher k = More Densely Connected Core)', 
                  fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ===== 2. K-Core vs Win% =====
    ax2 = fig.add_subplot(2, 2, 2)
    
    scatter = ax2.scatter(df['Max_K_Core'], df['Win_Pct'] * 100, 
                          s=df['Main_Core_Size'] * 20, c=df['Avg_Coreness'],
                          cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Annotate top teams
    for _, row in df.nlargest(8, 'Max_K_Core').iterrows():
        ax2.annotate(row['Team'], (row['Max_K_Core'], row['Win_Pct'] * 100),
                    fontsize=11, fontweight='bold', xytext=(5, 5), textcoords='offset points')
    
    # Correlation
    from scipy import stats
    corr, pval = stats.pearsonr(df['Max_K_Core'], df['Win_Pct'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.4f}', transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Maximum K-Core Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Win Percentage (%)', fontsize=14, fontweight='bold')
    ax2.set_title('K-CORE DEPTH vs WINNING\nDo Teams with Deeper Cores Win More?', 
                  fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Avg Coreness', fontsize=12)
    
    # ===== 3. Core Size Distribution =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    top_10 = df.nlargest(10, 'Max_K_Core')
    x = np.arange(len(top_10))
    width = 0.25
    
    ax3.bar(x - width, top_10['K2_Core_Size'], width, label='k=2 Core', color='#3498db', edgecolor='black')
    ax3.bar(x, top_10['K3_Core_Size'], width, label='k=3 Core', color='#e74c3c', edgecolor='black')
    ax3.bar(x + width, top_10['K4_Core_Size'], width, label='k=4 Core', color='#2ecc71', edgecolor='black')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_10['Team'], fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Players in Core', fontsize=14, fontweight='bold')
    ax3.set_title('CORE SIZE AT DIFFERENT K LEVELS\nHow Many Players in Each Team\'s Inner Circle?', 
                  fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== 4. Top Teams' Core Players =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Get top 5 teams by Max_K_Core
    top_5_teams = df.nlargest(5, 'Max_K_Core')['Team'].tolist()
    
    table_data = []
    for team in top_5_teams:
        info = core_players.get(team, {})
        players = info.get('players', [])[:3]
        players_str = ', '.join([unidecode(p.split()[-1]) for p in players]) if players else 'N/A'
        table_data.append([
            team,
            f"k={info.get('max_k', 0)}",
            f"{len(info.get('players', []))}",
            players_str
        ])
    
    columns = ['Team', 'Max K-Core', 'Core Size', 'Core Players (Top 3)']
    
    table = ax4.table(cellText=table_data,
                      colLabels=columns,
                      cellLoc='center',
                      loc='center',
                      colColours=['#3498db']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.2)
    
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if i == 0:
                table[(i+1, j)].set_facecolor('#d4edda')
    
    ax4.set_title('TOP 5 TEAMS: CORE PLAYERS\nThe "Inner Circle" of Each Offense', 
                  fontsize=16, fontweight='bold', pad=30)
    
    plt.suptitle('K-CORE ANALYSIS: Finding the Heart of NBA Offenses\n'
                 'Which Teams Have the Most Densely Connected Core?',
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kcore_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: kcore_analysis.png")


def print_insights(df, core_players):
    """Print k-core analysis insights."""
    
    print("\n" + "="*90)
    print("K-CORE ANALYSIS: THE INNER CIRCLE OF NBA OFFENSES")
    print("="*90)
    
    print("\n[WHAT IS K-CORE?]")
    print("-"*90)
    print("  K-core removes nodes iteratively based on degree:")
    print("  - k=2: Players with at least 2 passing connections")
    print("  - k=3: Players with at least 3 passing connections")
    print("  - Higher k = More deeply embedded in the offense")
    
    print("\n[TOP 10 TEAMS BY CORE DEPTH]")
    print("-"*90)
    print(f"  {'Rank':<6} {'Team':<6} {'Max K':<8} {'Core Size':<12} {'Avg Coreness':<14} {'Win%':<8}")
    print("  " + "-"*80)
    
    for rank, (_, row) in enumerate(df.nlargest(10, 'Max_K_Core').iterrows(), 1):
        print(f"  {rank:<6} {row['Team']:<6} {int(row['Max_K_Core']):<8} {int(row['Main_Core_Size']):<12} {row['Avg_Coreness']:<14.2f} {row['Win_Pct']:.1%}")
    
    # Correlation insight
    from scipy import stats
    corr, pval = stats.pearsonr(df['Max_K_Core'], df['Win_Pct'])
    
    print(f"\n[KEY FINDING]")
    print("-"*90)
    print(f"  Correlation between K-Core Depth and Win%: r = {corr:.3f} (p = {pval:.4f})")
    
    if corr > 0.2:
        print("  --> Teams with DEEPER cores (more interconnected) tend to WIN MORE")
    elif corr < -0.2:
        print("  --> Teams with SHALLOWER cores tend to WIN MORE (star-dependent)")
    else:
        print("  --> Core depth has WEAK correlation with winning")
    
    # Top team's core
    top_team = df.nlargest(1, 'Max_K_Core').iloc[0]
    top_info = core_players.get(top_team['Team'], {})
    
    print(f"\n[DEEPEST CORE: {top_team['Team']}]")
    print("-"*90)
    print(f"  Max K-Core: {int(top_team['Max_K_Core'])}")
    print(f"  Core Size: {int(top_team['Main_Core_Size'])} players")
    print(f"  Core Players: {', '.join([unidecode(p) for p in top_info.get('players', [])[:5]])}")
    print(f"  Insight: This team has {int(top_team['Main_Core_Size'])} players all connected to at least")
    print(f"           {int(top_team['Max_K_Core'])} other teammates - a tightly knit offensive unit!")
    
    print("\n" + "="*90)


def main():
    """Main execution."""
    print("="*70)
    print("K-CORE ANALYSIS")
    print("Finding the Core Structure of NBA Team Offenses")
    print("="*70)
    
    # Build graphs
    team_graphs = build_team_graphs()
    
    # K-core analysis
    kcore_df, core_players = analyze_kcore(team_graphs)
    
    # Merge with standings
    kcore_df = merge_with_standings(kcore_df)
    
    # Visualize
    print("\n[GENERATING VISUALIZATIONS]")
    plot_kcore_analysis(kcore_df, core_players)
    
    # Print insights
    print_insights(kcore_df, core_players)
    
    # Save data
    kcore_df.to_csv(OUTPUT_DIR / 'kcore_results.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
