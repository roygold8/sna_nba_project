"""
K-CORE ANALYSIS: Top 5 Players per Team
========================================
Find at which k-core level each team's top 5 players are embedded.
Are stars deeply integrated or isolated?
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
    """Build passing networks and identify top 5 players per team."""
    print("\n[BUILDING TEAM NETWORKS]")
    
    players_df = pd.read_csv(DATA_DIR / 'filtered_players.csv')
    
    team_graphs = {}
    team_top5 = {}
    
    # Group players by team with their stats
    teams = players_df.groupby('TEAM_ABBREVIATION')
    
    for team_abbr, team_players in teams:
        G = nx.Graph()
        player_degrees = {}
        
        # Add nodes for all players
        for _, player in team_players.iterrows():
            pid = player['PLAYER_ID']
            pname = player['PLAYER_NAME']
            G.add_node(pid, name=pname, minutes=player.get('MIN', 0), points=player.get('PTS', 0))
        
        # Load passing data and build edges
        for _, player in team_players.iterrows():
            pid = player['PLAYER_ID']
            passing_file = DATA_DIR / f"passing_{pid}.json"
            
            if not passing_file.exists():
                continue
            
            try:
                with open(passing_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                total_passes = 0
                
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
                            
                            total_passes += passes
                            
                            if receiver_id not in G.nodes():
                                G.add_node(receiver_id, name=receiver_name, minutes=0, points=0)
                            
                            if G.has_edge(pid, receiver_id):
                                G[pid][receiver_id]['weight'] += passes
                            else:
                                G.add_edge(pid, receiver_id, weight=passes)
                
                player_degrees[pid] = total_passes
                
            except:
                pass
        
        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Get top 5 players by total passing volume (weighted degree)
        if player_degrees:
            sorted_players = sorted(player_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_ids = [p[0] for p in sorted_players]
            top5_names = [G.nodes[p[0]].get('name', 'Unknown') for p in sorted_players]
            top5_degrees = [p[1] for p in sorted_players]
        else:
            top5_ids = []
            top5_names = []
            top5_degrees = []
        
        team_graphs[team_abbr] = G
        team_top5[team_abbr] = {
            'ids': top5_ids,
            'names': top5_names,
            'degrees': top5_degrees
        }
    
    print(f"  [OK] Built {len(team_graphs)} team networks")
    return team_graphs, team_top5


def analyze_top5_kcore(team_graphs, team_top5):
    """Find k-core level of each team's top 5 players."""
    print("\n[ANALYZING TOP 5 K-CORE LEVELS]")
    
    results = []
    
    for team, G in team_graphs.items():
        if G.number_of_nodes() < 5:
            continue
        
        top5 = team_top5.get(team, {})
        top5_ids = top5.get('ids', [])
        top5_names = top5.get('names', [])
        top5_degrees = top5.get('degrees', [])
        
        if len(top5_ids) < 5:
            continue
        
        # Get coreness of each node
        coreness = nx.core_number(G)
        
        # Get coreness of top 5 players
        top5_coreness = []
        for pid in top5_ids:
            if pid in coreness:
                top5_coreness.append(coreness[pid])
            else:
                top5_coreness.append(0)
        
        # Calculate metrics
        min_top5_kcore = min(top5_coreness) if top5_coreness else 0
        max_top5_kcore = max(top5_coreness) if top5_coreness else 0
        avg_top5_kcore = np.mean(top5_coreness) if top5_coreness else 0
        
        # Team's overall max k-core
        max_team_kcore = max(coreness.values()) if coreness else 0
        
        results.append({
            'Team': team,
            'Top5_Min_KCore': min_top5_kcore,
            'Top5_Max_KCore': max_top5_kcore,
            'Top5_Avg_KCore': avg_top5_kcore,
            'Team_Max_KCore': max_team_kcore,
            'Top5_Names': ', '.join([unidecode(n.split()[-1]) for n in top5_names]),
            'Top5_KCores': top5_coreness,
            'Player1': top5_names[0] if len(top5_names) > 0 else '',
            'Player1_KCore': top5_coreness[0] if len(top5_coreness) > 0 else 0,
            'Player2': top5_names[1] if len(top5_names) > 1 else '',
            'Player2_KCore': top5_coreness[1] if len(top5_coreness) > 1 else 0,
            'Player3': top5_names[2] if len(top5_names) > 2 else '',
            'Player3_KCore': top5_coreness[2] if len(top5_coreness) > 2 else 0,
            'Player4': top5_names[3] if len(top5_names) > 3 else '',
            'Player4_KCore': top5_coreness[3] if len(top5_coreness) > 3 else 0,
            'Player5': top5_names[4] if len(top5_names) > 4 else '',
            'Player5_KCore': top5_coreness[4] if len(top5_coreness) > 4 else 0,
        })
    
    df = pd.DataFrame(results)
    print(f"  [OK] Analyzed top 5 for {len(df)} teams")
    return df


def merge_standings(df):
    """Merge with standings data."""
    standings_df = pd.read_csv(DATA_DIR / 'standings.csv')
    players_df = pd.read_csv(DATA_DIR / 'filtered_players.csv')
    
    team_map = players_df.drop_duplicates('TEAM_ID')[['TEAM_ID', 'TEAM_ABBREVIATION']].set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()
    
    win_pcts = {}
    for team, team_id in team_map.items():
        standing = standings_df[standings_df['TeamID'] == team_id]
        if not standing.empty:
            win_pcts[team] = standing.iloc[0]['WinPCT']
    
    df['Win_Pct'] = df['Team'].map(win_pcts)
    return df.dropna(subset=['Win_Pct'])


def plot_top5_kcore(df):
    """Visualize top 5 k-core analysis."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # ===== 1. Top 5 Avg K-Core by Team =====
    ax1 = fig.add_subplot(2, 2, 1)
    
    df_sorted = df.sort_values('Top5_Avg_KCore', ascending=False).head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_sorted)))
    
    bars = ax1.barh(df_sorted['Team'], df_sorted['Top5_Avg_KCore'], color=colors, edgecolor='black')
    
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{row["Top5_Avg_KCore"]:.1f}', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Average K-Core of Top 5 Players', fontsize=14, fontweight='bold')
    ax1.set_title('HOW EMBEDDED ARE THE STARS?\nAverage K-Core Level of Top 5 Players', 
                  fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ===== 2. Top 5 K-Core vs Win% =====
    ax2 = fig.add_subplot(2, 2, 2)
    
    scatter = ax2.scatter(df['Top5_Avg_KCore'], df['Win_Pct'] * 100, 
                          s=150, c=df['Top5_Min_KCore'], cmap='RdYlGn',
                          alpha=0.7, edgecolors='black', linewidths=1.5)
    
    for _, row in df.iterrows():
        ax2.annotate(row['Team'], (row['Top5_Avg_KCore'], row['Win_Pct'] * 100),
                    fontsize=10, fontweight='bold', xytext=(3, 3), textcoords='offset points')
    
    from scipy import stats
    corr, pval = stats.pearsonr(df['Top5_Avg_KCore'], df['Win_Pct'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.4f}', transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Top 5 Players Average K-Core', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Win Percentage (%)', fontsize=14, fontweight='bold')
    ax2.set_title('STAR INTEGRATION vs WINNING\nDo Teams with More Embedded Stars Win More?', 
                  fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Min K-Core of Top 5', fontsize=12)
    
    # ===== 3. Individual Top 5 Player K-Cores (Top 10 Teams) =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    top_10 = df.nlargest(10, 'Win_Pct')
    
    x = np.arange(len(top_10))
    width = 0.15
    
    for i in range(5):
        kcores = [row[f'Player{i+1}_KCore'] for _, row in top_10.iterrows()]
        ax3.bar(x + i*width, kcores, width, label=f'Player {i+1}', 
                color=plt.cm.Set2(i/5), edgecolor='black', alpha=0.8)
    
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(top_10['Team'], fontsize=12, fontweight='bold')
    ax3.set_ylabel('K-Core Level', fontsize=14, fontweight='bold')
    ax3.set_title('TOP 10 WINNING TEAMS: Individual Player K-Cores\n(Player 1 = Highest Passing Volume)', 
                  fontsize=16, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== 4. Top 5 Players Table =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    top_5_teams = df.nlargest(5, 'Win_Pct')
    
    table_data = []
    for _, row in top_5_teams.iterrows():
        players = []
        for i in range(5):
            name = row.get(f'Player{i+1}', '')
            kcore = row.get(f'Player{i+1}_KCore', 0)
            if name:
                short_name = unidecode(name.split()[-1])
                players.append(f"{short_name}(k={int(kcore)})")
        
        table_data.append([
            row['Team'],
            f"{row['Win_Pct']:.1%}",
            f"{row['Top5_Avg_KCore']:.1f}",
            ', '.join(players[:3])
        ])
    
    columns = ['Team', 'Win%', 'Avg K-Core', 'Top 3 Players (with K-Core)']
    
    table = ax4.table(cellText=table_data,
                      colLabels=columns,
                      cellLoc='center',
                      loc='center',
                      colColours=['#3498db']*4)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2.2)
    
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if i == 0:
                table[(i+1, j)].set_facecolor('#d4edda')
    
    ax4.set_title('TOP 5 WINNING TEAMS: Core Players\nHow Deeply Embedded are the Stars?', 
                  fontsize=16, fontweight='bold', pad=30)
    
    plt.suptitle('K-CORE ANALYSIS: Top 5 Players Per Team\nFinding How Deeply Stars are Integrated into the Offense',
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kcore_top5_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: kcore_top5_analysis.png")


def print_top5_insights(df):
    """Print insights about top 5 players k-core."""
    
    print("\n" + "="*100)
    print("K-CORE ANALYSIS: TOP 5 PLAYERS PER TEAM")
    print("="*100)
    
    print("\n[TOP 10 TEAMS BY WIN% - Their Stars' K-Core Levels]")
    print("-"*100)
    
    top_10 = df.nlargest(10, 'Win_Pct')
    
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"\n  {rank}. {row['Team']} ({row['Win_Pct']:.1%} Win)")
        print(f"     Avg K-Core of Top 5: {row['Top5_Avg_KCore']:.1f}")
        print(f"     Players:")
        for i in range(5):
            name = row.get(f'Player{i+1}', '')
            kcore = row.get(f'Player{i+1}_KCore', 0)
            if name:
                print(f"       {i+1}. {unidecode(name)}: k={int(kcore)}")
    
    # Correlation
    from scipy import stats
    corr, pval = stats.pearsonr(df['Top5_Avg_KCore'], df['Win_Pct'])
    
    print("\n" + "="*100)
    print("[KEY FINDING]")
    print("="*100)
    print(f"  Correlation between Top 5 Avg K-Core and Win%: r = {corr:.3f} (p = {pval:.4f})")
    
    if corr > 0.2:
        print("\n  --> Teams whose STARS are MORE EMBEDDED in the network WIN MORE")
        print("  --> This suggests elite teams integrate their stars into cohesive systems")
    elif corr < -0.2:
        print("\n  --> Teams whose STARS are LESS EMBEDDED WIN MORE")
        print("  --> This supports the HELIOCENTRIC model - isolated stars dominate")
    else:
        print("\n  --> WEAK correlation - star integration doesn't strongly predict winning")
    
    print("\n" + "="*100)


def main():
    """Main execution."""
    print("="*70)
    print("K-CORE ANALYSIS: TOP 5 PLAYERS")
    print("How Deeply are Each Team's Stars Embedded?")
    print("="*70)
    
    # Build graphs
    team_graphs, team_top5 = build_team_graphs()
    
    # Analyze
    df = analyze_top5_kcore(team_graphs, team_top5)
    
    # Merge standings
    df = merge_standings(df)
    
    # Visualize
    print("\n[GENERATING VISUALIZATIONS]")
    plot_top5_kcore(df)
    
    # Print insights
    print_top5_insights(df)
    
    # Save
    df.to_csv(OUTPUT_DIR / 'kcore_top5_results.csv', index=False)
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
