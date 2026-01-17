"""
visualize_network.py - NBA Team Passing Network Visualizer

Generates aesthetically pleasing, publication-ready directed graphs of NBA
team passing networks for a given season.

Features:
- Node size proportional to player importance (Weighted Degree)
- Node color mapped to scoring ability (PTS)
- Edge thickness based on pass volume (logarithmic scale)
- Curved edges for bidirectional visibility
- Clean, readable labels with background boxes

Usage:
    python visualize_network.py
    
    Or import and call:
    from visualize_network import visualize_team_network
    visualize_team_network('2023-24', 'DEN', min_pass_threshold=10)

Author: NBA Network Analysis Project
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
IMAGES_DIR = Path("output_images")

# Visual settings
FIGURE_DPI = 200
FIGURE_SIZE = (16, 14)

# Color schemes
NODE_CMAP = 'plasma'  # Vibrant colormap for scoring
EDGE_COLOR = '#2C3E50'  # Dark blue-gray for edges
BACKGROUND_COLOR = '#FAFAFA'  # Light background


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


def shorten_name(name: str, max_len: int = 12) -> str:
    """Shorten player name for display."""
    name = format_player_name(name)
    if len(name) <= max_len:
        return name
    # Try first initial + last name
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name[:max_len]


# =============================================================================
# DATA LOADING
# =============================================================================
def load_player_metrics() -> pd.DataFrame:
    """Load player metrics from CSV."""
    # Try enhanced version first, then regular
    enhanced_path = OUTPUT_DIR / "nba_player_metrics_enhanced.csv"
    regular_path = OUTPUT_DIR / "nba_player_metrics.csv"
    
    if enhanced_path.exists():
        return pd.read_csv(enhanced_path)
    elif regular_path.exists():
        return pd.read_csv(regular_path)
    else:
        raise FileNotFoundError("Player metrics CSV not found. Run 02_build_metrics.py first.")


def load_filtered_players(season: str) -> pd.DataFrame:
    """Load filtered players CSV for a season."""
    filepath = DATA_DIR / season / "filtered_players.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Filtered players not found: {filepath}")
    return pd.read_csv(filepath)


def load_passing_data(season: str) -> Dict[int, dict]:
    """Load all passing JSON files for a season."""
    season_dir = DATA_DIR / season
    if not season_dir.exists():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    
    passing_data = {}
    json_files = list(season_dir.glob("passing_*.json"))
    
    for filepath in json_files:
        try:
            player_id = int(filepath.stem.replace("passing_", ""))
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and 'resultSets' in data:
                passing_data[player_id] = data
        except (ValueError, json.JSONDecodeError):
            continue
    
    return passing_data


def extract_passes_made(passing_data: dict) -> List[dict]:
    """Extract passes made from passing data structure."""
    passes = []
    try:
        result_sets = passing_data.get('resultSets', [])
        for result_set in result_sets:
            if result_set.get('name') == 'PassesMade':
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                for row in rows:
                    if len(row) == len(headers):
                        record = dict(zip(headers, row))
                        passes.append(record)
                break
    except Exception:
        pass
    return passes


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================
def build_team_network(
    season: str,
    team_abbr: str,
    min_pass_threshold: int = 5
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build a directed network graph for a specific team.
    
    Args:
        season: Season string (e.g., '2023-24')
        team_abbr: Team abbreviation (e.g., 'DEN')
        min_pass_threshold: Minimum passes to include an edge
        
    Returns:
        Tuple of (NetworkX DiGraph, team player info DataFrame)
    """
    print(f"[INFO] Building network for {team_abbr} ({season})...")
    
    # Load data
    player_info = load_filtered_players(season)
    passing_data = load_passing_data(season)
    player_metrics = load_player_metrics()
    
    # Filter to team
    team_players = player_info[player_info['TEAM_ABBREVIATION'] == team_abbr].copy()
    
    if team_players.empty:
        raise ValueError(f"Team {team_abbr} not found in {season} data")
    
    team_player_ids = set(team_players['PLAYER_ID'].tolist())
    
    # Get player metrics for this team/season
    team_metrics = player_metrics[
        (player_metrics['TEAM_ABBREVIATION'] == team_abbr) & 
        (player_metrics['SEASON'] == season)
    ]
    
    # Create player info lookup
    player_lookup = {}
    for _, row in team_players.iterrows():
        pid = row['PLAYER_ID']
        
        # Get metrics from master CSV
        metrics_row = team_metrics[team_metrics['PLAYER_ID'] == pid]
        
        if not metrics_row.empty:
            weighted_degree = metrics_row.iloc[0].get('Weighted_Degree', 0)
            pts = metrics_row.iloc[0].get('PTS', 0)
        else:
            weighted_degree = 0
            pts = row.get('PTS', 0)
        
        player_lookup[pid] = {
            'name': format_player_name(row['PLAYER_NAME']),
            'short_name': shorten_name(row['PLAYER_NAME']),
            'weighted_degree': weighted_degree,
            'pts': pts
        }
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for pid, info in player_lookup.items():
        G.add_node(
            pid,
            name=info['name'],
            short_name=info['short_name'],
            weighted_degree=info['weighted_degree'],
            pts=info['pts']
        )
    
    # Add edges from passing data
    for player_id in team_player_ids:
        if player_id not in passing_data:
            continue
        
        passes_made = extract_passes_made(passing_data[player_id])
        
        for pass_record in passes_made:
            teammate_id = pass_record.get('PASS_TEAMMATE_PLAYER_ID')
            pass_count = pass_record.get('PASS', 0)
            
            if teammate_id and teammate_id in team_player_ids:
                if pass_count >= min_pass_threshold:
                    G.add_edge(player_id, teammate_id, weight=pass_count)
    
    print(f"[OK] Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, team_players


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_team_network(
    season: str,
    team_abbr: str,
    min_pass_threshold: int = 5,
    show_plot: bool = True,
    save_plot: bool = True
) -> Optional[plt.Figure]:
    """
    Generate a publication-quality visualization of a team's passing network.
    
    Args:
        season: Season string (e.g., '2023-24')
        team_abbr: Team abbreviation (e.g., 'DEN')
        min_pass_threshold: Minimum passes to include an edge
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot to file
        
    Returns:
        Matplotlib Figure object
    """
    # Build the network
    G, team_info = build_team_network(season, team_abbr, min_pass_threshold)
    
    if G.number_of_edges() == 0:
        print(f"[WARNING] No edges found for {team_abbr} ({season}) with threshold {min_pass_threshold}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Calculate layout
    # Use spring layout with high k for spread, or kamada_kawai for balance
    try:
        pos = nx.kamada_kawai_layout(G, weight='weight')
    except Exception:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # -------------------------------------------------------------------------
    # NODE ATTRIBUTES
    # -------------------------------------------------------------------------
    
    # Get node attributes
    weighted_degrees = nx.get_node_attributes(G, 'weighted_degree')
    pts_values = nx.get_node_attributes(G, 'pts')
    names = nx.get_node_attributes(G, 'short_name')
    
    # Calculate node sizes (proportional to weighted degree)
    if weighted_degrees:
        wd_values = [weighted_degrees.get(n, 100) for n in G.nodes()]
        min_wd = max(min(wd_values), 1)
        max_wd = max(wd_values)
        
        # Scale to reasonable size range (300 to 3000)
        node_sizes = [
            300 + 2700 * ((weighted_degrees.get(n, min_wd) - min_wd) / (max_wd - min_wd + 1))
            for n in G.nodes()
        ]
    else:
        node_sizes = [800] * G.number_of_nodes()
    
    # Node colors (based on PTS)
    if pts_values:
        pts_list = [pts_values.get(n, 0) for n in G.nodes()]
        min_pts = min(pts_list)
        max_pts = max(pts_list)
        norm = Normalize(vmin=min_pts, vmax=max_pts)
        cmap = plt.cm.get_cmap(NODE_CMAP)
        node_colors = [cmap(norm(pts_values.get(n, 0))) for n in G.nodes()]
    else:
        node_colors = ['#3498DB'] * G.number_of_nodes()
    
    # -------------------------------------------------------------------------
    # EDGE ATTRIBUTES
    # -------------------------------------------------------------------------
    
    # Get edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        # Use logarithmic scale for edge widths (handles extreme differences)
        log_weights = [np.log1p(w) for w in edge_weights]
        log_min = min(log_weights)
        log_max = max(log_weights)
        
        # Scale to width range (0.5 to 6)
        edge_widths = [
            0.5 + 5.5 * ((np.log1p(w) - log_min) / (log_max - log_min + 0.001))
            for w in edge_weights
        ]
        
        # Scale to alpha range (0.2 to 0.9)
        edge_alphas = [
            0.2 + 0.7 * ((np.log1p(w) - log_min) / (log_max - log_min + 0.001))
            for w in edge_weights
        ]
    else:
        edge_widths = [1.0] * len(G.edges())
        edge_alphas = [0.5] * len(G.edges())
    
    # -------------------------------------------------------------------------
    # DRAW EDGES (with curved arrows)
    # -------------------------------------------------------------------------
    
    for i, (u, v) in enumerate(G.edges()):
        # Draw curved edge with arrow
        ax.annotate(
            '',
            xy=pos[v], xycoords='data',
            xytext=pos[u], textcoords='data',
            arrowprops=dict(
                arrowstyle='-|>',
                color=EDGE_COLOR,
                alpha=edge_alphas[i],
                lw=edge_widths[i],
                connectionstyle='arc3,rad=0.1',
                shrinkA=15,  # Shrink from start node
                shrinkB=15,  # Shrink to end node
                mutation_scale=10 + edge_widths[i] * 2
            )
        )
    
    # -------------------------------------------------------------------------
    # DRAW NODES
    # -------------------------------------------------------------------------
    
    # Draw node circles
    for i, node in enumerate(G.nodes()):
        circle = plt.Circle(
            pos[node],
            radius=0.03 + 0.05 * (node_sizes[i] / max(node_sizes)),
            facecolor=node_colors[i],
            edgecolor='white',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
    
    # Alternative: Use scatter for nodes (simpler approach)
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    
    scatter = ax.scatter(
        node_x, node_y,
        s=node_sizes,
        c=[pts_values.get(n, 0) for n in G.nodes()],
        cmap=NODE_CMAP,
        edgecolors='white',
        linewidths=2,
        zorder=10
    )
    
    # -------------------------------------------------------------------------
    # LABELS (with background boxes)
    # -------------------------------------------------------------------------
    
    # Sort nodes by weighted degree to label top players more prominently
    sorted_nodes = sorted(
        G.nodes(),
        key=lambda n: weighted_degrees.get(n, 0),
        reverse=True
    )
    
    # Label top players (adjust count based on team size)
    num_labels = min(len(sorted_nodes), max(8, len(sorted_nodes) // 2))
    
    for i, node in enumerate(sorted_nodes):
        name = names.get(node, f"Player {node}")
        x, y = pos[node]
        
        # Determine label position (offset from node)
        # Alternate positions to reduce overlap
        offset_x = 0.02 if i % 2 == 0 else -0.02
        offset_y = 0.04 if i % 3 == 0 else -0.04
        
        # Top players get larger, bolder labels
        if i < num_labels:
            fontsize = 11 if i < 5 else 9
            fontweight = 'bold' if i < 3 else 'normal'
            alpha = 1.0 if i < 5 else 0.9
            
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(x + offset_x, y + offset_y),
                fontsize=fontsize,
                fontweight=fontweight,
                ha='center',
                va='center',
                alpha=alpha,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.85
                ),
                zorder=20
            )
    
    # -------------------------------------------------------------------------
    # COLORBAR
    # -------------------------------------------------------------------------
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Points Per Game (PTS)', fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # TITLE AND STYLING
    # -------------------------------------------------------------------------
    
    # Get team win percentage if available
    player_metrics = load_player_metrics()
    team_metrics = player_metrics[
        (player_metrics['TEAM_ABBREVIATION'] == team_abbr) & 
        (player_metrics['SEASON'] == season)
    ]
    
    # Find star player (max weighted degree)
    star_player = ""
    if weighted_degrees:
        star_id = max(weighted_degrees, key=weighted_degrees.get)
        star_player = names.get(star_id, "")
    
    # Title
    ax.set_title(
        f'{team_abbr} Passing Network ({season})\n'
        f'Star Hub: {star_player} | Min. {min_pass_threshold} passes per edge',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
    # Remove axis
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    # Add legend for node sizes
    legend_elements = [
        plt.scatter([], [], s=500, c='gray', alpha=0.5, label='Low Pass Volume'),
        plt.scatter([], [], s=2000, c='gray', alpha=0.5, label='High Pass Volume'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower left',
        title='Node Size = Pass Involvement',
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9
    )
    
    # Add stats annotation
    total_passes = sum(edge_weights)
    stats_text = (
        f"Nodes: {G.number_of_nodes()} | "
        f"Edges: {G.number_of_edges()} | "
        f"Total Passes: {total_passes:,}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------------------------
    
    if save_plot:
        ensure_directory(IMAGES_DIR)
        filename = f"network_{team_abbr}_{season.replace('-', '_')}.png"
        filepath = IMAGES_DIR / filename
        fig.savefig(
            filepath,
            dpi=FIGURE_DPI,
            bbox_inches='tight',
            facecolor=BACKGROUND_COLOR,
            edgecolor='none'
        )
        print(f"[OK] Saved: {filepath}")
    
    if show_plot:
        plt.show()
    
    return fig


def visualize_multiple_teams(
    teams: List[Tuple[str, str]],
    min_pass_threshold: int = 10
) -> None:
    """
    Generate network visualizations for multiple teams.
    
    Args:
        teams: List of (season, team_abbr) tuples
        min_pass_threshold: Minimum passes to include an edge
    """
    for season, team_abbr in teams:
        try:
            visualize_team_network(
                season=season,
                team_abbr=team_abbr,
                min_pass_threshold=min_pass_threshold,
                show_plot=False,
                save_plot=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to visualize {team_abbr} ({season}): {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main entry point with example visualizations."""
    print("\n" + "="*60)
    print("NBA TEAM PASSING NETWORK VISUALIZER")
    print("="*60)
    
    # Example 1: Denver Nuggets 2022-23 (Championship season, Jokic hub)
    print("\n[1] Generating Denver Nuggets 2022-23 network (Championship team)...")
    try:
        visualize_team_network(
            season='2022-23',
            team_abbr='DEN',
            min_pass_threshold=15,
            show_plot=False,
            save_plot=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Example 2: Denver Nuggets 2023-24 (Jokic MVP season)
    print("\n[2] Generating Denver Nuggets 2023-24 network...")
    try:
        visualize_team_network(
            season='2023-24',
            team_abbr='DEN',
            min_pass_threshold=15,
            show_plot=False,
            save_plot=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Example 3: Golden State Warriors 2015-16 (73-9 season)
    print("\n[3] Generating Golden State Warriors 2015-16 network (73-9 season)...")
    try:
        visualize_team_network(
            season='2015-16',
            team_abbr='GSW',
            min_pass_threshold=15,
            show_plot=False,
            save_plot=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Example 4: Houston Rockets 2017-18 (Harden heliocentric)
    print("\n[4] Generating Houston Rockets 2017-18 network (Harden heliocentric)...")
    try:
        visualize_team_network(
            season='2017-18',
            team_abbr='HOU',
            min_pass_threshold=15,
            show_plot=False,
            save_plot=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nNetwork images saved to: {IMAGES_DIR.absolute()}")


if __name__ == "__main__":
    main()
