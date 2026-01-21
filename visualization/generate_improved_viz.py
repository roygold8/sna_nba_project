#!/usr/bin/env python3
"""
Generate improved NBA Assist Network visualization with team colors and weighted sizing.
Run this script directly: python generate_improved_viz.py
"""

import networkx as nx
import numpy as np
from pyvis.network import Network
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Helper function to safely get edge weight (GEXF stores as strings!)
def get_weight(data, default=1):
    """Safely get edge weight, converting from string if needed."""
    w = data.get('weight', default)
    try:
        return float(w)
    except (TypeError, ValueError):
        return default

# NBA Team Colors
NBA_TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#000000', 'CHA': '#1D1160',
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240',
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62',
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E',
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#F58426',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160',
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141',
    'UTA': '#002B5C', 'WAS': '#002B5C',
}

def get_team_from_node(node_name):
    parts = node_name.split(', ')
    return parts[1] if len(parts) >= 2 else 'UNK'

def filter_by_season(G, season):
    nodes_in_season = [n for n in G.nodes() if season in n]
    return G.subgraph(nodes_in_season).copy()

def visualize_team_network(G, output_file='network.html', min_edge_weight=3):
    """Create team-colored visualization with weighted node sizes."""
    
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter weak edges
    if min_edge_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                          if get_weight(d, 1) < min_edge_weight]
        G = G.copy()
        G.remove_edges_from(edges_to_remove)
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        print(f"After filtering (min weight {min_edge_weight}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        print("ERROR: No nodes left after filtering!")
        return None
    
    # Create PyVis network
    net = Network(height='900px', width='100%', bgcolor='#1a1a2e', 
                  font_color='white', directed=True, cdn_resources='in_line')
    
    # Calculate weighted degrees
    weighted_in = {}
    weighted_out = {}
    for node in G.nodes():
        weighted_out[node] = sum(get_weight(d, 0) for _, _, d in G.out_edges(node, data=True))
        weighted_in[node] = sum(get_weight(d, 0) for _, _, d in G.in_edges(node, data=True))
    
    weighted_total = {n: weighted_in.get(n, 0) + weighted_out.get(n, 0) for n in G.nodes()}
    max_weight = max(weighted_total.values()) if weighted_total else 1
    
    # Get max edge weight for scaling
    edge_weights = [get_weight(d, 1) for _, _, d in G.edges(data=True)]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    
    # Group by team
    teams = {}
    for node in G.nodes():
        team = get_team_from_node(node)
        if team not in teams:
            teams[team] = []
        teams[team].append(node)
    
    print(f"Found {len(teams)} teams: {sorted(teams.keys())}")
    
    # Team positions in a circle - MAXIMUM SPACING
    num_teams = len(teams)
    team_list = sorted(teams.keys())
    team_positions = {}
    radius = 2500  # Very large radius for team separation
    for i, team in enumerate(team_list):
        angle = 2 * np.pi * i / num_teams
        team_positions[team] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # Node positions - MAXIMUM SPACING WITHIN TEAMS
    node_positions = {}
    for team, team_nodes in teams.items():
        if len(team_nodes) <= 1:
            for node in team_nodes:
                node_positions[node] = team_positions[team]
        else:
            team_subgraph = G.subgraph(team_nodes)
            # k=5 gives maximum spacing between nodes
            team_pos = nx.spring_layout(team_subgraph, k=5, iterations=100, seed=42)
            team_center = team_positions[team]
            team_scale = 500  # Large scale for player separation
            for node, pos in team_pos.items():
                node_positions[node] = (
                    team_center[0] + pos[0] * team_scale,
                    team_center[1] + pos[1] * team_scale
                )
    
    # Add nodes
    print("Adding nodes...")
    for node in G.nodes():
        team = get_team_from_node(node)
        color = NBA_TEAM_COLORS.get(team, '#888888')
        
        w = weighted_total.get(node, 0)
        # More dramatic size difference based on weight
        if w > 0 and max_weight > 0:
            # Use power scaling for more dramatic differences
            ratio = w / max_weight
            size = 5 + 55 * (ratio ** 0.6)  # Power scaling makes differences more visible
        else:
            size = 5
        
        player_name = node.split(',')[0].strip()
        tooltip = f"<b>{node}</b><br>Team: {team}<br>Assists Given: {weighted_out.get(node, 0)}<br>Assists Received: {weighted_in.get(node, 0)}"
        
        pos = node_positions.get(node, (0, 0))
        net.add_node(node, label=player_name, title=tooltip, color=color, size=size,
                    x=pos[0], y=pos[1], font={'size': 12, 'color': 'white'})
    
    # Add edges
    print("Adding edges...")
    for source, target, data in G.edges(data=True):
        weight = get_weight(data, 1)
        # More dramatic edge width difference
        ratio = weight / max_edge_weight
        edge_width = 0.5 + 8 * (ratio ** 0.7)  # Power scaling for more contrast
        opacity = 0.2 + 0.7 * (ratio ** 0.5)   # More visible high-weight edges
        
        source_team = get_team_from_node(source)
        edge_color = NBA_TEAM_COLORS.get(source_team, '#888888')
        edge_title = f"{source.split(',')[0]} → {target.split(',')[0]}: {int(weight)} assists"
        
        net.add_edge(source, target, width=edge_width, title=edge_title,
                    color={'color': edge_color, 'opacity': opacity},
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.8, 'type': 'arrow'}})
    
    net.toggle_physics(False)
    net.set_options('''
    {
      "nodes": {
        "borderWidth": 2, 
        "font": {"size": 11, "strokeWidth": 2, "strokeColor": "#1a1a2e"}
      },
      "edges": {
        "smooth": {"enabled": true, "type": "curvedCW", "roundness": 0.2},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "color": {"inherit": false},
        "selectionWidth": 2
      },
      "interaction": {
        "hover": true, 
        "tooltipDelay": 50, 
        "zoomView": true, 
        "dragView": true,
        "zoomSpeed": 0.5
      }
    }
    ''')
    
    print(f"Saving to {output_file}...")
    net.save_graph(output_file)
    print(f"✅ Done! Open {output_file} in your browser.")
    return net

if __name__ == "__main__":
    print("Loading assist network...")
    network_path = os.path.join(BASE_DIR, 'output', 'nba_assist_network.gexf')
    G = nx.read_gexf(network_path)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter to 2023-24 season
    G_season = filter_by_season(G, '2023-24')
    print(f"2023-24 season: {G_season.number_of_nodes()} nodes, {G_season.number_of_edges()} edges")
    
    # Generate visualization (min_edge_weight=3 keeps connections with 3+ assists)
    visualize_team_network(G_season, 'assist_network_2023-24_improved.html', min_edge_weight=3)

