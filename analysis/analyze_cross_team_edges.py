import networkx as nx
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the network
print("Loading network...")
G = nx.read_gexf(os.path.join(BASE_DIR, 'output', 'nba_pass_network.gexf'))

print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")

# Find cross-team edges
cross_team_edges = []

for source, target in G.edges():
    # Extract team from node name: "Player Name, TEAM, Season"
    source_parts = source.split(', ')
    target_parts = target.split(', ')
    
    if len(source_parts) >= 3 and len(target_parts) >= 3:
        source_team = source_parts[1]
        target_team = target_parts[1]
        source_season = source_parts[2]
        target_season = target_parts[2]
        
        # Check if teams differ (but same season)
        if source_season == target_season and source_team != target_team:
            cross_team_edges.append({
                'source': source,
                'target': target,
                'source_team': source_team,
                'target_team': target_team,
                'season': source_season,
                'weight': G[source][target].get('weight', 1)
            })

print(f"\nFound {len(cross_team_edges)} cross-team edges")

if cross_team_edges:
    print("\nTop 10 examples of cross-team edges:")
    # Sort by weight to see most significant ones
    sorted_edges = sorted(cross_team_edges, key=lambda x: x['weight'], reverse=True)[:10]
    
    for i, edge in enumerate(sorted_edges, 1):
        print(f"\n{i}. Weight: {edge['weight']}")
        print(f"   {edge['source']}")
        print(f"   -> {edge['target']}")
        print(f"   Teams: {edge['source_team']} -> {edge['target_team']}")
        print(f"   Season: {edge['season']}")

print(f"\nPercentage of cross-team edges: {len(cross_team_edges) / G.number_of_edges() * 100:.2f}%")

# Analyze by season
from collections import defaultdict
by_season = defaultdict(int)
for edge in cross_team_edges:
    by_season[edge['season']] += 1

print("\nCross-team edges by season:")
for season in sorted(by_season.keys()):
    print(f"  {season}: {by_season[season]} edges")
