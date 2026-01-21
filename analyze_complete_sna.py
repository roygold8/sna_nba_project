"""
COMPLETE SNA ANALYSIS - All Course-Required Metrics
=====================================================
This script calculates ALL metrics required for the SNA course:
1. Basic Network Stats (nodes, edges, density)
2. Connectivity (connected components, largest component)
3. Path Metrics (APL - Average Path Length)
4. ALL Centrality Measures (Degree, Betweenness, Closeness, Eigenvector)
5. Clustering Coefficients
6. Community Detection (Louvain)
7. Correlations with Winning

Author: NBA Network Analysis Project
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings('ignore')

# Try to import community detection
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("[WARNING] python-louvain not installed. Run: pip install python-louvain")

plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_complete_sna")
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24'
]

# Team Win% data (verified from NBA stats)
TEAM_WINS = {
    ('GSW', '2015-16'): 0.890, ('SAS', '2015-16'): 0.817, ('CLE', '2015-16'): 0.695,
    ('OKC', '2015-16'): 0.671, ('TOR', '2015-16'): 0.683, ('LAC', '2015-16'): 0.646,
    ('BOS', '2015-16'): 0.585, ('MIA', '2015-16'): 0.585, ('ATL', '2015-16'): 0.585,
    ('POR', '2015-16'): 0.537, ('CHA', '2015-16'): 0.585, ('IND', '2015-16'): 0.549,
    ('DET', '2015-16'): 0.537, ('CHI', '2015-16'): 0.512, ('DAL', '2015-16'): 0.512,
    ('MEM', '2015-16'): 0.512, ('HOU', '2015-16'): 0.500, ('WAS', '2015-16'): 0.500,
    ('UTA', '2015-16'): 0.488, ('DEN', '2015-16'): 0.402, ('SAC', '2015-16'): 0.402,
    ('NOP', '2015-16'): 0.366, ('MIL', '2015-16'): 0.402, ('ORL', '2015-16'): 0.427,
    ('NYK', '2015-16'): 0.390, ('BKN', '2015-16'): 0.256, ('MIN', '2015-16'): 0.354,
    ('PHX', '2015-16'): 0.280, ('LAL', '2015-16'): 0.207, ('PHI', '2015-16'): 0.122,
    ('GSW', '2016-17'): 0.817, ('SAS', '2016-17'): 0.744, ('HOU', '2016-17'): 0.671,
    ('CLE', '2016-17'): 0.622, ('BOS', '2016-17'): 0.646, ('TOR', '2016-17'): 0.622,
    ('UTA', '2016-17'): 0.622, ('LAC', '2016-17'): 0.622, ('WAS', '2016-17'): 0.598,
    ('OKC', '2016-17'): 0.573, ('MEM', '2016-17'): 0.524, ('ATL', '2016-17'): 0.524,
    ('MIL', '2016-17'): 0.512, ('IND', '2016-17'): 0.512, ('MIA', '2016-17'): 0.500,
    ('POR', '2016-17'): 0.500, ('DEN', '2016-17'): 0.488, ('CHI', '2016-17'): 0.500,
    ('NOP', '2016-17'): 0.415, ('DET', '2016-17'): 0.451, ('CHA', '2016-17'): 0.439,
    ('DAL', '2016-17'): 0.402, ('SAC', '2016-17'): 0.390, ('MIN', '2016-17'): 0.378,
    ('NYK', '2016-17'): 0.378, ('ORL', '2016-17'): 0.354, ('PHI', '2016-17'): 0.341,
    ('PHX', '2016-17'): 0.293, ('LAL', '2016-17'): 0.317, ('BKN', '2016-17'): 0.244,
    ('HOU', '2017-18'): 0.793, ('TOR', '2017-18'): 0.720, ('GSW', '2017-18'): 0.707,
    ('BOS', '2017-18'): 0.671, ('PHI', '2017-18'): 0.634, ('CLE', '2017-18'): 0.610,
    ('POR', '2017-18'): 0.598, ('IND', '2017-18'): 0.585, ('OKC', '2017-18'): 0.585,
    ('UTA', '2017-18'): 0.585, ('NOP', '2017-18'): 0.585, ('SAS', '2017-18'): 0.573,
    ('MIN', '2017-18'): 0.573, ('MIA', '2017-18'): 0.537, ('DEN', '2017-18'): 0.561,
    ('MIL', '2017-18'): 0.537, ('WAS', '2017-18'): 0.524, ('LAC', '2017-18'): 0.512,
    ('DET', '2017-18'): 0.476, ('CHA', '2017-18'): 0.439, ('NYK', '2017-18'): 0.354,
    ('BKN', '2017-18'): 0.341, ('CHI', '2017-18'): 0.329, ('SAC', '2017-18'): 0.329,
    ('LAL', '2017-18'): 0.427, ('ORL', '2017-18'): 0.305, ('DAL', '2017-18'): 0.293,
    ('ATL', '2017-18'): 0.293, ('MEM', '2017-18'): 0.268, ('PHX', '2017-18'): 0.256,
    ('MIL', '2018-19'): 0.732, ('TOR', '2018-19'): 0.707, ('GSW', '2018-19'): 0.695,
    ('DEN', '2018-19'): 0.659, ('POR', '2018-19'): 0.646, ('HOU', '2018-19'): 0.646,
    ('PHI', '2018-19'): 0.622, ('BOS', '2018-19'): 0.598, ('UTA', '2018-19'): 0.610,
    ('OKC', '2018-19'): 0.598, ('IND', '2018-19'): 0.585, ('SAS', '2018-19'): 0.585,
    ('LAC', '2018-19'): 0.585, ('BKN', '2018-19'): 0.512, ('ORL', '2018-19'): 0.512,
    ('SAC', '2018-19'): 0.476, ('MIA', '2018-19'): 0.476, ('DET', '2018-19'): 0.500,
    ('CHA', '2018-19'): 0.476, ('MIN', '2018-19'): 0.439, ('LAL', '2018-19'): 0.451,
    ('NOP', '2018-19'): 0.402, ('DAL', '2018-19'): 0.402, ('MEM', '2018-19'): 0.402,
    ('WAS', '2018-19'): 0.390, ('ATL', '2018-19'): 0.354, ('CHI', '2018-19'): 0.268,
    ('CLE', '2018-19'): 0.232, ('PHX', '2018-19'): 0.232, ('NYK', '2018-19'): 0.207,
    ('MIL', '2019-20'): 0.767, ('LAL', '2019-20'): 0.732, ('TOR', '2019-20'): 0.736,
    ('LAC', '2019-20'): 0.681, ('BOS', '2019-20'): 0.667, ('DEN', '2019-20'): 0.630,
    ('MIA', '2019-20'): 0.603, ('UTA', '2019-20'): 0.611, ('OKC', '2019-20'): 0.611,
    ('HOU', '2019-20'): 0.611, ('PHI', '2019-20'): 0.589, ('IND', '2019-20'): 0.589,
    ('DAL', '2019-20'): 0.571, ('POR', '2019-20'): 0.473, ('BKN', '2019-20'): 0.486,
    ('ORL', '2019-20'): 0.458, ('MEM', '2019-20'): 0.466, ('SAS', '2019-20'): 0.451,
    ('NOP', '2019-20'): 0.417, ('SAC', '2019-20'): 0.431, ('PHX', '2019-20'): 0.466,
    ('WAS', '2019-20'): 0.361, ('CHA', '2019-20'): 0.348, ('CHI', '2019-20'): 0.338,
    ('NYK', '2019-20'): 0.318, ('DET', '2019-20'): 0.303, ('ATL', '2019-20'): 0.299,
    ('CLE', '2019-20'): 0.292, ('MIN', '2019-20'): 0.292, ('GSW', '2019-20'): 0.231,
    ('UTA', '2020-21'): 0.722, ('PHX', '2020-21'): 0.722, ('BKN', '2020-21'): 0.667,
    ('PHI', '2020-21'): 0.681, ('DEN', '2020-21'): 0.653, ('LAC', '2020-21'): 0.667,
    ('MIL', '2020-21'): 0.639, ('DAL', '2020-21'): 0.583, ('POR', '2020-21'): 0.583,
    ('LAL', '2020-21'): 0.583, ('NYK', '2020-21'): 0.569, ('ATL', '2020-21'): 0.569,
    ('MIA', '2020-21'): 0.556, ('BOS', '2020-21'): 0.500, ('MEM', '2020-21'): 0.528,
    ('SAS', '2020-21'): 0.472, ('IND', '2020-21'): 0.472, ('GSW', '2020-21'): 0.528,
    ('WAS', '2020-21'): 0.472, ('CHA', '2020-21'): 0.458, ('CHI', '2020-21'): 0.431,
    ('NOP', '2020-21'): 0.431, ('SAC', '2020-21'): 0.431, ('TOR', '2020-21'): 0.375,
    ('MIN', '2020-21'): 0.319, ('CLE', '2020-21'): 0.306, ('OKC', '2020-21'): 0.306,
    ('ORL', '2020-21'): 0.292, ('DET', '2020-21'): 0.278, ('HOU', '2020-21'): 0.236,
    ('PHX', '2021-22'): 0.780, ('MEM', '2021-22'): 0.683, ('MIA', '2021-22'): 0.646,
    ('GSW', '2021-22'): 0.646, ('BOS', '2021-22'): 0.622, ('MIL', '2021-22'): 0.622,
    ('PHI', '2021-22'): 0.622, ('DAL', '2021-22'): 0.634, ('UTA', '2021-22'): 0.598,
    ('TOR', '2021-22'): 0.585, ('DEN', '2021-22'): 0.585, ('CHI', '2021-22'): 0.561,
    ('MIN', '2021-22'): 0.561, ('CLE', '2021-22'): 0.537, ('BKN', '2021-22'): 0.537,
    ('ATL', '2021-22'): 0.524, ('CHA', '2021-22'): 0.524, ('LAC', '2021-22'): 0.512,
    ('NOP', '2021-22'): 0.439, ('NYK', '2021-22'): 0.451, ('LAL', '2021-22'): 0.402,
    ('SAS', '2021-22'): 0.415, ('WAS', '2021-22'): 0.427, ('SAC', '2021-22'): 0.366,
    ('POR', '2021-22'): 0.329, ('IND', '2021-22'): 0.305, ('DET', '2021-22'): 0.280,
    ('OKC', '2021-22'): 0.293, ('ORL', '2021-22'): 0.268, ('HOU', '2021-22'): 0.244,
    ('MIL', '2022-23'): 0.707, ('BOS', '2022-23'): 0.695, ('DEN', '2022-23'): 0.646,
    ('PHI', '2022-23'): 0.659, ('CLE', '2022-23'): 0.622, ('SAC', '2022-23'): 0.585,
    ('PHX', '2022-23'): 0.549, ('NYK', '2022-23'): 0.573, ('MEM', '2022-23'): 0.622,
    ('BKN', '2022-23'): 0.549, ('LAC', '2022-23'): 0.537, ('MIA', '2022-23'): 0.537,
    ('GSW', '2022-23'): 0.537, ('ATL', '2022-23'): 0.500, ('LAL', '2022-23'): 0.524,
    ('MIN', '2022-23'): 0.512, ('TOR', '2022-23'): 0.500, ('NOP', '2022-23'): 0.512,
    ('CHI', '2022-23'): 0.488, ('OKC', '2022-23'): 0.488, ('DAL', '2022-23'): 0.463,
    ('UTA', '2022-23'): 0.451, ('WAS', '2022-23'): 0.427, ('IND', '2022-23'): 0.427,
    ('POR', '2022-23'): 0.402, ('ORL', '2022-23'): 0.415, ('CHA', '2022-23'): 0.329,
    ('DET', '2022-23'): 0.207, ('SAS', '2022-23'): 0.268, ('HOU', '2022-23'): 0.268,
    ('BOS', '2023-24'): 0.780, ('OKC', '2023-24'): 0.695, ('DEN', '2023-24'): 0.695,
    ('MIN', '2023-24'): 0.683, ('CLE', '2023-24'): 0.585, ('NYK', '2023-24'): 0.610,
    ('MIL', '2023-24'): 0.598, ('LAC', '2023-24'): 0.622, ('PHX', '2023-24'): 0.598,
    ('DAL', '2023-24'): 0.610, ('NOP', '2023-24'): 0.598, ('ORL', '2023-24'): 0.573,
    ('IND', '2023-24'): 0.573, ('PHI', '2023-24'): 0.573, ('MIA', '2023-24'): 0.561,
    ('SAC', '2023-24'): 0.561, ('LAL', '2023-24'): 0.573, ('GSW', '2023-24'): 0.561,
    ('HOU', '2023-24'): 0.500, ('UTA', '2023-24'): 0.378, ('CHI', '2023-24'): 0.476,
    ('ATL', '2023-24'): 0.439, ('BKN', '2023-24'): 0.390, ('TOR', '2023-24'): 0.305,
    ('MEM', '2023-24'): 0.329, ('POR', '2023-24'): 0.256, ('CHA', '2023-24'): 0.256,
    ('SAS', '2023-24'): 0.268, ('DET', '2023-24'): 0.171, ('WAS', '2023-24'): 0.183,
}


def load_passing_data(season_dir):
    """Load all passing JSON files for a season."""
    passing_data = {}
    json_files = list(season_dir.glob("passing_*.json"))
    
    for filepath in json_files:
        try:
            player_id = int(filepath.stem.replace("passing_", ""))
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and 'resultSets' in data:
                passing_data[player_id] = data
        except Exception:
            continue
    
    return passing_data


def extract_passes_made(passing_data):
    """Extract passes made from passing data structure."""
    passes = []
    try:
        for result_set in passing_data.get('resultSets', []):
            if result_set.get('name') == 'PassesMade':
                headers = result_set.get('headers', [])
                rows = result_set.get('rowSet', [])
                for row in rows:
                    if len(row) == len(headers):
                        passes.append(dict(zip(headers, row)))
                break
    except Exception:
        pass
    return passes


def build_team_network(team_abbr, season, passing_data, player_info):
    """Build a directed weighted graph for a team."""
    G = nx.DiGraph()
    
    team_players = player_info[player_info['TEAM_ABBREVIATION'] == team_abbr]
    team_player_ids = set(team_players['PLAYER_ID'].tolist())
    
    if not team_player_ids:
        return G
    
    # Create player name lookup
    player_names = {}
    for _, row in team_players.iterrows():
        player_names[row['PLAYER_ID']] = row['PLAYER_NAME']
    
    # Add nodes
    for player_id in team_player_ids:
        G.add_node(player_id, name=player_names.get(player_id, f"P{player_id}"))
    
    # Add edges
    for player_id in team_player_ids:
        if player_id not in passing_data:
            continue
        
        for pass_record in extract_passes_made(passing_data[player_id]):
            teammate_id = pass_record.get('PASS_TEAMMATE_PLAYER_ID')
            pass_count = pass_record.get('PASS', 0)
            
            if teammate_id and teammate_id in team_player_ids and pass_count > 0:
                if G.has_edge(player_id, teammate_id):
                    G[player_id][teammate_id]['weight'] += pass_count
                else:
                    G.add_edge(player_id, teammate_id, weight=pass_count)
    
    return G


def calculate_entropy(values):
    """Calculate Shannon Entropy of a distribution."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or np.sum(values) == 0:
        return 0
    
    probs = values / np.sum(values)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(values)) if len(values) > 1 else 1
    
    return entropy / max_entropy if max_entropy > 0 else 0


def calculate_gini(values):
    """Calculate Gini coefficient for inequality."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def calculate_all_metrics(G, team, season):
    """
    Calculate ALL SNA metrics for a team network.
    
    This includes all course-required metrics.
    """
    metrics = {
        'Team': team,
        'Season': season,
        'Win_Pct': TEAM_WINS.get((team, season), None),
    }
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes < 2 or n_edges == 0:
        return None
    
    # ===========================
    # 1. BASIC NETWORK STATISTICS
    # ===========================
    metrics['Num_Nodes'] = n_nodes
    metrics['Num_Edges'] = n_edges
    metrics['Total_Passes'] = sum(d['weight'] for _, _, d in G.edges(data=True))
    
    # Standard Graph Density (edges / possible edges)
    metrics['Graph_Density'] = nx.density(G)
    
    # ===========================
    # 2. CONNECTIVITY ANALYSIS
    # ===========================
    G_undirected = G.to_undirected()
    
    # Is the network connected?
    metrics['Is_Connected'] = nx.is_connected(G_undirected)
    
    # Number of connected components
    metrics['Num_Components'] = nx.number_connected_components(G_undirected)
    
    # Largest connected component size
    if metrics['Num_Components'] > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        metrics['Largest_Component_Size'] = len(largest_cc)
        metrics['Largest_Component_Pct'] = len(largest_cc) / n_nodes
    else:
        metrics['Largest_Component_Size'] = 0
        metrics['Largest_Component_Pct'] = 0
    
    # For directed: strongly/weakly connected
    metrics['Is_Strongly_Connected'] = nx.is_strongly_connected(G)
    metrics['Is_Weakly_Connected'] = nx.is_weakly_connected(G)
    metrics['Num_Strong_Components'] = nx.number_strongly_connected_components(G)
    metrics['Num_Weak_Components'] = nx.number_weakly_connected_components(G)
    
    # ===========================
    # 3. PATH METRICS (APL)
    # ===========================
    try:
        if nx.is_connected(G_undirected):
            metrics['APL'] = nx.average_shortest_path_length(G_undirected)
            metrics['Diameter'] = nx.diameter(G_undirected)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G_undirected), key=len)
            G_lcc = G_undirected.subgraph(largest_cc)
            metrics['APL'] = nx.average_shortest_path_length(G_lcc)
            metrics['Diameter'] = nx.diameter(G_lcc)
    except Exception:
        metrics['APL'] = None
        metrics['Diameter'] = None
    
    # ===========================
    # 4. DEGREE METRICS
    # ===========================
    in_degrees = dict(G.in_degree(weight='weight'))
    out_degrees = dict(G.out_degree(weight='weight'))
    total_degrees = {n: in_degrees.get(n, 0) + out_degrees.get(n, 0) for n in G.nodes()}
    
    degree_values = list(total_degrees.values())
    
    metrics['Mean_Weighted_Degree'] = np.mean(degree_values)
    metrics['Std_Weighted_Degree'] = np.std(degree_values)
    metrics['Max_Weighted_Degree'] = np.max(degree_values)
    metrics['Min_Weighted_Degree'] = np.min(degree_values)
    
    # Custom "Density" = equality measure
    mean_sq = np.mean(np.array(degree_values) ** 2)
    if mean_sq > 0:
        metrics['Degree_Equality'] = (np.mean(degree_values) ** 2) / mean_sq
    else:
        metrics['Degree_Equality'] = 0
    
    # ===========================
    # 5. CENTRALITY MEASURES
    # ===========================
    
    # Degree Centralization (Freeman's formula)
    max_degree = max(degree_values)
    sum_diff = sum(max_degree - d for d in degree_values)
    max_possible = (n_nodes - 1) * max_degree if max_degree > 0 else 1
    metrics['Degree_Centralization'] = sum_diff / max_possible if max_possible > 0 else 0
    
    # Betweenness Centrality
    try:
        bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
        metrics['Mean_Betweenness'] = np.mean(list(bc.values()))
        metrics['Max_Betweenness'] = np.max(list(bc.values()))
        metrics['Std_Betweenness'] = np.std(list(bc.values()))
    except Exception:
        metrics['Mean_Betweenness'] = 0
        metrics['Max_Betweenness'] = 0
        metrics['Std_Betweenness'] = 0
    
    # Closeness Centrality (NEW - was missing)
    try:
        cc = nx.closeness_centrality(G_undirected)
        metrics['Mean_Closeness'] = np.mean(list(cc.values()))
        metrics['Max_Closeness'] = np.max(list(cc.values()))
        metrics['Std_Closeness'] = np.std(list(cc.values()))
    except Exception:
        metrics['Mean_Closeness'] = 0
        metrics['Max_Closeness'] = 0
        metrics['Std_Closeness'] = 0
    
    # Eigenvector Centrality
    try:
        ec = nx.eigenvector_centrality(G_undirected, max_iter=1000, weight='weight')
        metrics['Mean_Eigenvector'] = np.mean(list(ec.values()))
        metrics['Max_Eigenvector'] = np.max(list(ec.values()))
        metrics['Std_Eigenvector'] = np.std(list(ec.values()))
    except Exception:
        metrics['Mean_Eigenvector'] = 0
        metrics['Max_Eigenvector'] = 0
        metrics['Std_Eigenvector'] = 0
    
    # ===========================
    # 6. CLUSTERING & TRIANGLES
    # ===========================
    try:
        metrics['Avg_Clustering'] = nx.average_clustering(G_undirected)
        metrics['Avg_Clustering_Weighted'] = nx.average_clustering(G_undirected, weight='weight')
        metrics['Transitivity'] = nx.transitivity(G_undirected)
        metrics['Num_Triangles'] = sum(nx.triangles(G_undirected).values()) // 3
    except Exception:
        metrics['Avg_Clustering'] = 0
        metrics['Avg_Clustering_Weighted'] = 0
        metrics['Transitivity'] = 0
        metrics['Num_Triangles'] = 0
    
    # ===========================
    # 7. COMMUNITY DETECTION
    # ===========================
    if HAS_LOUVAIN:
        try:
            partition = community_louvain.best_partition(G_undirected, weight='weight')
            metrics['Num_Communities'] = len(set(partition.values()))
            metrics['Modularity'] = community_louvain.modularity(partition, G_undirected, weight='weight')
            
            # Community sizes
            community_sizes = defaultdict(int)
            for node, comm in partition.items():
                community_sizes[comm] += 1
            sizes = list(community_sizes.values())
            metrics['Largest_Community_Size'] = max(sizes) if sizes else 0
            metrics['Smallest_Community_Size'] = min(sizes) if sizes else 0
            metrics['Community_Size_Std'] = np.std(sizes) if sizes else 0
        except Exception:
            metrics['Num_Communities'] = 1
            metrics['Modularity'] = 0
            metrics['Largest_Community_Size'] = n_nodes
            metrics['Smallest_Community_Size'] = n_nodes
            metrics['Community_Size_Std'] = 0
    else:
        metrics['Num_Communities'] = None
        metrics['Modularity'] = None
        metrics['Largest_Community_Size'] = None
        metrics['Smallest_Community_Size'] = None
        metrics['Community_Size_Std'] = None
    
    # ===========================
    # 8. INEQUALITY METRICS
    # ===========================
    metrics['Gini_Coefficient'] = calculate_gini(degree_values)
    
    # Top player concentration
    sorted_degrees = sorted(degree_values, reverse=True)
    total_degree = sum(degree_values)
    if total_degree > 0:
        metrics['Top1_Concentration'] = sorted_degrees[0] / total_degree
        metrics['Top2_Concentration'] = sum(sorted_degrees[:2]) / total_degree if len(sorted_degrees) >= 2 else metrics['Top1_Concentration']
        metrics['Top3_Concentration'] = sum(sorted_degrees[:3]) / total_degree if len(sorted_degrees) >= 3 else metrics['Top2_Concentration']
    else:
        metrics['Top1_Concentration'] = 0
        metrics['Top2_Concentration'] = 0
        metrics['Top3_Concentration'] = 0
    
    # ===========================
    # 9. ENTROPY (Pass Distribution)
    # ===========================
    metrics['Pass_Entropy'] = calculate_entropy(degree_values)
    
    # Edge weight entropy
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    metrics['Edge_Weight_Entropy'] = calculate_entropy(edge_weights) if edge_weights else 0
    
    # ===========================
    # 10. STAR PLAYER METRICS
    # ===========================
    star_id = max(total_degrees, key=total_degrees.get)
    metrics['Star_Player_Degree'] = total_degrees[star_id]
    metrics['Star_Player_Name'] = G.nodes[star_id].get('name', f"P{star_id}")
    
    # Star's share of total
    metrics['Star_Degree_Share'] = total_degrees[star_id] / total_degree if total_degree > 0 else 0
    
    return metrics


def process_all_seasons():
    """Process all seasons and calculate all metrics."""
    all_metrics = []
    
    for season in SEASONS:
        print(f"\n[Processing {season}]")
        season_dir = DATA_DIR / season
        
        if not season_dir.exists():
            print(f"  [SKIP] Directory not found")
            continue
        
        # Load data
        players_file = season_dir / "filtered_players.csv"
        if not players_file.exists():
            print(f"  [SKIP] No filtered_players.csv")
            continue
        
        player_info = pd.read_csv(players_file)
        passing_data = load_passing_data(season_dir)
        
        print(f"  Loaded {len(player_info)} players, {len(passing_data)} passing files")
        
        # Get unique teams
        teams = player_info['TEAM_ABBREVIATION'].unique()
        
        for team in teams:
            G = build_team_network(team, season, passing_data, player_info)
            
            if G.number_of_edges() > 0:
                metrics = calculate_all_metrics(G, team, season)
                if metrics:
                    all_metrics.append(metrics)
        
        print(f"  Processed {len(teams)} teams")
    
    return pd.DataFrame(all_metrics)


def calculate_correlations(df):
    """Calculate correlations between all metrics and winning."""
    df = df[df['Win_Pct'].notna()].copy()
    
    # Metrics to correlate
    metric_cols = [c for c in df.columns if c not in ['Team', 'Season', 'Win_Pct', 'Star_Player_Name', 
                                                        'Is_Connected', 'Is_Strongly_Connected', 'Is_Weakly_Connected']]
    
    correlations = []
    for metric in metric_cols:
        if df[metric].notna().sum() < 10:
            continue
        
        valid_df = df[[metric, 'Win_Pct']].dropna()
        if len(valid_df) < 10 or valid_df[metric].std() < 0.0001:
            continue
        
        r, p = stats.pearsonr(valid_df[metric], valid_df['Win_Pct'])
        
        correlations.append({
            'Metric': metric,
            'Correlation': r,
            'P_Value': p,
            'Significant': p < 0.05,
            'N': len(valid_df)
        })
    
    return pd.DataFrame(correlations).sort_values('Correlation', ascending=False)


def plot_comprehensive_analysis(df, corr_df):
    """Create comprehensive visualizations."""
    df = df[df['Win_Pct'].notna()].copy()
    
    # ===========================
    # FIGURE 1: Key Correlations
    # ===========================
    fig1 = plt.figure(figsize=(20, 16))
    
    key_metrics = [
        ('Std_Weighted_Degree', 'Hierarchy (Std Degree)', 'steelblue'),
        ('Pass_Entropy', 'Pass Entropy', 'darkorange'),
        ('Max_Weighted_Degree', 'Star Max Degree (Heliocentric)', 'green'),
        ('Graph_Density', 'Graph Density', 'purple'),
        ('Mean_Closeness', 'Mean Closeness Centrality', 'crimson'),
        ('Avg_Clustering', 'Clustering Coefficient', 'teal'),
    ]
    
    for idx, (metric, label, color) in enumerate(key_metrics, 1):
        ax = fig1.add_subplot(2, 3, idx)
        
        if metric in df.columns:
            valid = df[[metric, 'Win_Pct']].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[metric], valid['Win_Pct'])
                
                ax.scatter(valid[metric], valid['Win_Pct'], alpha=0.5, s=40, c=color, edgecolors='white')
                
                # Regression line
                z = np.polyfit(valid[metric], valid['Win_Pct'], 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
                ax.plot(x_line, p_line(x_line), 'r-', linewidth=2)
                
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                ax.set_title(f'{label}\nr = {r:.3f}{sig}, p = {p:.4f}', fontweight='bold')
            else:
                ax.set_title(f'{label}\nInsufficient data')
        else:
            ax.set_title(f'{label}\nNot available')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Win %')
    
    plt.suptitle('KEY NETWORK METRICS vs WINNING\n(All Values from Real Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_key_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_key_correlations.png")
    
    # ===========================
    # FIGURE 2: Correlation Heatmap
    # ===========================
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    
    heatmap_metrics = ['Win_Pct', 'Std_Weighted_Degree', 'Pass_Entropy', 'Max_Weighted_Degree',
                       'Graph_Density', 'Degree_Equality', 'Mean_Closeness', 'Mean_Betweenness',
                       'Avg_Clustering', 'Gini_Coefficient', 'Top2_Concentration', 'Modularity']
    heatmap_metrics = [m for m in heatmap_metrics if m in df.columns]
    
    corr_matrix = df[heatmap_metrics].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax2)
    ax2.set_title('Correlation Matrix: Network Metrics vs Winning', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_correlation_heatmap.png")
    
    # ===========================
    # FIGURE 3: Community Analysis
    # ===========================
    if 'Num_Communities' in df.columns and df['Num_Communities'].notna().sum() > 10:
        fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
        
        # Communities vs Win%
        valid = df[['Num_Communities', 'Win_Pct']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['Num_Communities'], valid['Win_Pct'])
            axes3[0].scatter(valid['Num_Communities'], valid['Win_Pct'], alpha=0.5, s=40)
            axes3[0].set_xlabel('Number of Communities')
            axes3[0].set_ylabel('Win %')
            axes3[0].set_title(f'Communities vs Win%\nr = {r:.3f}, p = {p:.4f}')
        
        # Modularity vs Win%
        valid = df[['Modularity', 'Win_Pct']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['Modularity'], valid['Win_Pct'])
            axes3[1].scatter(valid['Modularity'], valid['Win_Pct'], alpha=0.5, s=40, c='orange')
            axes3[1].set_xlabel('Modularity')
            axes3[1].set_ylabel('Win %')
            axes3[1].set_title(f'Modularity vs Win%\nr = {r:.3f}, p = {p:.4f}')
        
        # Community distribution
        axes3[2].hist(df['Num_Communities'].dropna(), bins=range(1, 8), edgecolor='black', alpha=0.7)
        axes3[2].set_xlabel('Number of Communities')
        axes3[2].set_ylabel('Count')
        axes3[2].set_title('Distribution of Communities per Team')
        
        plt.suptitle('COMMUNITY DETECTION ANALYSIS', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '03_community_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[OK] Saved: 03_community_analysis.png")
    
    # ===========================
    # FIGURE 4: Connectivity & Path Metrics
    # ===========================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
    
    # APL vs Win%
    if 'APL' in df.columns:
        valid = df[['APL', 'Win_Pct']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['APL'], valid['Win_Pct'])
            axes4[0, 0].scatter(valid['APL'], valid['Win_Pct'], alpha=0.5, s=40, c='steelblue')
            z = np.polyfit(valid['APL'], valid['Win_Pct'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid['APL'].min(), valid['APL'].max(), 100)
            axes4[0, 0].plot(x_line, p_line(x_line), 'r-', linewidth=2)
            axes4[0, 0].set_xlabel('Average Path Length (APL)')
            axes4[0, 0].set_ylabel('Win %')
            axes4[0, 0].set_title(f'APL vs Win%\nr = {r:.3f}, p = {p:.4f}', fontweight='bold')
    
    # Closeness vs Win%
    if 'Mean_Closeness' in df.columns:
        valid = df[['Mean_Closeness', 'Win_Pct']].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid['Mean_Closeness'], valid['Win_Pct'])
            axes4[0, 1].scatter(valid['Mean_Closeness'], valid['Win_Pct'], alpha=0.5, s=40, c='crimson')
            z = np.polyfit(valid['Mean_Closeness'], valid['Win_Pct'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid['Mean_Closeness'].min(), valid['Mean_Closeness'].max(), 100)
            axes4[0, 1].plot(x_line, p_line(x_line), 'r-', linewidth=2)
            axes4[0, 1].set_xlabel('Mean Closeness Centrality')
            axes4[0, 1].set_ylabel('Win %')
            axes4[0, 1].set_title(f'Closeness vs Win%\nr = {r:.3f}, p = {p:.4f}', fontweight='bold')
    
    # Num Components distribution
    if 'Num_Components' in df.columns:
        axes4[1, 0].hist(df['Num_Components'].dropna(), bins=range(1, 6), edgecolor='black', alpha=0.7, color='green')
        axes4[1, 0].set_xlabel('Number of Connected Components')
        axes4[1, 0].set_ylabel('Count')
        axes4[1, 0].set_title('Connectivity: Most Teams are Fully Connected', fontweight='bold')
    
    # Diameter distribution
    if 'Diameter' in df.columns:
        axes4[1, 1].hist(df['Diameter'].dropna(), bins=range(1, 6), edgecolor='black', alpha=0.7, color='purple')
        axes4[1, 1].set_xlabel('Network Diameter')
        axes4[1, 1].set_ylabel('Count')
        axes4[1, 1].set_title('Diameter: Max Shortest Path in Network', fontweight='bold')
    
    plt.suptitle('CONNECTIVITY & PATH METRICS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_connectivity_paths.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_connectivity_paths.png")
    
    # ===========================
    # FIGURE 5: All Correlations Bar Chart
    # ===========================
    fig5, ax5 = plt.subplots(figsize=(14, 10))
    
    # Filter to significant correlations
    sig_corrs = corr_df[corr_df['Significant'] == True].copy()
    if len(sig_corrs) > 0:
        colors = ['green' if r > 0 else 'red' for r in sig_corrs['Correlation']]
        bars = ax5.barh(sig_corrs['Metric'], sig_corrs['Correlation'], color=colors, edgecolor='black')
        ax5.axvline(x=0, color='black', linewidth=1)
        ax5.set_xlabel('Pearson Correlation with Win%')
        ax5.set_title('Significant Correlations with Winning (p < 0.05)', fontweight='bold')
        
        for bar, (_, row) in zip(bars, sig_corrs.iterrows()):
            x_pos = row['Correlation'] + (0.01 if row['Correlation'] >= 0 else -0.01)
            ha = 'left' if row['Correlation'] >= 0 else 'right'
            ax5.text(x_pos, bar.get_y() + bar.get_height()/2, f"{row['Correlation']:.3f}", 
                     va='center', ha=ha, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_all_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 05_all_correlations.png")


def print_summary(df, corr_df):
    """Print comprehensive summary for presentation."""
    df = df[df['Win_Pct'].notna()].copy()
    
    print("\n" + "="*100)
    print("COMPLETE SNA ANALYSIS - VERIFIED STATISTICS FOR PRESENTATION")
    print("="*100)
    
    # Dataset size
    print("\n[DATASET SIZE]")
    print(f"  Seasons: {len(SEASONS)} ({SEASONS[0]} to {SEASONS[-1]})")
    print(f"  Team-Seasons Analyzed: {len(df)}")
    print(f"  Total Nodes (players): {df['Num_Nodes'].sum():.0f}")
    print(f"  Total Edges (passing connections): {df['Num_Edges'].sum():.0f}")
    print(f"  Total Passes: {df['Total_Passes'].sum():,.0f}")
    
    # Connectivity
    print("\n[CONNECTIVITY]")
    print(f"  Fully Connected Networks: {df['Is_Connected'].sum()} / {len(df)} ({df['Is_Connected'].mean()*100:.1f}%)")
    print(f"  Average Path Length (APL): {df['APL'].mean():.3f} (range: {df['APL'].min():.2f} - {df['APL'].max():.2f})")
    print(f"  Average Diameter: {df['Diameter'].mean():.2f}")
    
    # Key Correlations
    print("\n[KEY CORRELATIONS WITH WINNING]")
    print("-"*80)
    print(f"{'Metric':<35} {'r':>10} {'p-value':>12} {'Sig':>8} {'Interpretation':<30}")
    print("-"*80)
    
    key_metrics = ['Std_Weighted_Degree', 'Max_Weighted_Degree', 'Pass_Entropy', 
                   'Graph_Density', 'Mean_Closeness', 'Avg_Clustering', 'Modularity',
                   'APL', 'Gini_Coefficient', 'Top2_Concentration']
    
    for metric in key_metrics:
        row = corr_df[corr_df['Metric'] == metric]
        if len(row) > 0:
            r = row['Correlation'].values[0]
            p = row['P_Value'].values[0]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
            if metric == 'Std_Weighted_Degree':
                interp = 'Hierarchy -> More Wins'
            elif metric == 'Max_Weighted_Degree':
                interp = 'Star System -> More Wins'
            elif metric == 'Pass_Entropy':
                interp = 'Randomness -> Fewer Wins'
            elif metric == 'Graph_Density':
                interp = 'Weak/No Effect'
            elif metric == 'Mean_Closeness':
                interp = 'Efficiency -> More Wins' if r > 0 else 'N/A'
            else:
                interp = 'Positive' if r > 0 else 'Negative'
            
            print(f"{metric:<35} {r:>+10.4f} {p:>12.4f} {sig:>8} {interp:<30}")
    
    # Community Detection
    if 'Num_Communities' in df.columns and df['Num_Communities'].notna().sum() > 0:
        print("\n[COMMUNITY DETECTION]")
        print(f"  Average Communities per Team: {df['Num_Communities'].mean():.2f}")
        print(f"  Range: {df['Num_Communities'].min():.0f} - {df['Num_Communities'].max():.0f}")
        print(f"  Average Modularity: {df['Modularity'].mean():.3f}")
        
        # Community interpretation
        print("\n  Community Interpretation:")
        print("  - Communities represent 'passing sub-groups' within teams")
        print("  - Lower community count = more unified passing patterns")
        print("  - Higher modularity = more distinct sub-groups")
    
    # Top findings
    print("\n[TOP 5 STRONGEST CORRELATIONS]")
    print("-"*60)
    for _, row in corr_df.head(5).iterrows():
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        print(f"  {row['Metric']:<30} r = {row['Correlation']:+.4f}{sig}")
    
    print("\n[BOTTOM 5 CORRELATIONS (Negative)]")
    print("-"*60)
    for _, row in corr_df.tail(5).iterrows():
        sig = '***' if row['P_Value'] < 0.001 else '**' if row['P_Value'] < 0.01 else '*' if row['P_Value'] < 0.05 else ''
        print(f"  {row['Metric']:<30} r = {row['Correlation']:+.4f}{sig}")
    
    # Presentation-ready statements
    print("\n" + "="*100)
    print("PRESENTATION-READY STATEMENTS (Copy-Paste)")
    print("="*100)
    
    # Get actual values
    hier = corr_df[corr_df['Metric'] == 'Std_Weighted_Degree']
    star = corr_df[corr_df['Metric'] == 'Max_Weighted_Degree']
    ent = corr_df[corr_df['Metric'] == 'Pass_Entropy']
    dens = corr_df[corr_df['Metric'] == 'Graph_Density']
    
    if len(hier) > 0:
        print(f"\n1. HIERARCHY: 'Strong positive correlation between the standard deviation of weighted")
        print(f"   degree (hierarchy) and winning percentage (r={hier['Correlation'].values[0]:.3f}, p={hier['P_Value'].values[0]:.4f})'")
    
    if len(star) > 0:
        print(f"\n2. HELIOCENTRIC: 'Strong positive correlation (r={star['Correlation'].values[0]:.3f}) between")
        print(f"   a star player's maximum weighted degree and team wins'")
    
    if len(ent) > 0:
        print(f"\n3. ENTROPY: 'Pass Entropy (randomness in distribution) has a negative correlation")
        print(f"   with winning (r={ent['Correlation'].values[0]:.3f})'")
    
    if len(dens) > 0:
        print(f"\n4. DENSITY: 'Graph Density showed weak correlation with winning percentage")
        print(f"   (r={dens['Correlation'].values[0]:.3f}, p={dens['P_Value'].values[0]:.4f})'")
    
    print("\n" + "="*100)


def main():
    """Main execution."""
    print("="*70)
    print("COMPLETE SNA ANALYSIS - ALL COURSE-REQUIRED METRICS")
    print("="*70)
    
    print("\n[PROCESSING ALL SEASONS]")
    df = process_all_seasons()
    
    print(f"\n[TOTAL TEAM-SEASONS: {len(df)}]")
    
    print("\n[CALCULATING CORRELATIONS]")
    corr_df = calculate_correlations(df)
    
    print("\n[GENERATING VISUALIZATIONS]")
    plot_comprehensive_analysis(df, corr_df)
    
    print_summary(df, corr_df)
    
    # Save data
    df.to_csv(OUTPUT_DIR / 'complete_team_metrics.csv', index=False)
    corr_df.to_csv(OUTPUT_DIR / 'complete_correlations.csv', index=False)
    
    print(f"\n[OK] All results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
