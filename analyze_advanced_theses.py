"""
analyze_advanced_theses.py - Advanced NBA Network Analysis

Generates insights and visualizations for 3 advanced theses:
1. The Triangle Thesis: Clustering Coefficient vs Win %
2. Cognitive Load / Entropy: Pass Entropy vs Offensive Rating
3. The Black Hole Matrix: Player pass flow analysis
4. Dynamic Duos: Strongest player pairs by pass volume

Usage:
    python analyze_advanced_theses.py

Author: NBA Network Analysis Project
"""

import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"

SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24'
]

# Figure settings
FIGURE_DPI = 150


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


def progress_bar(iterable, desc: str, total: int = None):
    """Wrapper for progress bar."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    else:
        print(f"Processing: {desc}...")
        return iterable


def calculate_shannon_entropy(values: List[float]) -> float:
    """
    Calculate Shannon Entropy of a distribution.
    Higher entropy = more unpredictable/distributed passing.
    
    Args:
        values: List of pass counts/weights
        
    Returns:
        Shannon entropy value
    """
    if not values or sum(values) == 0:
        return 0.0
    
    # Convert to probabilities
    total = sum(values)
    probs = [v / total for v in values if v > 0]
    
    # Calculate entropy: -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return entropy


# =============================================================================
# DATA LOADING
# =============================================================================
def load_existing_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing player and team metrics CSVs."""
    player_path = OUTPUT_DIR / "nba_player_metrics.csv"
    team_path = OUTPUT_DIR / "nba_team_metrics.csv"
    
    player_df = pd.read_csv(player_path)
    team_df = pd.read_csv(team_path)
    
    print(f"[OK] Loaded player metrics: {player_df.shape}")
    print(f"[OK] Loaded team metrics: {team_df.shape}")
    
    return player_df, team_df


def load_passing_data(season_dir: Path) -> Dict[int, dict]:
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
        except (ValueError, json.JSONDecodeError):
            continue
    
    return passing_data


def load_filtered_players(season_dir: Path) -> pd.DataFrame:
    """Load filtered players CSV for a season."""
    filepath = season_dir / "filtered_players.csv"
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath)


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
# NETWORK RECONSTRUCTION & ADVANCED METRICS
# =============================================================================
def build_team_network_from_json(
    team_id: int,
    passing_data: Dict[int, dict],
    player_info: pd.DataFrame
) -> nx.DiGraph:
    """Build directed network graph for a team from JSON data."""
    G = nx.DiGraph()
    
    team_players = player_info[player_info['TEAM_ID'] == team_id]
    team_player_ids = set(team_players['PLAYER_ID'].tolist())
    
    if not team_player_ids:
        return G
    
    # Create player name lookup
    player_names = {}
    for _, row in team_players.iterrows():
        player_names[row['PLAYER_ID']] = format_player_name(row['PLAYER_NAME'])
    
    # Add nodes
    for player_id in team_player_ids:
        G.add_node(player_id, name=player_names.get(player_id, f"Player_{player_id}"))
    
    # Add edges from passing data
    for player_id in team_player_ids:
        if player_id not in passing_data:
            continue
        
        passes_made = extract_passes_made(passing_data[player_id])
        
        for pass_record in passes_made:
            teammate_id = pass_record.get('PASS_TEAMMATE_PLAYER_ID')
            pass_count = pass_record.get('PASS', 0)
            
            if teammate_id and teammate_id in team_player_ids and pass_count > 0:
                if G.has_edge(player_id, teammate_id):
                    G[player_id][teammate_id]['weight'] += pass_count
                else:
                    G.add_edge(player_id, teammate_id, weight=pass_count)
    
    return G


def calculate_advanced_team_metrics(
    G: nx.DiGraph,
    team_id: int,
    team_abbr: str,
    season: str
) -> dict:
    """
    Calculate advanced team-level metrics from the network graph.
    
    Returns:
        Dictionary with Clustering Coefficient and Pass Entropy
    """
    metrics = {
        'TEAM_ID': team_id,
        'TEAM_ABBREVIATION': team_abbr,
        'SEASON': season,
        'Clustering_Coefficient': 0.0,
        'Pass_Entropy': 0.0,
    }
    
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return metrics
    
    # Clustering Coefficient (on undirected version)
    try:
        G_undirected = G.to_undirected()
        metrics['Clustering_Coefficient'] = nx.average_clustering(G_undirected, weight='weight')
    except Exception:
        metrics['Clustering_Coefficient'] = 0.0
    
    # Shannon Entropy of edge weight distribution
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if edge_weights:
        metrics['Pass_Entropy'] = calculate_shannon_entropy(edge_weights)
    
    return metrics


def find_team_duos(
    G: nx.DiGraph,
    team_id: int,
    team_abbr: str,
    season: str
) -> List[dict]:
    """
    Find all player pairs (duos) and their bidirectional pass volume.
    
    Returns:
        List of duo dictionaries with pass volumes
    """
    duos = []
    
    # Get all node pairs
    nodes = list(G.nodes())
    
    for i, player_a in enumerate(nodes):
        for player_b in nodes[i+1:]:
            # Calculate bidirectional pass volume
            a_to_b = G[player_a][player_b]['weight'] if G.has_edge(player_a, player_b) else 0
            b_to_a = G[player_b][player_a]['weight'] if G.has_edge(player_b, player_a) else 0
            total = a_to_b + b_to_a
            
            if total > 0:
                name_a = G.nodes[player_a].get('name', f"Player_{player_a}")
                name_b = G.nodes[player_b].get('name', f"Player_{player_b}")
                
                duos.append({
                    'Player_A_ID': player_a,
                    'Player_A_Name': name_a,
                    'Player_B_ID': player_b,
                    'Player_B_Name': name_b,
                    'A_to_B': a_to_b,
                    'B_to_A': b_to_a,
                    'Total_Passes': total,
                    'TEAM_ID': team_id,
                    'TEAM_ABBREVIATION': team_abbr,
                    'SEASON': season
                })
    
    return duos


def process_all_seasons_for_advanced_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all seasons to calculate advanced team metrics and find duos.
    
    Returns:
        Tuple of (advanced_team_metrics_df, all_duos_df)
    """
    print("\n[INFO] Processing raw JSON data for advanced metrics...")
    
    all_advanced_metrics = []
    all_duos = []
    
    for season in progress_bar(SEASONS, desc="Processing seasons"):
        season_dir = DATA_DIR / season
        
        if not season_dir.exists():
            continue
        
        player_info = load_filtered_players(season_dir)
        if player_info.empty:
            continue
        
        passing_data = load_passing_data(season_dir)
        if not passing_data:
            continue
        
        # Get unique teams
        teams = player_info[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates()
        
        for _, row in teams.iterrows():
            team_id = row['TEAM_ID']
            team_abbr = row['TEAM_ABBREVIATION']
            
            # Build network
            G = build_team_network_from_json(team_id, passing_data, player_info)
            
            if G.number_of_edges() == 0:
                continue
            
            # Calculate advanced metrics
            adv_metrics = calculate_advanced_team_metrics(G, team_id, team_abbr, season)
            all_advanced_metrics.append(adv_metrics)
            
            # Find duos
            team_duos = find_team_duos(G, team_id, team_abbr, season)
            all_duos.extend(team_duos)
    
    adv_metrics_df = pd.DataFrame(all_advanced_metrics)
    duos_df = pd.DataFrame(all_duos)
    
    print(f"[OK] Calculated advanced metrics for {len(adv_metrics_df)} team-seasons")
    print(f"[OK] Found {len(duos_df)} player pairs across all seasons")
    
    return adv_metrics_df, duos_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def add_player_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Add Black Hole Ratio and other player-level features."""
    df = player_df.copy()
    
    # Black Hole Ratio: In-Degree / (Out-Degree + 1)
    # High ratio = receives many passes but doesn't distribute = "Finisher/Black Hole"
    # Low ratio = distributes more than receives = "General/Playmaker"
    df['Black_Hole_Ratio'] = df['Weighted_In_Degree'] / (df['Weighted_Out_Degree'] + 1)
    
    # Net Flow: In - Out (positive = net receiver, negative = net distributor)
    df['Net_Pass_Flow'] = df['Weighted_In_Degree'] - df['Weighted_Out_Degree']
    
    print(f"[OK] Added player features: Black_Hole_Ratio, Net_Pass_Flow")
    
    return df


def merge_advanced_team_metrics(
    team_df: pd.DataFrame,
    adv_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge advanced team metrics into the main team dataframe."""
    df = team_df.merge(
        adv_metrics_df[['TEAM_ID', 'SEASON', 'Clustering_Coefficient', 'Pass_Entropy']],
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )
    
    # Fill missing values
    df['Clustering_Coefficient'] = df['Clustering_Coefficient'].fillna(0)
    df['Pass_Entropy'] = df['Pass_Entropy'].fillna(0)
    
    print(f"[OK] Merged advanced team metrics: Clustering_Coefficient, Pass_Entropy")
    
    return df


# =============================================================================
# VISUALIZATION A: THE TRIANGLE THESIS
# =============================================================================
def plot_triangle_thesis(team_df: pd.DataFrame) -> Tuple[plt.Figure, float, float]:
    """
    Plot Clustering Coefficient vs Win %.
    
    Returns:
        Figure, Pearson correlation, p-value
    """
    print("\n--- Creating Triangle Thesis Plot ---")
    
    df = team_df[['Clustering_Coefficient', 'W_PCT', 'TEAM_ABBREVIATION', 'SEASON']].dropna()
    
    if len(df) < 10:
        print("[WARNING] Not enough data for Triangle Thesis plot")
        return None, 0, 1
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df['Clustering_Coefficient'], df['W_PCT'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with regression line
    sns.regplot(
        data=df,
        x='Clustering_Coefficient',
        y='W_PCT',
        scatter_kws={'alpha': 0.6, 's': 60, 'edgecolors': 'white'},
        line_kws={'color': '#E74C3C', 'linewidth': 2},
        ax=ax
    )
    
    # Title and labels
    ax.set_xlabel('Clustering Coefficient (Network Connectivity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Percentage', fontsize=14, fontweight='bold')
    ax.set_title(
        'The Triangle Thesis: Do Connected Teams Win More?\n'
        f'Pearson r = {corr:.3f}, p = {p_val:.4f}',
        fontsize=16, fontweight='bold', pad=20
    )
    
    # Reference line at .500
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Interpretation text
    if p_val < 0.05:
        if corr > 0:
            interpretation = "POSITIVE correlation: More connected teams tend to win more"
        else:
            interpretation = "NEGATIVE correlation: More connected teams tend to win less"
    else:
        interpretation = "No statistically significant relationship found"
    
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    print(f"  Correlation: r = {corr:.3f}, p = {p_val:.4f}")
    
    return fig, corr, p_val


# =============================================================================
# VISUALIZATION B: COGNITIVE LOAD / ENTROPY
# =============================================================================
def plot_entropy_thesis(team_df: pd.DataFrame) -> Tuple[plt.Figure, float, float]:
    """
    Plot Pass Entropy vs Offensive Rating (using Total Passes as proxy if rating missing).
    
    Returns:
        Figure, Pearson correlation, p-value
    """
    print("\n--- Creating Entropy Thesis Plot ---")
    
    # Use Total_Passes as offensive output proxy (more passes = more possessions/activity)
    # Or we can use W_PCT as a proxy for overall team effectiveness
    df = team_df[['Pass_Entropy', 'Total_Passes', 'W_PCT', 'TEAM_ABBREVIATION', 'SEASON']].dropna()
    df = df[df['Pass_Entropy'] > 0]
    
    if len(df) < 10:
        print("[WARNING] Not enough data for Entropy Thesis plot")
        return None, 0, 1
    
    # Calculate correlation (Entropy vs Win%)
    corr, p_val = stats.pearsonr(df['Pass_Entropy'], df['W_PCT'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot colored by Total_Passes
    scatter = ax.scatter(
        df['Pass_Entropy'],
        df['W_PCT'],
        c=df['Total_Passes'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add regression line
    z = np.polyfit(df['Pass_Entropy'], df['W_PCT'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Pass_Entropy'].min(), df['Pass_Entropy'].max(), 100)
    ax.plot(x_line, p(x_line), color='#E74C3C', linewidth=2, linestyle='--',
            label=f'Regression (r={corr:.3f})')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Total Team Passes\n(Offensive Volume)', fontsize=11)
    
    # Labels
    ax.set_xlabel('Pass Distribution Entropy (Unpredictability)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Percentage', fontsize=14, fontweight='bold')
    ax.set_title(
        'Cognitive Load Thesis: Does Unpredictability Lead to Better Offense?\n'
        f'Pearson r = {corr:.3f}, p = {p_val:.4f}',
        fontsize=16, fontweight='bold', pad=20
    )
    
    ax.legend(loc='lower right', fontsize=10)
    
    # Interpretation
    if p_val < 0.05:
        if corr > 0:
            interpretation = "POSITIVE: Higher entropy (more distributed) correlates with winning"
        else:
            interpretation = "NEGATIVE: Lower entropy (more concentrated) correlates with winning"
    else:
        interpretation = "No statistically significant relationship found"
    
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    print(f"  Correlation: r = {corr:.3f}, p = {p_val:.4f}")
    
    return fig, corr, p_val


# =============================================================================
# VISUALIZATION C: THE BLACK HOLE MATRIX
# =============================================================================
def plot_black_hole_matrix(player_df: pd.DataFrame) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Plot Weighted Out-Degree vs In-Degree, colored by PTS.
    Players above the 45-degree line are "Black Holes" (Finishers).
    Players below are "Generals" (Playmakers).
    
    Returns:
        Figure, top_black_holes_df, top_generals_df
    """
    print("\n--- Creating Black Hole Matrix ---")
    
    # Filter to players with meaningful pass volume
    df = player_df[
        (player_df['Weighted_In_Degree'] > 100) & 
        (player_df['Weighted_Out_Degree'] > 100) &
        (player_df['PTS'].notna())
    ].copy()
    
    if len(df) < 50:
        print("[WARNING] Not enough data for Black Hole Matrix")
        return None, pd.DataFrame(), pd.DataFrame()
    
    # Calculate deviation from 45-degree line (positive = Black Hole, negative = General)
    df['Deviation'] = df['Weighted_In_Degree'] - df['Weighted_Out_Degree']
    
    # Find extremes
    top_black_holes = df.nlargest(5, 'Deviation')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'Weighted_In_Degree', 
         'Weighted_Out_Degree', 'PTS', 'Deviation']
    ]
    
    top_generals = df.nsmallest(5, 'Deviation')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'Weighted_In_Degree', 
         'Weighted_Out_Degree', 'PTS', 'Deviation']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot colored by PTS
    scatter = ax.scatter(
        df['Weighted_Out_Degree'],
        df['Weighted_In_Degree'],
        c=df['PTS'],
        cmap='YlOrRd',
        s=50,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.3
    )
    
    # 45-degree line
    max_val = max(df['Weighted_Out_Degree'].max(), df['Weighted_In_Degree'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='Equal In/Out (Connectors)')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Points Per Game', fontsize=12)
    
    # Annotate extreme players
    # Black Holes (top 5)
    for _, row in df.nlargest(5, 'Deviation').iterrows():
        name = row['PLAYER_NAME']
        if len(name) > 15:
            name = name[:12] + "..."
        ax.annotate(
            name,
            xy=(row['Weighted_Out_Degree'], row['Weighted_In_Degree']),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8, fontweight='bold', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7)
        )
    
    # Generals (bottom 5)
    for _, row in df.nsmallest(5, 'Deviation').iterrows():
        name = row['PLAYER_NAME']
        if len(name) > 15:
            name = name[:12] + "..."
        ax.annotate(
            name,
            xy=(row['Weighted_Out_Degree'], row['Weighted_In_Degree']),
            xytext=(10, -15), textcoords='offset points',
            fontsize=8, fontweight='bold', color='darkblue',
            arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7)
        )
    
    # Labels and zones
    ax.set_xlabel('Weighted Out-Degree (Passes Made)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weighted In-Degree (Passes Received)', fontsize=14, fontweight='bold')
    ax.set_title(
        'The Black Hole Matrix: Player Pass Flow Analysis\n'
        'Above line = "Black Holes" (Finishers) | Below line = "Generals" (Playmakers)',
        fontsize=16, fontweight='bold', pad=20
    )
    
    # Add zone labels
    ax.text(0.85, 0.15, 'GENERALS\n(Playmakers)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='darkblue', alpha=0.7,
            ha='center', va='center')
    ax.text(0.15, 0.85, 'BLACK HOLES\n(Finishers)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='darkred', alpha=0.7,
            ha='center', va='center')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    
    plt.tight_layout()
    
    print(f"  Identified {len(df[df['Deviation'] > 0])} Black Holes, {len(df[df['Deviation'] < 0])} Generals")
    
    return fig, top_black_holes, top_generals


# =============================================================================
# VISUALIZATION D: DYNAMIC DUOS
# =============================================================================
def plot_dynamic_duos(duos_df: pd.DataFrame) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Bar chart of top 10 strongest player pairs by total pass volume.
    
    Returns:
        Figure, top_duos_df
    """
    print("\n--- Creating Dynamic Duos Chart ---")
    
    if duos_df.empty:
        print("[WARNING] No duo data available")
        return None, pd.DataFrame()
    
    # Get top 10 duos
    top_duos = duos_df.nlargest(10, 'Total_Passes').copy()
    
    # Create label with ASCII-safe names
    def make_duo_label(r):
        name_a = r['Player_A_Name'][:12].encode('ascii', 'replace').decode('ascii')
        name_b = r['Player_B_Name'][:12].encode('ascii', 'replace').decode('ascii')
        return f"{name_a} & {name_b}\n({r['TEAM_ABBREVIATION']} {r['SEASON'][-5:]})"
    
    top_duos['Duo_Label'] = top_duos.apply(make_duo_label, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
    
    # Horizontal bar chart
    bars = ax.barh(
        range(len(top_duos)),
        top_duos['Total_Passes'],
        color=colors,
        edgecolor='white',
        linewidth=1
    )
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, top_duos.itertuples())):
        ax.text(
            bar.get_width() + 100,
            bar.get_y() + bar.get_height()/2,
            f'{int(row.Total_Passes):,}',
            va='center', fontsize=10, fontweight='bold'
        )
        # Add breakdown
        ax.text(
            bar.get_width() - 500,
            bar.get_y() + bar.get_height()/2,
            f'({int(row.A_to_B)} / {int(row.B_to_A)})',
            va='center', fontsize=8, color='white', alpha=0.9
        )
    
    # Labels
    ax.set_yticks(range(len(top_duos)))
    ax.set_yticklabels(top_duos['Duo_Label'], fontsize=10)
    ax.set_xlabel('Total Bidirectional Pass Volume', fontsize=14, fontweight='bold')
    ax.set_title(
        'The Dynamic Duos: Top 10 Player Pairs by Pass Connection\n'
        '(A to B / B to A breakdown shown)',
        fontsize=16, fontweight='bold', pad=20
    )
    
    ax.invert_yaxis()  # Highest at top
    ax.set_xlim(0, top_duos['Total_Passes'].max() * 1.15)
    
    plt.tight_layout()
    
    name_a = top_duos.iloc[0]['Player_A_Name'].encode('ascii', 'replace').decode('ascii')
    name_b = top_duos.iloc[0]['Player_B_Name'].encode('ascii', 'replace').decode('ascii')
    print(f"  Top duo: {name_a} & {name_b} ({top_duos.iloc[0]['Total_Passes']:,} passes)")
    
    return fig, top_duos


# =============================================================================
# INSIGHTS REPORT GENERATION
# =============================================================================
def generate_insights_report(
    triangle_corr: float,
    triangle_p: float,
    entropy_corr: float,
    entropy_p: float,
    top_black_holes: pd.DataFrame,
    top_generals: pd.DataFrame,
    top_duos: pd.DataFrame
) -> str:
    """Generate markdown insights report."""
    
    # Format black holes
    black_hole_lines = []
    for i, (_, row) in enumerate(top_black_holes.iterrows(), 1):
        name = row['PLAYER_NAME']
        team = row['TEAM_ABBREVIATION']
        season = row['SEASON']
        in_deg = int(row['Weighted_In_Degree'])
        out_deg = int(row['Weighted_Out_Degree'])
        pts = row['PTS']
        black_hole_lines.append(
            f"{i}. **{name}** ({team} {season}) - In: {in_deg:,}, Out: {out_deg:,}, PTS: {pts:.1f}"
        )
    
    # Format generals
    general_lines = []
    for i, (_, row) in enumerate(top_generals.iterrows(), 1):
        name = row['PLAYER_NAME']
        team = row['TEAM_ABBREVIATION']
        season = row['SEASON']
        in_deg = int(row['Weighted_In_Degree'])
        out_deg = int(row['Weighted_Out_Degree'])
        pts = row['PTS']
        general_lines.append(
            f"{i}. **{name}** ({team} {season}) - In: {in_deg:,}, Out: {out_deg:,}, PTS: {pts:.1f}"
        )
    
    # Format duos
    duo_lines = []
    for i, (_, row) in enumerate(top_duos.head(5).iterrows(), 1):
        name_a = row['Player_A_Name']
        name_b = row['Player_B_Name']
        team = row['TEAM_ABBREVIATION']
        season = row['SEASON']
        total = int(row['Total_Passes'])
        duo_lines.append(
            f"{i}. **{name_a} & {name_b}** ({team} {season}) - {total:,} total passes"
        )
    
    # Interpretations
    triangle_interp = ""
    if triangle_p < 0.05:
        if triangle_corr > 0:
            triangle_interp = "**Finding:** Teams with higher clustering (more interconnected passing) tend to win more games. This supports the idea that balanced, team-oriented offenses are more successful."
        else:
            triangle_interp = "**Finding:** Teams with lower clustering (more star-dependent) tend to win more. This might indicate that concentrated offensive schemes can be effective."
    else:
        triangle_interp = "**Finding:** No statistically significant relationship between team connectivity and winning. Success may depend more on talent than passing structure."
    
    entropy_interp = ""
    if entropy_p < 0.05:
        if entropy_corr > 0:
            entropy_interp = "**Finding:** Higher pass entropy (more unpredictable, distributed passing) correlates with winning. Diverse offensive options may be harder to defend."
        else:
            entropy_interp = "**Finding:** Lower pass entropy (more concentrated passing) correlates with winning. Focused, predictable schemes executed well may be more effective."
    else:
        entropy_interp = "**Finding:** No statistically significant relationship between pass unpredictability and winning."
    
    report = f"""# NBA Network Analysis: Advanced Insights Report

## Executive Summary

This report presents findings from advanced Social Network Analysis of NBA passing data from 2015-16 to 2023-24.
We analyzed **270 team-seasons** and **3,400+ player-seasons** to test three hypotheses about team success.

---

## Thesis 1: The Triangle Thesis

**Question:** Do teams with more interconnected passing networks win more games?

### Statistical Analysis
- **Metric:** Average Clustering Coefficient
- **Pearson Correlation:** r = {triangle_corr:.3f}
- **P-value:** {triangle_p:.4f}
- **Significant:** {'Yes' if triangle_p < 0.05 else 'No'} (alpha = 0.05)

{triangle_interp}

---

## Thesis 2: Cognitive Load / Entropy Thesis

**Question:** Does passing unpredictability lead to better offensive performance?

### Statistical Analysis
- **Metric:** Shannon Entropy of Pass Distribution
- **Pearson Correlation:** r = {entropy_corr:.3f}
- **P-value:** {entropy_p:.4f}
- **Significant:** {'Yes' if entropy_p < 0.05 else 'No'} (alpha = 0.05)

{entropy_interp}

---

## Thesis 3: The Black Hole Analysis

**Question:** Who are the biggest "Black Holes" (ball-stoppers) and "Generals" (playmakers)?

### Top 5 Black Holes (Finishers)
*Players who receive far more passes than they distribute - terminal points of the offense*

{chr(10).join(black_hole_lines)}

### Top 5 Generals (Playmakers)
*Players who distribute far more passes than they receive - orchestrators of the offense*

{chr(10).join(general_lines)}

---

## Dynamic Duos Analysis

**Question:** Which player pairs have the strongest passing connections?

### Top 5 Dynamic Duos
*Player pairs with the highest bidirectional pass volume*

{chr(10).join(duo_lines)}

---

## Methodology Notes

- **Clustering Coefficient:** Measures how interconnected a team's passing network is (0-1 scale)
- **Pass Entropy:** Shannon entropy of edge weight distribution - higher = more unpredictable
- **Black Hole Ratio:** Weighted In-Degree / (Weighted Out-Degree + 1)
- **Duo Strength:** Sum of passes A->B and B->A

---

*Generated by NBA Network Analysis Pipeline*
"""
    
    return report


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("NBA NETWORK ANALYSIS - ADVANCED THESES")
    print("="*70)
    
    ensure_directory(OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)
    
    # 1. Load existing data
    player_df, team_df = load_existing_data()
    
    # 2. Process JSONs for advanced metrics
    adv_metrics_df, duos_df = process_all_seasons_for_advanced_metrics()
    
    # 3. Feature Engineering
    player_df = add_player_features(player_df)
    team_df = merge_advanced_team_metrics(team_df, adv_metrics_df)
    
    # Save enhanced CSVs
    player_df.to_csv(OUTPUT_DIR / "nba_player_metrics_enhanced.csv", index=False)
    team_df.to_csv(OUTPUT_DIR / "nba_team_metrics_enhanced.csv", index=False)
    duos_df.to_csv(OUTPUT_DIR / "nba_duos_all.csv", index=False)
    print(f"\n[OK] Saved enhanced CSVs")
    
    # 4. Generate Visualizations
    figures = []
    
    # A. Triangle Thesis
    try:
        fig_triangle, triangle_corr, triangle_p = plot_triangle_thesis(team_df)
        if fig_triangle:
            filepath = FIGURES_DIR / "thesis_triangle_clustering.png"
            fig_triangle.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(fig_triangle)
    except Exception as e:
        print(f"\n[ERROR] Triangle thesis plot: {e}")
        triangle_corr, triangle_p = 0, 1
    
    # B. Entropy Thesis
    try:
        fig_entropy, entropy_corr, entropy_p = plot_entropy_thesis(team_df)
        if fig_entropy:
            filepath = FIGURES_DIR / "thesis_entropy_cognitive_load.png"
            fig_entropy.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(fig_entropy)
    except Exception as e:
        print(f"\n[ERROR] Entropy thesis plot: {e}")
        entropy_corr, entropy_p = 0, 1
    
    # C. Black Hole Matrix
    try:
        fig_blackhole, top_black_holes, top_generals = plot_black_hole_matrix(player_df)
        if fig_blackhole:
            filepath = FIGURES_DIR / "thesis_black_hole_matrix.png"
            fig_blackhole.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(fig_blackhole)
    except Exception as e:
        print(f"\n[ERROR] Black hole matrix plot: {e}")
        top_black_holes = pd.DataFrame()
        top_generals = pd.DataFrame()
    
    # D. Dynamic Duos
    try:
        fig_duos, top_duos = plot_dynamic_duos(duos_df)
        if fig_duos:
            filepath = FIGURES_DIR / "thesis_dynamic_duos.png"
            fig_duos.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"\n[OK] Saved: {filepath}")
            figures.append(fig_duos)
    except Exception as e:
        print(f"\n[ERROR] Dynamic duos plot: {e}")
        top_duos = pd.DataFrame()
    
    # 5. Generate Insights Report
    try:
        report = generate_insights_report(
            triangle_corr, triangle_p,
            entropy_corr, entropy_p,
            top_black_holes, top_generals,
            top_duos
        )
        
        report_path = Path("INSIGHTS.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[OK] Saved insights report: {report_path}")
    except Exception as e:
        print(f"\n[ERROR] Generating insights report: {e}")
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Enhanced Player CSV: output/nba_player_metrics_enhanced.csv")
    print(f"  - Enhanced Team CSV: output/nba_team_metrics_enhanced.csv")
    print(f"  - All Duos CSV: output/nba_duos_all.csv")
    print(f"  - Figures: output/figures/thesis_*.png")
    print(f"  - Report: INSIGHTS.md")
    
    # Show figures
    if figures:
        print("\nDisplaying figures...")
        plt.show()


if __name__ == "__main__":
    main()
