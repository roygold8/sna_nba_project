import os
import json
import networkx as nx
import pandas as pd
import glob

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
ASSIST_NETWORK_FILE = 'nba_assist_network.gexf'
PASS_NETWORK_FILE = 'nba_pass_network.gexf'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_available_seasons():
    """
    Scans the data directory and returns a list of available season folders.
    """
    seasons = []
    if os.path.exists(DATA_DIR):
        for item in sorted(os.listdir(DATA_DIR)):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path) and '-' in item:  # e.g., "2023-24"
                seasons.append(item)
    return seasons

def load_qualified_players(season):
    """
    Loads the filtered_players.csv to get the list of qualified players for a season.
    Returns a set of player IDs that meet the criteria.
    """
    csv_path = os.path.join(DATA_DIR, season, "filtered_players.csv")
    if not os.path.exists(csv_path):
        return set()
    
    df = pd.read_csv(csv_path)
    return set(df['PLAYER_ID'].astype(int).tolist())

def load_passing_data(season):
    """
    Loads all JSON files from the data directory for the given season.
    Returns a list of tuples: (data_dict, season_string).
    """
    search_path = os.path.join(DATA_DIR, season, "passing_*.json")
    files = glob.glob(search_path)
    
    all_data = []
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                all_data.append((data, season))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    return all_data

def build_networks(data_list, qualified_player_ids_by_season):
    """
    Constructs two DiGraphs: Assist Network and Pass Network.
    Nodes: "PlayerName, Team, Season"
    Edges: Weighted by AST or PASS
    
    data_list: List of (data_dict, season_string) tuples
    qualified_player_ids_by_season: Dict mapping season -> set of qualified player IDs
    """
    G_assist = nx.DiGraph()
    G_pass = nx.DiGraph()
    
    print("Building networks...")
    
    skipped_source = 0
    skipped_target = 0
    
    for entry, season in data_list:
        qualified_ids = qualified_player_ids_by_season.get(season, set())
        
        try:
            result_sets = entry['resultSets']
            passes_made_set = next((rs for rs in result_sets if rs['name'] == 'PassesMade'), None)
            
            if not passes_made_set:
                continue
                
            headers = passes_made_set['headers']
            row_set = passes_made_set['rowSet']
            
            def get_idx(col_name):
                return headers.index(col_name)
            
            idx_pid = get_idx('PLAYER_ID')
            idx_pname = get_idx('PLAYER_NAME_LAST_FIRST')
            idx_team = get_idx('TEAM_ABBREVIATION')
            idx_to_pid = get_idx('PASS_TEAMMATE_PLAYER_ID')
            idx_to_name = get_idx('PASS_TO')
            idx_ast = get_idx('AST')
            idx_pass = get_idx('PASS')
            
            for row in row_set:
                p_id = int(row[idx_pid])
                to_id = int(row[idx_to_pid])
                
                # Filter by qualified players
                if qualified_ids and p_id not in qualified_ids:
                    skipped_source += 1
                    continue
                
                if qualified_ids and to_id not in qualified_ids:
                    skipped_target += 1
                    continue
                
                p_name = row[idx_pname]
                p_team = row[idx_team]
                
                if ',' in p_name:
                    last, first = p_name.split(', ')
                    p_name_fmt = f"{first} {last}"
                else:
                    p_name_fmt = p_name

                source_node = f"{p_name_fmt}, {p_team}, {season}"
                
                to_name = row[idx_to_name]
                
                if ',' in to_name:
                    last, first = to_name.split(', ')
                    to_name_fmt = f"{first} {last}"
                else:
                    to_name_fmt = to_name
                
                target_node = f"{to_name_fmt}, {p_team}, {season}"
                
                # Add Nodes
                G_assist.add_node(source_node, type='Player', season=season, team=p_team, player_id=p_id)
                G_assist.add_node(target_node, type='Player', season=season, team=p_team, player_id=to_id)
                
                G_pass.add_node(source_node, type='Player', season=season, team=p_team, player_id=p_id)
                G_pass.add_node(target_node, type='Player', season=season, team=p_team, player_id=to_id)
                
                # Pass Weight
                passes = row[idx_pass]
                if passes > 0:
                    G_pass.add_edge(source_node, target_node, weight=passes)
                    
                # Assist Weight
                assists = row[idx_ast]
                if assists > 0:
                    G_assist.add_edge(source_node, target_node, weight=assists)
                    
        except Exception as e:
            continue
    
    print(f"Skipped {skipped_source} rows (source not qualified), {skipped_target} rows (target not qualified)")
            
    return G_assist, G_pass

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 1. Get available seasons
    seasons = get_available_seasons()
    if not seasons:
        print("No season data found. Run fetch_nba_data.py first.")
        return
        
    print(f"Found {len(seasons)} seasons: {seasons}")
    
    # 2. Load qualified player IDs for each season
    qualified_by_season = {}
    for season in seasons:
        qualified_by_season[season] = load_qualified_players(season)
        print(f"  {season}: {len(qualified_by_season[season])} qualified players")
    
    # 3. Load all passing data from all seasons
    all_data = []
    for season in seasons:
        season_data = load_passing_data(season)
        print(f"  {season}: {len(season_data)} data files")
        all_data.extend(season_data)
    
    print(f"Total data files: {len(all_data)}")
    
    if not all_data:
        print("No data found. Run fetch_nba_data.py first.")
        return
        
    # 4. Build Networks (combining all seasons)
    G_ast, G_pass = build_networks(all_data, qualified_by_season)
    
    print(f"\nAssist Network: {G_ast.number_of_nodes()} nodes, {G_ast.number_of_edges()} edges.")
    print(f"Pass Network: {G_pass.number_of_nodes()} nodes, {G_pass.number_of_edges()} edges.")
    
    # 5. Save
    nx.write_gexf(G_ast, os.path.join(OUTPUT_DIR, ASSIST_NETWORK_FILE))
    nx.write_gexf(G_pass, os.path.join(OUTPUT_DIR, PASS_NETWORK_FILE))
    print(f"Saved networks to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
