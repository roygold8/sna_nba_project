import os
import networkx as nx
from pyvis.network import Network
import sys

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
INPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FILES = ['nba_assist_network.gexf', 'nba_pass_network.gexf']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_gexf(file_name):
    print(f"Visualizing {file_name}...")
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load Graph
    try:
        G = nx.read_gexf(file_path)
    except Exception as e:
        print(f"Error reading GEXF: {e}")
        return

    # Create PyVis Network with in_line resources and directed=True for arrows
    net = Network(height='90vh', width='100%', bgcolor='#222222', font_color='white', 
                  select_menu=True, filter_menu=True, cdn_resources='in_line', directed=True)
    
    # Pre-calculate layout using NetworkX
    print("Calculating layout (this may take a moment)...")
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Calculate out-degree for node sizing
    out_degrees = dict(G.out_degree())
    max_out_degree = max(out_degrees.values()) if out_degrees else 1
    min_out_degree = min(out_degrees.values()) if out_degrees else 0
    
    # Use logarithmic scaling for better visual differentiation
    import math
    min_size = 5
    max_size = 100  # Increased range for more contrast
    
    # Apply coordinates and node properties
    scale = 1000
    for node, coords in pos.items():
        G.nodes[node]['x'] = coords[0] * scale
        G.nodes[node]['y'] = coords[1] * scale
        
        # Size based on out-degree (logarithmic scaling)
        out_deg = out_degrees.get(node, 0)
        if out_deg > 0:
            # Log scale: log(1 + out_deg) for better distribution
            log_deg = math.log(1 + out_deg)
            max_log = math.log(1 + max_out_degree)
            node_size = min_size + (log_deg / max_log) * (max_size - min_size)
        else:
            node_size = min_size
        
        G.nodes[node]['size'] = node_size
        
        # Labels
        G.nodes[node]['label'] = node.split(',')[0]  # Just the name
        G.nodes[node]['title'] = f"{node}\nOut-degree: {out_deg}"  # Tooltip with info

    # Add edge labels with weights
    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 1)
        G[source][target]['label'] = str(int(weight))  # Show weight as edge label
        G[source][target]['title'] = f"Weight: {weight}"  # Tooltip

    net.from_nx(G)
    
    # Disable physics
    net.toggle_physics(False)
    
    # Set options for better visibility with directed arrows
    net.set_options('''
    {
      "nodes": {
        "font": {
          "size": 14,
          "strokeWidth": 2,
          "strokeColor": "#222222"
        },
        "scaling": {
          "label": {
            "enabled": true,
            "min": 10,
            "max": 30
          }
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "color": {
          "inherit": "from",
          "opacity": 0.6
        },
        "smooth": {
          "enabled": true,
          "type": "curvedCW",
          "roundness": 0.1
        },
        "width": 1,
        "font": {
          "size": 10,
          "align": "middle",
          "background": "rgba(0,0,0,0.7)",
          "strokeWidth": 0
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true,
        "hideEdgesOnDrag": true,
        "hideEdgesOnZoom": true
      }
    }
    ''')
    
    output_name = file_name.replace('.gexf', '.html')
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    print(f"Saving to {output_path}...")
    try:
        net.write_html(output_path)
        
        # Now inject custom JavaScript for neighbor filtering
        with open(output_path, 'r') as f:
            html_content = f.read()
        
        # Custom JS for filtering neighbors on node click
        custom_js = '''
<style>
  #filter-controls {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    background: rgba(50,50,50,0.9);
    padding: 10px;
    border-radius: 5px;
    color: white;
    font-family: Arial, sans-serif;
  }
  #filter-controls button {
    margin: 5px;
    padding: 8px 12px;
    cursor: pointer;
    background: #4a90d9;
    border: none;
    color: white;
    border-radius: 3px;
  }
  #filter-controls button:hover {
    background: #357abd;
  }
  #selected-node {
    margin-top: 10px;
    font-size: 12px;
    color: #aaa;
  }
</style>
<div id="filter-controls">
  <button id="btn-filter">Filter to Selected</button>
  <button id="btn-reset">Reset View</button>
  <div id="selected-node">Click a node to select</div>
</div>
<script type="text/javascript">
  // Store original data for reset
  var originalNodes = null;
  var originalEdges = null;
  var selectedNodeId = null;
  
  // Wait for network to be ready
  network.once('stabilized', function() {
    originalNodes = nodes.get();
    originalEdges = edges.get();
  });
  
  // Also capture immediately in case already stable
  setTimeout(function() {
    if (!originalNodes) {
      originalNodes = nodes.get();
      originalEdges = edges.get();
    }
  }, 500);
  
  // Track selected node
  network.on('selectNode', function(params) {
    if (params.nodes.length > 0) {
      selectedNodeId = params.nodes[0];
      document.getElementById('selected-node').innerText = 'Selected: ' + selectedNodeId;
    }
  });
  
  // Filter button
  document.getElementById('btn-filter').addEventListener('click', function() {
    if (!selectedNodeId) {
      alert('Please select a node first by clicking on it.');
      return;
    }
    
    // Get connected nodes
    var connectedNodes = network.getConnectedNodes(selectedNodeId);
    connectedNodes.push(selectedNodeId); // Include the selected node itself
    
    // Get connected edges
    var connectedEdges = network.getConnectedEdges(selectedNodeId);
    
    // Hide all nodes except connected ones
    var nodeUpdates = [];
    originalNodes.forEach(function(node) {
      if (connectedNodes.indexOf(node.id) === -1) {
        nodeUpdates.push({id: node.id, hidden: true});
      } else {
        nodeUpdates.push({id: node.id, hidden: false});
      }
    });
    nodes.update(nodeUpdates);
    
    // Hide all edges except connected ones
    var edgeUpdates = [];
    originalEdges.forEach(function(edge) {
      if (connectedEdges.indexOf(edge.id) === -1) {
        edgeUpdates.push({id: edge.id, hidden: true});
      } else {
        edgeUpdates.push({id: edge.id, hidden: false});
      }
    });
    edges.update(edgeUpdates);
    
    // Fit view to visible nodes
    network.fit({nodes: connectedNodes, animation: true});
  });
  
  // Reset button
  document.getElementById('btn-reset').addEventListener('click', function() {
    if (!originalNodes) return;
    
    // Show all nodes
    var nodeUpdates = originalNodes.map(function(node) {
      return {id: node.id, hidden: false};
    });
    nodes.update(nodeUpdates);
    
    // Show all edges
    var edgeUpdates = originalEdges.map(function(edge) {
      return {id: edge.id, hidden: false};
    });
    edges.update(edgeUpdates);
    
    // Fit all
    network.fit({animation: true});
    
    selectedNodeId = null;
    document.getElementById('selected-node').innerText = 'Click a node to select';
  });
</script>
'''
        
        # Inject before </body>
        html_content = html_content.replace('</body>', custom_js + '</body>')
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print("Success.")
    except Exception as e:
        print(f"Error saving HTML: {e}")

def main():
    ensure_dir(OUTPUT_DIR)
    for f in FILES:
        visualize_gexf(f)

if __name__ == "__main__":
    main()
