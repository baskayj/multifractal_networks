import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from collections import defaultdict

def load_graph(uploaded_file):
    """Load a graph from the uploaded file."""
    try:
        if uploaded_file.name.endswith(".graphml"):
            # Load GraphML file
            graph = nx.read_graphml(uploaded_file)
        elif uploaded_file.name.endswith((".csv", ".tsv", ".txt")):
            # Determine the delimiter based on the file extension
            delimiter = "," if uploaded_file.name.endswith(".csv") else "\t" if uploaded_file.name.endswith(".tsv") else None
            # Load edge list file
            graph = nx.read_edgelist(uploaded_file, delimiter=delimiter, nodetype=int)
        else:
            raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        return graph
    except Exception as e:
        raise ValueError(f"Error loading graph: {e}")

def plot_single_motif(adj_mat):
    fig, ax = plt.subplots(figsize=(3, 3))
    
    G = nx.from_numpy_array(adj_mat)
    def exact_minimize_crossings(G):
        """Minimize edge crossings in a circular layout."""
        # Get all possible circular orderings of the nodes
        nodes = list(G.nodes)
        _permutations = list(permutations(nodes))
        adj_matrix = nx.to_numpy_array(G)

        def count_crossings(G):
            """Count the number of edge crossings for a given node order."""
            crossing_count = 0
            edges = list(G.edges)
            nodes = list(G.nodes)
        
            # Map each node to its position in the circular order
            positions = {node: i for i, node in enumerate(nodes)}
            
            # Check all pairs of edges for crossings
            for i, (u1, v1) in enumerate(edges):
                for u2, v2 in edges[i + 1:]:
                    # Ensure edges do not share a node
                    if len({u1, v1, u2, v2}) < 4:
                        continue  # Skip if the edges share a node
                    
                    # Get positions of the endpoints of both edges
                    pos_u1, pos_v1 = positions[u1], positions[v1]
                    pos_u2, pos_v2 = positions[u2], positions[v2]
                    
                    # Edge (u1, v1) crosses edge (u2, v2) if their endpoints are interleaved
                    if (pos_u1 < pos_u2 < pos_v1 < pos_v2) or (pos_u2 < pos_u1 < pos_v2 < pos_v1):
                        crossing_count += 1
            
            return crossing_count
        # Find the ordering with the minimum number of crossings
        min_crossings = float('inf')
        best_G = None
        for order in _permutations:
            # Permute the adjacency matrix based on the current permutation
            permuted_adj_matrix = adj_matrix[np.ix_(order, order)]
            # Create a new graph from the permuted adjacency matrix
            permuted_G = nx.from_numpy_array(permuted_adj_matrix)
            crossings = count_crossings(permuted_G)
            if crossings < min_crossings:
                min_crossings = crossings
                best_G = permuted_G
        return best_G, min_crossings
    G,_ = exact_minimize_crossings(G)
    # Get the circular layout and reorder the positions according to the best order
    pos = nx.circular_layout(G)
            
    # Compute node degrees and create labels based on degree
    labels = {node: G.degree(node) for node in G.nodes()}
            
    # Draw the graph with node degree labels
    nx.draw(
        G,
        pos,
        ax=ax,
        labels=labels,
        node_color='lightblue',
        edge_color='gray',
        node_size=300,  # Smaller node size
        font_size=8,    # Smaller font size for labels
    )

    plt.tight_layout()
    
    return fig