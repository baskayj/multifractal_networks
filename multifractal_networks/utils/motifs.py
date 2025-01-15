import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from collections import defaultdict

# TODO: NOT ALL 6-MOTIFS GET GENERATED (5 or lower is fine, tho...)

class MotifGenerator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.motifs = self._generate_connected_graphs(num_nodes)
        self.named_motifs = self._assign_names_to_motifs()

    def _generate_connected_graphs(self, num_nodes):
        """Generate all connected non-isomorphic graphs for a given number of nodes."""
        nodes = list(range(num_nodes))
        all_graphs = []

        def _is_isomorphic_to_any(graph, graph_list):
            """Check if the graph is isomorphic to any graph in the graph list."""
            for g in graph_list:
                if nx.is_isomorphic(graph, g):
                    return True
            return False
        min_edges = num_nodes - 1
        max_edges = (num_nodes * (num_nodes - 1)) / 2
        for num_edges in range(num_nodes - 1, 2 * num_nodes + 1):  # Adjust edge range as needed
            for edges in combinations(combinations(nodes, 2), num_edges):
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)

                if nx.is_connected(G) and not _is_isomorphic_to_any(G, all_graphs):
                    all_graphs.append(G)

        return all_graphs

    def _degree_sequence_name(self, graph):
        """Create a base name for the graph based on its degree sequence."""
        degree_sequence = sorted([d for _, d in graph.degree()], reverse=True)
        name = '_'.join(map(str, degree_sequence)) + '_motif'
        return name

    def _assign_names_to_motifs(self):
        """Assign unique names to motifs based on their degree sequences."""
        named_motifs = []
        degree_motif_groups = defaultdict(list)

        # Group motifs by their degree sequence base name
        for idx, G in enumerate(self.motifs, 1):
            degree_sequence_name = self._degree_sequence_name(G)
            degree_motif_groups[degree_sequence_name].append(G)

        # Now, assign names, adding suffix only if necessary
        for base_name, graphs in degree_motif_groups.items():
            if len(graphs) == 1:
                # If there's only one motif, no need for numbering
                adj_matrix = nx.to_numpy_array(graphs[0], dtype=int)
                named_motifs.append({'name': base_name, 'matrix': adj_matrix})
            else:
                # If more than one, append numbering
                for i, graph in enumerate(graphs, 1):
                    adj_matrix = nx.to_numpy_array(graph, dtype=int)
                    named_motifs.append({'name': f"{base_name}_{i}", 'matrix': adj_matrix})

        return named_motifs

    def number_of_motifs(self):
        """Return the number of motifs found."""
        return len(self.named_motifs)

    def get_motifs(self):
        """Return all motifs with their names and adjacency matrices."""
        return self.named_motifs

    def exact_minimize_crossings(self, G):
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

    def plot_motifs(self, rows, cols, streamlit = False):
        """Plot all motifs in a grid of user-defined size with a chosen layout and optional edge crossing minimization."""
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        for i, motif in enumerate(self.named_motifs):
            ax = axes[i]
            G = nx.from_numpy_array(motif['matrix'])

            # Minimize crossinggs
            G,_ = self.exact_minimize_crossings(G)
            # Get the circular layout and reorder the positions according to the best order
            pos = nx.circular_layout(G)
            ax.set_title(f"{motif['name']}")
            
            # Compute node degrees and create labels based on degree
            labels = {node: G.degree(node) for node in G.nodes()}
            
            # Draw the graph with node degree labels
            nx.draw(G, pos, ax=ax, labels=labels, node_color='lightblue', edge_color='gray', node_size=500)
            ax.set_title(motif['name'])

        # Clear remaining empty plots
        for i in range(len(self.named_motifs), rows * cols):
            axes[i].axis('off')

        plt.tight_layout()
        if streamlit:
            return fig
        else:
            plt.show()