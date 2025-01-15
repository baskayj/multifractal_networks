import json
import os
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from scipy.signal import savgol_filter
from matplotlib.colors import Normalize, LogNorm
from itertools import combinations

import concurrent.futures

from multifractal_networks.utils.metrics import measure_degree_distribution, measure_nearest_neighbor_avg_degree
from multifractal_networks.utils.metrics import analytical_degree_distribution, analytical_nearest_neighbor_avg_degree, MRE
from multifractal_networks.utils.metrics import MotifAnalyzer
from multifractal_networks.utils.misc import find_max_d, kron_pow, correct_correlations, sort_state, keep_largest_component
from multifractal_networks.utils.visualization import plot_curves, plot_metric, plot_motif_spectra
from multifractal_networks.utils.motifs import MotifGenerator


class MultifractalNetworkGenerator:
    def __init__(self, P, l, k, N=None, avg_degree=None, num_workers = 1):
        assert len(l) == len(P), 'P and l should have the same dimension'
        assert num_workers != 0, 'Set num workers to a positive integer, or -1 to use all cores'
        start_time = time()
        print('Initializing...')
        self.P = (P + P.T) / 2    # Ensure P is symmetric
        #self.P /= np.sum(self.P)  # Normalize P
        self.l = l
        self.l /= np.sum(self.l)  # Normalize l
        self.P, self.l = sort_state(self.P, self.l)
        self.k = k
        self.N = N
        self.G = None
        if num_workers == -1:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
        
        if N is None and avg_degree is None and k is None:
            raise ValueError("From N, avg_degree and k at least two must be provided.")

        k_was_estimated = False
        if self.k is None:
            self.k = self.estimate_k(self.N, avg_degree)
            print(f'The k to get an avg_degree of {avg_degree} is {self.k}')
            k_was_estimated = True


        self.P_k = kron_pow(self.P, self.k)
        self.l_k = kron_pow(self.l, self.k)
        
        if avg_degree is not None and k_was_estimated == False:
            self.N = self.calculate_N(avg_degree)
            print(f'The recomended network size to get an avg_degree of {avg_degree} is {self.N}')
        else:
            avg_deg = self.calculate_avg_degree()
            if avg_deg < 1:
                warnings.warn(f'Expected avg_degree {avg_deg}')

        end_time = time() - start_time
        print(end_time)

    def calculate_N(self, avg_degree):
        edge_proba = np.sum(np.outer(self.l_k, self.l_k) * self.P_k)
        return int(avg_degree/edge_proba)

    def calculate_avg_degree(self):
        edge_proba = np.sum(np.outer(self.l_k, self.l_k) * self.P_k)
        return self.N * edge_proba

    def estimate_k(self, N, desired_avg_degree):
        max_iterations = int(np.log(N) / np.log(len(self.P)))
        for k in range(1, max_iterations + 1):
            P_k = kron_pow(self.P, k)
            l_k = kron_pow(self.l, k)
            edge_proba = np.sum(np.outer(l_k, l_k) * P_k)
            avg_degree = N * edge_proba
            if avg_degree <= desired_avg_degree:
                return k
        raise ValueError(f"Could not estimate a suitable k value within {max_iterations} iterations")

    def generate_edges(self, node_pairs, indices):
        edges = []
        for i, j in node_pairs:
            i_idx = indices[i]
            j_idx = indices[j]
            link_proba = self.P_k[i_idx, j_idx]
            if np.random.rand() <= link_proba:
                edges.append((i, j))
        return edges
    
    def __call__(self, multiprocess = True, clean_graph = False):
        points = np.random.rand(self.N)
        
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.N))

        def get_index(l, point):
            cum_sum_l = np.cumsum(l)
            index = np.searchsorted(cum_sum_l, point, side = 'right')
            return index

        '''for i in tqdm(range(self.N)):
            for j in range(i+1, self.N):
                i_idx = get_index(self.l_k, points[i])
                j_idx = get_index(self.l_k, points[j])

                link_proba = self.P_k[i_idx, j_idx]
                
                if np.random.rand() <= link_proba:
                    self.G.add_edge(i, j)'''

        indices = [get_index(self.l_k, point) for point in points]

        if multiprocess:
            # Generate all node pairs
            print('Preparing node pairs...')
            start_time = time()
            node_pairs = list(combinations(range(self.N), 2))
            end_time = time() - start_time
            print(end_time)
            
            # Use concurrent futures for parallel processing
            chunk_size = len(node_pairs) // self.num_workers
    
            print('Commencing edge generation...')
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(self.num_workers):
                    chunk = node_pairs[i * chunk_size: (i + 1) * chunk_size]
                    futures.append(executor.submit(self.generate_edges, chunk, indices))
                
                all_edges = []
                for future in tqdm(concurrent.futures.as_completed(futures), total=self.num_workers):
                    all_edges.extend(future.result())
            
            self.G.add_edges_from(all_edges)

        else:
            for node_pair in tqdm(combinations(range(self.N), 2)):
                edges = self.generate_edges([node_pair], indices)
                self.G.add_edges_from(edges)

        if clean_graph:
            # Remove dangling nodes
            nodes = max(nx.connected_components(self.G), key=len)
            self.G = nx.subgraph(self.G, nodes)
            # Update N accordingly
            self.N = len(self.G)

    def save_network(self, filename):
        if self.G is None:
            raise ValueError("Generate the network before saving")
        
        self.G.graph['P'] = json.dumps(self.P.tolist())
        self.G.graph['l'] = json.dumps(self.l.tolist())
        self.G.graph['k'] = self.k
        self.G.graph['N'] = self.N
        
        nx.write_graphml(self.G, filename)

    def load_network(self, filename):
        self.G = nx.read_graphml(filename)
        self.P = np.array(json.loads(self.G.graph['P']))
        self.l = np.array(json.loads(self.G.graph['l']))
        self.k = int(self.G.graph['k'])
        self.N = int(self.G.graph['N'])

    def plot_generated_network(self,
                               layout_func = None,
                               draw_options = None,
                               figsize = (10, 10),
                               title = None,
                               filename = None,
                               streamlit = False,
                               largest_component_only=True,
                               smart_viz=True
                              ):
        """
        Plots the generated network using a given layout function, allowing customization of node and edge appearance.
        
        Parameters:
        -----------
        layout_func : callable, optional
            The layout function used to position nodes (e.g., nx.spring_layout, nx.circular_layout).
            You can also pass a lambda function for Graphviz layouts or other layouts with specific parameters.
            
        draw_options : dict, optional
            Dictionary of drawing options to control the visualization. Defaults to:
            {
                'node_color': 'blue',
                'edge_color': 'gray',
                'node_size':  100,
                'node_alpha': 1.0,
                'edge_alpha': 0.4
            }.
    
        figsize : tuple, optional
            Figure size for the plot. Default is (20, 20).
            
        title : str, optional
            Title for the plot. If None, no title will be set.
            
        filename : str, optional
            If provided, saves the plot to the specified file.
            
        Raises:
        -------
        ValueError
            If the graph (`self.G`) has not been generated.
    
        Example Usage:
        --------------
        # Use a lambda function for a Graphviz layout
        plot_generated_network(layout_func=lambda G: nx.graphviz_layout(G, prog='dot'))
        
        # Use a lambda function for spring layout with specific parameters
        plot_generated_network(layout_func=lambda G: nx.spring_layout(G, k=0.1, iterations=50))
        
        # Customize alpha values
        plot_generated_network(draw_options={
            'node_color': 'red', 'node_alpha': 0.8,
            'edge_color': 'black', 'edge_alpha': 0.5
        })
        largest_component_only : bool, optional
        Whether to only show the largest connected component.

        smart_viz : bool, optional
            Whether to use smart visualization with degree-based sizing and colors.
        """
        if self.G is None:
            raise ValueError("Generate the network first!")

        plot_graph = self.G
        if largest_component_only:
            # Keep only largest component
            largest_cc = max(nx.connected_components(plot_graph), key=len)
            plot_graph = plot_graph.subgraph(largest_cc).copy()
        
        if smart_viz:
            # Sort nodes by degree for better visualization
            sorted_nodes = sorted(plot_graph.nodes(), key=lambda x: plot_graph.degree[x], reverse=True)
            mapping = {node: i for i, node in enumerate(sorted_nodes)}
            plot_graph = nx.relabel_nodes(plot_graph, mapping)
            
            # Calculate degrees for node attributes
            degree = dict(plot_graph.degree())
            nx.set_node_attributes(plot_graph, degree, 'degree')
            
            # Node size based on degree (logarithmic scaling)
            degrees = np.array(list(degree.values()))
            degrees = degrees + 1  # Avoid log(0)
            node_sizes = 10 + 100 * (np.log(degrees) - np.log(degrees.min())) / (
                np.log(degrees.max()) - np.log(degrees.min()) if degrees.max() != degrees.min() else 1
            )
            
            # Node color based on degree (logarithmic scaling)
            cmap = plt.cm.viridis
            if degrees.max() != degrees.min():
                node_colors = cmap((np.log(degrees) - np.log(degrees.min())) / 
                                 (np.log(degrees.max()) - np.log(degrees.min())))
            else:
                node_colors = cmap(np.zeros_like(degrees))
            
            # Calculate edge betweenness for edge attributes
            print('Calculating betweenness...')
            edge_betweenness = nx.edge_betweenness_centrality(plot_graph)
            print('Done.')
            edge_widths = [1 + 7 * edge_betweenness[edge] for edge in plot_graph.edges()]
            edge_alphas = [0.1 + 0.6 * edge_betweenness[edge] for edge in plot_graph.edges()]
        else:
            # Use provided draw options or defaults
            if draw_options is None:
                draw_options = {
                    'node_color': 'blue',
                    'edge_color': 'black',
                    'node_size': 50,
                    'node_alpha': 1.0,
                    'edge_alpha': 0.1
                }
        
        # Set up layout
        if layout_func is None:
            layout_func = lambda G: nx.spring_layout(G)
        print('Calculating node positions...')
        pos = layout_func(plot_graph)
        print('Done.')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if smart_viz:
            # Draw edges with scaled widths and alphas
            for edge, width, alpha in zip(plot_graph.edges(), edge_widths, edge_alphas):
                nx.draw_networkx_edges(
                    plot_graph,
                    pos,
                    edgelist=[edge],
                    width=width,
                    alpha=alpha,
                    edge_color="gray",
                    ax=ax
                )
            
            # Draw nodes with size and color mapped to degree
            nx.draw_networkx_nodes(
                plot_graph,
                pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=cmap,
                edgecolors='white',
                ax=ax
            )
            
        else:
            nx.draw_networkx_nodes(
                plot_graph, pos, ax=ax,
                node_color=draw_options['node_color'],
                node_size=draw_options['node_size'],
                alpha=draw_options['node_alpha']
            )
            nx.draw_networkx_edges(
                plot_graph, pos, ax=ax,
                edge_color=draw_options['edge_color'],
                alpha=draw_options['edge_alpha']
            )
        
        if title is not None:
            plt.title(title, color='white' if smart_viz else 'black')
        
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=1000, facecolor=fig.get_facecolor())
        
        if streamlit:
            return fig
        else:
            plt.show()
            return None

    def network_stats(self,
                      log_scale = False,
                      grid = False,
                      motifs = [
                          {'name': 'edge', 'matrix': np.array([[0,1],[1,0]]), 'size': 2},
                          {'name': 'wedge', 'matrix': np.array([[0,1,1],[1,0,0],[1,0,0]]), 'size': 3},
                          {'name': 'triangle', 'matrix': np.array([[0,1,1],[1,0,1],[1,1,0]]), 'size': 3},
                          {'name': 'three_star', 'matrix': np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]), 'size': 4},
                          {'name': 'four_chain', 'matrix': np.array([[0, 1, 0, 0],[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0]]), 'size': 4},
                          {'name': 'three_loop_out', 'matrix': np.array([[0, 1, 1, 1],[1, 0, 1, 0],[1, 1, 0, 0],[1, 0, 0, 0]]), 'size': 4},
                          {'name': 'four_loop', 'matrix': np.array([[0, 1, 0, 1],[1, 0, 1, 0],[0, 1, 0, 1],[1, 0, 1, 0]]), 'size': 4},
                          {'name': 'four_semi_clique', 'matrix': np.array([[0, 1, 1, 1],[1, 0, 0, 1],[1, 0, 0, 1],[1, 1, 1, 0]]), 'size': 4},
                          {'name': 'four_clique', 'matrix': np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]), 'size': 4},
                      ],
                      sampling_ratios = {3: 0.001,
                                         4: 0.00001},
                      motif_analyzer_args = {'exact': False, 'bootstrap_iters': None, 'ci': None}, # By default CI bootstrapping is skipped.
                      filename = None):
        if self.G is None:
            raise ValueError("Generate the network first!")
        # --------------------
        # MEASURE
        # --------------------
        if log_scale:
            plot_metric(self.P_k, self.l_k, grid = grid, norm = LogNorm)
        else:
            plot_metric(self.P_k, self.l_k, grid = grid, norm = Normalize)
        # --------------------
        # Degree distribution and correlations
        # --------------------
        max_d = find_max_d(self.G)
        d_range = list(range(0,max_d+1))
        rho_measured = measure_degree_distribution(self.G, [0,max_d])
        #rho_measured = savgol_filter(rho_measured, window_length=10, polyorder=3)
        dnn_measured = measure_nearest_neighbor_avg_degree(self.G, [0,max_d])
        #dnn_measured = correct_correlations(dnn_measured)
        rho_analytical,_ = analytical_degree_distribution(self.P_k, self.l_k,self. N, [0,max_d])
        dnn_analytical,_ = analytical_nearest_neighbor_avg_degree(self.P_k, self.l_k, rho_analytical, self.N, [0,max_d])
        # TODO: ADD SUPPORT FOR FILENAMES AND SAVING THE RESULTS
        plot_curves(d_range, rho_measured, rho_analytical, log_scale=False, analytical_ci=None, display_mre=True)
        plot_curves(d_range, dnn_measured, dnn_analytical, log_scale=False, analytical_ci=None, display_mre=True)
        # --------------------
        # MOTIFS
        # --------------------
        # Check the analyzer arguments for bootstrapping
        if motif_analyzer_args['bootstrap_iters'] is None or motif_analyzer_args['bootstrap_iters'] == 1 or motif_analyzer_args['ci'] is None or motif_analyzer_args['ci'] == 0:
            motif_count_ci = False
        else:
            motif_count_ci = True
        # Order them according to size
        motifs.sort(key=lambda motif: motif['size'])
        # Motif sizes
        motif_sizes = sorted(set([motif['size'] for motif in motifs]))
        # Process motifs by size in ascending order
        for size in motif_sizes:
            # Filter motifs of the current size
            motifs_of_size = [motif for motif in motifs if motif['size'] == size]
            # Edges are handled separately
            if size == 2:
                if motif_count_ci:
                    number_of_edges = self.G.number_of_edges()
                    results = {'edge':{'count': number_of_edges,'count_ci_low': number_of_edges,'count_ci_high': number_of_edges}}
                else:
                    results = {'edge':{'count':self.G.number_of_edges()}}
            else:
                # Initialize MotifAnalyzer for the current size
                analyzer = MotifAnalyzer(
                    adj_matrices=[motif['matrix'] for motif in motifs_of_size],
                    names=[motif['name'] for motif in motifs_of_size]
                )
                # Run count_specific_motifs with extra arguments (for bootstrapping, etc.)
                sampling_ratio = sampling_ratios[size]
                results = analyzer.count_specific_motifs(self.G, sampling_ratio = sampling_ratio, **motif_analyzer_args)
            # Store results in motifs
            for motif in motifs_of_size:
                motif_name = motif['name']
                if motif_name in results:
                    analyzer = MotifAnalyzer(
                    adj_matrices=[motif['matrix'] for motif in motifs_of_size],
                    names=[motif['name'] for motif in motifs_of_size]
                )
                    motif_analytical = analyzer.expected_number_of_motifs(self.P, self.l, self.k, self.N)
                    motif_measured = results[motif_name]['count']
                    mre_motif = MRE(motif_measured, motif_analytical[motif_name])
                    motif['MRE'] = mre_motif   
                    # Optionally check for confidence intervals
                    if 'count_ci_low' in results[motif_name] and 'count_ci_high' in results[motif_name]:
                        motif_measured_low = results[motif_name]['count_ci_low']
                        motif['MRE_ci_low'] = mre_motif * (1 - MRE(motif_measured, motif_measured_low))
                        motif_measured_high = results[motif_name]['count_ci_high']
                        motif['MRE_ci_high'] = mre_motif * (1 + MRE(motif_measured, motif_measured_high))
                        if motif['MRE_ci_low'] > motif['MRE_ci_high']:
                            motif['MRE_ci_low'], motif['MRE_ci_high'] = motif['MRE_ci_high'], motif['MRE_ci_low']
        # Plotting
        plot_motif_spectra(motifs)
        