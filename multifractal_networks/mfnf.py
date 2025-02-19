import random
import time
import datetime
import optuna
import pickle
import os
import gc

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.signal import savgol_filter
from itertools import combinations, permutations
from collections import Counter
from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.special import comb, factorial
from tqdm import tqdm
from graph_tool import Graph
from graph_tool.clustering import motifs
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LogNorm

from joblib import Parallel, delayed, cpu_count
from tqdm_joblib import tqdm_joblib

from multifractal_networks.utils.misc import sort_state, kron_pow, find_max_d, find_min_d
from multifractal_networks.utils.misc import correct_correlations, smooth_data, smooth_data_log_scale, log_transform

from multifractal_networks.utils.visualization import plot_metric, plot_curves, plot_motif_spectra

from multifractal_networks.utils.metrics import MRE, KL_divergence
from multifractal_networks.utils.metrics import analytical_degree_distribution, measure_degree_distribution
from multifractal_networks.utils.metrics import analytical_nearest_neighbor_avg_degree, measure_nearest_neighbor_avg_degree
from multifractal_networks.utils.metrics import expected_number_of_d_stars, count_d_stars
from multifractal_networks.utils.metrics import expected_number_of_t_cliques, count_t_cliques
from multifractal_networks.utils.metrics import MotifAnalyzer

from multifractal_networks.mfng import MultifractalNetworkGenerator

sns.set_theme(context = 'notebook',
              style = 'darkgrid',
              palette = 'deep',
              font = 'sans-serif',
              font_scale = 1,
              color_codes = True,
              rc = None)

def unwrap_state(state, m, k, kronpow = True):
    upper_triangle_indices = np.triu_indices(m)
    
    P_flat = state[:len(upper_triangle_indices[0])]
    l = state[len(upper_triangle_indices[0]):]
    
    P = np.zeros((m, m))
    P[upper_triangle_indices] = P_flat
    P = P + P.T - np.diag(P.diagonal())
    P = (P + P.T) / 2

    P, l = sort_state(P, l)

    if kronpow:
        P_k = kron_pow(P, k)
        l_k = kron_pow(l.reshape(-1, 1), k).flatten()
        return P,l,P_k,l_k
    else:
        return P,l

def wrap_state(P, l, m):
    upper_triangle_indices = np.triu_indices(m)
    state = np.concatenate([P[upper_triangle_indices], l])
    return state

def energy(P, l, k, P_k, l_k, N,
           rho_d_range,
           dnn_d_range,
           dnn_index_range,
           rho_measured,
           deg_nn_measured,
           motifs_measured,
           weights = {'rho':1,
                      'deg_nn':1,
                      'motifs':1},
           log_scale = False,
           debug = False):
    error = 0
    if debug:
        error_dict = {}
    
    if not weights['rho'] == 0:
        rho_analytical,_ = analytical_degree_distribution(P_k, l_k, N, rho_d_range)
        if log_scale:
            rho_measured_tf,_ = log_transform(rho_measured, rho_d_range)
            rho_analytical_tf,_ = log_transform(rho_analytical, rho_d_range)
        else:
            rho_measured_tf = rho_measured
            rho_analytical_tf = rho_analytical
        rho_mre = MRE(rho_measured_tf, rho_analytical_tf)
        error += weights['rho'] * rho_mre
        if debug:
            error_dict['rho'] = rho_mre
    
    if not weights['deg_nn'] == 0:
        if weights['rho'] == 0:
            rho_analytical,_ = analytical_degree_distribution(P_k, l_k, N, dnn_d_range)
        else:
            rho_analytical = rho_analytical[dnn_index_range[0] : dnn_index_range[1]]
        deg_nn_analytical,_ = analytical_nearest_neighbor_avg_degree(P_k, l_k, rho_analytical, N, dnn_d_range)
        if log_scale:
            deg_nn_measured_tf,_ = log_transform(deg_nn_measured, dnn_d_range)
            deg_nn_analytical_tf,_ = log_transform(deg_nn_analytical, dnn_d_range)
        else:
            deg_nn_measured_tf = deg_nn_measured
            deg_nn_analytical_tf = deg_nn_analytical
        deg_nn_mre = MRE(deg_nn_measured_tf, deg_nn_analytical_tf)
        error += weights['deg_nn'] * deg_nn_mre
        if debug:
            error_dict['deg_nn'] = deg_nn_mre

    if not weights['motifs'] == 0 and len(motifs_measured) > 0:
        motif_sizes = sorted(set([motif['size'] for motif in motifs_measured]))
        if debug:
            error_dict['motifs'] = {}

        for size in motif_sizes:
            motifs_of_size = [motif for motif in motifs_measured if motif['size'] == size]
            # Initialize the MotifAnalyzer
            analyzer = MotifAnalyzer(
                adj_matrices=[motif['matrix'] for motif in motifs_of_size],
                names=[motif['name'] for motif in motifs_of_size]
            )
            # Calculate expected motif counts
            motif_analytical = analyzer.expected_number_of_motifs(P, l, k, N)
            # Add error term for each motif
            for motif in motifs_of_size:
                motif_name = motif['name']
                measured_count = motif['count']
                analytical_count = motif_analytical[motif_name]
                motif_mre = MRE(measured_count, analytical_count)
                if motif_name in weights.keys():
                    motif_multiplier = weights[motif_name]
                else:
                    motif_multiplier = 1.0
                error += motif_multiplier * weights['motifs'] * motif_mre
                if debug:
                    error_dict['motifs'][motif_name] = motif_mre
    
    if debug:
        return (error, error_dict)
    else:
        return error

def kl_energy(P, l, k, P_k, l_k, N,
              rho_d_range,
              dnn_d_range,
              dnn_index_range,
              rho_measured,
              deg_nn_measured,
              motifs_measured,
              weights = {'rho':1,
                         'deg_nn':1,
                         'motifs':1},
              log_scale = False,
              debug = False):
    error = 0
    if debug:
        error_dict = {}
    
    if not weights['rho'] == 0:
        rho_analytical,_ = analytical_degree_distribution(P_k, l_k, N, rho_d_range)
        rho_measured_tf = rho_measured
        rho_analytical_tf = rho_analytical
        rho_kl = KL_divergence(rho_measured_tf, rho_analytical_tf)
        error += weights['rho'] * rho_kl
        if debug:
            error_dict['rho'] = rho_kl
    
    if not weights['deg_nn'] == 0:
        if weights['rho'] == 0:
            rho_analytical,_ = analytical_degree_distribution(P_k, l_k, N, dnn_d_range)
        else:
            rho_analytical = rho_analytical[dnn_index_range[0] : dnn_index_range[1]]
        deg_nn_analytical,_ = analytical_nearest_neighbor_avg_degree(P_k, l_k, rho_analytical, N, dnn_d_range)
        if log_scale:
            deg_nn_measured_tf,_ = log_transform(deg_nn_measured, dnn_d_range)
            deg_nn_analytical_tf,_ = log_transform(deg_nn_analytical, dnn_d_range)
        else:
            deg_nn_measured_tf = deg_nn_measured
            deg_nn_analytical_tf = deg_nn_analytical
        deg_nn_mre = MRE(deg_nn_measured_tf, deg_nn_analytical_tf)
        error += weights['deg_nn'] * deg_nn_mre
        if debug:
            error_dict['deg_nn'] = deg_nn_mre

    if not weights['motifs'] == 0 and len(motifs_measured) > 0:
        motif_sizes = sorted(set([motif['size'] for motif in motifs_measured]))
        if debug:
            error_dict['motifs'] = {}

        for size in motif_sizes:
            motifs_of_size = [motif for motif in motifs_measured if motif['size'] == size]
            # Initialize the MotifAnalyzer
            analyzer = MotifAnalyzer(
                adj_matrices=[motif['matrix'] for motif in motifs_of_size],
                names=[motif['name'] for motif in motifs_of_size]
            )
            # Calculate expected motif counts
            motif_analytical = analyzer.expected_number_of_motifs(P, l, k, N)
            # Add error term for each motif
            for motif in motifs_of_size:
                motif_name = motif['name']
                measured_count = motif['count']
                analytical_count = motif_analytical[motif_name]
                motif_mre = MRE(measured_count, analytical_count)
                if motif_name in weights.keys():
                    motif_multiplier = weights[motif_name]
                else:
                    motif_multiplier = 1.0
                error += motif_multiplier * weights['motifs'] * motif_mre
                if debug:
                    error_dict['motifs'][motif_name] = motif_mre
    
    if debug:
        return (error, error_dict)
    else:
        return error

def kl_divergence(P_k, l_k, N, d_range, rho_measured):
    rho_analytical,_ = analytical_degree_distribution(P_k, l_k, N, d_range)
    return KL_divergence(rho_measured, rho_analytical)


class MultifractalNetworkFitter:
    def __init__(self, G,
                 d_range = None,
                 apply_corrections=True,
                 filter_params = None,
                 correction_params = None,
                 motifs=None,
                 sampling_ratios = {3: 0.001,
                                    4: 0.00001},
                 motif_analyzer_args=None,
                 population_size = 100,
                 weights = {'rho':1,
                            'deg_nn':1,
                            'motifs': 1},
                 log_scale = False,
                 pre_loaded_motifs = None,
                 n_jobs = 10
                 ):
        self.G = G
        self.N = len(G)
        self.max_d = find_max_d(self.G)
        self.min_d = find_min_d(self.G)
        if d_range is None:
            self.rho_d_range = [self.min_d, self.max_d]
            self.dnn_d_range = [self.min_d, self.max_d]
        else:
            self.rho_d_range = d_range
            self.dnn_d_range = d_range
            
        self.rho_index_range = [0,self.rho_d_range[1] - self.rho_d_range[0] + 1]
        self.dnn_index_range = [0,self.dnn_d_range[1] - self.dnn_d_range[0] + 1]

        filter_params = filter_params or {'window_length': 10, 'polyorder': 3}
        self.filter_params = filter_params
        correction_params = correction_params or {'noise_factor': 0.1}
        self.correction_params = correction_params
        self.apply_corrections = apply_corrections
        self.motif_analyzer_args = motif_analyzer_args or {'exact': False, 'bootstrap_iters': None, 'ci': None} # By default CI bootstrapping is skipped.
        if self.motif_analyzer_args['bootstrap_iters'] is None or self.motif_analyzer_args['bootstrap_iters'] == 1 or self.motif_analyzer_args['ci'] is None or self.motif_analyzer_args['ci'] == 0:
            self.motif_count_ci = False
        else:
            self.motif_count_ci = True
        
        print('Measuring graph properties...')
        print('Degree distribution...')
        self.rho_measured = measure_degree_distribution(self.G, self.rho_d_range)
        if self.apply_corrections:
            self.rho_measured = savgol_filter(self.rho_measured,**self.filter_params)
        print('Degree correlations...')
        self.deg_nn_measured = measure_nearest_neighbor_avg_degree(self.G, self.dnn_d_range)
        if self.apply_corrections:
            d_original = np.arange(self.dnn_d_range[0],self.dnn_d_range[1] + 1)
            self.deg_nn_measured, d_sliced = correct_correlations(d_original,
                                                                  self.deg_nn_measured,
                                                                  **self.correction_params)
            start = np.where(d_original== d_sliced[0])[0][0]
            end = start + len(d_sliced)
            self.dnn_d_range = [d_sliced[0],d_sliced[-1]]
            self.dnn_index_range = [start,end]

        print('Motifs...')
        if pre_loaded_motifs is None:
            # Default motifs
            self.motifs = [
                {'name': 'edge', 'matrix': np.array([[0,1],[1,0]]), 'size': 2},
                {'name': 'wedge', 'matrix': np.array([[0,1,1],[1,0,0],[1,0,0]]), 'size': 3},
                {'name': 'triangle', 'matrix': np.array([[0,1,1],[1,0,1],[1,1,0]]), 'size': 3},
                {'name': 'three_star', 'matrix': np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]), 'size': 4},
                {'name': 'four_clique', 'matrix': np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]), 'size': 4},
            ]
            
            if motifs is not None:
                print(f"Checking compatibility with {len(motifs)} user-provided motifs.")
                for custom_motif in motifs:
                    # Check that each custom motif has the required keys
                    if not isinstance(custom_motif, dict) or 'name' not in custom_motif or 'matrix' not in custom_motif:
                        print(f"Custom motif {custom_motif} is missing 'name' or 'matrix' keys. Skipping.")
                        continue
                    # Check if the graph is connected
                    if not nx.is_connected(nx.from_numpy_array(custom_motif['matrix'])):
                        print(f"Custom motif '{custom_motif['name']}' is not connected. Skipping.")
                        continue
                    # Proceed with isomorphism check if structure is valid. We also check against the cliques, we want to exclude those.
                    is_isomorphic = False
                    for default_motif in self.motifs:
                        if nx.is_isomorphic(nx.from_numpy_array(custom_motif['matrix']),
                                            nx.from_numpy_array(default_motif['matrix'])):
                            print(f"Custom motif '{custom_motif['name']}' is isomorphic to default '{default_motif['name']}' - Skipping.")
                            is_isomorphic = True
                            break
                    # If not isomorphic, add to motifs
                    if not is_isomorphic:
                        print(f"Adding custom motif '{custom_motif['name']}'.")
                        custom_motif['size'] = custom_motif['matrix'].shape[0]  # Store motif size
                        self.motifs.append(custom_motif)
    
            # Order them according to size
            self.motifs.sort(key=lambda motif: motif['size'])
            
            # Motif sizes
            self.motif_sizes = sorted(set([motif['size'] for motif in self.motifs]))
    
            # Process motifs by size in ascending order
            for size in self.motif_sizes:
                # Filter motifs of the current size
                motifs_of_size = [motif for motif in self.motifs if motif['size'] == size]
                # Edges are handled separately
                if size == 2:
                    if self.motif_count_ci:
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
                    results = analyzer.count_specific_motifs(self.G, sampling_ratio = sampling_ratio, **self.motif_analyzer_args)
                # Store results in self.motifs
                for motif in motifs_of_size:
                    motif_name = motif['name']
                    if motif_name in results:
                        motif['count'] = results[motif_name]['count']
                        # Optionally check for confidence intervals
                        if 'count_ci_low' in results[motif_name] and 'count_ci_high' in results[motif_name]:
                            motif['count_ci_low'] = results[motif_name]['count_ci_low']
                            motif['count_ci_high'] = results[motif_name]['count_ci_high']

        else:
            with open(pre_loaded_motifs, 'rb') as f:
                data = pickle.load(f)
            # Assertions to validate the structure of the loaded data
            assert isinstance(data, list), "Loaded data must be a list."
            for motif in data:
                assert isinstance(motif, dict), "Each motif must be a dictionary."
                assert 'name' in motif and isinstance(motif['name'], str), "'name' must be a string."
                assert 'matrix' in motif, "'matrix' key is missing."
                assert 'size' in motif and isinstance(motif['size'], int), "'size' must be an integer."
                assert 'count' in motif and isinstance(motif['count'], int), "'count' must be an integer."
            # Assign to self.motifs after passing all assertions
            self.motifs = data
            print(f'Succesfully loaded motifs for network G from {pre_loaded_motifs}')

        self.population_size = population_size #Number of restarts in find_P_l
        self.n_jobs = n_jobs
        self.weights = weights
        self.log_scale = log_scale
        self.bootstrap = False # Will be set to true,one bootstrap_P_l_stability is run.
        self.ground_truth = None
        self.rho_analytical = {}
        self.deg_nn_analytical = {}
        self.P = None
        self.l = None
        self.k = None

    def save_motifs(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.motifs, f)

    def initialize_state(self, m, k,
                         P_init='m x m',
                         l_init='dirichlet'
                        ):
        # Initialize P
        if isinstance(P_init, np.ndarray):
            if len(np.shape(P_init)) == 2:
                P_init= P_init
            else:
                raise ValueError("Valid values for P_init are: 'diag', 'm x m', or a 2D numpy array")
        else:
            if P_init == 'diag':
                P_init = np.diag(np.random.rand(m))
            elif P_init == 'm x m':
                P_init = np.random.rand(m, m)
            else:
                raise ValueError("Valid values for P_init are: 'diag', 'm x m', or a 2D numpy array")
        # Initialize l
        if isinstance(l_init, np.ndarray):
            if len(np.shape(l_init)) == 1:
                l = l_init
            else:
                raise ValueError("Valid values for l_init are: 'dirichlet', or a 1D numpy array")
        else:
            if l_init == 'dirichlet':
                l = np.random.dirichlet(np.ones(m))
            else:
                raise ValueError("Valid values for l_init are: 'dirichlet', or a 1D numpy array")

        P_init, l = sort_state(P_init, l)
        state = wrap_state(P_init,l,m)

        return state
    
    def _get_energy(self,P,l,k,P_k,l_k):
        return energy(P, l, k, P_k, l_k, self.N,
                      self.rho_d_range, self.dnn_d_range, self.dnn_index_range,
                      self.rho_measured,
                      self.deg_nn_measured,
                      motifs_measured=self.motifs,
                      weights=self.weights,
                      log_scale=self.log_scale)

    def _get_kl_energy(self,P,l,k,P_k,l_k):
        return kl_energy(P, l, k, P_k, l_k, self.N,
                         self.rho_d_range, self.dnn_d_range, self.dnn_index_range,
                         self.rho_measured,
                         self.deg_nn_measured,
                         motifs_measured=self.motifs,
                         weights=self.weights,
                         log_scale=self.log_scale)

    def _get_kl_divergence(self,P,l,k,P_k,l_k):
        return kl_divergence(P_k, l_k, self.N, self.rho_d_range, self.rho_measured)

    def single_run(self, m, k,
                   P_init = 'm x m',
                   l_init = 'dirichlet',
                   return_population = False
                  ):
        seed = (int(time.time() * 1000) + os.getpid()) % (2**32 - 1)
        np.random.seed(seed)
        init_state = self.initialize_state(m,k,P_init,l_init)
        #print(init_state)
    
        def inner_objective(x):
            P,l,P_k,l_k = unwrap_state(x,m,k)
            return energy(P, l, k, P_k, l_k, self.N,
                          self.rho_d_range,
                          self.dnn_d_range,
                          self.dnn_index_range,
                          self.rho_measured,
                          self.deg_nn_measured,
                          motifs_measured=self.motifs,
                          weights=self.weights,
                          log_scale=self.log_scale)
        
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x[-m:]) - 1}) # Constraint for sum of l=1
        result = minimize(inner_objective, init_state, constraints=cons, bounds=[(0, 1)] * len(init_state))

        if return_population:
            # TODO: Implement this
            raise NotImplementedError('Returning the population for every step of the optimization process for stability analysis is not yet supported!')
        
        final_state = result.x
        P,l,P_k,l_k = unwrap_state(final_state,m,k)
        final_state = wrap_state(P,l,m) # Always return a sorted state
        final_energy = self._get_energy(P, l, k, P_k, l_k)
        
        return final_state, final_energy

    def _calculate_metrics(self):
        P, l = self.P, self.l
        P_k = kron_pow(self.P, self.k)
        l_k = kron_pow(self.l.reshape(-1, 1), self.k).flatten()

        # degree distribution and correlation
        rho_analytical, _ = analytical_degree_distribution(P_k, l_k, self.N, self.rho_d_range)
        self.rho_analytical['medoid'] = rho_analytical
        self.rho_analytical['mre'] = MRE(self.rho_measured, rho_analytical)
        self.rho_analytical['kld'] = KL_divergence(self.rho_measured, rho_analytical)
        deg_nn_analytical,_ = analytical_nearest_neighbor_avg_degree(P_k, l_k, rho_analytical[self.dnn_index_range[0]:self.dnn_index_range[1]], self.N, self.dnn_d_range)
        self.deg_nn_analytical['medoid'] = deg_nn_analytical
        self.deg_nn_analytical['mre'] = MRE(self.deg_nn_measured, deg_nn_analytical)

        #motifs
        for motif in self.motifs:
            motif_name = motif['name']
            analyzer = MotifAnalyzer(adj_matrices=[motif['matrix']], names=[motif_name])
            motif_analytical = analyzer.expected_number_of_motifs(P, l, self.k, self.N)
            motif['expected_count'] = motif_analytical[motif_name]
            motif_measured = next(motif_dict['count'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
            mre_motif = MRE(motif_measured, motif_analytical[motif_name])
            motif['MRE'] = mre_motif    
            if self.motif_count_ci:
                motif_measured_low = next(motif_dict['count_ci_low'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                motif['MRE_ci_low'] = mre_motif * (1 - MRE(motif_measured, motif_measured_low))
                motif_measured_high = next(motif_dict['count_ci_high'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                motif['MRE_ci_high'] = mre_motif * (1 + MRE(motif_measured, motif_measured_high))
                if motif['MRE_ci_low'] > motif['MRE_ci_high']:
                    motif['MRE_ci_low'], motif['MRE_ci_high'] = motif['MRE_ci_high'], motif['MRE_ci_low']
    
    def find_P_l(self, m, k,
                 P_init = 'm x m',
                 l_init = 'dirichlet',
                 verbose = True,
                 bootstrap = False,
                 one_shot = False,
                 return_population = False
                 ):
        with Parallel(n_jobs=self.n_jobs, backend='multiprocessing') as parallel:
            results = parallel(delayed(self.single_run)(m, k, P_init, l_init) for _ in range(self.population_size))

        candidates, energies = zip(*results)
        del results
        gc.collect()

        if return_population:
            # TODO: Implement this.
            raise NotImplementedError('Returning the population for every step of the optimization process for stability analysis is not yet supported!')

        if bootstrap:
            return candidates, energies
        
        energies = np.array(energies)
        best_run = np.argmin(energies)
        final_state = candidates[best_run]
        P, l, P_k, l_k = unwrap_state(final_state,m,k,kronpow = True)
        final_energy = energies[best_run]
        final_kl_divergence = self._get_kl_divergence(P, l, k, P_k, l_k)
        final_kl_energy = self._get_kl_energy(P, l, k, P_k, l_k)
        
        if one_shot:
            print('Energy:',final_energy)
            print('KL-divergence:',final_kl_divergence)
            print('KLD augumented energy:',final_kl_energy)
            self.P,self.l,_,_ = unwrap_state(final_state,m,k)
            self.k = k
            self._calculate_metrics()
        else:
            return ({'energy':final_energy, 'kl_divergence':final_kl_divergence, 'kl_energy': final_kl_energy}, {'P':P,'l':l})
        
    def objective(self, trial, m_list, k_range, P_init, l_init,
                  optimize_for = 'kl_divergence',
                  grid_search = True, 
                  verbose = False
                 ):
        
        m = trial.suggest_categorical('m', m_list)
        if grid_search:
            k = trial.suggest_categorical('k', k_range)
        else:
            if len(k_range) == 2:
                k = trial.suggest_int('k', k_range[0], k_range[1])
            elif len(k_range) == 1:
                k = k_range[0]
            else:
                raise ValueError("Valid k_range values are: 'auto', or a list of either 1 or 2 (min, max) elements.")
        metric, results = self.find_P_l(m, k, P_init = P_init, l_init = l_init, verbose = verbose, one_shot = False)
        trial.set_user_attr('P', results['P'])
        trial.set_user_attr('l', results['l'])
        trial.set_user_attr('energy', metric['energy'])
        trial.set_user_attr('kl_divergence', metric['kl_divergence'])
        trial.set_user_attr('kl_energy', metric['kl_energy'])
        return metric[optimize_for]

    def _find_m_k_P_l(self, m_list, k_range,
                      n_trials=20,
                      P_init = 'm x m',
                      l_init = 'dirichlet',
                      optimize_for = 'kl_divergence',
                      grid_search = True,
                      verbose = False
                     ):
        
        if optimize_for != 'kl_divergence' and optimize_for != 'energy' and optimize_for != 'kl_energy':
            raise ValueError("'kl_divergence', 'energy' and 'kl_energy' can be used in deciding the best (m,k) values!)")
        
        if grid_search:
            if len(k_range) == 2:
                k_range = list(range(k_range[0],k_range[1]))
            else:
                raise ValueError("Valid k_range values are: 'auto', or a list of either 1 or 2 (min, max) elements.")
            search_space = {'m': m_list,
                            'k': k_range}
            print(f"Grid Search requires {len(m_list)*len(k_range)} trials...")
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(lambda trial: self.objective(trial, m_list, k_range, P_init, l_init, optimize_for, grid_search, verbose), n_trials=len(m_list)*len(k_range))
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, m_list, k_range, P_init, l_init, optimize_for, grid_search, verbose), n_trials=n_trials)

        best_trial = study.best_trial
        if len(k_range) >= 2 or k_range == 'auto':
            k = best_trial.params['k']
        else:
            k = k_range[0]
        P = best_trial.user_attrs['P']
        l = best_trial.user_attrs['l']
        
        return k,l,P,best_trial.value,study

    def find_m_k_P_l(self, m_k_dict,
                     n_trials=None,
                     P_init = 'm x m',
                     l_init = 'dirichlet',
                     optimize_for = 'kl_divergence',
                     grid_search = True,
                     verbose = False
                    ):
        ks,ls,Ps,energies,studies = [],[],[],[],[]
        for m_k in m_k_dict:
            k,l,P,energy,study = self._find_m_k_P_l(m_list = [m_k['m']],
                                                    k_range = m_k['k'],
                                                    n_trials = n_trials,
                                                    P_init = P_init,
                                                    l_init = l_init,
                                                    optimize_for = optimize_for,
                                                    grid_search = grid_search,
                                                    verbose = verbose)
            ks.append(k)
            ls.append(l)
            Ps.append(P)
            energies.append(energy)
            studies.append(study)

        energies = np.array(energies)

        best_index = np.argmin(energies)
        self.study = studies[best_index]
        best_energy = energies[best_index]
        self.k = ks[best_index]
        self.P = Ps[best_index]
        self.l = ls[best_index]

        print("Best parameters: ", self.study.best_trial.params)
        print("Best energy: ", self.study.best_trial.user_attrs['energy'])
        print("Best KL-divergence: ", self.study.best_trial.user_attrs['kl_divergence'])
        print("Best KLD augumented energy:", self.study.best_trial.user_attrs['kl_energy'])
        print("Best P matrix:\n", self.P)
        print("Best l vector:\n", self.l)
        self._calculate_metrics()

    def bootstrap_P_l_stability(self,
                                P_init = 'm x m',
                                l_init = 'dirichlet',
                                verbose = True,
                                resample_size = 100,
                                ci = 0.95,
                                medoid = 'mean'
                                ):
        if medoid != 'mean' and medoid != 'median' and medoid != 'lowest':
            raise ValueError("medoid can be 'mean', 'median' or 'lowest'!")
        
        # Prepare an expanding list to hold all states (P, l) and energies
        all_candidates = []
        all_energies = []

        m = self.P.shape[0]
        if resample_size < self.population_size:
            raise ValueError('resample_size should be larger than the population_size!')
        reruns = (resample_size // self.population_size) * 30
    
        # Run find_P_l multiple times to gather a large set of states and energies
        print(f"Gathering statistics ({reruns} runs)...")
        for _ in tqdm(range(reruns)):
            candidates, energies = self.find_P_l(m, self.k, P_init, l_init,
                                                 bootstrap=True,
                                                 verbose = False,
                                                 one_shot = False,
                                                 return_population = False)
            all_candidates.extend(candidates)
            all_energies.extend(energies)
    
        all_candidates = np.array(all_candidates)
        all_energies = np.array(all_energies)
    
        # Parallelized resampling with joblib
        print(f"Bootstrapping: Resampling {resample_size} times...")

        def _resample_and_find_best(all_candidates, all_energies, population_size):
            # Resample with replacement
            resampled_indices = np.random.choice(len(all_candidates), size=population_size, replace=True)
            resampled_candidates = all_candidates[resampled_indices]
            resampled_energies = all_energies[resampled_indices]
            # Pick the best (minimum energy) from the resample
            best_idx = np.argmin(resampled_energies)
            return resampled_candidates[best_idx], resampled_energies[best_idx]
        
        start_time = datetime.datetime.now()
        resampled_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_resample_and_find_best)(all_candidates, all_energies, self.population_size)
            for _ in range(resample_size)
        )
        end_time = datetime.datetime.now()
        print(f"Bootstrapping finished in {end_time - start_time}")
    
        # Separate candidates and energies
        resampled_best_candidates, resampled_best_energies = zip(*resampled_results)
        self.resampled_best_candidates = np.array(resampled_best_candidates)
        self.resampled_best_energies = np.array(resampled_best_energies)

        self._calculate_bootstrap_metrics(ci = ci, medoid = medoid)

        # Set bootstrap to true, to tell the plotting functions to add the CI-s!
        self.bootstrap = True

    def _calculate_bootstrap_metrics(self, ci, medoid):
        m = self.P.shape[0]
        
        # Compute confidence intervals for P, l, and energy
        ci_low = (1 - ci) / 2 * 100
        ci_high = (ci + (1 - ci) / 2) * 100

        # Find the best state in the bunch
        if medoid == 'lowest':
            lowest_index = np.argmin(self.resampled_best_energies)

        # Find the point estimate based on user choice
        if medoid == 'median':
            medoid_state = np.median(self.resampled_best_candidates, axis=0)
        elif medoid == 'mean':
            medoid_state = np.mean(self.resampled_best_candidates, axis=0)
        elif medoid == 'lowest':
            medoid_state = self.resampled_best_candidates[lowest_index]
        P_medoid, l_medoid = unwrap_state(medoid_state, m, self.k, kronpow = False)
        
        ci_low_state = np.percentile(self.resampled_best_candidates, ci_low, axis=0)
        P_ci_low, l_ci_low, P_k_low, l_k_low = unwrap_state(ci_low_state, m, self.k, kronpow = True)
        
        ci_high_state = np.percentile(self.resampled_best_candidates, ci_high, axis=0)
        P_ci_high, l_ci_high, P_k_high, l_k_high = unwrap_state(ci_high_state, m, self.k, kronpow = True)

        if medoid == 'median':
            energy_medoid = np.median(self.resampled_best_energies)
        elif medoid == 'mean':
            energy_medoid = np.mean(self.resampled_best_energies)
        elif medoid == 'lowest':
            energy_medoid = self.resampled_best_energies[lowest_index]
        
        energy_ci_low = np.percentile(self.resampled_best_energies, ci_low)
        if medoid == 'lowest' and energy_ci_low > energy_medoid:
            # No harm done.
            energy_ci_low = energy_medoid
        energy_ci_high = np.percentile(self.resampled_best_energies, ci_high)

        # Compute the CI-s for the metrics
        # Start by getting them for all 'resampled_best_candidates'
        resampled_rho_analyticals = []
        resampled_rho_analytical_mres = []
        resampled_rho_analytical_klds = []
        resampled_deg_nn_analyticals = []
        resampled_deg_nn_analytical_mres = []
        resampled_expected_counts = {motif['name']: [] for motif in self.motifs}
        resampled_mres = {motif['name']: [] for motif in self.motifs}
        if self.ground_truth:
            resampled_ground_truth_rho_analytical_mres = []
            resampled_ground_truth_rho_analytical_klds = []
            resampled_ground_truth_deg_nn_analytical_mres = []
            resampled_ground_truth_mres = {motif['name']: [] for motif in self.motifs}
        for resampled_state in self.resampled_best_candidates:
            P_resampled, l_resampled, P_k_resampled, l_k_resampled = unwrap_state(resampled_state, m, self.k, kronpow = True)
            # degree distribution and correlation
            resampled_rho_analytical, _ = analytical_degree_distribution(P_k_resampled, l_k_resampled, self.N, self.rho_d_range)
            resampled_rho_analyticals.append(resampled_rho_analytical)
            resampled_rho_analytical_mres.append(MRE(self.rho_measured, resampled_rho_analytical))
            resampled_rho_analytical_klds.append(KL_divergence(self.rho_measured, resampled_rho_analytical))
            if self.ground_truth:
                resampled_ground_truth_rho_analytical_mres.append(MRE(self.rho_analytical['ground_truth'], resampled_rho_analytical))
                resampled_ground_truth_rho_analytical_klds.append(KL_divergence(self.rho_analytical['ground_truth'], resampled_rho_analytical))
            resampled_deg_nn_analytical,_ = analytical_nearest_neighbor_avg_degree(P_k_resampled, l_k_resampled, resampled_rho_analytical[self.dnn_index_range[0]:self.dnn_index_range[1]], self.N, self.dnn_d_range)
            resampled_deg_nn_analyticals.append(resampled_deg_nn_analytical)
            resampled_deg_nn_analytical_mres.append(MRE(self.deg_nn_measured, resampled_deg_nn_analytical))
            if self.ground_truth:
                resampled_ground_truth_deg_nn_analytical_mres.append(MRE(self.deg_nn_analytical['ground_truth'], resampled_deg_nn_analytical))
            # MRE for motifs
            for motif in self.motifs:
                motif_name = motif['name']
                analyzer = MotifAnalyzer(adj_matrices=[motif['matrix']], names=[motif_name])
                motif_analytical = analyzer.expected_number_of_motifs(P_resampled, l_resampled, self.k, self.N)
                resampled_expected_counts[motif_name].append(motif_analytical[motif_name])
                motif_measured = next(motif_dict['count'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                mre_motif = MRE(motif_measured, motif_analytical[motif_name])
                resampled_mres[motif_name].append(mre_motif)
                if self.motif_count_ci:
                    motif_measured_low = next(motif_dict['count_ci_low'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                    mre_motif_low = MRE(motif_measured_low, motif_analytical[motif_name])
                    resampled_mres[motif_name].append(mre_motif_low)
                    motif_measured_high = next(motif_dict['count_ci_high'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                    mre_motif_high = MRE(motif_measured_high, motif_analytical[motif_name])
                    resampled_mres[motif_name].append(mre_motif_high)
                if self.ground_truth:
                    # If ground truth is already set, we can compare!
                    motif_ground_truth = analyzer.expected_number_of_motifs(self.ground_truth['P'], self.ground_truth['l'], self.ground_truth['k'], self.N)
                    ground_truth_mre_motif = MRE(motif_ground_truth[motif_name], motif_analytical[motif_name])
                    resampled_ground_truth_mres[motif_name].append(ground_truth_mre_motif)
                

        resampled_rho_analyticals = np.array(resampled_rho_analyticals)
        resampled_rho_analytical_mres = np.array(resampled_rho_analytical_mres)
        resampled_rho_analytical_klds = np.array(resampled_rho_analytical_klds)
        resampled_deg_nn_analyticals = np.array(resampled_deg_nn_analyticals)
        resampled_deg_nn_analytical_mres = np.array(resampled_deg_nn_analytical_mres)
        if self.ground_truth:
            resampled_ground_truth_rho_analytical_mres = np.array(resampled_ground_truth_rho_analytical_mres)
            resampled_ground_truth_rho_analytical_klds = np.array(resampled_ground_truth_rho_analytical_klds)
            resampled_ground_truth_deg_nn_analytical_mres = np.array(resampled_ground_truth_deg_nn_analytical_mres)
        # Set the model results accordingly
        self.P, self.l = sort_state(P_medoid, l_medoid)
        if medoid == 'median':
            resampled_rho_analytical_medoid = np.median(resampled_rho_analyticals, axis=0)
            resampled_rho_analytical_mre_medoid = np.median(resampled_rho_analytical_mres, axis=0)
            resampled_rho_analytical_kld_medoid = np.median(resampled_rho_analytical_klds, axis=0)
            resampled_deg_nn_analytical_medoid = np.median(resampled_deg_nn_analyticals, axis=0)
            resampled_deg_nn_analytical_mre_medoid = np.median(resampled_deg_nn_analytical_mres, axis=0)
            if self.ground_truth:
                resampled_ground_truth_rho_analytical_mre_medoid = np.median(resampled_ground_truth_rho_analytical_mres, axis=0)
                resampled_ground_truth_rho_analytical_kld_medoid = np.median(resampled_ground_truth_rho_analytical_klds, axis=0)
                resampled_ground_truth_deg_nn_analytical_mre_medoid = np.median(resampled_ground_truth_deg_nn_analytical_mres, axis=0)
        elif medoid == 'mean':
            resampled_rho_analytical_medoid = np.mean(resampled_rho_analyticals, axis=0)
            resampled_rho_analytical_mre_medoid = np.mean(resampled_rho_analytical_mres, axis=0)
            resampled_rho_analytical_kld_medoid = np.mean(resampled_rho_analytical_klds, axis=0)
            resampled_deg_nn_analytical_medoid = np.mean(resampled_deg_nn_analyticals, axis=0)
            resampled_deg_nn_analytical_mre_medoid = np.mean(resampled_deg_nn_analytical_mres, axis=0)
            if self.ground_truth:
                resampled_ground_truth_rho_analytical_mre_medoid = np.mean(resampled_ground_truth_rho_analytical_mres, axis=0)
                resampled_ground_truth_rho_analytical_kld_medoid = np.mean(resampled_ground_truth_rho_analytical_klds, axis=0)
                resampled_ground_truth_deg_nn_analytical_mre_medoid = np.mean(resampled_ground_truth_deg_nn_analytical_mres, axis=0)                
        elif medoid == 'lowest':
            resampled_rho_analytical_medoid = resampled_rho_analyticals[lowest_index]
            resampled_rho_analytical_mre_medoid = resampled_rho_analytical_mres[lowest_index]
            resampled_rho_analytical_kld_medoid = resampled_rho_analytical_klds[lowest_index]
            resampled_deg_nn_analytical_medoid = resampled_deg_nn_analyticals[lowest_index]
            resampled_deg_nn_analytical_mre_medoid = resampled_deg_nn_analytical_mres[lowest_index]
            if self.ground_truth:
                resampled_ground_truth_rho_analytical_mre_medoid = resampled_ground_truth_rho_analytical_mres[lowest_index]
                resampled_ground_truth_rho_analytical_kld_medoid = resampled_ground_truth_rho_analytical_klds[lowest_index]
                resampled_ground_truth_deg_nn_analytical_mre_medoid = resampled_ground_truth_deg_nn_analytical_mres[lowest_index]
        self.rho_analytical['medoid'] = resampled_rho_analytical_medoid
        self.rho_analytical['ci_low'] = np.percentile(resampled_rho_analyticals, ci_low, axis=0)
        self.rho_analytical['ci_high'] = np.percentile(resampled_rho_analyticals, ci_high, axis=0)
        self.rho_analytical['confidence_level'] = ci
        self.rho_analytical['mre'] = resampled_rho_analytical_mre_medoid
        self.rho_analytical['mre_ci_high'] = np.percentile(resampled_rho_analytical_mres, ci_high, axis=0)
        self.rho_analytical['mre_ci_low'] = np.percentile(resampled_rho_analytical_mres, ci_low, axis=0)
        if self.rho_analytical['mre_ci_low'] > self.rho_analytical['mre_ci_high']:
            self.rho_analytical['mre_ci_low'], self.rho_analytical['mre_ci_high'] = self.rho_analytical['mre_ci_high'], self.rho_analytical['mre_ci_low']
        if medoid == 'lowest' and self.rho_analytical['mre_ci_low'] > self.rho_analytical['mre']:
            self.rho_analytical['mre_ci_low'] = self.rho_analytical['mre']
        if self.ground_truth:
            self.rho_analytical['ground_truth_mre'] = resampled_ground_truth_rho_analytical_mre_medoid
            self.rho_analytical['ground_truth_mre_ci_high'] = np.percentile(resampled_ground_truth_rho_analytical_mres, ci_high, axis=0)
            self.rho_analytical['ground_truth_mre_ci_low'] = np.percentile(resampled_ground_truth_rho_analytical_mres, ci_low, axis=0)
            if self.rho_analytical['ground_truth_mre_ci_low'] > self.rho_analytical['ground_truth_mre_ci_high']:
                self.rho_analytical['ground_truth_mre_ci_low'], self.rho_analytical['ground_truth_mre_ci_high'] = self.rho_analytical['ground_truth_mre_ci_high'], self.rho_analytical['ground_truth_mre_ci_low']
            if medoid == 'lowest' and self.rho_analytical['ground_truth_mre_ci_low'] > self.rho_analytical['ground_truth_mre']:
                self.rho_analytical['ground_truth_mre_ci_low'] = self.rho_analytical['ground_truth_mre']
        self.rho_analytical['kld'] = resampled_rho_analytical_kld_medoid
        self.rho_analytical['kld_ci_high'] = np.percentile(resampled_rho_analytical_klds, ci_high, axis=0)
        self.rho_analytical['kld_ci_low'] = np.percentile(resampled_rho_analytical_klds, ci_low, axis=0)
        if self.rho_analytical['kld_ci_low'] > self.rho_analytical['kld_ci_high']:
            self.rho_analytical['kld_ci_low'], self.rho_analytical['kld_ci_high'] = self.rho_analytical['kld_ci_high'], self.rho_analytical['kld_ci_low']
        if self.ground_truth:
            self.rho_analytical['ground_truth_kld'] = resampled_ground_truth_rho_analytical_kld_medoid
            self.rho_analytical['ground_truth_kld_ci_high'] = np.percentile(resampled_ground_truth_rho_analytical_klds, ci_high, axis=0)
            self.rho_analytical['ground_truth_kld_ci_low'] = np.percentile(resampled_ground_truth_rho_analytical_klds, ci_low, axis=0)
            if self.rho_analytical['ground_truth_kld_ci_low'] > self.rho_analytical['ground_truth_kld_ci_high']:
                self.rho_analytical['ground_truth_kld_ci_low'], self.rho_analytical['ground_truth_kld_ci_high'] = self.rho_analytical['ground_truth_kld_ci_high'], self.rho_analytical['ground_truth_kld_ci_low']
            if medoid == 'lowest' and self.rho_analytical['ground_truth_kld_ci_low'] > self.rho_analytical['ground_truth_kld']:
                self.rho_analytical['ground_truth_kld_ci_low'] = self.rho_analytical['ground_truth_kld']            
        if medoid == 'lowest' and self.rho_analytical['kld_ci_low'] > self.rho_analytical['kld']:
            self.rho_analytical['kld_ci_low'] = self.rho_analytical['kld']
        self.deg_nn_analytical['medoid'] = resampled_deg_nn_analytical_medoid
        self.deg_nn_analytical['ci_low'] = np.percentile(resampled_deg_nn_analyticals, ci_low, axis=0)
        self.deg_nn_analytical['ci_high'] = np.percentile(resampled_deg_nn_analyticals, ci_high, axis=0)
        self.deg_nn_analytical['confidence_level'] = ci
        self.deg_nn_analytical['mre'] = resampled_deg_nn_analytical_mre_medoid
        self.deg_nn_analytical['mre_ci_high'] = np.percentile(resampled_deg_nn_analytical_mres, ci_high, axis=0)
        self.deg_nn_analytical['mre_ci_low'] = np.percentile(resampled_deg_nn_analytical_mres, ci_low, axis=0)
        if self.deg_nn_analytical['mre_ci_low'] > self.deg_nn_analytical['mre_ci_high']:
            self.deg_nn_analytical['mre_ci_low'], self.deg_nn_analytical['mre_ci_high'] = self.deg_nn_analytical['mre_ci_high'], self.deg_nn_analytical['mre_ci_low']
        if medoid == 'lowest' and self.deg_nn_analytical['mre_ci_low'] > self.deg_nn_analytical['mre']:
            self.deg_nn_analytical['mre_ci_low'] = self.deg_nn_analytical['mre']
        if self.ground_truth:
            self.deg_nn_analytical['ground_truth_mre'] = resampled_ground_truth_deg_nn_analytical_mre_medoid
            self.deg_nn_analytical['ground_truth_mre_ci_high'] = np.percentile(resampled_ground_truth_deg_nn_analytical_mres, ci_high, axis=0)
            self.deg_nn_analytical['ground_truth_mre_ci_low'] = np.percentile(resampled_ground_truth_deg_nn_analytical_mres, ci_low, axis=0)
            if self.deg_nn_analytical['ground_truth_mre_ci_low'] > self.deg_nn_analytical['ground_truth_mre_ci_high']:
                self.deg_nn_analytical['ground_truth_mre_ci_low'], self.deg_nn_analytical['ground_truth_mre_ci_high'] = self.deg_nn_analytical['ground_truth_mre_ci_high'], self.deg_nn_analytical['ground_truth_mre_ci_low']
            if medoid == 'lowest' and self.deg_nn_analytical['ground_truth_mre_ci_low'] > self.deg_nn_analytical['ground_truth_mre']:
                self.deg_nn_analytical['ground_truth_mre_ci_low'] = self.deg_nn_analytical['ground_truth_mre']
        # Including the motifs...
        for motif in self.motifs:
            motif_name = motif['name']
            _resempled_expected_counts = np.array(resampled_expected_counts[motif_name])
            _resampled_mres = np.array(resampled_mres[motif_name])
            if self.ground_truth:
                _resampled_ground_truth_mres = np.array(resampled_ground_truth_mres[motif_name])
            if medoid == 'median':
                motif['expected_count'] = np.median(_resempled_expected_counts)
                motif['MRE'] = np.median(_resampled_mres)
                if self.ground_truth:
                    motif['fit_ground_truth_MRE'] = np.median(_resampled_ground_truth_mres)
            elif medoid == 'mean':
                motif['expected_count'] = np.mean(_resempled_expected_counts)
                motif['MRE'] = np.mean(_resampled_mres)
                if self.ground_truth:
                    motif['fit_ground_truth_MRE'] = np.mean(_resampled_ground_truth_mres)
            elif medoid == 'lowest':
                motif['expected_count'] = _resempled_expected_counts[lowest_index]
                motif['MRE'] = _resampled_mres[lowest_index]
                if self.ground_truth:
                    motif['fit_ground_truth_MRE'] = _resampled_ground_truth_mres[lowest_index]
            # CI for counts
            motif['expected_count_ci_low'] = np.percentile(_resempled_expected_counts, ci_low)
            motif['expected_count_ci_high'] = np.percentile(_resempled_expected_counts, ci_high)
            if motif['expected_count_ci_low'] > motif['expected_count_ci_high']:
                motif['expected_count_ci_low'], motif['expected_count_ci_high'] = motif['expected_count_ci_high'], motif['expected_count_ci_low']
            if medoid == 'lowest' and motif['expected_count_ci_low'] > motif['expected_count']:
                motif['expected_count_ci_low'] = motif['expected_count']
            # CI for errors
            motif['MRE_ci_low'] = np.percentile(_resampled_mres, ci_low)
            motif['MRE_ci_high'] = np.percentile(_resampled_mres, ci_high)
            if motif['MRE_ci_low'] > motif['MRE_ci_high']:
                motif['MRE_ci_low'], motif['MRE_ci_high'] = motif['MRE_ci_high'], motif['MRE_ci_low']
            if medoid == 'lowest' and motif['MRE_ci_low'] > motif['MRE']:
                motif['MRE_ci_low'] = motif['MRE']
            if self.ground_truth:
                motif['fit_ground_truth_MRE_ci_low'] = np.percentile(_resampled_ground_truth_mres, ci_low)
                motif['fit_ground_truth_MRE_ci_high'] = np.percentile(_resampled_ground_truth_mres, ci_high)
                if motif['fit_ground_truth_MRE_ci_low'] > motif['fit_ground_truth_MRE_ci_high']:
                    motif['fit_ground_truth_MRE_ci_low'], motif['fit_ground_truth_MRE_ci_high'] = motif['fit_ground_truth_MRE_ci_high'], motif['fit_ground_truth_MRE_ci_low']
                if medoid == 'lowest' and motif['fit_ground_truth_MRE_ci_low'] > motif['fit_ground_truth_MRE']:
                    motif['fit_ground_truth_MRE_ci_low'] = motif['fit_ground_truth_MRE']

        # Print results
        print(f"Energy = {energy_medoid:.2f} ({energy_ci_low:.2f}, {energy_ci_high:.2f})")
        
        print("l = [", end="")
        for i, l_val in enumerate(l_medoid):
            print(f"{l_val:.2f} ({l_ci_low[i]:.2f}, {l_ci_high[i]:.2f})", end="")
            if i < len(l_medoid) - 1:
                print(", ", end="")
        print("]")
    
        print("P = [")
        for i in range(P_medoid.shape[0]):
            print("[", end="")
            for j in range(P_medoid.shape[1]):
                print(f"{P_medoid[i, j]:.2f} ({P_ci_low[i, j]:.2f}, {P_ci_high[i, j]:.2f})", end="")
                if j < P_medoid.shape[1] - 1:
                    print(", ", end="")
            print("]", end="")
            if i < P_medoid.shape[0] - 1:
                print(",\n")
        print("\n]")

    def set_ground_truth(self, P, l, k,
                         ci = 0.95,
                         medoid = 'mean'):
        # For bootstrap calculations
        if medoid != 'mean' and medoid != 'median':
            raise ValueError("medoid can be 'mean' or 'median'!")
        # For bootstrap CI calculation
        ci_low = (1 - ci) / 2 * 100
        ci_high = (ci + (1 - ci) / 2) * 100

        # Set the ground truth values as the input
        self.ground_truth = {'P': P,
                             'l': l,
                             'k': k,
                             'P_k': kron_pow(P,k),
                             'l_k': kron_pow(l,k)}
        
        # degree distribution and correlation
        rho_analytical, _ = analytical_degree_distribution(self.ground_truth['P_k'], self.ground_truth['l_k'], self.N, self.rho_d_range)
        self.rho_analytical['ground_truth'] = rho_analytical
        self.rho_analytical['ground_truth_mre'] = MRE(self.rho_measured, rho_analytical)
        self.rho_analytical['ground_truth_kld'] = KL_divergence(self.rho_measured, rho_analytical)
        deg_nn_analytical,_ = analytical_nearest_neighbor_avg_degree(self.ground_truth['P_k'], self.ground_truth['l_k'], rho_analytical[self.dnn_index_range[0]:self.dnn_index_range[1]], self.N, self.dnn_d_range)
        self.deg_nn_analytical['ground_truth'] = deg_nn_analytical
        self.deg_nn_analytical['ground_truth_mre'] = MRE(self.deg_nn_measured, deg_nn_analytical)
        #motifs
        for motif in self.motifs:
            # ground truth vs measured
            motif_name = motif['name']
            analyzer = MotifAnalyzer(adj_matrices=[motif['matrix']], names=[motif_name])
            motif_analytical = analyzer.expected_number_of_motifs(self.ground_truth['P'], self.ground_truth['l'], self.ground_truth['k'], self.N)
            motif['ground_truth_analytical'] = motif_analytical[motif_name]
            motif_measured = next(motif_dict['count'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
            mre_motif = MRE(motif_measured, motif_analytical[motif_name])
            motif['ground_truth_MRE'] = mre_motif   
            if self.motif_count_ci:
                motif_measured_low = next(motif_dict['count_ci_low'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                motif['ground_truth_MRE_ci_low'] = mre_motif * (1 - MRE(motif_measured, motif_measured_low))
                motif_measured_high = next(motif_dict['count_ci_high'] for motif_dict in self.motifs if motif_dict['name'] == motif_name)
                motif['ground_truth_MRE_ci_high'] = mre_motif * (1 + MRE(motif_measured, motif_measured_high))
                if motif['ground_truth_MRE_ci_low'] > motif['ground_truth_MRE_ci_high']:
                    motif['ground_truth_MRE_ci_low'], motif['ground_truth_MRE_ci_high'] = motif['ground_truth_MRE_ci_high'], motif['ground_truth_MRE_ci_low']
            # If the modell has already been fitted, calculate the MRE between the fit and the ground truth
            if self.P is not None and not self.bootstrap:
                motif_analytical_fit = analyzer.expected_number_of_motifs(self.P, self.l, self.k, self.N)
                mre_motif = MRE(motif_analytical[motif_name], motif_analytical_fit[motif_name])
                motif['fit_ground_truth_MRE'] = mre_motif
            # If it's also bootstrapped we can also get CI-s
            elif self.bootstrap:
                resampled_ground_truth_mres = []
                for resampled_state in self.bootstrap_states:
                    P_resampled, l_resampled, P_k_resampled, l_k_resampled = unwrap_state(resampled_state, m, self.k, kronpow = True)
                    motif_analytical_fit = analyzer.expected_number_of_motifs(P_resampled, l_resampled, self.k, self.N)
                    mre_motif = MRE(motif_analytical[motif_name], motif_analytical_fit[motif_name])
                    resampled_ground_truth_mres.append(mre_motif)
                resampled_ground_truth_mres = np.array(resampled_ground_truth_mres)
                if medoid == 'median':
                    motif['fit_ground_truth_MRE'] = np.median(resampled_ground_truth_mres)
                elif medoid == 'mean':
                    motif['fit_ground_truth_MRE'] = np.mean(resampled_ground_truth_mres)
                motif['fit_ground_truth_MRE_ci_low'] = np.percentile(resampled_ground_truth_mres, ci_low)
                motif['fit_ground_truth_MRE_ci_high'] = np.percentile(resampled_ground_truth_mres, ci_high)
                if motif['fit_ground_truth_MRE_ci_low'] > motif['fit_ground_truth_MRE_ci_high']:
                        motif['fit_ground_truth_MRE_ci_low'], motif['fit_ground_truth_MRE_ci_high'] = motif['fit_ground_truth_MRE_ci_high'], motif['fit_ground_truth_MRE_ci_low']

    def plot_P_l_result(self,
                        log_scale = False,
                        grid = False,
                        display_mre = True,
                        display_kld = False,
                        plot_ground_truth = False,
                        plot_cumulative_error = False,
                        filename = None,
                        xlim = None,
                        ylim = None,
                        streamlit = False
                       ):
        """
        Plots results for rho, degree-nearest-neighbor, motif spectra, and metric distributions.
    
        Parameters
        ----------
        log_scale : bool, optional
            If True, use logarithmic scale for the plots.
        grid : bool, optional
            If True, show grid lines in the metric plot.
        display_mre : bool, optional
            If True, display the mean relative error (MRE) annotations in the rho and degree-nearest-neighbor plots.
        display_kld : bool, optional
            If True, display the KL-divergence annotations in the rho and degree-nearest-neighbor plots.
        plot_cumulative_error : bool, optional
            If True, display cumulative errors in the motif spectra plot.
        filename : str, optional
            Base filename for saving plots. Separate filenames will be generated for different plots.
        xlim : tuple, optional
            Limits for the x-axis as a tuple (xmin, xmax). Default is None.
        ylim : tuple, optional
            Limits for the y-axis as a tuple (ymin, ymax). Default is None.
        """
    
        if filename:
            base, extension = os.path.splitext(filename)
            filename_rho = f"{base}_rho{extension}"
            filename_deg_nn = f"{base}_deg_nn{extension}"
            filename_motifs = f"{base}_motifs{extension}"
            filename_measure = f"{base}_measure{extension}"
        else:
            filename_rho, filename_deg_nn, filename_motifs, filename_measure = None, None, None, None
    
        self.P, self.l = sort_state(self.P, self.l)
        
        P_k = kron_pow(self.P, self.k)
        l_k = kron_pow(self.l.reshape(-1, 1), self.k).flatten()
    
        if self.bootstrap:
            rho_analytical = self.rho_analytical['medoid']
            rho_analytical_ci = self.rho_analytical
            deg_nn_analytical = self.deg_nn_analytical['medoid']
            deg_nn_analytical_ci = self.deg_nn_analytical
            if self.ground_truth and plot_ground_truth:
                rho_ground_truth = self.rho_analytical['ground_truth']
                deg_nn_ground_truth = self.deg_nn_analytical['ground_truth']
            else:
                rho_ground_truth = None
                deg_nn_ground_truth = None
        else:
            rho_analytical = self.rho_analytical['medoid']
            rho_analytical_ci = None
            deg_nn_analytical = self.deg_nn_analytical['medoid']
            deg_nn_analytical_ci = None
            if self.ground_truth and plot_ground_truth:
                rho_ground_truth = self.rho_analytical['ground_truth']
                deg_nn_ground_truth = self.deg_nn_analytical['ground_truth']
            else:
                rho_ground_truth = None
                deg_nn_ground_truth = None
    
        # Plot rho curves
        fig1 = plot_curves(np.arange(self.rho_d_range[0], self.rho_d_range[1] + 1),
                    self.rho_measured,
                    rho_analytical,
                    self.rho_analytical,
                    xlabel='d',
                    ylabel=r'$\rho^{k}(d)$',
                    log_scale=log_scale,
                    analytical_ci=rho_analytical_ci,
                    ground_truth=rho_ground_truth,
                    display_mre=display_mre,
                    display_kld=display_kld,
                    filename=filename_rho,
                    xlim=xlim,
                    ylim=ylim,
                    streamlit=streamlit)
    
        # Plot degree-nearest-neighbor curves
        fig2 = plot_curves(np.arange(self.dnn_d_range[0], self.dnn_d_range[1] + 1),
                    self.deg_nn_measured,
                    deg_nn_analytical,
                    self.deg_nn_analytical,
                    xlabel='d',
                    ylabel=r'$d^{NN}(d)$',
                    log_scale=log_scale,
                    analytical_ci=deg_nn_analytical_ci,
                    ground_truth=deg_nn_ground_truth,
                    display_mre=display_mre,
                    display_kld=False,
                    filename=filename_deg_nn,
                    streamlit=streamlit)#,
                    #xlim=xlim,
                    #ylim=ylim)
    
        # Plot motif spectra
        fig3 = plot_motif_spectra(self.motifs,
                                  plot_ground_truth = plot_ground_truth,
                                  plot_cumulative_error=plot_cumulative_error,
                                  log_scale=log_scale,
                                  filename=filename_motifs,
                                  streamlit=streamlit)
        
        # Plot metric distributions
        if log_scale:
            fig4 = plot_metric(P_k, l_k,
                               grid=grid,
                               norm=LogNorm,
                               filename=filename_measure,
                               streamlit=streamlit)
        else:
            fig4 = plot_metric(P_k, l_k,
                               grid=grid,
                               norm=Normalize,
                               filename=filename_measure,
                               streamlit=streamlit)
        if streamlit:
            return fig1,fig2,fig3,fig4

    def generate_variants(self,
                          plot_input_graphs = True,
                          number_of_variants = 3,
                          plot_options = {'layout_func': lambda G: nx.spring_layout(G),
                                          'draw_options': {'node_color': 'blue',
                                                           'edge_color': 'black',
                                                           'node_size': 50,
                                                           'node_alpha': 1.0,
                                                           'edge_alpha': 0.1},
                                           'figsize': (10,10),
                                           'filename':  None,
                                           'streamlit': False,
                                           'largest_component_only': False,
                                           'smart_viz': True
                                         },
                          filename = None,
                          streamlit = False
                         ):
        generator = MultifractalNetworkGenerator(self.P,self.l,self.k,self.N,num_workers=-1)
        figs = []
        graphs = []
        # First (optionally) plot the original network.
        if plot_input_graphs:
            plot_options_original = plot_options.copy()
            plot_options_original['title'] = 'Input Graph'
            if plot_options_original['filename'] is not None:
                base, extension = os.path.splitext(plot_options_original['filename'])
                plot_options_original['filename'] = f"{base}_input_graph{extension}"
            generator.G = self.G
            fig = generator.plot_generated_network(**plot_options_original)
            figs.append(fig)
            graphs.append(self.G)
        # Then let's generate the variants!
        for i in range(1, number_of_variants + 1):
            generator()
            if plot_options is not None:
                plot_options_variant = plot_options.copy()
                plot_options_variant['title'] = f"Variant Graph {i}"
                if plot_options_variant['filename'] is not None:
                    base, extension = os.path.splitext(plot_options_variant['filename'])
                    plot_options_variant['filename'] = f"{base}_variant_graph_{i}{extension}"
                fig = generator.plot_generated_network(**plot_options_variant)
                figs.append(fig)
            if filename is not None:
                base, extension = os.path.splitext(filename)
                filename_variant = f"{base}_variant_graph_{i}{extension}"
                generator.save_network(filename_variant)
            graphs.append(generator.G.copy())
        if streamlit:
            return figs, graphs 
        else:
            return None