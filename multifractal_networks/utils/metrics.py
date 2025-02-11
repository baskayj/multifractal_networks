import warnings
warnings.filterwarnings('ignore')

import random
import datetime

import numpy as np
import networkx as nx

from itertools import combinations
from collections import Counter
from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.special import comb, factorial
from graph_tool import Graph
from graph_tool.clustering import motifs

from joblib import Parallel, delayed, cpu_count

# -------------------
# Mean relative error
# -------------------
def MRE(truth, pred, debug=False):
    """
    Computes the Mean Relative Error (MRE) between two arrays, `truth` and `pred`, 
    ignoring any positions where both arrays have zero values.
    
    Parameters
    ----------
    truth : array-like or scalar
        The ground truth values. Can be a single float/int or an array-like object.
    pred : array-like or scalar
        The predicted values. Can be a single float/int or an array-like object.
    debug : bool, optional
        If True, return additional debug information (relative errors and cumulative sum).
        
    Returns
    -------
    float or tuple
        - If debug is False, return the mean of the relative errors.
        - If debug is True, return a tuple containing:
            - mean of the relative errors,
            - dictionary with:
                - 'errors': the relative error array (with `None` for invalid indices),
                - 'cumulative_errors': the cumulative sum of relative errors, with `None` treated as 0.
            
    Notes
    -----
    - The function excludes any index where both `truth` and `pred` are zero to avoid division by zero.
    - If all elements in `truth` and `pred` are zero, the function returns `None`.
    """
    # Convert inputs to numpy arrays if they are scalar (float or int)
    truth = np.asarray(truth) if np.ndim(truth) > 0 else np.array([truth])
    pred = np.asarray(pred) if np.ndim(pred) > 0 else np.array([pred])

    # Set negative values to zero.
    truth = np.where(truth < 0, 0, truth)
    pred = np.where(pred < 0, 0, pred)

    # Set a lower bound
    if len(truth) > 1:
        lower_bound = min(truth) if min(truth) < 1 else None
        if lower_bound is not None:
            pred = np.where(pred <= lower_bound, 0, pred)
            truth = np.where(truth <= lower_bound, 0, truth)

    # Identify valid indices where neither truth nor pred is NaN and neither is zero
    both_zero = (truth == 0) & (pred == 0)
    #either_zero = ((truth == 0) | (pred == 0)) & ~both_zero
    invalid = np.isnan(pred) | np.isnan(truth) | np.isinf(pred) | np.isinf(truth)
    #valid_indices = ~ (either_zero | invalid | both_zero)
    valid_indices = ~ (invalid | both_zero)

    # Initialize relative errors with ones
    relative_errors = np.full_like(truth, 1.0, dtype=float)
    
    # Set zero error where both values are zero
    relative_errors[both_zero] = 0.0

    # Compute relative errors for valid indices
    if np.any(valid_indices):
        numerator = np.abs(truth[valid_indices] - pred[valid_indices])
        denominator = np.maximum(truth[valid_indices], pred[valid_indices])
        relative_errors[valid_indices] = numerator / denominator

    # Calculate mean including zeros from both_zero cases
    mean_relative_error = np.mean(relative_errors)

    if debug:
        cumsum = np.cumsum(relative_errors)
        return (mean_relative_error, {'errors': relative_errors, 'cumulative_errors': cumsum})

    return mean_relative_error

def KL_divergence(truth, pred, debug=False):
    """
    Computes the Kullback-Leibler (KL) divergence between two arrays, `truth` and `pred`, 
    excluding any positions where either array has zero or invalid values.
    
    Parameters
    ----------
    truth : array-like
        The ground truth distribution (reference distribution). Should be a valid probability distribution.
    pred : array-like
        The predicted distribution. Should have the same shape as `truth`.
    debug : bool, optional
        If True, return additional debug information (individual KL terms and cumulative sum).
        
    Returns
    -------
    float or tuple
        - If debug is False, return the KL divergence.
        - If debug is True, return a tuple containing:
            - KL divergence value,
            - dictionary with:
                - 'errors': the KL divergence terms array (with `None` for invalid indices),
                - 'cumulative_errors': the cumulative sum of the valid KL terms, with `None` treated as 0.
            
    Notes
    -----
    - The function excludes any index where either `truth` or `pred` is zero or invalid (e.g., infinity).
    - If all elements in `truth` or `pred` are invalid, the function returns `None`.
    - KL divergence is defined as:
        .. math::
            D_{KL}(P || Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)}\right)
    """
    
    truth = np.asarray(truth)
    pred = np.asarray(pred)

    # Handle invalid (negative or zero) values
    truth = np.where(truth < 0, 0, truth)
    pred = np.where(pred < 0, 0, pred)

    # Identify invalid indices (where truth or pred are zero or infinite or None)
    invalid_indices = (truth == 0) | (pred == 0) | np.isinf(truth) | np.isinf(pred) | np.isnan(truth) | np.isnan(pred)

    # Exclude invalid indices
    valid_indices = ~invalid_indices

    # If no valid indices are left, return None
    if not np.any(valid_indices):
        return np.inf if not debug else (np.inf, {'errors': [np.inf] * len(truth), 'cumulative_errors': [np.inf] * len(truth)})

    # Normalize the valid parts of the distributions
    truth_valid = truth[valid_indices]
    pred_valid = pred[valid_indices]
    
    truth_valid = truth_valid / np.sum(truth_valid)
    pred_valid = pred_valid / np.sum(pred_valid)

    # Calculate KL divergence terms
    kl_terms = np.full_like(truth, None, dtype=object)
    kl_terms[valid_indices] = truth_valid * np.log(truth_valid / pred_valid)

    # Calculate total KL divergence
    kl_divergence = np.sum(kl_terms[valid_indices])

    if debug:
        # Compute cumulative sum, treating None as 0
        kl_terms_filled = np.nan_to_num(np.array(kl_terms, dtype=float))
        cumulative_kl = np.cumsum(kl_terms_filled)
        
        return (kl_divergence,
                {'errors': kl_terms,
                 'cumulative_errors': cumulative_kl}
               )

    return kl_divergence

# -------------------
# Palla et al. 2010
# -------------------

# Degree distribution
"""
Analytical expressions for the degree distribution, nearest neighbor average degree,
and clustering coefficient in Kronecker graphs based on Palla et al. (2010).
"""

def analytical_degree_distribution(P_k, l_k, N, d_range):
    """
    Computes the analytical degree distribution for a Kronecker graph.

    The degree distribution is given by:

    .. math::

        \\rho^{(k)}(d) = \\sum_{i=1}^{m^k} \\rho_i^{(k)}(d) \\cdot l_i^{\\otimes k}

    where:

    .. math::

        \\rho_i^{(k)}(d) = \\frac{(\\mathbb{E}[d_i^{(k)}])^d e^{-\\mathbb{E}[d_i^{(k)}]}}{d!}

    and:

    .. math::

        \\mathbb{E}[d_i^{(k)}] = N \\sum_j p^{\\otimes k}_{ij} \\cdot l^{\\otimes k}_j

    Parameters
    ----------
    P_k : numpy.ndarray
        The Kronecker power of the initiator matrix P.
    l_k : numpy.ndarray
        The Kronecker power of the initiator degree distribution l.
    N : int
        The number of nodes in the graph.
    d_range : tuple of int
        The range of degrees to compute the distribution over, given as (min_d, max_d).

    Returns
    -------
    rho_k : numpy.ndarray
        The analytical degree distribution over the specified degree range.
    d : list of int
        The degrees corresponding to the values in rho_k.

    References
    ----------
    Palla, G., Lovász, L.,& Vicsek, T. (2010).
    Multifractal network generator. Proceedings of the National Academy of Sciences of the United States of America 107 (17): 7640–45.

    """
    min_d, max_d = d_range
    m_k = P_k.shape[0]
    # Expected degree for each node
    d_k = N * np.sum(P_k * l_k[:, np.newaxis], axis=0)
    # Generate all degree values in the range at once
    d = np.arange(min_d, max_d + 1)
    # Use broadcasting to compute Poisson PMF for all values of d in one go
    rho_k_d = poisson.pmf(d[:, np.newaxis], d_k)
    # Multiply by l_k and sum across the node dimension to get rho_k for each degree d
    rho_k = np.sum(rho_k_d * l_k, axis=1)
    return rho_k, d

"""
# Expected degree for each node
d_k = N * np.sum(P_k * l_k[:, np.newaxis], axis=0)

# Generate all degree values in the range at once
d = np.arange(min_d, max_d + 1)

# Use broadcasting to compute Poisson PMF for all values of d in one go
rho_k_d = poisson.pmf(d[:, np.newaxis], d_k)

# Multiply by l_k and sum across the node dimension to get rho_k for each degree d
rho_k = np.sum(rho_k_d * l_k, axis=1)
"""

def measure_degree_distribution(G, d_range):
    """
    Measures the degree distribution of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    d_range : tuple of int
        The range of degrees to compute the distribution over, given as (min_d, max_d).

    Returns
    -------
    rho_measured : numpy.ndarray
        The measured degree distribution over the specified degree range.

    """
    min_d, max_d = d_range
    
    degree_sequence = [d for n, d in G.degree()]
    degree_count = Counter(degree_sequence)
    rho_measured = []
    for d in range(min_d, max_d + 1):
        rho_measured.append(degree_count.get(d, 0) / G.number_of_nodes())
    rho_measured = np.array(rho_measured)
    
    return rho_measured

# Nearest neighbor average degree
"""
Computes the nearest neighbor average degree, which is a measure of degree correlations
in the network.

The analytical expression is:

.. math::

    d_{NN}^{(k)}(d) = \\frac{1}{\\rho^{(k)}(d)} \\sum_{i=1}^{m^k} l_i^{\\otimes k} \\rho_i^{(k)}(d)
    \\frac{\\sum_{j=1}^{m^k} p_{ij}^{\\otimes k} l_j^{\\otimes k} d_j^{(k)}}{\\sum_{j=1}^{m^k} p_{ij}^{\\otimes k} l_j^{\\otimes k}}
"""

def analytical_nearest_neighbor_avg_degree(P_k, l_k, rho_k, N, d_range):
    """
    Computes the analytical nearest neighbor average degree for a Kronecker graph.

    Parameters
    ----------
    P_k : numpy.ndarray
        The Kronecker power of the initiator matrix P.
    l_k : numpy.ndarray
        The Kronecker power of the initiator degree distribution l.
    rho_k : numpy.ndarray
        The analytical degree distribution computed over d_range.
    N : int
        The number of nodes in the graph.
    d_range : tuple of int
        The range of degrees to compute over, given as (min_d, max_d).

    Returns
    -------
    nearest_neighbor_avg_degree : numpy.ndarray
        The analytical nearest neighbor average degree over the specified degree range.
    d : list of int
        The degrees corresponding to the values in nearest_neighbor_avg_degree.

    """
    min_d, max_d = d_range
    m_k = P_k.shape[0]
    # Expected degree for each node
    d_k = N * np.sum(P_k * l_k[:, np.newaxis], axis=0)
    # Generate all degree values in the range at once
    d = np.arange(min_d, max_d + 1)
    # Use broadcasting to compute Poisson PMF for all values of d in one go
    rho_k_d = poisson.pmf(d[:, np.newaxis], d_k)
    # Calculate the inner sums
    inner_sum_num = np.einsum('ij,j,j->i', P_k, l_k, d_k)
    inner_sum_den = np.einsum('ij,j->i', P_k, l_k)
    # Calculate the weighted sum
    weighted_sum = np.einsum('i,di,i->d', l_k, rho_k_d, inner_sum_num / inner_sum_den)
    # Compute nearest neighbor average degree
    nearest_neighbor_avg_degree = weighted_sum / rho_k
    return nearest_neighbor_avg_degree, d

def measure_nearest_neighbor_avg_degree(G, d_range):
    """
    Measures the nearest neighbor average degree in a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    d_range : tuple of int
        The range of degrees to compute over, given as (min_d, max_d).

    Returns
    -------
    nearest_neighbor_avg_degree : numpy.ndarray
        The measured nearest neighbor average degree over the specified degree range.

    """
    min_d, max_d = d_range
    # Get the degrees of all nodes
    degrees = dict(G.degree())
    # Compute sum of neighbor degrees for each node
    neighbors_degree_sum = {n: sum(degrees[neighbor] for neighbor in G.neighbors(n)) for n in G.nodes()}
    # Initialize an array to store the average nearest neighbor degrees for each degree
    nearest_neighbor_avg_degree = np.zeros(max_d - min_d + 1)
    # Compute the average nearest neighbor degree for nodes of each degree in the range
    for d in range(min_d, max_d + 1):
        neighbor_degrees = [neighbors_degree_sum[n] / degrees[n] for n in G.nodes() if degrees[n] == d]
        if len(neighbor_degrees) > 0:
            nearest_neighbor_avg_degree[d - min_d] = np.mean(neighbor_degrees)
        else:
            nearest_neighbor_avg_degree[d - min_d] = 0  # If no nodes with degree d
    return nearest_neighbor_avg_degree

# Clustering coefficient
"""
Computes the clustering coefficient, which quantifies the tendency of nodes to form clusters or triangles.

The analytical expression is:

.. math::

    \\langle C_i(k) \\rangle = \\frac{
    \\frac{1}{2} \\sum_j [l_j^{\\otimes k}]^2 [p_{ij}^{\\otimes k}]^2 p_{jj}^{\\otimes k} +
    \\sum_j \\sum_{q > j} l_j^{\\otimes k} l_q^{\\otimes k} p_{ij}^{\\otimes k} p_{iq}^{\\otimes k} p_{jq}^{\\otimes k}
    }{
    \\frac{1}{2} \\sum_j [l_j^{\\otimes k}]^2 [p_{ij}^{\\otimes k}]^2 +
    \\sum_j \\sum_{q > j} l_j^{\\otimes k} l_q^{\\otimes k} p_{ij}^{\\otimes k} p_{iq}^{\\otimes k}
    }

Note
----
This computation is slow for both analytical and empirical calculations, especially on large networks.
Therefore, its use is not recommended for large-scale graphs.
"""

def compute_C_i(P, l, k):
    """
    Computes the clustering coefficient for each node in a Kronecker graph.

    Parameters
    ----------
    P : numpy.ndarray
        The initiator matrix.
    l : numpy.ndarray
        The initiator degree distribution.
    k : int
        The number of Kronecker iterations.

    Returns
    -------
    C_i_values : numpy.ndarray
        The clustering coefficient for each node.

    """
    # Compute the Kronecker powers
    P_k = kron_pow(P, k)
    l_k = kron_pow(l.reshape(-1, 1), k).flatten()

    m_k = P_k.shape[0]

    # Upper triangular matrix U with ones above the diagonal
    U = np.triu(np.ones((m_k, m_k)), k=1)

    # Compute l_k^2 and P_k^2
    l_k_sq = l_k ** 2
    P_k_sq = P_k ** 2

    # First term in numerator for all i
    numerator_1 = np.sum((l_k_sq[:, None] * P_k_sq) * np.diag(P_k), axis=0)

    numerator_2 = np.zeros(m_k)
    denominator_1 = np.sum(l_k_sq[:, None] * P_k_sq, axis=0)
    denominator_2 = np.zeros(m_k)

    for i in range(m_k):
        l_P_k_i = l_k * P_k[i, :]
        for j in range(m_k):
            P_k_i_j = P_k[i, j]
            l_j = l_k[j]
            U_j = U[j, :]
            P_k_j = P_k[j, :]
            numerator_2[i] += np.sum(U_j * l_j * l_P_k_i * P_k_i_j * P_k_j)
            denominator_2[i] += np.sum(U_j * l_j * l_P_k_i * P_k_i_j)

    numerator = 0.5 * numerator_1 + numerator_2
    denominator = 0.5 * denominator_1 + denominator_2

    C_i_values = numerator / denominator
    C_i_values[denominator == 0] = 0

    return C_i_values

def analytical_clustering(P, l, k):
    """
    Computes the average clustering coefficient for a Kronecker graph.

    Parameters
    ----------
    P : numpy.ndarray
        The initiator matrix.
    l : numpy.ndarray
        The initiator degree distribution.
    k : int
        The number of Kronecker iterations.

    Returns
    -------
    float
        The average clustering coefficient.

    Notes
    -----
    - This function may be slow for large `k` or larger initiator matrices `P`.
    - For larger values of `k`, an exponential fit is used to estimate the clustering coefficient.

    """
    m = len(P)
    if (m == 2 and k < 8) or (m > 2 and k < 3):
        return np.mean(compute_C_i(P, l, k))
    else:
        # Quick estimate using exponential fit
        if m == 2:
            k_values = range(1, 8)
        else:
            k_values = range(1, 3)
        mean_C_i_values = []
        for k_ in k_values:
            C_i_values = compute_C_i(P, l, k_)
            mean_C_i_values.append(np.mean(C_i_values))
        mean_C_i_values = np.array(mean_C_i_values)
        
        def exponential_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        params, _ = curve_fit(exponential_func, k_values, mean_C_i_values)

        return exponential_func(k, *params)

def measure_clustering(G):
    """
    Measures the average clustering coefficient of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.

    Returns
    -------
    float
        The average clustering coefficient.

    """
    return nx.average_clustering(G)

# -------------------
# Benson et al. 2014
# -------------------

"""
Calculations for motifs such as d-stars and t-cliques based on Benson et al. (2014).
"""

# d-Stars
"""
Computes the expected number of d-stars in a Kronecker graph.

The analytical expression is:

.. math::

    \\mathbb{E}[S_d^{(k)}] = n \\binom{n-1}{d}
    \\left(
    \\sum_{i_1, i_2, \\dots, i_{d+1} \\in [m]}
    \\prod_{j=2}^{d+1} p_{i_1, i_j}
    \\prod_{j=1}^{d+1} l_{i_j}
    \\right)^{k}

Note
----
- Can be calculated efficiently up to \( d = 15 \).
- Recommended to use \( d = 2, 3, 4, 5 \) for practical computations.
"""

def expected_number_of_d_stars(P, l, k, n, d):
    """
    Computes the expected number of d-stars in a Kronecker graph.

    Parameters
    ----------
    P : numpy.ndarray
        The initiator matrix.
    l : numpy.ndarray
        The initiator degree distribution.
    k : int
        The number of Kronecker iterations.
    n : int
        The number of nodes in the graph.
    d : int
        The degree of the star (number of edges connected to the central node).

    Returns
    -------
    float
        The expected number of d-stars.

    Notes
    -----
    - The function computes the expectation based on combinatorial calculations.
    - Efficient for small values of \( d \).

    References
    ----------
    Benson, A. R., Riquelme, C.,& Schmit, S.(2014).
    Learning Multifractal Structure in Large Networks.
    Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1326–35.

    """
    #k += 1  # Adjust for programming convention
    m = len(P)

    total_sum = 0
    
    # Iterate over all possible combinations of (i_1, ..., i_{d+1}) in [m]
    for indices in np.ndindex(*([m] * (d + 1))):
        # Compute the product of p_{i_1,i_j} for j from 2 to d+1
        product_p = np.prod([P[indices[0], indices[j]] for j in range(1, d + 1)])
        
        # Compute the product of l_{i_j} for j from 1 to d+1
        product_l = np.prod([l[indices[j]] for j in range(d + 1)])
        
        # Add to the total sum
        total_sum += product_p * product_l
    
    binomial_factor = comb(n - 1, d, exact=False)
    correction = 1  # May adjust based on specific requirements
    return n * binomial_factor * np.power(total_sum, k) * correction

def count_d_stars(G, d):
    """
    Counts the number of d-stars in the graph G.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    d : int
        The degree of the star (number of edges connected to the central node).

    Returns
    -------
    int
        The number of d-stars in the graph.

    """
    d_star_count = 0
    # Iterate over all nodes in the graph
    for node in G.nodes():
        degree = G.degree(node)
        if degree >= d:
            # Compute the combinations
            d_star_count += comb(degree, d, exact=False)
    return d_star_count

# t-Cliques
"""
Computes the expected number of t-cliques in a Kronecker graph.

The analytical expression is:

.. math::

    \\mathbb{E}[C_t^{(k)}] = \\binom{n}{t}
    \\left(
    \\sum_{i_1, i_2, \\dots, i_t \\in [m]}
    \\prod_{\\substack{j, q \\in [t] \\\\ j < q}} p_{i_j, i_q}
    \\prod_{j=1}^{t} l_{i_j}
    \\right)^{k}

Note
----
- Efficient computation, suitable for \( t \geq 2 \).
- Use values of \( t \) where actual cliques of that size exist in the network.
"""

def expected_number_of_t_cliques(P, l, k, n, t):
    """
    Computes the expected number of t-cliques in a Kronecker graph.

    Parameters
    ----------
    P : numpy.ndarray
        The initiator matrix.
    l : numpy.ndarray
        The initiator degree distribution.
    k : int
        The number of Kronecker iterations.
    n : int
        The number of nodes in the graph.
    t : int
        The size of the cliques to count.

    Returns
    -------
    float
        The expected number of t-cliques.

    Notes
    -----
    - Efficient for practical values of \( t \) where cliques exist.
    - The function uses combinatorial calculations over the initiator matrix.

    References
    ----------
    Benson, A. R., Gleich, D. F., & Leskovec, J. (2014).
    Higher-order organization of complex networks.
    Science, 353(6295), 163-166.

    """
    #k += 1  # Adjust for programming convention
    m = len(l)  # The number of nodes in the initiator matrix P
    
    s_t = 0  # Initialize s_t
    
    # Iterate over all combinations of i_1, ..., i_t in [m]
    for indices in np.ndindex(*([m] * t)):
        product_p = 1
        product_l = 1
        
        # Calculate the product of p_{i_j, i_q} for j < q
        for j in range(t):
            for q in range(j + 1, t):
                product_p *= P[indices[j], indices[q]]
        
        # Calculate the product of l_{i_j}
        product_l = np.prod([l[indices[j]] for j in range(t)])
        
        # Accumulate s_t
        s_t += product_p * product_l
    
    # Raise s_t to the power of k
    s_t = s_t ** k
    
    # Calculate the binomial coefficient
    binomial_coefficient = comb(n, t, exact=False)
    # Final expected value
    expected_cliques = binomial_coefficient * s_t
    
    return expected_cliques

def count_t_cliques(G, t):
    """
    Counts the number of unique t-cliques in the graph G.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    t : int
        The size of the cliques to count.

    Returns
    -------
    int
        The number of unique t-cliques in the graph.

    Notes
    -----
    - The function finds all maximal cliques and then enumerates all possible t-sized cliques within them.
    - It ensures uniqueness by storing cliques as sorted tuples in a set.

    """
    cliques = set()  # To store unique cliques

    # Find all maximal cliques in the graph
    maximal_cliques = list(nx.find_cliques(G))

    # Iterate over each maximal clique
    for clique in maximal_cliques:
        s = len(clique)
        if s >= t:
            # Generate all combinations of nodes of size t within the maximal clique
            for sub_clique in combinations(clique, t):
                # Add the sorted tuple of nodes to the set to ensure uniqueness
                cliques.add(tuple(sorted(sub_clique)))

    # The number of unique t-cliques is the size of the set
    return len(cliques)

# -------------------
# MOTIFS (to be implemented)
# -------------------
class MotifAnalyzer:
    def __init__(self, adj_matrices, names):
        """
        Initializes the MotifAnalyzer with a list of adjacency matrices and motif names.

        Parameters
        ----------0
        adj_matrices : list of numpy.ndarray
            List of adjacency matrices for the motifs.
        names : list of str
            List of descriptive names for the motifs.
        """
        if not all(m.shape == adj_matrices[0].shape for m in adj_matrices):
            raise ValueError("All motifs must have the same size.")

        self.motifs = [{'name': name, 'adj_mat': adj_mat, 'graph': nx.from_numpy_array(adj_mat),
                        'factor': self._compute_motif_factor(adj_mat)} 
                       for name, adj_mat in zip(names, adj_matrices)]

    def _nx_to_graphtool(self, G):
        G_gt = Graph(directed=False)
        node_map = {}
        for node in G.nodes():
            v = G_gt.add_vertex()
            node_map[node] = v
        for edge in G.edges():
            u, v = edge
            G_gt.add_edge(node_map[u], node_map[v])
        return G_gt

    def _graphtool_to_numpy(self, G):
        num_vertices = G.num_vertices()
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for edge in G.edges():
            source = int(edge.source())
            target = int(edge.target())
            adj_matrix[source, target] = 1
            adj_matrix[target, source] = 1
        return adj_matrix

    # TODO: FIX THIS
    def count_specific_motifs(self, G,
                              exact=False,
                              sampling_ratio=0.0001,
                              bootstrap_iters = 100,
                              ci = 0.95,
                              num_jobs=-1):
        motif_size = self.motifs[0]['adj_mat'].shape[0]
        
        if exact:
            G_gt = self._nx_to_graphtool(G)
            print(f"Counting motifs of size {motif_size}... grab a tea or watch a movie, this might take a while.")
            start_time = datetime.datetime.now()
            patterns, counts = motifs(G_gt, motif_size)
            end_time = datetime.datetime.now()
            print(f"Motif counting finished in {end_time - start_time}")
            
            result_counts = {motif['name']: {'count': 0} for motif in self.motifs}

            for pattern, count in zip(patterns, counts):
                # Convert graph-tool pattern to numpy array
                adj_mat = self._graphtool_to_numpy(pattern)
                # Convert to networkx for isomorphism checking
                adj_graph = nx.from_numpy_array(adj_mat)
                # Find which motif this pattern matches
                for motif in self.motifs:
                    if self._is_isomorphic(motif['graph'], adj_graph):
                        result_counts[motif['name']]['count'] = count
            return result_counts
        else:
            return self._parallel_stochastic_count(G, motif_size, sampling_ratio, bootstrap_iters, ci, num_jobs)

    def _parallel_stochastic_count(self, G, motif_size, sampling_ratio, bootstrap_iters, ci, num_jobs):
        total_nodes = len(G.nodes())
        total_combinations = comb(total_nodes, motif_size, exact=False)
        sample_size = int(sampling_ratio * total_combinations)
        
        if sample_size < 1:
            sample_size = 1

        if num_jobs == -1:
            num_jobs = cpu_count()

        batch_size = sample_size // num_jobs
        batches = [batch_size] * num_jobs

        print(f"Counting motifs of size {motif_size}...")
        start_time = datetime.datetime.now()
        with Parallel(n_jobs=num_jobs) as parallel:
            results = parallel(delayed(self._sample_and_count_motifs)(G, motif_size, batch) 
                               for batch in batches)
        end_time = datetime.datetime.now()
        print(f"Motif counting finished in {end_time - start_time}")
        
        # Concatenate results from all batches
        binary_vectors = {motif['name']: [] for motif in self.motifs}
        for batch_result in results:
            for motif_name, motif_vector in batch_result.items():
                binary_vectors[motif_name].extend(motif_vector)

        # Check if we need to skip bootstrapping
        if bootstrap_iters is None or bootstrap_iters == 1 or ci is None or ci == 0:
            print("Skipping bootstrapping.")
            # Just sum up the binary vectors without bootstrapping
            return {motif_name: {'count':((sum(binary_vector)/len(binary_vector))*total_combinations)} for motif_name, binary_vector in binary_vectors.items()}

        # Bootstrap and compute confidence intervals
        print("Bootstrapping the results...")
        start_time = datetime.datetime.now()
        ci_results = self._bootstrap_confidence_intervals(binary_vectors, total_combinations, bootstrap_iters, ci, num_jobs)
        end_time = datetime.datetime.now()
        print(f"Bootstrapping finished in {end_time - start_time}")
        
        return ci_results

    def _sample_and_count_motifs(self, G, motif_size, sample_size):
        nodes = list(G.nodes())
        binary_lists = {motif['name']: [] for motif in self.motifs}

        for _ in range(sample_size):
            sampled_nodes = random.sample(nodes, motif_size)
            sampled_subgraph = G.subgraph(sampled_nodes)
            if nx.is_connected(sampled_subgraph):
                for motif in self.motifs:
                    if self._is_isomorphic(motif['graph'], sampled_subgraph):
                        binary_lists[motif['name']].append(1)  # Found the motif
                    else:
                        binary_lists[motif['name']].append(0)  # Did not find the motif
            else:
                for motif in self.motifs:
                    binary_lists[motif['name']].append(0)  # Did not find the motif (graph is disconnected)

        return binary_lists

    def _bootstrap_confidence_intervals(self, binary_vectors, total_combinations, bootstrap_iters, ci, num_jobs):
        results = {}
        ci_low = (1 - ci) / 2 * 100  # Lower bound percentile
        ci_high = (ci + (1 - ci) / 2) * 100  # Upper bound percentile
        
        # Convert the dictionary of binary vectors into a 2D numpy array (matrix)
        motifs = list(binary_vectors.keys())
        binary_matrix = np.array(list(binary_vectors.values()))

        n = binary_matrix.shape[1]  # Number of columns, which are the length of the binary vectors
        
        def resample_and_sum_column(col_index):
            resampled = np.random.choice(np.arange(n), size=n, replace=True)
            return binary_matrix[:, resampled].sum(axis=1)  # Sum motifs across the resampled columns
        
        # Parallelize the resampling process
        bootstrap_counts = Parallel(n_jobs=num_jobs)(
            delayed(resample_and_sum_column)(i) for i in range(bootstrap_iters)
        )
        
        bootstrap_counts = np.array(bootstrap_counts)  # Convert to numpy array for easier manipulation
        bootstrap_counts = bootstrap_counts.T  # Transpose to get motifs in rows
        
        # Iterate over each motif and compute counts and confidence intervals
        for idx, motif_name in enumerate(motifs):
            motif_vector = binary_matrix[idx]
            
            count = (np.sum(motif_vector) / len(motif_vector)) * total_combinations
            lower_ci = (np.percentile(bootstrap_counts[idx], ci_low) / len(motif_vector)) * total_combinations
            upper_ci = (np.percentile(bootstrap_counts[idx], ci_high) / len(motif_vector)) * total_combinations
            
            results[motif_name] = {
                'count': count,
                'count_ci_low': lower_ci,
                'count_ci_high': upper_ci
            }

        return results

    def _is_isomorphic(self, motif_graph, subgraph):
        return nx.isomorphism.GraphMatcher(motif_graph, subgraph).is_isomorphic()

    def _compute_motif_factor(self, adj_mat):
        possible_labelings = factorial(adj_mat.shape[0], exact=False)
        automorphism_number = self._compute_automorphisms(adj_mat)
        return possible_labelings / automorphism_number

    def _compute_automorphisms(self, adj_mat):
        motif_graph = nx.from_numpy_array(adj_mat)
        matcher = nx.isomorphism.GraphMatcher(motif_graph, motif_graph)
        automorphisms = sum(1 for _ in matcher.isomorphisms_iter())
        return automorphisms

    def expected_number_of_motifs(self, P, l, k, n):
        '''Calculate the expexted number of motifs in a Multifractal Network.'''
        motif_size = self.motifs[0]['adj_mat'].shape[0]
        results = {}
        for motif in self.motifs:
            m = P.shape[0]
            all_indices = np.array(np.meshgrid(*[np.arange(m)] * motif_size, indexing='ij'))
            all_indices = all_indices.reshape(motif_size, -1)
            upper_tri_mask = np.triu(motif['adj_mat'], k=1)
            edge_indices = np.where(upper_tri_mask == 1)
            i_edges, j_edges = edge_indices
            P_values = P[all_indices[i_edges][:, np.newaxis], all_indices[j_edges][:, np.newaxis]]
            edge_product = np.prod(P_values, axis=0) 
            l_product = np.prod(l[all_indices], axis=0)
            S = np.sum(edge_product * l_product)
            combinatorial_factor = comb(n, motif_size, exact=False) * motif['factor']
            expected_motif_count = combinatorial_factor * (S ** k)
            results[motif['name']] = int(expected_motif_count) # A count of 0.4 is still 0 in most cases.
        return results