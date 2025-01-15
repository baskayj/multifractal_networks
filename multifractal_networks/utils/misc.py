import numpy as np
import networkx as nx

def sort_state(P, l, ascending=False):
    """
    Sorts the state variables `P` and `l` based on `l` in ascending or descending order.

    Parameters
    ----------
    P : numpy.ndarray
        A 2D square array (matrix) of shape (n, n), representing probabilities,
        weights, or interactions between states. The matrix should correspond to `l`
        such that `P[i, j]` relates to the interaction between states `i` and `j`.
    l : array-like
        A 1D array of length `n`, representing the sizes, weights, or probabilities
        associated with each state.
    ascending : bool, optional
        If True, sorts in ascending order. If False, sorts in descending order.
        Default is False.

    Returns
    -------
    P_sorted : numpy.ndarray
        The matrix `P` with both its rows and columns permuted to match the sorted
        order of `l`. The resulting matrix maintains the alignment between `P` and `l`.
    l_sorted : numpy.ndarray
        The array `l` sorted according to the specified order.

    Notes
    -----
    - The function sorts the array `l` in ascending or descending order based on the
      `ascending` parameter.
    - It then applies the same permutation to both the rows and columns of `P`, ensuring
      that the relationships in `P` correspond to the sorted `l`.
    - This is useful when you need to reorder states based on their importance, size, or
      any other metric represented by `l`.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0.1, 0.2, 0.3],
    ...               [0.2, 0.5, 0.4],
    ...               [0.3, 0.4, 0.6]])
    >>> l = np.array([0.3, 0.1, 0.6])
    >>> # Sort in ascending order
    >>> P_sorted_asc, l_sorted_asc = sort_state(P, l, ascending=True)
    >>> P_sorted_asc
    array([[0.5, 0.2, 0.4],
           [0.2, 0.1, 0.3],
           [0.4, 0.3, 0.6]])
    >>> l_sorted_asc
    array([0.1, 0.3, 0.6])
    >>> # Sort in descending order
    >>> P_sorted_desc, l_sorted_desc = sort_state(P, l, ascending=False)
    >>> P_sorted_desc
    array([[0.6, 0.3, 0.4],
           [0.3, 0.1, 0.2],
           [0.4, 0.2, 0.5]])
    >>> l_sorted_desc
    array([0.6, 0.3, 0.1])
    """

    # Sort l and get the sorted indices
    sorted_indices = np.argsort(l)
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    l_sorted = l[sorted_indices]

    # Apply the same permutation to P (both rows and columns)
    P_sorted = P[sorted_indices, :][:, sorted_indices]

    return P_sorted, l_sorted

def kron_pow(mat, k):
    """
    Computes the Kronecker power of a matrix.

    Parameters:
    ----------
    mat : np.ndarray
        The input square matrix to be raised to the Kronecker power.
    k : int
        The exponent specifying the number of times to apply the Kronecker product.
        Must be a non-negative integer. When k=0, returns a scalar 1.

    Returns:
    -------
    np.ndarray or int
        The resulting matrix after applying the Kronecker power.
        If k=0, returns 1.

    Examples:
    --------
    >>> mat = np.array([[1, 2], [3, 4]])
    >>> kron_pow(mat, 2)
    array([[ 1,  2,  2,  4],
           [ 3,  4,  6,  8],
           [ 3,  6,  4,  8],
           [ 9, 12, 12, 16]])
    """
    if k == 0:
        return 1
    current = mat
    for _ in range(1, k):
        current = np.kron(current, mat)
    return current

def find_max_d(G):
    """
    Finds the maximum degree among all nodes in a graph.

    Parameters:
    ----------
    G : networkx.Graph
        The input graph.

    Returns:
    -------
    int
        The maximum degree found among all nodes in the graph.

    Examples:
    --------
    >>> G = nx.star_graph(5)
    >>> find_max_d(G)
    5
    """
    degree_sequence = [d for _, d in G.degree()]
    return max(degree_sequence)

def keep_largest_component(G):
    """
    Extracts the largest connected component from a graph.

    Parameters:
    ----------
    G : networkx.Graph
        The input graph, which may be disconnected.

    Returns:
    -------
    networkx.Graph
        A subgraph containing only the nodes in the largest connected component.

    Examples:
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    >>> largest_subgraph = keep_largest_component(G)
    >>> list(largest_subgraph.nodes)
    [1, 2, 3]
    """
    # Find all connected components
    components = list(nx.connected_components(G))
    
    # Find the largest component
    largest_component = max(components, key=len)
    
    # Create a subgraph containing only the nodes in the largest component
    largest_subgraph = G.subgraph(largest_component).copy()
    
    return largest_subgraph

def smooth_data(data, window_size, mode = 'same'):
    """
    Smooths a one-dimensional data array using a moving average.

    Parameters:
    ----------
    data : array-like
        The input data to be smoothed.
    window_size : int
        The size of the moving window. Must be a positive integer.
        If zero or negative, no smoothing is applied.

    Returns:
    -------
    np.ndarray
        The smoothed data array.

    Examples:
    --------
    >>> data = [1, 2, 3, 4, 5]
    >>> smooth_data(data, window_size=3)
    array([1.        , 2.        , 3.        , 4.        , 3.33333333])
    """
    data = np.asarray(data)
    if window_size > 0:
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode = mode)
    else:
        smoothed_data = data
    return smoothed_data

def smooth_data_log_scale(data, window_size):
    """
    Smooths data using a moving average with a window size that scales logarithmically.

    Parameters:
    ----------
    data : array-like
        The input data to be smoothed.
    window_size : int
        The base window size for smoothing.

    Returns:
    -------
    np.ndarray
        The smoothed data array with logarithmically scaled window sizes.

    Examples:
    --------
    >>> data = np.arange(1, 11)
    >>> smooth_data_log_scale(data, window_size=3)
    array([1. , 1.5, 2. , 2.5, 3. , 5. , 7. , 8. , 9. , 9.5])
    """
    data = np.asarray(data)
    n = len(data)
    log_indices = np.logspace(0, np.log10(n - 1), n).astype(int)
    # Ensure indices are within bounds
    log_indices = np.clip(log_indices, 0, n - 1)

    smoothed_data = np.zeros_like(data, dtype=float)
    
    for idx in range(n):
        # Calculate window size for the current index
        log_window_size = int(window_size * (log_indices[idx] / n))
        log_window_size = max(1, log_window_size)  # Ensure window size is at least 1
        start = max(0, idx - log_window_size // 2)
        end = min(n, idx + log_window_size // 2 + 1)
        
        smoothed_data[idx] = np.mean(data[start:end])
    
    return smoothed_data

def log_transform(func_values, d_range, grid_size=15):
    """
    Transforms function values into logarithmically spaced bins.

    Parameters:
    ----------
    func_values : array-like
        The function values to be binned.
    d_range : tuple of int or float
        The range (min, max) of the domain over which `func_values` are defined.
    grid_size : int, optional
        The number of bins to use in the logarithmic scale. Defaults to 15.

    Returns:
    -------
    tuple of np.ndarray
        - binned_values: The binned function values corresponding to the logarithmic bins.
        - bin_centers: The centers of the logarithmic bins.

    Examples:
    --------
    >>> func_values = np.linspace(1, 100, 100)
    >>> binned_values, bin_centers = log_transform(func_values, (1, 100), grid_size=10)
    """
    func_values = np.asarray(func_values)
    # Define the logarithmic bins
    log_bins = np.logspace(np.log10(d_range[0]), np.log10(d_range[1]), grid_size + 1)
    
    # Calculate the bin indices for each function value
    domain = np.linspace(d_range[0], d_range[1], len(func_values))
    bin_indices = np.digitize(domain, log_bins) - 1
    
    # Initialize the binned values array
    binned_values = np.zeros(grid_size)
    
    # Aggregate function values into bins
    for i in range(grid_size):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            binned_values[i] = np.mean(func_values[bin_mask])
        else:
            binned_values[i] = np.nan  # Assign NaN if no values fall into the bin
    
    # Calculate bin centers
    bin_centers = (log_bins[:-1] + log_bins[1:]) / 2

    return binned_values, bin_centers

"""def correct_correlations(data, noise_factor=0.5):

    Corrects degree correlation data by filling zeros with a combination of:
    - Constant extrapolation for zeros at the beginning and end
    - Linear interpolation for intermediate zeros
    - Adds noise based on the variance in the non-zero data points.
    
    Parameters:
    - data: np.array of degree correlation data with zeros representing missing values.
    - noise_factor: multiplier to control the noise level in the added noise to interpolated/extrapolated values.
    
    Returns:
    - corrected_data: np.array with the corrected data.


    # Copy of the original data
    corrected_data = np.copy(data)
    
    # Identify the non-zero data points for noise calculation
    non_zero_data = corrected_data[corrected_data > 0]
    noise_std = np.std(non_zero_data)  # Calculate the standard deviation (noise) of non-zero data points

    # Extrapolate on the left side (constant extrapolation if starting with zeros)
    if corrected_data[0] == 0:
        first_non_zero = np.argmax(corrected_data > 0)  # Find the first non-zero index
        corrected_data[:first_non_zero] = corrected_data[first_non_zero]  # Extrapolate using the first non-zero value
    
    # Extrapolate on the right side (constant extrapolation if ending with zeros)
    if corrected_data[-1] == 0:
        last_non_zero = len(corrected_data) - np.argmax(corrected_data[::-1] > 0) - 1  # Find the last non-zero index
        corrected_data[last_non_zero+1:] = corrected_data[last_non_zero]  # Extrapolate using the last non-zero value
    
    # Linear interpolation for intermediate zeros
    zero_indices = np.where(corrected_data == 0)[0]
    for idx in zero_indices:
        # Find the nearest non-zero neighbors
        left_idx = np.max(np.where(corrected_data[:idx] > 0)[0]) if np.any(corrected_data[:idx] > 0) else None
        right_idx = np.min(np.where(corrected_data[idx:] > 0)[0]) + idx if np.any(corrected_data[idx:] > 0) else None
        
        if left_idx is not None and right_idx is not None:
            # Linear interpolation
            corrected_data[idx] = corrected_data[left_idx] + (corrected_data[right_idx] - corrected_data[left_idx]) * ((idx - left_idx) / (right_idx - left_idx))
    
    # Add noise to the interpolated and extrapolated values
    interpolated_indices = np.where((data == 0) & (corrected_data > 0))[0]
    noise = np.random.normal(loc=0, scale=noise_std * noise_factor, size=len(interpolated_indices))
    corrected_data[interpolated_indices] += noise

    return corrected_data"""

def correct_correlations(x, data, noise_factor=0.1, return_without_slicing = False):
    # Copy of the original data
    corrected_data = np.copy(data)
    
    # Identify the non-zero data points for noise calculation
    non_zero_data = corrected_data[corrected_data > 0]
    noise_std = np.std(non_zero_data)  # Calculate the standard deviation (noise) of non-zero data points

    # Linear interpolation for intermediate zeros
    zero_indices = np.where(corrected_data == 0)[0]
    for idx in zero_indices:
        # Find the nearest non-zero neighbors
        left_idx = np.max(np.where(corrected_data[:idx] > 0)[0]) if np.any(corrected_data[:idx] > 0) else None
        right_idx = np.min(np.where(corrected_data[idx:] > 0)[0]) + idx if np.any(corrected_data[idx:] > 0) else None
        
        if left_idx is not None and right_idx is not None:
            # Linear interpolation
            corrected_data[idx] = corrected_data[left_idx] + (corrected_data[right_idx] - corrected_data[left_idx]) * ((idx - left_idx) / (right_idx - left_idx))
    
    # Add noise to the interpolated and extrapolated values
    interpolated_indices = np.where((data == 0) & (corrected_data > 0))[0]
    noise = np.random.normal(loc=0, scale=noise_std * noise_factor, size=len(interpolated_indices))
    corrected_data[interpolated_indices] += noise

    # Find the range within where the data is non-zero
    first_non_zero = np.argmax(corrected_data > 0)  # Find the first non-zero index
    last_non_zero = len(corrected_data) - np.argmax(corrected_data[::-1] > 0) - 1  # Find the last non-zero index

    if return_without_slicing:
        return corrected_data
    else:
        return (corrected_data[first_non_zero:last_non_zero+1],x[first_non_zero:last_non_zero+1])

def find_new_k(N, P, l, k, avg_degree):
    """
    Finds the k for a network of size N that produces the given average degree.
    """

    def calculate_avg_degree(N, P, l, k):
        """
        Calculates the average degree for a given N, P, l, and k.
        """
        P_k = kron_pow(P, k)
        l_k = kron_pow(l, k)
        edge_proba = np.sum(np.outer(l_k, l_k) * P_k)
        return N * edge_proba
    
    new_k = k + 1  # Start searching from k+1
    while True:
        avg_degree_N = calculate_avg_degree(N, P, l, new_k)
        if avg_degree_N >= avg_degree:
            break
        new_k += 1
    return new_k

def recommend_N_for_k_range(P, l, k, avg_degree, k_limit):
    """
    Returns the recommended N for each k in the range [k+1, k_limit].
    """

    def calculate_N(P, l, k, avg_degree):
        """
        Calculates the network size N for a given average degree.
        """
        P_k = kron_pow(P, k)
        l_k = kron_pow(l, k)
        edge_proba = np.sum(np.outer(l_k, l_k) * P_k)
        return int(avg_degree / edge_proba)
    
    recommendations = {}
    for new_k in range(k + 1, k_limit + 1):
        recommended_N = calculate_N(P, l, new_k, avg_degree)
        recommendations[new_k] = recommended_N
    return recommendations