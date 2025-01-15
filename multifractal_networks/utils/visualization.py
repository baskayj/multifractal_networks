import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, LogNorm
from itertools import permutations
from math import floor, ceil

from multifractal_networks.utils.metrics import MRE, KL_divergence
from multifractal_networks.utils.misc import smooth_data, smooth_data_log_scale

sns.set_theme(
    context='paper',
    style='darkgrid',
    palette='deep',
    font='sans-serif',
    font_scale=1,
    color_codes=True,
    rc=None
)

def plot_metric(P_k, l_k,
                grid = False,
                norm = LogNorm,
                cmap = plt.colormaps.get_cmap('magma'),
                filename = None,
                streamlit = False
               ):
    """
    Plots the generating metric using a rectangular grid representation.

    Parameters
    ----------
    P_k : numpy.ndarray
        A 2D square array representing the matrix of probabilities or weights.
        It should be of shape (n, n), where `n` is the number of bins.
    l_k : array-like
        A 1D array of length `n` representing the sizes (lengths) of the bins along each axis.
        The values in `l_k` should sum to 1 to fit within the unit square.
    grid : bool, optional
        If True, displays grid lines between the rectangles.
        Default is False.
    norm : matplotlib.colors.Normalize or subclass, optional
        A normalization instance to scale the values in `P_k` for colormap mapping.
        Default is `matplotlib.colors.LogNorm`.
    cmap : matplotlib.colors.Colormap or str, optional
        The colormap instance or name used to map normalized data values to RGBA colors.
        Default is the 'magma' colormap.

    Returns
    -------
    None
        Displays the plot using `matplotlib.pyplot.show()`.

    Notes
    -----
    - This function visualizes the metric defined by `P_k` and `l_k` by drawing rectangles
      whose positions and sizes are determined by the cumulative sums of `l_k`. The color
      of each rectangle is determined by the corresponding value in `P_k` using the specified colormap.
    - The colormap can be customized by passing a different `cmap` parameter.
    - The plot is displayed within a unit square (0 to 1 on both axes).
    - If `grid` is set to True, grid lines are displayed between the rectangles.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import LogNorm
    >>> P_k = np.random.rand(5, 5)
    >>> l_k = np.ones(5) / 5  # Equal-sized bins
    >>> cmap = plt.get_cmap('viridis')
    >>> plot_metric(P_k, l_k, grid=True, norm=LogNorm, cmap=cmap)
    """
    # Plot the generating metric (P_k, l_k)
    fig, ax = plt.subplots()

    # Calculate the cumulative sum to get the positions of bin edges
    x_edges = np.cumsum([0] + list(l_k))
    y_edges = np.cumsum([0] + list(l_k))

    # Normalize the P_k values to avoid issues with zero values
    P_k = P_k.copy()  # Avoid modifying the original P_k
    min_nonzero = next((x for x in np.sort(P_k.flatten()) if x > 0), None)
    if min_nonzero is None:
        min_nonzero = 0
    else:
        min_nonzero /= 10
    P_k += min_nonzero

    # Set up the normalization and colormap
    nrm = norm(vmin=np.min(P_k), vmax=np.max(P_k))

    # Ensure cmap is a Colormap instance
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Loop over each cell in P_k and draw the rectangle
    for i in range(P_k.shape[0]):
        for j in range(P_k.shape[1]):
            if grid:
                edge_clr = None
            else:
                edge_clr = cmap(nrm(P_k[i, j]))
            rect = Rectangle(
                (x_edges[j], y_edges[i]),
                width=l_k[j],
                height=l_k[i],
                facecolor=cmap(nrm(P_k[i, j])),
                edgecolor=edge_clr
            )
            ax.add_patch(rect)

    # Add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=nrm)
    sm.set_array([])  # Only needed for matplotlib < 3.1
    cbar = fig.colorbar(sm, ax=ax)

    # Set the limits and aspect ratio
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Hide grid lines and axis numbers
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if filename:
        plt.savefig(filename, dpi = 1000)
    if streamlit:
        return fig
    else:
        plt.show()
        return None

def plot_curves(d, measured, analytical,
                xlabel = 'x',
                ylabel = 'y',
                title = None,
                log_scale = False,
                analytical_ci = None,
                ground_truth = None,
                display_mre = False,
                display_kld = False,
                debug = False,
                display_cumulative = False,
                filename = None,
                xlim = None,
                ylim = None,
                streamlit = False # TODO actually have fig object
               ):
    """
    Plots measured and analytical data curves for comparison, including optional confidence bands, MRE/KL error annotations, 
    and error curves (debug mode). Also adds stacked annotations when both MRE and KL are displayed.

    Parameters
    ----------
    d : array-like
        The domain values, typically representing degrees or another independent variable. Should be sorted in ascending order.
    measured : array-like
        The measured values corresponding to `d`.
    analytical : array-like
        The analytical or expected values corresponding to `d`.
    xlabel : str, optional
        Label for the x-axis. Default is 'x'.
    ylabel : str, optional
        Label for the y-axis. Default is 'y'.
    title : str, optional
        Title of the plot.
    log_scale : bool, optional
        If True, sets the x-axis to logarithmic scale. Default is False.
    analytical_ci: dict, optional
        Dictionary containing the confidence interval of the analytical curve with keys 'ci_low', 'ci_high', and 'confidence_level'.
    ground_truth : array-like, optional
        The ground truth values corresponding to `d`, used for additional MRE/KL error calculations and plotting. Default is None.
    display_mre: bool, optional
        Whether or not to display the mean relative error (MRE).
    display_kld: bool, optional
        Whether or not to display the KL-divergence.
    debug: bool, optional
        If True, MRE/KLD will return detailed error info and display `errors` or `cumulative_errors`.
    display_cumulative: bool, optional
        If True, displays cumulative errors instead of individual relative errors. Only relevant if `debug=True`.
    xlim : tuple, optional
        Limits for the x-axis as a tuple (xmin, xmax). Default is None.
    ylim : tuple, optional
        Limits for the y-axis as a tuple (ymin, ymax). Default is None.
    filename : str, optional
        If specified, saves the plot to the given filename.

    Returns
    -------
    None
        Displays the plot using `matplotlib.pyplot.show()`.
    """

    if display_mre and display_kld and debug:
        raise ValueError("Cannot display both MRE and KLD in debug mode. Choose one.")

    # Create figure and axes objects
    fig, ax1 = plt.subplots()
    
    # Plot the main curves
    ax1.plot(d, measured, label='Measured', color='#006BA4', alpha=0.7, marker='.', linestyle='')
    if ground_truth is not None:
        ax1.plot(d, ground_truth, label='Ground Truth', color='#006BA4')
    ax1.plot(d, analytical, label='Analytical', color='#FF800E')

    # Plot confidence intervals if provided
    if analytical_ci is not None:
        ci_low = analytical_ci.get('ci_low', None)
        ci_high = analytical_ci.get('ci_high', None)
        confidence_level = analytical_ci.get('confidence_level', None)
        if ci_low is not None and ci_high is not None:
            ax1.fill_between(d, ci_low, ci_high, color='#FF800E', alpha=0.3)  # CI band

    # Logarithmic scaling if requested
    if log_scale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    # Set axis limits if specified
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(facecolor='white', loc='upper left')

    # Compute errors or cumulative errors and plot in debug mode
    if debug and (display_mre or display_kld):
        ax2 = ax1.twinx()
        ax2.set_ylabel('Error')  # Label for twin y-axis

        if display_mre:
            mre_value, mre_debug = MRE(measured, analytical, debug=True)
            if display_cumulative:
                ax2.plot(d, mre_debug['cumulative_errors'], color='#ABABAB', linestyle='--', label='Cumulative MRE Errors')
            else:
                ax2.plot(d, mre_debug['errors'], color='#ABABAB', marker='.', linestyle='', label='MRE Errors')
        elif display_kld:
            kld_value, kld_debug = KL_divergence(measured, analytical, debug=True)
            if display_cumulative:
                ax2.plot(d, kld_debug['cumulative_errors'], color='#ABABAB', linestyle='--', label='Cumulative KLD Errors')
            else:
                ax2.plot(d, kld_debug['errors'], color='#ABABAB', marker='.', linestyle='', label='KLD Errors')

        ax2.legend(loc='upper right')

        # Add annotation for debug mode
        if display_mre:
            annotation_text = f'MRE: {mre_value:.3f}'
        elif display_kld:
            annotation_text = f'KL Divergence: {kld_value:.3f}'

        ax1.annotate(annotation_text, xy=(0.02, 0.15), xycoords='axes fraction',
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))

    # Handle non-debug annotations, including CI handling for MRE
    # Handle confidence level interpretation
    confidence_str = None
    if analytical_ci is not None and 'confidence_level' in analytical_ci:
        confidence_level = analytical_ci['confidence_level']
        if 0 < confidence_level < 1:
            confidence_str = f"{confidence_level * 100:.1f}% CI"
        elif 1 <= confidence_level <= 100:
            confidence_str = f"{confidence_level:.1f}% CI"
    
    if not debug:
        # Determine vertical positioning based on which metrics are displayed
        base_y_position = 0.07  # Lower base position
        annotation_gap = 0.08   # Gap between annotations when both present
        # Calculate y positions based on what's being displayed
        if display_mre and display_kld:
            mre_y_position = base_y_position + annotation_gap
            kld_y_position = base_y_position
        else:
            # If only one is displayed, use the base position
            single_y_position = base_y_position
        if display_mre:
            mre_value = MRE(measured, analytical)
            # Check if CI exists, calculate MRE for CI
            if analytical_ci is not None and 'ci_low' in analytical_ci and 'ci_high' in analytical_ci:
                mre_ci_low = MRE(measured, analytical_ci['ci_low'])
                mre_ci_high = MRE(measured, analytical_ci['ci_high'])
                if mre_ci_low > mre_ci_high:
                    mre_ci_low, mre_ci_high = mre_ci_high, mre_ci_low
                if confidence_str is not None:
                    mre_str = f'MRE: {mre_value:.3f} ({confidence_str}: {mre_ci_low:.3f} - {mre_ci_high:.3f})'
                else:
                    mre_str = f'MRE: {mre_value:.3f} (CI: {mre_ci_low:.3f} - {mre_ci_high:.3f})'
            else:
                mre_str = f'MRE: {mre_value:.3f}'
            # If ground_truth is present add it!
            if ground_truth is not None:
                mre_gt = MRE(ground_truth, analytical)
                if analytical_ci is not None and 'ci_low' in analytical_ci and 'ci_high' in analytical_ci:
                    mre_gt_ci_low = MRE(ground_truth, analytical_ci['ci_low'])
                    mre_gt_ci_high = MRE(ground_truth, analytical_ci['ci_high'])
                    if mre_gt_ci_low > mre_gt_ci_high:
                        mre_gt_ci_low, mre_gt_ci_high = mre_gt_ci_high, mre_gt_ci_low
                    if confidence_str is not None:
                        mre_str += f'\nMRE (vs GT): {mre_gt:.3f} ({confidence_str}: {mre_gt_ci_low:.3f} - {mre_gt_ci_high:.3f})'
                    else:
                        mre_str += f'\nMRE (vs GT): {mre_gt:.3f} (CI: {mre_gt_ci_low:.3f} - {mre_gt_ci_high:.3f})'
                else:
                    mre_str += f'\nMRE (vs GT): {mre_gt:.3f}'

            ax1.annotate(mre_str, xy=(0.02, mre_y_position if display_kld else single_y_position), xycoords='axes fraction',
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))
        if display_kld:
            kld_value = KL_divergence(measured, analytical)
            # Check if CI exists, calculate KLD for CI
            if analytical_ci is not None and 'ci_low' in analytical_ci and 'ci_high' in analytical_ci:
                kld_ci_low = KL_divergence(measured, analytical_ci['ci_low'])
                kld_ci_high = KL_divergence(measured, analytical_ci['ci_high'])
                if kld_ci_low > kld_ci_high:
                    kld_ci_low, kld_ci_high = kld_ci_high, kld_ci_low
                # Handle confidence level interpretation
                if confidence_str is not None:
                    kld_str = f'KL Divergence: {kld_value:.3f} ({confidence_str}: {kld_ci_low:.3f} - {kld_ci_high:.3f})'
                else:
                    kld_str = f'KL Divergence: {kld_value:.3f} (CI: {kld_ci_low:.3f} - {kld_ci_high:.3f})'
            else:
                kld_str = f'KL Divergence: {kld_value:.3f}'
            # If ground_truth is present, calculate KLD for it
            if ground_truth is not None:
                kld_gt = KL_divergence(ground_truth, analytical)
                # If CI exists, calculate KLD for ground_truth vs CI
                if analytical_ci is not None and 'ci_low' in analytical_ci and 'ci_high' in analytical_ci:
                    kld_gt_ci_low = KL_divergence(ground_truth, analytical_ci['ci_low'])
                    kld_gt_ci_high = KL_divergence(ground_truth, analytical_ci['ci_high'])
                    if kld_gt_ci_low > kld_gt_ci_high:
                        kld_gt_ci_low, kld_gt_ci_high = kld_gt_ci_high, kld_gt_ci_low
                    if confidence_str is not None:
                        kld_str += f'\nKL Divergence (vs GT): {kld_gt:.3f} ({confidence_str}: {kld_gt_ci_low:.3f} - {kld_gt_ci_high:.3f})'
                    else:
                        kld_str += f'\nKL Divergence (vs GT): {kld_gt:.3f} (CI: {kld_gt_ci_low:.3f} - {kld_gt_ci_high:.3f})'
                else:
                    kld_str += f'\nKL Divergence (vs GT): {kld_gt:.3f}'
                    
            ax1.annotate(kld_str, xy=(0.02, kld_y_position if display_mre else single_y_position), xycoords='axes fraction',
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))
    if title:
        ax1.set_title(title)
    ax1.grid(True)
    
    if filename:
        fig.savefig(filename, dpi=1000)

    if streamlit:
        return fig
    else:
        plt.show()
        plt.close(fig)
        return None

def plot_motif_spectra(all_motifs,
                       plot_cumulative_error = True,
                       filename = None,
                       streamlit = False
                      ):
    """
    Plots the mean relative error (MRE) of different motifs along with optional confidence intervals (CI) 
    and displays motif graphs underneath the corresponding data points on the x-axis.
    
    Parameters:
        all_motifs (list of dicts): A list where each entry is a dictionary containing data for a motif.
                                    Required fields:
                                    - 'name': The name of the motif (for x-axis labels).
                                    - 'MRE': Mean Absolute Error value.
                                    - 'matrix': The adjacency matrix of the motif (for plotting the motif graph).
                                    Optional fields:
                                    - 'MRE_ci_low': Lower bound of the confidence interval for the MRE.
                                    - 'MRE_ci_high': Upper bound of the confidence interval for the MRE.
    
    Functionality:
        - MRE values are plotted on the y-axis with the corresponding motif names on the x-axis.
        - If available, confidence intervals for the MRE values are displayed using error bars.
        - Motif graphs are drawn as small inset plots below each MRE data point using circular layouts.
        - A method is used to minimize edge crossings in motif graphs before drawing.
    """
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjusted figure size for better visibility

    # Prepare MRE values and optional error bars
    mres = []
    ci_low = []
    ci_high = []
    has_ci = False

    # Prepare optional comparisons with GT
    gt = False
    # GT vs. measured
    mres_gt = []
    gt_ci_low = []
    gt_ci_high = []
    gt_has_ci = False
    # GT vs. fitted
    mres_fit_gt = []
    fit_gt_ci_low = []
    fit_gt_ci_high = []
    fit_gt_has_ci = False

    # X-axis labels and positions
    motif_names = []

    # Iterate through motifs to collect data and drawings
    for idx, motif in enumerate(all_motifs):
        motif_name = motif['name']
        mres.append(motif['MRE'])
        motif_names.append(motif_name)
        if 'MRE_ci_low' in motif and 'MRE_ci_high' in motif:
            l = motif['MRE'] - motif['MRE_ci_low']
            l = l if l >= 0 else abs(l) # Negative values for yerr are forbidden
            h = motif['MRE_ci_high'] - motif['MRE']
            h = h if h >= 0 else abs(h) # Negative values for yerr are forbidden
            ci_low.append(l)
            ci_high.append(h)
            has_ci = True
        else:
            ci_low.append(0)
            ci_high.append(0)
        # Check for the presence of GT MRE and FIT GT MRE
        if 'ground_truth_MRE' in motif:
            mres_gt.append(motif['ground_truth_MRE'])
            gt = True
            if 'ground_truth_MRE_ci_low' in motif and 'ground_truth_MRE_ci_high' in motif:
                l = motif['ground_truth_MRE'] - motif['ground_truth_MRE_ci_low']
                l = l if l >= 0 else abs(l) # Negative values for yerr are forbidden 
                h = motif['ground_truth_MRE_ci_high'] - motif['ground_truth_MRE']
                h = h if h >= 0 else abs(h) # Negative values for yerr are forbidden
                gt_ci_low.append(l)
                gt_ci_high.append(h)
                gt_has_ci = True
            else:
                gt_ci_low.append(0)
                gt_ci_high.append(0)
        else:
            mres_gt.append(0)
        if 'fit_ground_truth_MRE' in motif:
            mres_fit_gt.append(motif['fit_ground_truth_MRE'])
            if 'fit_ground_truth_MRE_ci_low' in motif and 'fit_ground_truth_MRE_ci_high' in motif:
                l = motif['fit_ground_truth_MRE'] - motif['fit_ground_truth_MRE_ci_low']
                l = l if l >= 0 else abs(l) # Negative values for yerr are forbidden 
                h = motif['fit_ground_truth_MRE_ci_high'] - motif['fit_ground_truth_MRE']
                h = h if h >= 0 else abs(h) # Negative values for yerr are forbidden
                fit_gt_ci_low.append(l)
                fit_gt_ci_high.append(h)
                fit_gt_has_ci = True
            else:
                fit_gt_ci_low.append(0)
                fit_gt_ci_high.append(0)
        else:
            mres_fit_gt.append(0)
        

    # Plot the MRE values with or without error bars
    x = np.arange(len(all_motifs))
    mres = np.array(mres)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    mres_gt = np.array(mres_gt)
    gt_ci_low = np.array(gt_ci_low)
    gt_ci_high = np.array(gt_ci_high)

    mres_fit_gt = np.array(mres_fit_gt)
    fit_gt_ci_low = np.array(fit_gt_ci_low)
    fit_gt_ci_high = np.array(fit_gt_ci_high)

    if gt:
        if gt_has_ci:
            ax.errorbar(x, mres_gt, yerr=[gt_ci_low, gt_ci_high], fmt='s', capsize=5, label = 'Measured vs. Ground Truth')
        else:
            ax.plot(x, mres_gt, 's', label = 'Measured vs. Ground Truth')
        if fit_gt_has_ci:
            ax.errorbar(x, mres_fit_gt, yerr=[fit_gt_ci_low, fit_gt_ci_high], fmt='^', capsize=5, label = 'Ground Truth vs. Fitted')
        else:
            ax.plot(x, mres_fit_gt, '^', label = 'Ground Truth vs. Fitted')

    if has_ci:
        ax.errorbar(x, mres, yerr=[ci_low, ci_high], fmt='o', capsize=5, label = 'Measured vs. Fitted')
    else:
        ax.plot(x, mres, 'o', label = 'Measured vs. Fitted')

    # (Optional) Plot the cumulative error
    if plot_cumulative_error:
        ax2 = ax.twinx()
        ax2.set_ylabel('Cumulative Relative Error')  # Label for twin y-axis
        ax2.plot(x, np.cumsum(mres), color='#ABABAB', linestyle='--')
        if has_ci:
            ax2.fill_between(x, (np.cumsum(mres)- np.cumsum(ci_low)), (np.cumsum(mres) + np.cumsum(ci_high)), color='#ABABAB', alpha=0.3)

    # Set x-axis to match motif positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels(motif_names, rotation=0, fontsize=10, ha='center')  # Horizontal labels

    # Adjust the tick length and handle staggered positioning of labels
    ax.tick_params(axis='x', length=10)  # Increase the tick length for better readability

    for label in ax.get_xticklabels():
        idx = ax.get_xticklabels().index(label)
        if idx % 2 == 1:  # Every second label moves slightly lower
            label.set_y(-0.08)

    ax.set_ylabel('← Mean Relative Error (MRE)')

    # Adjust the main plot to provide room for the motif plots underneath
    plt.subplots_adjust(bottom=0.55)

    # Padding parameters for the insets
    inset_size = 0.06  # Size of each motif inset (both width and height, in normalized figure units)

    # Get the x-tick positions in data coordinates
    x_ticks = ax.get_xticks()

    # Draw each motif graph underneath the corresponding x-tick
    for idx, motif in enumerate(all_motifs):
        G = nx.from_numpy_array(motif['matrix'])
        
        # Reduce crossings if possible
        def exact_minimize_crossings(G):
            nodes = list(G.nodes)
            _permutations = list(permutations(nodes))
            adj_matrix = nx.to_numpy_array(G)

            def count_crossings(G):
                crossing_count = 0
                edges = list(G.edges)
                nodes = list(G.nodes)
                positions = {node: i for i, node in enumerate(nodes)}

                for i, (u1, v1) in enumerate(edges):
                    for u2, v2 in edges[i + 1:]:
                        if len({u1, v1, u2, v2}) < 4:
                            continue
                        pos_u1, pos_v1 = positions[u1], positions[v1]
                        pos_u2, pos_v2 = positions[u2], positions[v2]
                        if (pos_u1 < pos_u2 < pos_v1 < pos_v2) or (pos_u2 < pos_u1 < pos_v2 < pos_v1):
                            crossing_count += 1
                return crossing_count

            min_crossings = float('inf')
            best_G = None
            for order in _permutations:
                permuted_adj_matrix = adj_matrix[np.ix_(order, order)]
                permuted_G = nx.from_numpy_array(permuted_adj_matrix)
                crossings = count_crossings(permuted_G)
                if crossings < min_crossings:
                    min_crossings = crossings
                    best_G = permuted_G

            return best_G, min_crossings
        
        G, _ = exact_minimize_crossings(G)
        pos = nx.circular_layout(G)

        # Get the x-tick position for the current motif and transform it to figure coordinates
        tick_x_pos = x_ticks[idx]
        tick_x_pos_display = ax.transData.transform((tick_x_pos, 0))  # Get x-coordinate in display units
        tick_x_pos_normalized = (tick_x_pos_display[0] - fig.bbox.xmin) / fig.bbox.width  # Normalize to figure

        # Set the y position for the inset below the x-axis labels
        ax_pos = ax.get_position()
        inset_y = ax_pos.y0 - 0.15  # Position closer to the x-ticks

        # Add the inset axis with equal height and width for aspect ratio consistency
        ax_inset = fig.add_axes([tick_x_pos_normalized - inset_size / 2, inset_y, inset_size, inset_size], 
                                transform=fig.transFigure)

        # Draw motif graph within the inset axis
        nx.draw(G, pos, ax=ax_inset, node_color='black', edge_color='gray', node_size=20)
        ax_inset.axis('off')

    ax.legend(loc='upper left', facecolor='white', edgecolor='gray')

    if filename:
        plt.savefig(filename, dpi = 1000)
    if streamlit:
        return fig
    else:
        plt.show()
        return None