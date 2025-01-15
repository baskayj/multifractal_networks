#####################################################################################
'''


import streamlit as st
import networkx as nx
import numpy as np
import os
import json
from utils import load_graph  # Placeholder for your custom graph loader
from your_module import MultifractalNetworkFitter  # Replace with actual module path

# App Title
st.title("Multifractal Network Generator")

# Sidebar Configuration
st.sidebar.header("Configuration")
upload_file = st.sidebar.file_uploader("Upload Graph File (GraphML or TXT)", type=["graphml", "txt"])
motif_json = st.sidebar.file_uploader("Upload Cached Motifs (Optional)", type=["json"])
export_json = st.sidebar.button("Export Fitted Parameters to JSON")

# Parameters
st.sidebar.subheader("Fitting Parameters")
population_size = st.sidebar.slider("Population Size", min_value=10, max_value=500, value=200)
log_scale = st.sidebar.checkbox("Log Scale", value=True)
grid_search = st.sidebar.checkbox("Enable Grid Search", value=True)

# Weights
st.sidebar.subheader("Weights")
weight_rho = st.sidebar.slider("Weight (ρ)", min_value=0, max_value=10, value=9)
weight_deg_nn = st.sidebar.slider("Weight (deg_nn)", min_value=0, max_value=10, value=9)
weight_motifs = st.sidebar.slider("Weight (Motifs)", min_value=0, max_value=10, value=1)

# Options
st.sidebar.subheader("Optional Configurations")
optimize_for = st.sidebar.selectbox("Optimize For", options=["kl_divergence", "energy", "kl_energy"], index=2)
ci_value = st.sidebar.slider("Confidence Interval (%)", min_value=80, max_value=99, value=95)
num_variants = st.sidebar.slider("Number of Variants", min_value=1, max_value=10, value=3)

# Load Graph
if upload_file:
    st.header("Uploaded Graph")
    try:
        # Placeholder for loading graph (ensure it outputs a networkx Graph)
        graph = load_graph(upload_file)
        st.write(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        nx.draw(graph, with_labels=False, node_size=10, alpha=0.7)
        st.pyplot()
    except Exception as e:
        st.error(f"Error loading graph: {e}")

# Instantiate Network Fitter
if upload_file and st.button("Initialize Fitter"):
    motifs = [
        {'name': 'four_chain', 'matrix': np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])},
        {'name': 'three_loop_out', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])},
        {'name': 'four_loop', 'matrix': np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])},
        {'name': 'four_semi_clique', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])},
    ]
    mfnf = MultifractalNetworkFitter(
        graph,
        motifs=motifs,
        sampling_ratios={3: 0.01, 4: 0.0001},
        population_size=population_size,
        weights={'rho': weight_rho, 'deg_nn': weight_deg_nn, 'motifs': weight_motifs},
        motif_analyzer_args={'exact': True, 'bootstrap_iters': None, 'ci': None},
    )
    st.success("Fitter Initialized")

# Fit Parameters
if st.button("Run Grid Search"):
    if mfnf:
        mfnf.find_m_k_P_l(
            m_k_dict=[
                {'m': 2, 'k': [1, 10]},
                {'m': 3, 'k': [1, 7]},
                {'m': 5, 'k': [1, 5]},
                {'m': 7, 'k': [1, 4]},
                {'m': 11, 'k': [1, 3]},
            ],
            n_trials=None,
            grid_search=grid_search,
            optimize_for=optimize_for,
        )
        st.success("Fitting Completed")

# Plot Results
if st.button("Plot Results"):
    if mfnf:
        mfnf.plot_P_l_result(log_scale=log_scale)
        st.pyplot()

# Bootstrapping Analysis
if st.button("Run Bootstrapping Analysis"):
    if mfnf:
        ci = ci_value / 100
        mfnf.bootstrap_P_l_stability(
            P_init='m x m',
            l_init='dirichlet',
            verbose=True,
            resample_size=population_size,
            ci=ci,
            medoid='lowest',
        )
        st.success("Bootstrapping Analysis Completed")
        st.pyplot()

# Generate Variants
if st.button("Generate Network Variants"):
    if mfnf:
        filename_base = st.text_input("Base Filename for Variants", value="variant_networks/variant")
        layout_func = lambda G: nx.spring_layout(G)  # Default layout
        draw_options = {
            'node_color': 'blue',
            'edge_color': 'gray',
            'node_size': 50,
            'node_alpha': 1.0,
            'edge_alpha': 0.1,
        }
        mfnf.generate_variants(
            plot_input_graphs=False,  # Set to False by default
            number_of_variants=num_variants,
            plot_options={
                'layout_func': layout_func,
                'draw_options': draw_options,
                'figsize': (10, 10),
                'filename': None,
            },
            filename=filename_base + ".graphml",
        )
        st.success(f"Generated {num_variants} network variants saved as {filename_base}.graphml")




###############################################################################################


# Weights
st.sidebar.subheader("Weights")
weight_rho = st.sidebar.slider("Weight (ρ)", min_value=0, max_value=10, value=9)
weight_deg_nn = st.sidebar.slider("Weight (deg_nn)", min_value=0, max_value=10, value=9)
weight_motifs_global = st.sidebar.slider("Global Weight (Motifs)", min_value=0, max_value=10, value=1)

# Specify individual motif weights
st.sidebar.subheader("Motif Weights")
default_motifs = [
    'edge',
    'wedge',
    'triangle',
    'three_star',
    'four_clique',
]  # Default motifs provided in the backend
motif_weights = {}

for motif in default_motifs:
    motif_weights[motif] = st.sidebar.slider(f"Weight ({motif})", min_value=0, max_value=10, value=1)

# Motif Definitions (Optional Customization)
st.sidebar.subheader("Custom Motif Definitions")
custom_motif_name = st.sidebar.text_input("Custom Motif Name")
custom_motif_matrix_input = st.sidebar.text_area(
    "Adjacency Matrix for Custom Motif (comma-separated rows)", value=""
)
add_custom_motif = st.sidebar.button("Add Custom Motif")

if add_custom_motif:
    try:
        # Convert input matrix to numpy array
        custom_matrix = np.array(
            [list(map(int, row.split(","))) for row in custom_motif_matrix_input.strip().split("\n")]
        )
        if custom_matrix.shape[0] != custom_matrix.shape[1]:
            st.error("Adjacency matrix must be square!")
        else:
            custom_motifs.append({'name': custom_motif_name, 'matrix': custom_matrix})
            st.success(f"Added custom motif: {custom_motif_name}")
    except Exception as e:
        st.error(f"Invalid matrix input: {e}")

# Backend Integration for Motifs
if upload_file and st.button("Initialize Fitter"):
    default_motif_list = [
        {'name': 'edge', 'matrix': np.array([[0, 1], [1, 0]])},
        {'name': 'wedge', 'matrix': np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])},
        {'name': 'triangle', 'matrix': np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])},
        {'name': 'three_star', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])},
        {'name': 'four_clique', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])},
    ]
    custom_motif_list = custom_motifs if 'custom_motifs' in locals() else []
    all_motifs = default_motif_list + custom_motif_list

    weights = {'rho': weight_rho, 'deg_nn': weight_deg_nn, 'motifs': weight_motifs_global}
    for motif in all_motifs:
        motif_name = motif['name']
        if motif_name in motif_weights:
            weights[motif_name] = motif_weights[motif_name]

    mfnf = MultifractalNetworkFitter(
        graph,
        motifs=all_motifs,
        sampling_ratios={3: 0.01, 4: 0.0001},
        population_size=population_size,
        weights=weights,
        motif_analyzer_args={'exact': True, 'bootstrap_iters': None, 'ci': None},
    )
    st.success("Fitter Initialized")



'''