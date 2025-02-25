import pickle
import copy                    
import sys
import logging
import datetime
import matplotlib

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd

from itertools import combinations
from math import ceil
from io import BytesIO, StringIO
from contextlib import contextmanager

from multifractal_networks.utils.app_utils import load_graph, plot_single_motif
from multifractal_networks.utils.motifs import MotifGenerator
from multifractal_networks.mfnf import MultifractalNetworkFitter

SAVE_BUTTON = True

class StreamlitWriter:
    def __init__(self, text_display):
        self.text_display = text_display
        self.buffer = []
        
    def write(self, text):
        self.buffer.append(text)
        combined_output = "".join(self.buffer)
        self.text_display.code(combined_output, language="plaintext")
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    def get_buffer_content(self):
        return "".join(self.buffer)

class StreamlitLogHandler(logging.Handler):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.setFormatter(logging.Formatter('[%(levelname)s %(asctime)s] %(message)s'))
        
    def emit(self, record):
        msg = self.format(record)
        self.writer.write(msg + '\n')

@contextmanager
def capture_output(text_display):
    """Context manager to capture both stdout and logging output."""
    # Set up stdout capture
    original_stdout = sys.stdout
    writer = StreamlitWriter(text_display)
    sys.stdout = writer
    
    # Set up logging capture - explicitly handle optuna logger
    log_handler = StreamlitLogHandler(writer)
    optuna_logger = logging.getLogger("optuna")  # Get Optuna's logger specifically
    
    # Save original settings
    original_handlers = optuna_logger.handlers.copy()
    original_level = optuna_logger.level
    original_propagate = optuna_logger.propagate
    
    # Configure optuna logger
    optuna_logger.handlers = []
    optuna_logger.addHandler(log_handler)
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.propagate = False  # Prevent double logging
    
    try:
        yield writer
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        
        # Restore optuna logger settings
        optuna_logger.handlers = original_handlers
        optuna_logger.setLevel(original_level)
        optuna_logger.propagate = original_propagate

def render_navigation_sidebar():
    """Renders the navigation sidebar with table of contents"""
    st.sidebar.header("Table of Contents")
    
    # Force a refresh of completion status by accessing session state
    graph_status = st.session_state.get("graph_uploaded", False) or "graph" in st.session_state
    motif_status = "saved_motifs" in st.session_state and len(st.session_state.get("saved_motifs", [])) > 0
    model_status = st.session_state.get("mfnf_initialized", False)
    analysis_status = "analyses" in st.session_state and len(st.session_state.get("analyses", [])) > 0
    
    # Define steps with their titles and completion conditions
    steps = [
        {
            "title": "Load Network",
            "complete": graph_status
        },
        {
            "title": "Configure Motifs",
            "complete": motif_status
        },
        {
            "title": "Initialize Model",
            "complete": model_status
        },
        {
            "title": "Analyze Network",
            "complete": analysis_status
        }
    ]
    
    # Create buttons for each step
    for i, step in enumerate(steps, 1):
        # Determine if this step should be enabled
        enabled = all(steps[j]["complete"] for j in range(i-1))
        
        # Button key that won't conflict with session state
        button_key = f"nav_button_{i}_{step['complete']}"  # Include completion state in key
        
        # Create a button with the appropriate styling
        if enabled:
            if st.sidebar.button(
                f"{i}. {step['title']}" + (" ✓" if step["complete"] else ""),
                key=button_key,
                disabled=False,
                use_container_width=True,
            ):
                st.session_state["current_step"] = i
        else:
            # Show disabled button
            st.sidebar.button(
                f"{i}. {step['title']}", 
                disabled=True,
                key=button_key,
                use_container_width=True,
            )
    
    st.sidebar.divider()

def save_analysis(state, filename="output.mfnf"):
    """Save the current session state as a pickle file."""
    state_to_save = {key: value for key, value in state.items() if key not in ["current_step"]}  # Exclude navigation
    buffer = BytesIO()
    pickle.dump(state_to_save, buffer)
    buffer.seek(0)
    return buffer

#def load_analysis(uploaded_file):
#    """Load a pickled session state file."""
#    try:
#        state = pickle.load(uploaded_file)
#        for key, value in state.items():
#            st.session_state[key] = value
#        st.session_state["current_step"] = 1
#        st.success("Analysis loaded successfully!")
#    except Exception as e:
#        st.error(f"Error loading analysis: {e}")

def load_analysis(uploaded_file):
    """Load a pickled session state file."""
    try:
        state = pickle.load(uploaded_file)
        
        # Clear existing session state (except current_step)
        for key in list(st.session_state.keys()):
            if key != "current_step" and not key.startswith("nav_button_"):
                del st.session_state[key]
        
        # Update session state with new values
        for key, value in state.items():
            if key != "current_step" and not key.startswith("nav_button_"):
                st.session_state[key] = value
                
        st.success("Analysis loaded successfully!")
    except Exception as e:
        st.error(f"Error loading analysis: {e}")

def save_button():
    """Render a Save Analysis button on each page."""
    st.sidebar.subheader("Save Analysis")
    st.sidebar.markdown("""
    **Save Analysis**:
    - Save your current progress to a file (default: `analysis.mfnf`).
    - You can reload the saved analysis later to continue from where you left off.
    """)
    # Let the user specify a filename, defaulting to 'analysis.mfnf'
    filename = st.sidebar.text_input("Filename", value="analysis.mfnf")
    buffer = save_analysis(st.session_state)

    # Button to download the analysis file
    st.sidebar.download_button(
        label="Save Analysis",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream",
    )

# Define default motifs
DEFAULT_MOTIFS = [
    {'name': 'edge', 'matrix': np.array([[0, 1], [1, 0]])},
    {'name': 'wedge', 'matrix': np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])},
    {'name': 'triangle', 'matrix': np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])},
    {'name': 'three_star', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])},
    {'name': 'four_clique', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])},
    {'name': 'four_chain', 'matrix': np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])},
    {'name': 'three_loop_out', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])},
    {'name': 'four_loop', 'matrix': np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])},
    {'name': 'four_semi_clique', 'matrix': np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])},
]

# Initialize session state for saved motifs
if "saved_motifs" not in st.session_state:
    st.session_state["saved_motifs"] = DEFAULT_MOTIFS.copy()

#############################################################
# S T E P :  1
#############################################################
def step_1():
    st.header("Step 1: Start Your Analysis")
    st.subheader("Overview of the Multifractal Network Analysis Process")
    with st.expander("Process Overview"):
        st.markdown("""
            1. **Load Network** (Current Step)
                - Upload your network data or load a previously saved analysis
                - Supported formats: GraphML, CSV, TSV, or Link-List TXT
            
            2. **Configure Motifs**
                - Select and customize network motifs (subgraph patterns)
                - Choose from pre-defined motifs or generate new ones
            
            3. **Initialize Model**
                - Set up the multifractal network model parameters
                - Configure motif counting and analysis settings
            
            4. **Analysis**
                - Run various analyses including:
                    - Grid search for optimal parameters
                    - Direct parameter fitting
                    - Bootstrap confidence estimation
                    - Generate similar network variants
                    - Visualize and compare results
        """)
    
    # Option to load a saved analysis
    st.subheader("Load a Saved Analysis")
    saved_file = st.file_uploader("Upload a saved analysis file (.mfnf)", type=["mfnf"])
    if saved_file and not st.session_state.get("just_loaded", False):
        load_analysis(saved_file)
        st.session_state["just_loaded"] = True
        st.rerun()
    elif saved_file and st.session_state.get("just_loaded"):
        st.session_state["just_loaded"] = False

    st.subheader("Upload a Graph File")
    st.markdown("""
        #### Supported File Formats:
        - **GraphML**: Standard graph markup language, preserves node/edge attributes
        - **CSV/TSV**: Edge list with comma/tab separators
        - **TXT**: Space-separated edge list
        
        #### File Size Considerations:
        - For best performance, graphs should have < 100,000 nodes
        - Larger graphs may require more computation time
        - Sparse graphs (few edges per node) work best with this analysis
    """)

    # Continue with graph upload if no analysis is loaded
    #st.subheader("Upload a Graph File")
    if "graph_filename" in st.session_state:
        st.success(f"Graph file '{st.session_state['graph_filename']}' is already uploaded!")
    else:
        uploaded_file = st.file_uploader(
            "Upload a graph file (GraphML, CSV, TSV, or Link-List TXT)",
            type=["graphml", "csv", "tsv", "txt"],
            help="Choose a network file to analyze. GraphML preserves all network attributes, while CSV/TSV/TXT should contain edge lists (pairs of connected nodes)."
        )
        st.markdown("""
                    **Expected Formats for Link-List Files**:
                    The file should represent the graph as an edge list, where each row corresponds to an edge.
                
                    - **Example** (for a graph with 3 edges):
                    ```plaintext
                    node1 node2
                    node2 node3
                    node3 node1
                    ```
                
                    - **CSV Example** (comma-separated):
                    ```csv
                    node1,node2
                    node2,node3
                    node3,node1
                    ```
                
                    - **TSV Example** (tab-separated):
                    ```tsv
                    node1\tnode2
                    node2\tnode3
                    node3\tnode1
                    ```
                
                    Ensure that:
                    - Nodes are labeled consistently (no spaces in names).
                    - The delimiter matches the file type (space, comma, or tab).
                    - Headers are optional but will be ignored.
                    """)
        if uploaded_file:
            try:
                graph = load_graph(uploaded_file)
                st.session_state["graph"] = graph
                st.session_state["graph_filename"] = uploaded_file.name
                st.session_state["graph_uploaded"] = True
                st.success(f"Graph file '{uploaded_file.name}' uploaded and stored successfully!")
            except ValueError as e:
                st.error(str(e))

    # Navigation
    col1, col2 = st.columns([1, 1])  # Equal-width columns for back and next buttons
    if st.session_state.get("graph_uploaded") or "graph" in st.session_state:
        with col2:
            def go_to_next_step():
                st.session_state.current_step = 2
            st.button("Next →", on_click=go_to_next_step)

#############################################################
# S T E P :  2
#############################################################
def step_2():
    st.header("Step 2: Motif Selection")
    st.markdown("""
        ### Understanding Network Motifs
        Network motifs are recurring, significant patterns of interconnections in your network.
        They serve as building blocks that reveal the network's underlying structure.
        
        #### Pre-defined Motifs Include:
        - **Edge**: Basic connection between two nodes
        - **Wedge**: Two edges sharing a node (important for clustering)
        - **Triangle**: Three fully connected nodes (indicates clustering)
        - **Stars**: Central node connected to multiple others (shows hierarchy)
        - **Cliques**: Fully connected subgroups (shows tight communities)
        
        #### Why Motifs Matter:
        - Help identify functional building blocks in your network
        - Capture local network structure and organization
        - Essential for accurate network modeling and comparison
    """)
    if SAVE_BUTTON:
        save_button()

    # Add new motif sizes dynamically
    st.subheader("Add New Motif Sizes")
    st.markdown("""
        #### Working with Custom Motifs
        The Motif Generator tool helps you explore and create larger network patterns beyond the pre-defined motifs.
        
        #### Generator Features:
        - Creates all possible connected patterns for a given size
        - Visualizes each motif to help you understand its structure
        - Allows selection of specific motifs to include in your analysis
        - Supports generating motifs for multiple sizes in one session
        
        #### Important Considerations:
        - Larger motifs (5+ nodes) capture more complex network patterns
        - Be cautious with size:
            - Counting larger motifs requires more computation time
            - Very large motifs (7+ nodes) may be impractical for large networks
            - Generation time increases significantly with motif size
        - Consider using sampling ratios to balance accuracy and speed
        
        #### Tips for Motif Selection:
        - Start with smaller motifs (2-4 nodes) for initial analysis
        - Add larger motifs based on your specific research questions
        - Choose motifs that represent important structures in your domain
        - Be patient during generation of larger motifs
        - Save selected motifs before moving to the next size
    """)
    if "arbitrary_sizes" not in st.session_state:
        st.session_state["arbitrary_sizes"] = []

    # Button to add a new motif size menu
    if st.button("Add Motif Size"):
        st.session_state["arbitrary_sizes"].append(len(st.session_state["arbitrary_sizes"]) + 5)  # Start with size 5

    # Generate and select motifs for each added size
    for size in st.session_state["arbitrary_sizes"]:
        with st.expander(f"Motif Generator: {size} Nodes"):
            if st.button(
                f"Generate Motifs for {size} Nodes",
                key=f"generate_{size}",
                help="Generate all possible connected patterns with this number of nodes. Generation time increases with size."
            ):
                motif_gen = MotifGenerator(size)
                motifs = motif_gen.get_motifs()
                st.session_state[f"motifs_{size}"] = motifs
                st.success(f"Generated {len(motifs)} motifs for {size} nodes!")

                # Calculate rows for 3 columns
                num_motifs = len(motifs)
                cols = 3
                rows = (num_motifs + cols - 1) // cols  # Ceiling division

                # Plot motifs
                with st.spinner("Processing motifs..."):
                    fig = motif_gen.plot_motifs(rows=rows, cols=cols,streamlit=True)
                    st.pyplot(fig)

                

            if f"motifs_{size}" in st.session_state:
                motifs = st.session_state[f"motifs_{size}"]

                # Selection
                st.subheader(f"Select Motifs for {size} Nodes")
                selected_motifs = st.multiselect(
                    f"Select motifs for {size} nodes",
                    options=[motif["name"] for motif in motifs],
                    default=[motif["name"] for motif in motifs],
                    key=f"select_{size}",
                    help="Choose which motifs to include in your analysis. Selected motifs will be counted in your network."
                )

                # Save Selected Motifs
                if st.button(
                    f"Save Selected {size}-Node Motifs", 
                    key=f"save_{size}",
                    help="Add selected motifs to your analysis. You can generate and save motifs of different sizes before proceeding."
                ):
                    saved_motifs = [
                        motif for motif in motifs if motif["name"] in selected_motifs
                    ]
                    st.session_state["saved_motifs"].extend(saved_motifs)
                    st.success(f"Saved {len(saved_motifs)} motifs for {size} nodes.")

    # Display selected motifs
    st.subheader("Selected Motifs")
    for motif in st.session_state.get("saved_motifs", []):
        with st.expander(f"{motif['name']}"):
            col1, col2 = st.columns([1, 3])  # Adjust column proportions as needed

            # Adjacency Matrix in col1
            with col1:
                st.write("Adjacency Matrix:")
                st.write(motif["matrix"])

            # Plot of the motif in col2
            with col2:
                fig = plot_single_motif(motif["matrix"])  # Assume this returns a matplotlib figure
                st.pyplot(fig, bbox_inches="tight")

    # Navigation
    col1, col2 = st.columns([1, 1])  # Equal-width columns for back and next buttons
    with col1:
        st.button("← Back", on_click=lambda: st.session_state.update({"current_step": 1}))
    with col2:
        def go_to_next_step():
            st.session_state.current_step = 3
        st.button("Next →", on_click = go_to_next_step)


#############################################################
# S T E P :  3
#############################################################
def step_3():
    st.header("Step 3: Configure Model Parameters")
    if SAVE_BUTTON:
        save_button()

    # Section for motif_analyzer_args
    st.markdown("""
        ### Model Configuration Guide
        The multifractal network model requires several key parameters to effectively capture your network's structure.
        
        #### Motif Analysis Settings:
        - **Exact Counting vs Sampling**:
            - Exact: Precise but slower, better for small networks
            - Sampling: Faster but approximate, necessary for large networks
        
        - **Bootstrap Parameters**:
            - Helps estimate uncertainty in motif counts
            - More iterations = more accurate confidence intervals
            - Recommended: 100 iterations for initial analysis
        
        ##### Sampling Ratios:
        - Controls what fraction of possible motifs to check
        - Larger ratios = more accurate but slower
        - Recommended starting values:
            - Size 3 motifs: 0.001 (0.1%)
            - Size 4 motifs: 0.0001 (0.01%)
            - Larger motifs: Even smaller ratios
    """)
    exact_count = st.checkbox(
        "Use Exact Counting",
        value=True,
        help="If checked, performs exact motif counting. Uncheck for larger networks to use sampling-based counting instead."
    )

    # Show additional options only if exact_count is False
    if not exact_count:
        bootstrap_iters = st.slider(
            "Bootstrap Iterations (0 for no bootstrapping)",
            min_value=0,
            max_value=1000,
            value=100,
            step=1,
            help="Number of resamples for confidence interval estimation. More iterations give more reliable intervals but take longer."
        )
        ci = st.slider(
            "Confidence Interval",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            help="Confidence level for interval estimation (e.g., 0.95 for 95% confidence intervals)."
        )

         # Section for sampling_ratios
        st.subheader("Sampling Ratios")
        sampling_ratios = {}
        for size in range(2, max(len(motif["matrix"]) for motif in st.session_state["saved_motifs"]) + 1):
            ratio = st.text_input(
                f"Sampling Ratio for Motifs of Size {size}",
                value="0.0001",
                key=f"sampling_ratio_{size}",
                help="Fraction of possible motif locations to sample. Smaller ratios are faster but less accurate. For size 3: try 0.001, for size 4+: use 0.0001 or smaller."
            )
            try:
                sampling_ratios[size] = float(ratio)
            except ValueError:
                st.warning(f"Invalid sampling ratio for size {size}. Using default value of 0.0001.")
                sampling_ratios[size] = 0.0001
        
    else:
        bootstrap_iters, ci = None, None
        sampling_ratios = {}
        for size in range(2, max(len(motif["matrix"]) for motif in st.session_state["saved_motifs"]) + 1):
            sampling_ratios[size] = 0.0001

    motif_analyzer_args = {
        "exact": exact_count,
        "bootstrap_iters": bootstrap_iters,
        "ci": ci,
    }

    # Population Size
    st.subheader("Population Size")
    st.markdown("""
    - The number of population restarts for the optimization process.
    - Higher values can improve the likelihood of finding a better solution but increase computation time.
    - This parameter can be adjusted later if needed.
    """)
    population_size = st.text_input(
        "Population Size",
        value="10",
        key="population_size",
        help="Number of random restarts in optimization. Larger values may find better solutions but take longer."
    )
    try:
        population_size = int(population_size)
    except ValueError:
        st.warning("Invalid population size. Using default value of 10.")
        population_size = 10

    # Weights
    st.subheader("Weights")
    st.markdown("""
        #### Understanding Weight Parameters
        Weights determine how different network properties influence the fitting process.
        
        #### Key Components:
        1. **Degree Distribution** (typically higher weight):
            - Controls how well the model matches node connectivity patterns
            - Higher values prioritize matching the overall network structure
        
        2. **Nearest Neighbor Degree**:
            - Influences degree correlations between connected nodes
            - Important for capturing network assortativity/dissortativity
        
        3. **Motifs**:
            - Individual weights for each selected motif
            - Higher weights make specific patterns more important in the fit
        
        #### Tips for Weight Selection:
        - Start with equal weights for basic features
        - Increase weights for properties most important to your analysis
        - Use higher weights for more reliable/important measurements
    """)
    weights = {}

    # Default weights
    default_weights = {"rho": len(st.session_state["saved_motifs"]),
                       "deg_nn": len(st.session_state["saved_motifs"]),
                       "motifs": 1
                      }

    # Proper names for the default weights
    default_weight_names = {
        "rho": "Degree Distribution",
        "deg_nn": "Nearest Neighbor Average Degree",
        "motifs": "Global Motifs"
    }

    # Add text inputs for default weights
    for key, display_name in default_weight_names.items():
        default_value = len(st.session_state["saved_motifs"]) if key in ["rho", "deg_nn", "edge"] else 1
        weights[key] = st.text_input(
            f"Weight for {display_name}",
            value=str(default_value),
            key=f"weight_{key}",
            help={"rho": "Weight for matching degree distribution. Higher values prioritize matching overall connectivity patterns.",
          "deg_nn": "Weight for matching degree correlations. Higher values ensure similar connection patterns between high/low degree nodes.",
          "motifs": "Global weight applied to all motif counts. Multiplies with individual motif weights."}[key]
        )
        try:
            weights[key] = float(weights[key])
        except ValueError:
            st.warning(f"Invalid weight for {key}. Using default value of {default_value}.")
            weights[key] = default_value

    # Add weights for each user-selected motif
    st.markdown(r"""
                **Custom Motifs**:
                - These are individual weights for each selected motif.
                - The loss function applies weights **multiplicatively**:
                  - $ \text{Weight}_{\text{Global Motifs}} \times \text{Weight}_{\text{Given Motif}} \times \text{Error}_{\text{Given Motif}} $
                - Adjust these values to control the relative importance of specific motifs in the analysis.
                """)
    for motif in st.session_state["saved_motifs"]:
        name = motif["name"]
        if name == 'edge':
            default_value = len(st.session_state["saved_motifs"])
            weights[name] = st.text_input(
                f"Weight for Motif {name}",
                value=str(default_value),
                key=f"weight_{name}",
            )
            st.divider()
        else:
            weights[name] = st.text_input(
                f"Weight for Motif {name}",
                value="1",
                key=f"weight_{name}",
                help="Individual weight for this motif. Final motif weight = Global Motif Weight × This Weight"
            )
        try:
            weights[name] = float(weights[name])
        except ValueError:
            st.warning(f"Invalid weight for motif {name}. Using default value of 1.")
            weights[name] = 1

    # Save the configuration in session state
    st.session_state["mfnf_config"] = {
        "motif_analyzer_args": motif_analyzer_args,
        "sampling_ratios": sampling_ratios,
        "population_size": population_size,
        "weights": weights,
    }

    st.subheader("🚀 Finalize Configuration and Start Motif Counting")
    st.markdown("""
                This step will initialize the **Multifractal Network Fitter (MFNF)** using your provided graph, selected motifs, and settings.
                - **Important**: This process may take some time depending on the size of the graph and the complexity of the motifs.
                - Once complete, you can view the detailed initialization output for transparency.
                """)
    
    # Red button styling specifically for this button
    button_style = """
    <style>
    #start-counting-btn button {
        background-color: #ff4b4b;
        color: white;
    }
    #start-counting-btn button:hover {
        background-color: #ff2b2b;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    # Button to initialize mfnf
    # Check if initialization has been completed
    # Show permanent initialization output if completed
    if st.session_state.get("init_output", None) and st.session_state.get("mfnf_initialized", True):
        # Success message and captured output in a dropdown
        with st.expander("Initialization Output"):
            output_display = st.code(st.session_state["init_output"], language="plaintext")
    # Start initialization otherwise...
    if not st.session_state.get("mfnf_initialized", False):
        # Button to initialize mfnf
        if st.button("Start Motif Counting", key="start-counting-btn"):
            st.markdown("### Initializing Multifractal Network Fitter")
        
            # Add a spinner while the motif counting is happening
            with st.spinner("Counting motifs and initializing model. This may take some time..."):
                try:
                    with st.expander("Initialization Output"):
                        output_display = st.code(st.empty(), language="plaintext")
                    
                    with capture_output(output_display) as writer:
                        mfnf = MultifractalNetworkFitter(
                        st.session_state["graph"],
                        motifs=st.session_state["saved_motifs"],
                        sampling_ratios=st.session_state["mfnf_config"]["sampling_ratios"],
                        motif_analyzer_args=st.session_state["mfnf_config"]["motif_analyzer_args"],
                        population_size=st.session_state["mfnf_config"]["population_size"],
                        weights=st.session_state["mfnf_config"]["weights"],
                    )
        
                    # Store mfnf in the session state
                    st.session_state["mfnf"] = mfnf
                    st.session_state["mfnf_initialized"] = True
                    st.session_state["init_output"] = writer.get_buffer_content()
                    st.success("Initialization complete!")
                    
                except Exception as e:
                    sys.stdout = sys.__stdout__  # Ensure stdout is reset in case of failure
                    st.error(f"An error occurred during initialization: {e}")

    col1, col2 = st.columns([1, 1])  # Equal-width columns for back and next buttons
    with col1:
        st.button("← Back", on_click=lambda: st.session_state.update({"current_step": 2}))
    # Show the "Next →" button only if initialization is complete
    if st.session_state.get("mfnf_initialized", False):
        with col2:
            st.button("Next →", on_click=lambda: st.session_state.update({"current_step": 4}))
            

#########################################################################
# Modular helper functions for Step 4
#########################################################################
def init_analysis_state():
    """Initialize session state variables if they don't exist"""
    if "analyses" not in st.session_state:
        st.session_state.analyses = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

def store_analysis_result(analysis_id: int, result_type: str, data):
    """Store analysis result in session state"""
    if analysis_id not in st.session_state.analysis_results:
        st.session_state.analysis_results[analysis_id] = {}
    st.session_state.analysis_results[analysis_id][result_type] = data

def get_analysis_result(analysis_id: int, result_type: str):
    """Retrieve analysis result from session state"""
    if analysis_id in st.session_state.analysis_results:
        return st.session_state.analysis_results[analysis_id].get(result_type)
    return None

def switch_to_results_mode(analysis_id):
    """Switch an analysis from settings mode to results mode"""
    mode_key = f"analysis_mode_{analysis_id}"
    st.session_state[mode_key] = "results"

def switch_to_settings_mode(analysis_id):
    """Switch an analysis from results mode to settings mode"""
    mode_key = f"analysis_mode_{analysis_id}"
    st.session_state[mode_key] = "settings"

def get_analysis_mode(analysis_id):
    """Get the current mode for an analysis"""
    mode_key = f"analysis_mode_{analysis_id}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "settings"
    return st.session_state[mode_key]

#########################################################################
# Analysis functions for Step 4
#########################################################################
# find_P_l
def render_find_pl_settings(analysis):
    """Render the settings interface for find_p_l analysis"""
    st.markdown("""
        #### Parameters:
        - **m**: Matrix dimension - controls model complexity
            - Smaller m (2-3): Simpler model, faster computation
            - Larger m (5+): More complex structure, slower computation
            - It is recommended to avoid proper prime powers (2^2=4, 2^3=8, ... etc.) as k choice will ensure they are checked.
        - **k**: Recursion depth - controls structural detail level
            - Too large: May overfit or be too slow
        - **P initialization**: How to start the optimization
            - 'm x m': Random full matrix (more flexible)
            - 'diag': Random diagonal matrix (simpler starting point)
            """)
    
    # Retrieve or initialize settings
    settings = analysis["settings"]
    if "m" not in settings:
        settings["m"] = 5
    if "k" not in settings:
        settings["k"] = 2
    
    settings["m"] = st.slider("m", 2, 15, settings["m"], 
                           key=f"m_slider_{analysis['id']}", 
                           help="Dimension of the initiator matrix")
    settings["k"] = st.slider("k", 1, 15, settings["k"], 
                           key=f"k_slider_{analysis['id']}", 
                           help="Number of recursive levels to apply")
    settings["P_init"] = st.selectbox(
        "P Initialization",
        ["m x m", "diag"],
        index=0,
        key=f"P_init_find_p_l_{analysis['id']}",
        help=("How to initialize P: 'm x m' uses random m×m matrix, " 
              "'diag' uses random diagonal matrix")
    )
    
    # Run button with callback
    st.button("Run", 
              key=f"run_find_pl_{analysis['id']}", 
              on_click=switch_to_results_mode,
              args=(analysis['id'],))

def display_matrix(matrix, label="Matrix", analysis_id="0", format="{:.2e}"):
    """Displays a matrix with nice formatting"""
    df = pd.DataFrame(matrix)
    # Apply styling to the table elements
    styled_df = df.style.format(format).hide(axis=0).hide(axis=1)
    
    html = f"""
        {styled_df.to_html()}
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(label)
    with col2:
        # Create a buffer to store the CSV data
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, header=False)
        csv_buffer.seek(0)
        
        # Add download button aligned with the label
        file_name = label.lower().replace(' ', '_')
        st.download_button(
            label="📥",  # Download icon
            key=f"download_button_{file_name}_{analysis_id}",
            data=csv_buffer,
            file_name=f"{file_name}.csv",
            mime="text/csv",
            help=f"Download {label} as CSV"
        )
    st.write(html, unsafe_allow_html=True)
    
def render_find_pl_results(analysis, performance_mode):
    """Render the results interface for find_p_l analysis"""
    P = get_analysis_result(analysis["id"], "P")
    l = get_analysis_result(analysis["id"], "l")
    k = get_analysis_result(analysis["id"], "k")
        
    if all(x is not None for x in [P, l, k]):
        # Display the results
        st.markdown("**Recursive Level k:**")
        st.write(f"k = {k}")
            
        col1, col2 = st.columns([2, 1])
            
        with col1:
            display_matrix(P, label="Probability Matrix P", analysis_id=analysis["id"])
                
        with col2:
            display_matrix(l, label="Length Vector l", analysis_id=analysis["id"])

def run_find_pl_analysis(analysis):
    """Execute the find_p_l analysis and store results"""
    with st.spinner("Finding P and l..."):
        buffer = StringIO()
        sys.stdout = buffer
        analysis["mfnf"].find_P_l(m=analysis["settings"]["m"], 
                                  k=analysis["settings"]["k"], 
                                  one_shot=True, 
                                  P_init=analysis["settings"]["P_init"])
        sys.stdout = sys.__stdout__
        st.code(buffer.getvalue(), language="plaintext")

        # Store all results
        store_analysis_result(analysis["id"], "output", buffer.getvalue())
        store_analysis_result(analysis["id"], "P", analysis["mfnf"].P)
        store_analysis_result(analysis["id"], "l", analysis["mfnf"].l)
        store_analysis_result(analysis["id"], "k", analysis["mfnf"].k)
        store_analysis_result(analysis["id"], "timestamp", datetime.datetime.now().isoformat())

################################
# grid_search
def render_grid_search_settings(analysis):
    """Render the settings interface for grid_search analysis"""
    st.markdown("""    
        Instead of examining a single combination of parameters, this approach:
        - Allows you to specify multiple matrix dimensions (m)
        - For each m, define a range of recursive levels (k) to explore
        - Automatically searches through all combinations to find the best fit
        """)

    # Retrieve or initialize settings
    settings = analysis["settings"]
    if "m_k_dict" not in settings:
        # Get graph length from mfnf
        graph_len = len(analysis["mfnf"].G)
        # Initialize with optimal defaults
        settings["m_k_dict"] = [
            {"m": 2, "k": [1, int(np.log(graph_len) / np.log(2))]},
            {"m": 3, "k": [1, int(np.log(graph_len) / np.log(3))]},
        ]

    st.markdown("""
        #### Matrix Dimensions and Recursive Levels
        For each matrix dimension (m), specify:
        - A minimum k value (where to start searching)
        - A maximum k value (where to stop searching)
        """)

    # Container for m-k pairs
    m_k_container = st.container()
    
    # Display and edit existing m-k pairs
    new_m_k_dict = []
    for i, m_k_entry in enumerate(settings["m_k_dict"]):
        with m_k_container.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                new_m = st.number_input(
                    "Matrix Dimension (m)", 
                    min_value=2,
                    value=m_k_entry["m"],
                    key=f"m_{analysis['id']}_{i}"
                )
            with col2:
                k_min = st.number_input(
                    "Min k",
                    min_value=1,
                    value=m_k_entry["k"][0],
                    key=f"k_min_{analysis['id']}_{i}",
                    help="Minimum recursion depth to try. Higher values create more complex structure but increase computation time."
                )
            with col3:
                k_max = st.number_input(
                    "Max k",
                    min_value=k_min,
                    value=m_k_entry["k"][1],
                    key=f"k_max_{analysis['id']}_{i}",
                    help="Maximum recursion depth to try. The optimal k often lies near log_m(N) where N is the number of nodes."
                )
            with col4:

                def delete_matrix_dimension(analysis_id, index, m_k_dict):
                    """Callback function to delete a matrix dimension"""
                    m_k_dict.pop(index)
                    # Update the settings in session state 
                    st.session_state.analyses[analysis_id - 1]["settings"]["m_k_dict"] = m_k_dict
                
                st.button(
                    "🗑️", 
                    key=f"delete_{analysis['id']}_{i}",
                    on_click=delete_matrix_dimension,
                    args=(analysis['id'], i, settings["m_k_dict"])
                )
            new_m_k_dict.append({"m": new_m, "k": [k_min, k_max]})
    
    def add_matrix_dimension(analysis_id, m_k_dict, graph_len):
        """Callback function to add a new matrix dimension"""
        new_m = max([entry["m"] for entry in m_k_dict]) + 1 if m_k_dict else 2
        m_k_dict.append({
            "m": new_m,
            "k": [1, int(np.log(graph_len) / np.log(new_m))]
        })
        # Update the settings in session state
        st.session_state.analyses[analysis_id - 1]["settings"]["m_k_dict"] = m_k_dict
    
    # In the render function, replace the if st.button block with:
    st.button(
        "➕ Add Matrix Dimension", 
        key=f"add_m_k_{analysis['id']}", 
        on_click=add_matrix_dimension,
        args=(analysis['id'], new_m_k_dict, len(analysis["mfnf"].G))
    )
    
    # Update settings
    settings["m_k_dict"] = new_m_k_dict

    st.markdown("""
        #### Optimization Criteria
        Choose how to evaluate the quality of fit:
        - **KL Energy**: Combines Kullback-Leibler divergence for degree distributions with mean relative errors for other features. Best balance between capturing both distribution shape and exact counts.
        - **Energy**: Uses mean relative errors for all features (including the degree distribution). Better when exact feature counts are most important.
        - **KL Divergence**: Focuses purely on degree distribution shape. Best when degree distribution is the primary concern.
        """)
    settings["optimize_for"] = st.selectbox(
        "Optimize For",
        ["kl_energy", "energy", "kl_divergence"],
        index=0,
        key=f"optimize_for_{analysis['id']}",
        help="""
        - kl_energy: Uses mean relative errors for motifs and degree correlation, 
          KL divergence for degree distribution
        - energy: Uses mean relative errors for all features
        - kl_divergence: Focuses on optimizing the KL divergence between distributions 
          when deciding optimal grid points
        """
    )
    
    settings["P_init"] = st.selectbox(
        "P Initialization",
        ["m x m", "diag"],
        index=0,
        key=f"P_init_grid_search_{analysis['id']}",
        help=("How to initialize P: 'm x m' uses random m×m matrix, " 
              "'diag' uses random diagonal matrix")
    )
    
    # Run button with callback
    st.button("Run Grid Search", 
              key=f"run_grid_search_{analysis['id']}", 
              on_click=switch_to_results_mode,
              args=(analysis['id'],))
        
def render_grid_search_results(analysis, performance_mode):
    """Render the results interface for grid_search analysis"""
    P = get_analysis_result(analysis["id"], "P")
    l = get_analysis_result(analysis["id"], "l")
    k = get_analysis_result(analysis["id"], "k")
        
    if all(x is not None for x in [P, l, k]):
        # Display the results
        st.markdown("**Recursive Level k:**")
        st.write(f"k = {k}")
            
        col1, col2 = st.columns([2, 1])
            
        with col1:
            display_matrix(P, label="Probability Matrix P", analysis_id=analysis["id"])
                
        with col2:
            display_matrix(l, label="Length Vector l", analysis_id=analysis["id"])

def run_grid_search_analysis(analysis):
    """Execute the grid search analysis and store results"""
    with st.spinner("Running Grid Search..."):
        # Create an expander to show the output
        output_display = st.code(st.empty(), language="plaintext")
            
        # Capture and display output in real-time
        with capture_output(output_display) as writer:
            # Fix the exclusive higher threshold by adding + 1
            for m_k_entry in analysis["settings"]["m_k_dict"]:
                m_k_entry["k"][1] += 1
            analysis["mfnf"].find_m_k_P_l(
                m_k_dict=analysis["settings"]["m_k_dict"],
                P_init=analysis["settings"]["P_init"],
                optimize_for=analysis["settings"]["optimize_for"],
            )

        # Store all results
        store_analysis_result(analysis["id"], "output", writer.get_buffer_content())
        store_analysis_result(analysis["id"], "P", analysis["mfnf"].P)
        store_analysis_result(analysis["id"], "l", analysis["mfnf"].l)
        store_analysis_result(analysis["id"], "k", analysis["mfnf"].k)
        store_analysis_result(analysis["id"], "timestamp", datetime.datetime.now().isoformat())

################################
# plot_results
def render_plot_results_settings(analysis):
    st.markdown("""
        #### Display Options
        Configure how results are visualized:
        - **Log Scale**: Use logarithmic axes. Better for visualizing heavy-tailed distributions.
        - **Mean Relative Error (MRE)**: Show how well fitted values match measured values on average.
        - **KL-Divergence**: Show how well fitted distribution shapes match measured ones.
        - **Custom Axis Limits**: Focus on specific ranges of values in degree distribution plots.
        """)
    # Retrieve or initialize settings
    settings = analysis["settings"]
    
    # Get list of previous analyses that are find_p_l, grid_search or bootstrap_confidence type
    available_analyses = [
        a for a in st.session_state.analyses 
        if (a["type"] in ["find_p_l", "grid_search", "bootstrap_confidence"] and 
            get_analysis_result(a["id"], "P") is not None)
    ]
    
    if not available_analyses:
        st.error("No completed find_p_l, grid_search or bootstrap_confidence analyses found")
        return
        
    # Analysis selection
    analysis_options = {f"Analysis {a['id']} ({a['type']})": a["id"] for a in available_analyses}
    selected_analysis = st.selectbox(
        "Select source analysis for k,P,l",
        options=list(analysis_options.keys()),
        key=f"source_analysis_select_{analysis['id']}"
    )
    
    # Update source analysis id in settings
    source_id = analysis_options[selected_analysis]
    settings["source_analysis"] = source_id
    
    # Copy the MFNF from selected analysis
    source_analysis = next(a for a in available_analyses if a["id"] == source_id)
    analysis["mfnf"] = copy.deepcopy(source_analysis["mfnf"])

    # Get source analysis results to check wether it actually succeeded...
    P = get_analysis_result(source_id, "P")
    l = get_analysis_result(source_id, "l")
    k = get_analysis_result(source_id, "k")
    
    if any(x is None for x in [P, l, k]):
        st.error("Source analysis results not found")
        return

    # Plot settings
    settings["log_scale"] = st.checkbox(
        "Log Scale", 
        value=settings.get("log_scale", False),
        key=f"log_scale_{analysis['id']}"
    )
    
    settings["display_mre"] = st.checkbox(
        "Display Mean Relative Error (MRE)",
        value=settings.get("display_mre", True),
        key=f"display_mre_{analysis['id']}"
    )
    
    settings["display_kld"] = st.checkbox(
        "Display KL-Divergence",
        value=settings.get("display_kld", False),
        key=f"display_kld_{analysis['id']}"
    )
    
    # Optional axis limits for degree distribution
    use_custom_limits = st.checkbox(
        "Use custom axis limits for degree distribution",
        value=settings.get("use_custom_limits", False),
        key=f"use_custom_limits_{analysis['id']}"
    )
    settings["use_custom_limits"] = use_custom_limits
    
    if use_custom_limits:
        col1, col2 = st.columns(2)
        with col1:
            xlim_min = st.number_input(
                "X-axis Minimum",
                min_value=1,
                value=settings.get("xlim_min", 1),
                key=f"xlim_min_{analysis['id']}"
            )
            xlim_max = st.number_input(
                "X-axis Maximum",
                min_value=xlim_min + 1,
                value=settings.get("xlim_max", 100),
                key=f"xlim_max_{analysis['id']}"
            )
            settings["xlim"] = (xlim_min, xlim_max)
        
        with col2:
            ylim_min = st.number_input(
                "Y-axis Minimum",
                min_value=0.0,
                max_value=1.0,
                value=settings.get("ylim_min", 0.0),
                key=f"ylim_min_{analysis['id']}"
            )
            ylim_max = st.number_input(
                "Y-axis Maximum",
                min_value=ylim_min,
                max_value=1.0,
                value=settings.get("ylim_max", 1.0),
                key=f"ylim_max_{analysis['id']}"
            )
            settings["ylim"] = (ylim_min, ylim_max)
    else:
        settings["xlim"] = None
        settings["ylim"] = None
    
    # Run button
    st.button("Plot", 
              key=f"run_plot_results_{analysis['id']}", 
              on_click=switch_to_results_mode,
              args=(analysis['id'],))

def render_plot_results_results(analysis, performance_mode):
    """Render the plot results with cached display"""
    figs = get_analysis_result(analysis["id"], "figures")
    if not figs:
        return
    
    plot_titles = [
        "Degree Distribution Comparison",
        "Degree Correlation Comparison",  
        "Motif Comparison",
        "Generating Metric"
    ]

    if performance_mode:
        st.warning("⚡️ Performance Mode: Limited visibility for plots. Click on Show Plot to view them.")
    
    # Display each figure with cached rendering
    for i, (fig, title) in enumerate(zip(figs, plot_titles)):
        st.write(f"### {title}")
        if performance_mode:
            # Create container after the button
            container = st.empty()
            if container.button(label="Show Plot", key=f"show_plot_button_{i}_{analysis['id']}"):
                container.pyplot(fig)
        else:
            st.pyplot(fig)
    
def run_plot_results_analysis(analysis):
    with st.spinner("Generating plots..."):
        # Generate plots
        figs = analysis["mfnf"].plot_P_l_result(
            log_scale=analysis["settings"]["log_scale"],
            display_mre=analysis["settings"]["display_mre"],
            display_kld=analysis["settings"]["display_kld"],
            xlim=analysis["settings"]["xlim"],
            ylim=analysis["settings"]["ylim"],
            streamlit=True,
        )
            
        # Store and display results
        store_analysis_result(analysis["id"], "figures", figs)
        store_analysis_result(analysis["id"], "timestamp", datetime.datetime.now().isoformat())

################################
# bootstrap_confidence
def render_bootstrap_confidence_settings(analysis):
    """Run bootstrap confidence analysis and store results in session state"""
    st.write("### Configure Bootstrap Confidence")
    st.markdown("""
        #### Parameters
        - **Confidence Interval**: Width of the confidence band around estimates. A 95% CI means we're 95% confident the true value lies in this range.
        - **Resample Size**: Number of bootstrap samples to generate. Larger values give more reliable confidence intervals but take longer to compute.
        - **Aggregation Type**: How to combine results across bootstrap samples:
            - Mean: Average values (most stable)
            - Median: Middle value (robust to outliers)
            - Lowest: Best (lowest energy) individual result (may be less stable for small sample sizes)
    """)
    
    #Retrieve or initialize settings
    settings = analysis["settings"]
    
    # Get list of previous analyses that are either find_p_l or grid_search type
    available_analyses = [
        a for a in st.session_state.analyses 
        if (a["type"] in ["find_p_l", "grid_search"] and 
            get_analysis_result(a["id"], "P") is not None)
    ]
    
    if not available_analyses:
        st.error("No completed find_p_l or grid_search analyses found")
        return
        
    # Analysis selection
    analysis_options = {f"Analysis {a['id']} ({a['type']})": a["id"] for a in available_analyses}
    selected_analysis = st.selectbox(
        "Select source analysis for k,P,l",
        options=list(analysis_options.keys()),
        key=f"source_analysis_select_{analysis['id']}"
    )
    
    # Update source analysis id in settings
    source_id = analysis_options[selected_analysis]
    settings["source_analysis"] = source_id
    
    # Copy the MFNF from selected analysis
    source_analysis = next(a for a in available_analyses if a["id"] == source_id)
    analysis["mfnf"] = copy.deepcopy(source_analysis["mfnf"])
    analysis["settings"]["P_init"] = copy.deepcopy(source_analysis["settings"]["P_init"])

    # Get source analysis results to check wether it actually succeeded...
    P = get_analysis_result(source_id, "P")
    l = get_analysis_result(source_id, "l")
    k = get_analysis_result(source_id, "k")
    
    if any(x is None for x in [P, l, k]):
        st.error("Source analysis results not found")
        return
    # Other settings
    if "ci" not in settings:
        settings["ci"] = 0.95
    if "medoid" not in settings:
        settings["medoid"] = "mean"
    if "resample_size" not in settings:
        settings["resample_size"] = 100

    # Bootstrap settings
    settings["ci"] = st.slider(
        "Confidence Interval", 0.5, 0.99, settings["ci"], 
        key=f"ci_slider_{analysis['id']}"
    )
    
    settings["medoid"] = st.selectbox(
        "Aggregation Type", 
        ["mean", "median", "lowest"],
        index=["mean", "median", "lowest"].index(settings["medoid"]),
        key=f"medoid_{analysis['id']}",
        help="How do you want to aggregate the different optimum (P,l)-s. 'mean' and 'median' take the population mean or median. Lowest chooses the single lowest energy result instead."
    )
    
    settings["resample_size"] = st.number_input(
        "Resample Size",
        min_value=analysis["mfnf"].population_size,
        max_value=analysis["mfnf"].population_size*100,
        value=settings["resample_size"],
        key=f"resample_size_{analysis['id']}",
        help="How many times do you want to sample from the popultion."
    )

    # Run button with callback
    st.button("Run Bootstrapping", 
              key=f"run_bootstrap_confidence_{analysis['id']}", 
              on_click=switch_to_results_mode,
              args=(analysis['id'],))

def run_bootstrap_confidence(analysis):
    with st.spinner("Running bootstrap analysis..."):
        # Create an expander to show the output
        output_display = st.code(st.empty(), language="plaintext")
            
        # Capture and display output in real-time
        # TODO: Rewrite the bootstrap_P_l_stability with Streamlit's tqdm implementation (too)
        with capture_output(output_display) as writer:
            analysis["mfnf"].bootstrap_P_l_stability(
                P_init=analysis["settings"]["P_init"],
                ci=analysis["settings"]["ci"],
                medoid=analysis["settings"]["medoid"],
                resample_size=analysis["settings"]["resample_size"]
            )

        # Store all results
        print(analysis["mfnf"].rho_analytical.keys())
        store_analysis_result(analysis["id"], "output", writer.get_buffer_content())
        store_analysis_result(analysis["id"], "P", analysis["mfnf"].P)
        store_analysis_result(analysis["id"], "l", analysis["mfnf"].l)
        store_analysis_result(analysis["id"], "k", analysis["mfnf"].k)
        store_analysis_result(analysis["id"], "timestamp", datetime.datetime.now().isoformat())

def render_bootstrap_confidence_results(analysis, performance_mode):
    return 0

################################
# generate_variants
def render_generate_variants_settings(analysis):
    """Render the settings interface for generate_variants analysis"""
    st.markdown("""
        #### Configure Variant Generation
        """)
    
    # Get list of previous analyses that are find_p_l, grid_search or bootstrap_confidence type
    available_analyses = [
        a for a in st.session_state.analyses 
        if (a["type"] in ["find_p_l", "grid_search", "bootstrap_confidence"] and 
            get_analysis_result(a["id"], "P") is not None)
    ]
    
    if not available_analyses:
        st.error("No completed find_p_l, grid_search or bootstrap_confidence analyses found")
        return
        
    # Analysis selection
    analysis_options = {f"Analysis {a['id']} ({a['type']})": a["id"] for a in available_analyses}
    selected_analysis = st.selectbox(
        "Select source analysis for k,P,l",
        options=list(analysis_options.keys()),
        key=f"source_analysis_select_{analysis['id']}"
    )
    
    # Update source analysis id in settings
    source_id = analysis_options[selected_analysis]
    settings = analysis["settings"]
    settings["source_analysis"] = source_id
    
    # Copy the MFNF from selected analysis
    source_analysis = next(a for a in available_analyses if a["id"] == source_id)
    analysis["mfnf"] = copy.deepcopy(source_analysis["mfnf"])
    
    # Get source analysis results to check whether it actually succeeded...
    P = get_analysis_result(source_id, "P")
    l = get_analysis_result(source_id, "l")
    k = get_analysis_result(source_id, "k")
    
    if any(x is None for x in [P, l, k]):
        st.error("Source analysis results not found")
        return

    # Retrieve or initialize settings
    if "num_variants" not in settings:
        settings["num_variants"] = 1
    if "layout_type" not in settings:
        settings["layout_type"] = "Spring Layout"
    if "smart_viz" not in settings:
        settings["smart_viz"] = True
    if "largest_component_only" not in settings:
        settings["largest_component_only"] = True

    # Variant settings
    settings["num_variants"] = st.slider(
        "Number of Variants", 
        1, 10, settings["num_variants"],
        key=f"num_variants_{analysis['id']}"
    )

    st.markdown("""
        ##### Visualization Options:
        - **Layout Types**: Different ways to arrange nodes
        """)
    
    settings["layout_type"] = st.selectbox(
        "Layout",
        ["Spring Layout", "Circular Layout"],
        index=["Spring Layout", "Circular Layout"].index(settings["layout_type"]),
        key=f"layout_type_{analysis['id']}"
    )

    # Advanced visualization settings checkbox
    use_advanced_settings = st.checkbox(
        "Use advanced visualization settings",
        value=settings.get("use_advanced_settings", False),
        key=f"use_advanced_settings_{analysis['id']}"
    )
    settings["use_advanced_settings"] = use_advanced_settings
    
    # Show advanced settings if checkbox is checked
    if use_advanced_settings:
        st.markdown("""
            - **Smart Visualization**: Automatically adjusts node sizes and colors based on node degree and edge betweenness
            - **Component Display**: Option to focus on main network structure
            """)
        settings["smart_viz"] = st.checkbox(
            "Use smart visualization",
            value=settings.get("smart_viz", True),
            help="When enabled, uses degree-based node sizing and coloring",
            key=f"smart_viz_{analysis['id']}"
        )
        
        settings["largest_component_only"] = st.checkbox(
            "Show only largest component",
            value=settings.get("largest_component_only", True),
            help="When enabled, only shows the largest connected component",
            key=f"largest_component_{analysis['id']}"
        )
        
        # Show manual controls only if smart visualization is disabled
        if not settings["smart_viz"]:
            col1, col2 = st.columns(2)
            with col1:
                settings["node_color"] = st.color_picker(
                    "Node Color", 
                    value=settings.get("node_color", "#0000FF"),
                    key=f"node_color_{analysis['id']}"
                )
                node_size = st.text_input(
                    "Node Size",
                    value=str(settings.get("node_size", 50)),
                    key=f"node_size_{analysis['id']}"
                )
                try:
                    settings["node_size"] = int(node_size)
                except ValueError:
                    st.warning("Invalid node size. Using default value of 50.")
                    settings["node_size"] = 50
            with col2:
                settings["edge_color"] = st.color_picker(
                    "Edge Color",
                    value=settings.get("edge_color", "#000000"),
                    key=f"edge_color_{analysis['id']}"
                )
                settings["edge_alpha"] = st.slider(
                    "Edge Transparency",
                    min_value=0.0, max_value=1.0, 
                    value=settings.get("edge_alpha", 0.1),
                    key=f"edge_alpha_{analysis['id']}"
                )
    else:
        # Set smart defaults when advanced settings are not used
        settings["smart_viz"] = True
        settings["largest_component_only"] = True
        settings["node_color"] = "#0000FF"
        settings["node_size"] = 50
        settings["edge_color"] = "#000000"
        settings["edge_alpha"] = 0.1

    # Run button with callback
    st.button("Generate Variants", 
              key=f"run_generate_variants_{analysis['id']}", 
              on_click=switch_to_results_mode,
              args=(analysis['id'],))

def render_generate_variants_results(analysis, performance_mode):
    """Render the results interface for generate_variants analysis"""
    # Get the stored figures
    figs = get_analysis_result(analysis["id"], "figures")
    graphs = get_analysis_result(analysis["id"], "graphs")
    if not figs:
        return
    if performance_mode:
        st.warning("⚡️ Performance Mode: Limited visibility for plots. Click on Show Plot to view them.")
    
    # Display each figure with proper handling for performance mode
    for i, (fig, graph) in enumerate(zip(figs, graphs)):
        if i == 0:
            st.write("### Input Graph")
        else:
            st.write(f"### Variant {i}")
        
        # Create container for the plot and download button
        plot_container = st.container()
        col1, col2 = plot_container.columns([3, 1])
        
        with col1:
            if performance_mode:
                # Create container after the button
                container = st.empty()
                if container.button(label="Show Plot", key=f"show_variant_plot_button_{i}_{analysis['id']}"):
                    container.pyplot(fig)
            else:
                st.pyplot(fig)
                
        with col2:
            # Add download button for each graph
            buf = BytesIO()
            nx.write_graphml(graph, buf)
            buf.seek(0)
            
            st.download_button(
                key=f"download_variant_graph_button_{i}_{analysis['id']}",
                label=f"Download {('Input Graph' if i==0 else f'Graph Variant {i}')}",
                data=buf.getvalue(),
                file_name=f"graph_{'input' if i==0 else f'variant_{i}'}.graphml",
                mime="application/octet-stream",
            )

def run_generate_variants_analysis(analysis):
    """Execute the generate_variants analysis and store results"""
    # Get source analysis id and results
    source_id = analysis["settings"]["source_analysis"]
    P = get_analysis_result(source_id, "P")
    l = get_analysis_result(source_id, "l")
    k = get_analysis_result(source_id, "k")

    with st.spinner("Generating graph variants..."):
        # Set the parameters in mfnf from the source analysis
        analysis["mfnf"].P = P
        analysis["mfnf"].l = l
        analysis["mfnf"].k = k
        
        # Prepare plot options
        plot_options = {
            "layout_func": (nx.spring_layout if analysis["settings"]["layout_type"] == "Spring Layout" 
                          else nx.circular_layout),
            "draw_options": {
                "node_color": analysis["settings"]["node_color"],
                "edge_color": analysis["settings"]["edge_color"],
                "node_size": analysis["settings"]["node_size"],
                "node_alpha": 1.0,
                "edge_alpha": analysis["settings"]["edge_alpha"]
            },
            "figsize": (10, 10),
            "filename": None,
            "streamlit": True,
            "largest_component_only": analysis["settings"]["largest_component_only"],
            "smart_viz": analysis["settings"]["smart_viz"]
        }
        
        # Generate variants
        figs, graphs = analysis["mfnf"].generate_variants(
            plot_input_graphs=True,
            number_of_variants=analysis["settings"]["num_variants"],
            plot_options=plot_options,
            streamlit=True
        )

        st.code(f"Generated {analysis['settings']['num_variants']} variant networks from the fitted measure.", language="plaintext")
        
        # Store results
        if figs is not None:  # Add error handling
            store_analysis_result(analysis["id"], "output", f"Generated {analysis['settings']['num_variants']} variant networks from the fitted measure.")
            store_analysis_result(analysis["id"], "figures", figs)
            store_analysis_result(analysis["id"], "graphs", graphs)
            store_analysis_result(analysis["id"], "timestamp", datetime.datetime.now().isoformat())
        else:
            st.error("Failed to generate variant figures")

def render_analysis(analysis, performance_mode):
    """Main function to render an analysis in either settings or results mode"""
    st.write(f"## {analysis['type'].replace('_', ' ').title()}")

    if analysis["type"] == "find_p_l":
        st.markdown("""
            Direct parameter optimization for a specific matrix dimension (m) and recursion depth (k).
            This is the most straightforward approach to fitting your network.
        """)
    elif analysis["type"] == "grid_search":
        st.markdown("""
            Grid search systematically explores multiple combinations of matrix dimensions (m) and recursive levels (k) 
            to find the optimal network representation.
            """)
    elif analysis["type"] == "plot_results":
        st.markdown("""
            Visualize and compare fitted results with your original network. This analysis helps you understand 
            how well your fitted parameters capture different network properties.
            """)
    elif analysis["type"] == "bootstrap_confidence":
        st.markdown("""
            This analysis estimates the uncertainty in fitted parameters using bootstrap resampling.
            It helps understand how reliable your fitted parameters are and provides confidence intervals.
        """)
    elif analysis["type"] == "generate_variants":
        st.markdown("""
            Creates new networks using your fitted parameters. These variants share statistical properties 
            with your original network while having different specific structures.
        """)
    
    mode = get_analysis_mode(analysis["id"])
    
    if mode == "settings":
        if analysis["type"] == "find_p_l":
            render_find_pl_settings(analysis)
        elif analysis["type"] == "grid_search":
            render_grid_search_settings(analysis)
        elif analysis["type"] == "plot_results":
            render_plot_results_settings(analysis)
        elif analysis["type"] == "bootstrap_confidence":
            render_bootstrap_confidence_settings(analysis)
        elif analysis["type"] == "generate_variants":
            render_generate_variants_settings(analysis)
        
    else:  # mode == "results"
        if analysis["type"] == "find_p_l":
            # Show existing output if it exists
            if output := get_analysis_result(analysis["id"], "output"):
                st.code(output, language="plaintext")
            # If we just switched to results mode, run the analysis
            if not get_analysis_result(analysis["id"], "output"):
                run_find_pl_analysis(analysis)
            render_find_pl_results(analysis, performance_mode)
        elif analysis["type"] == "grid_search":
            if output := get_analysis_result(analysis["id"], "output"):
                st.code(output, language="plaintext")
            if not get_analysis_result(analysis["id"], "output"):
                run_grid_search_analysis(analysis)
            render_grid_search_results(analysis, performance_mode)
        elif analysis["type"] == "plot_results":
            if not get_analysis_result(analysis["id"], "output"):
                run_plot_results_analysis(analysis)
            render_plot_results_results(analysis, performance_mode)
        elif analysis["type"] == "bootstrap_confidence":
            if output := get_analysis_result(analysis["id"], "output"):
                st.code(output, language="plaintext")
            if not get_analysis_result(analysis["id"], "output"):
                run_bootstrap_confidence(analysis)
            render_bootstrap_confidence_results(analysis, performance_mode)
        elif analysis["type"] == "generate_variants":
            if output := get_analysis_result(analysis["id"], "output"):
                st.code(output, language="plaintext")
            if not get_analysis_result(analysis["id"], "output"):
                run_generate_variants_analysis(analysis)
            render_generate_variants_results(analysis, performance_mode)

    # Create columns for the buttons
    col1, col2 = st.columns([1, 1])

    def reset_analysis(analysis_id):
        """Reset analysis to initial state"""
        # Modify the analysis directly in the session state's analyses list
        for i, analysis in enumerate(st.session_state.analyses):
            if analysis["id"] == analysis_id:
                st.session_state.analyses[i] = {
                    "id": analysis_id,
                    "type": None,
                    "settings": {},
                    "mfnf": copy.deepcopy(st.session_state["mfnf"]),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                switch_to_settings_mode(analysis_id)
                # Clean up any stored results
                if analysis_id in st.session_state.analysis_results:
                    del st.session_state.analysis_results[analysis_id]
                break

    def back_to_settings(analysis_id):
        """Return to analysis settings while preserving the type"""
        # Modify the analysis directly in the session state's analyses list
        for i, analysis in enumerate(st.session_state.analyses):
            if analysis["id"] == analysis_id:
                analysis_type = analysis["type"]  # preserve the type
                st.session_state.analyses[i] = {
                    "id": analysis_id,
                    "type": analysis_type,
                    "settings": {},
                    "mfnf": copy.deepcopy(st.session_state["mfnf"]) if analysis_type in ["find_p_l", "grid_search", "bootstrap_confidence"] else analysis["mfnf"],
                    "timestamp": datetime.datetime.now().isoformat()
                }
                switch_to_settings_mode(analysis_id)
                # Clean up stored results
                if analysis_id in st.session_state.analysis_results:
                    del st.session_state.analysis_results[analysis_id]
                break

    with col1:
        st.button("Reset Analysis", 
                  key=f"reset_{analysis['id']}_{analysis['type']}", 
                  on_click=reset_analysis,
                  args=(analysis['id'],),
                  help="Completely reset this analysis slot. You'll need to choose a new analysis type.")

    if mode == "results":
        with col2:
            st.button("Back to Settings", 
                      key=f"back_{analysis['id']}_{analysis['type']}", 
                      on_click=back_to_settings,
                      args=(analysis['id'],),
                      help="Return to the analysis settings.")

#############################################################
# S T E P :  4
#############################################################
def step_4():
    st.header("Step 4: Analysis")
    if SAVE_BUTTON:
        save_button()
    st.markdown("""
        ### Analysis Overview
        In this step, you can run various types of analyses using the initialized Multifractal Network Fitter (MFNF).
        Each analysis is preserved and can be referenced by later analyses.
    
        #### Available Analysis Types:
        - **Grid Search**: Systematically explore different matrix dimensions and recursive levels
        - **Find P and l**: Direct optimization for specific parameters
        - **Plot Results**: Visualize and compare fitted results with original network
        - **Bootstrap Confidence**: Estimate uncertainty in fitted parameters
        - **Generate Variants**: Create similar networks using fitted parameters
    """)

    if "mfnf" not in st.session_state:
        st.error("You need to initialize the Multifractal Network Fitter (MFNF) in Step 3 before proceeding.")
        return

    # Add performance mode checkbox
    performance_mode = st.checkbox(
        "⚡️ Performance Mode: Limit the visibility of plots (recommended for smoother experience)",
        value=True,  # Default to performance mode
        help="When enabled, plots are hidden by default and can be shown on demand. This significantly improves app responsiveness when working with multiple analyses."
    )
    
    init_analysis_state()
    
    def new_analysis():
        analysis_id = len(st.session_state.analyses) + 1
        st.session_state.analyses.append({
            "id": analysis_id,
            "type": None,
            "settings": {},
            "mfnf": copy.deepcopy(st.session_state["mfnf"]),
            "timestamp": datetime.datetime.now().isoformat()
        })
    st.button(
        "New Analysis",
        key="new_analysis_button",
        on_click=new_analysis,
        help="Create a new analysis slot. You can run multiple analyses in parallel to compare different measures."
    )

    # Define the mapping between display names and internal types
    ANALYSIS_TYPES = {
        "Grid Search": "grid_search",
        "Find P and l": "find_p_l",
        "Plot Results": "plot_results",
        "Bootstrap Confidence": "bootstrap_confidence",
        "Generate Variants": "generate_variants"
    }

    for analysis in st.session_state.analyses:
        with st.expander(f"Analysis {analysis['id']} - {analysis['type'] or 'New'}", expanded=True):
            if analysis["type"] is None:
                st.markdown("### Choose Analysis Type")

                # Check if we already have a plot-type analysis
                #has_plot_analysis = any(
                #    a["type"] in ["plot_results", "generate_variants"] 
                #    for a in st.session_state.analyses
                #)
                
                # Previous analysis results needed for these types
                needs_previous = any(a["type"] in ["grid_search", "find_p_l"] and 
                                   get_analysis_result(a["id"], "P") is not None 
                                   for a in st.session_state.analyses)
                
                # Define available choices based on whether we have previous results
                available_types = ["Find P and l", "Grid Search"]
                if needs_previous:
                    # In performance mode, only show plot types if none exist
                    #if performance_mode and has_plot_analysis:
                    #    available_types.extend(["Bootstrap Confidence"])
                    #    st.warning("⚡️ Performance Mode: Delete existing plot/visualization analysis to create a new one")  
                    #else:
                    #    available_types.extend(["Plot Results", "Bootstrap Confidence", "Generate Variants"])
                    available_types.extend(["Plot Results", "Bootstrap Confidence", "Generate Variants"])
                
                # Use the stored selection
                selected_type = st.radio(
                    "Select analysis type:",
                    options=available_types,
                    index=None,
                    key=f"selection_radio_{analysis['id']}"
                )

                def change_type(type_selection, analysis_id):
                    # Find and modify the correct analysis in session state
                    for i, analysis in enumerate(st.session_state.analyses):
                        if analysis["id"] == analysis_id:
                            st.session_state.analyses[i]["type"] = ANALYSIS_TYPES[type_selection]
                            switch_to_settings_mode(analysis_id)
                            break
                
                st.button("Confirm Selection",
                          key=f"confirm_analysis_{analysis['id']}", 
                          on_click=change_type,
                          args=[selected_type, analysis["id"]])              

            else:
                render_analysis(analysis, performance_mode)
                
    if len(st.session_state.analyses) > 0:
        #has_plot_analysis = any(a["type"] in ["plot_results", "generate_variants"] for a in st.session_state.analyses)
        #if not performance_mode or not has_plot_analysis:
        #    st.button("New Analysis", key="new_analysis_button_bottom", on_click=new_analysis)
        #elif has_plot_analysis:
        #    st.info("⚡️ Performance Mode: Delete existing plot/visualization analysis to create a new one")
        st.button("New Analysis", key="new_analysis_button_bottom", on_click=new_analysis)
    
    # Navigation
    st.button("← Back", on_click=lambda: st.session_state.update({"current_step": 3}))


#####################################
# Multi-Page Navigation
#####################################
render_navigation_sidebar()
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1  # Default to Step 1

current_step = st.session_state["current_step"]

if current_step == 1:
    step_1()
elif current_step == 2:
    step_2()
elif current_step == 3:
    step_3()
elif current_step == 4:
    step_4()