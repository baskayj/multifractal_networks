# Multifractal Networks

A Python framework for analyzing and modeling complex networks using multifractal measures. This tool allows you to:
- Fit multifractal measures to real-world networks
  ![Measure Example](measure_placeholder.png)
- Analyze network motifs and structural patterns
  ![Motifs Example](motifs_placeholder.png)
- Create statistically similar network variants
  ![Synthetic Network Example](synth_net_placeholder.png)
- Estimate uncertainty in fitted parameters
  ![Bootstrap Example](bootstrap_placeholder.png)

## Quick Start

The easiest way to get started is using our pre-built Docker container:

```bash
docker pull placeholder.fornow/multifractal-networks
docker run -p 8501:8501 -p 8888:8888 placeholder.fornow/multifractal-networks
```

By default, this will start the Streamlit interface at http://localhost:8501.

## Running the Container

The container supports two interfaces:

### Streamlit Interface (Recommended)

```bash
docker run -p 8501:8501 placeholder.fornow/multifractal-networks
```
Then open `http://localhost:8501` in your browser. The Streamlit interface provides:

- Step-by-step guidance through the analysis process
- Interactive visualization of results
- Easy parameter tuning and model fitting
- Real-time feedback on analysis progress

### Jupyter Lab Interface

```bash
docker run -p 8888:8888 placeholder.fornow/multifractal-networks ./start.sh jupyter
```
Then open `http://localhost:8888` in your browser. The Jupyter interface is useful for:

- Custom analysis scripts
- Direct access to the Python API
- Development and experimentation
- Advanced customization

## Building from Source

If you prefer to build the container yourself:

```bash
git clone https://github.com/username/multifractal-networks.git
cd multifractal-networks
docker build -t multifractal-networks .
```

## Features

- **Multifractal Measure Fitting**: Find optimal parameters to model real networks
- **Motif Analysis**: Analyze and count network substructures
- **Bootstrap Confidence**: Estimate uncertainty in fitted parameters
- **Network Generation**: Create synthetic networks with similar properties
- **Visualization**: Interactive plotting and network visualization tools
- **Scalable**: Handles networks with up to ~100,000 nodes
- **Extensible**: Easy to add custom motifs and analyses

## Citation

If you use this software in your research, please cite:
```bibtex
@article{placeholder2024,
  title={Multifractal Network Analysis Framework},
  author={Author1 and Author2},
  journal={Journal Name},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE.md for details.