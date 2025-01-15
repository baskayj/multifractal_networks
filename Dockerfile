FROM ubuntu:24.04

# Set environment variables for non-interactive apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set environment variables for localization
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Set a working directory
WORKDIR /app

# Install necessary system packages and add the Skewed repository for Graph-tool
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    update-ca-certificates && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget dpkg apt-transport-https gnupg curl && \
    wget --no-check-certificate https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_noble.deb -O skewed-keyring.deb && \
    dpkg -i skewed-keyring.deb && \
    echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt noble main" > /etc/apt/sources.list.d/skewed.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools \
    python3-graph-tool python3-matplotlib python3-cairo \
    git htop openssl libxml2 bash vim less && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* skewed-keyring.deb

# Set Bash as the default shell
SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash

# Install Python packages inside the venv
RUN pip install --break-system-packages \
    jupyterlab jupyterlab-git ipywidgets \
    tqdm tqdm_joblib \
    numpy scipy pandas scikit-learn optuna \
    seaborn plotly \
    networkx \
    streamlit

# Copy the project files
COPY multifractal_networks/ /app/multifractal_networks/
COPY test_networks/ /app/test_networks/
COPY saved_analyses/ /app/saved_analyses/
COPY variant_networks/ /app/variant_networks/
COPY app.py /app/
COPY README.md /app/
COPY start.sh /app/

# Create necessary directories if they don't exist
RUN mkdir -p /app/test_networks /app/saved_analyses /app/variant_networks

# Set permissions
RUN chmod -R 755 /app && \
    chmod +x /app/start.sh

# Expose the ports for Jupyter Lab and Streamlit
EXPOSE 8888
EXPOSE 8501

# Default to running Streamlit
CMD ["/app/start.sh"]