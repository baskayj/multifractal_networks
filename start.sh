#!/bin/bash

# Start either Jupyter Lab or Streamlit based on argument
if [ "$1" = "jupyter" ]; then
    jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''
else
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
fi