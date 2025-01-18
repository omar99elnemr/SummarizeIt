#!/bin/bash

# Install Rust and Cargo
curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add Cargo to PATH
source $HOME/.cargo/env

# Continue with Streamlit installation
pip install -r requirements.txt
