#!/bin/bash

# This script initializes the development environment inside the dev container.
echo "Initializing development environment..."

# Update package lists
pip install --upgrade pip \
    && pip install -r requirements.txt 
echo "Python packages installed."

  
wget -qO- https://quarto.org/download/latest/quarto-linux-amd64.deb -O /tmp/quarto.deb
sudo dpkg -i /tmp/quarto.deb
rm /tmp/quarto.deb
QUARTO_PYTHON="/usr/local/python/current/bin/python3"
echo "Quarto installed."