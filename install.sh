#!/bin/bash
#
# Script to Install
# Linux System Tools and Basic Python Components
# as well as to
# Start Jupyter Lab Server
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
# GENERAL LINUX
apt-get update     # updates the package index cache
apt-get upgrade -y # updates packages
# install system tools
apt-get install -y gcc git htop         # system tools
apt-get install -y screen htop vim wget # system tools
apt-get upgrade -y bash                 # upgrades bash if necessary
apt-get clean                           # cleans up the package index cache

# INSTALLING MINICONDA
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O Miniconda.sh
bash Miniconda.sh -b # installs Miniconda
rm -rf Miniconda.sh  # removes the installer
# prepends the new path for current session
export PATH="/root/miniconda3/bin:$PATH"
# prepends the new path in the shell configuration
cat >>~/.profile <<EOF
export PATH="/root/miniconda3/bin:$PATH"
EOF

# INSTALLING PYTHON LIBRARIES
conda install -y jupyter      # interactive data analytics in the browser
conda install -y jupyterlab   # Jupyter Lab environment
conda install -y numpy        #  numerical computing package
conda install -y pytables     # wrapper for HDF5 binary storage
conda install -y pandas       #  data analysis package
conda install -y matplotlib   # standard plotting library
conda install -y seaborn      # statistical plotting library
conda install -y scikit-learn # machine learning library
conda install -c conda-forge tqdm -y
conda install -c conda-forge python-graphviz -y
conda install -c conda-forge pydotplus -y
conda install -c conda-forge six -y
conda install -c conda-forge sweetviz -y
conda install -c conda-forge imbalanced-learn -y
conda install -c conda-forge black -y
conda install -c anaconda flake8 -y

pip install --upgrade pip # upgrading the package manager
pip install seaborn-analyzer

# COPYING FILES AND CREATING DIRECTORIES
mkdir /root/.jupyter
mkdir /root/.jupyter/custom
wget http://hilpisch.com/custom.css
mv custom.css /root/.jupyter/custom
jupyter notebook --generate-config
mkdir /root/notebook
cd /root/notebook
