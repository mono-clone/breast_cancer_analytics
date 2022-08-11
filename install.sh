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

