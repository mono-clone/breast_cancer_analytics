#
# Building a Docker Image with
# the Latest Ubuntu Version and
# Basic Python Install
# 
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#

# latest Ubuntu version
FROM ubuntu:latest  
#FROM --platform=linux/amd64 ubuntu:latest

WORKDIR /home/breast-cancer-analytics

# setup Ubuntu env 
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y \
    gcc \
    git \
    screen \
    htop \
    vim \
    wget \
    bash 
RUN apt-get clean

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh\
    -O Miniconda.sh
RUN bash Miniconda.sh -b
RUN rm -rf Miniconda.sh 

# prepend the new path
ENV PATH /root/miniconda3/bin:$PATH

# install pip package
RUN pip install --upgrade pip
# update anaconda 
RUN conda update -n base -c defaults conda -y
# update conda packages
RUN conda update --all -y
# create conda virtual env
COPY conda_env.yml .
RUN conda env create -f=conda_env.yml
RUN rm -rf conda_env.yml
RUN conda config --add channels conda-forge
RUN conda init && echo "conda activate breast-cancer-analytics" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV breast-cancer-analytics && PATH /root/conda/envs/breast-cancer-analytics/bin:$PATH

RUN echo 'Conda env is built. please relunch this terminal by using this command "docker restart <container name>"'
RUN echo 'You can check <container name> by using "docker ps"'
