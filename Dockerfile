<<<<<<< HEAD
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
#FROM ubuntu:latest
FROM --platform=linux/amd64 ubuntu:latest

WORKDIR /home/breast-cancer-analytics

COPY ./install.sh ./
RUN chmod u+x ./install.sh &&\ 
    ./install.sh

ENV PATH /root/miniconda3/bin:$PATH
