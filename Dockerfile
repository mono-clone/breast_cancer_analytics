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
# FROM ubuntu:latest  
FROM --platform=linux/amd64 ubuntu:latest

# add the bash script
ADD install.sh /
# change rights for the script
RUN chmod u+x /install.sh
# run the bash script
# took 50 minutes......(using m1 MacBookAir, 16GB RAM)
RUN /install.sh
# prepend the new path
ENV PATH /root/miniconda3/bin:$PATH

# update anaconda 
RUN conda update -n base conda -y
# update conda packages
RUN conda update --all -y
# install pip package
RUN pip3 install --upgrade pip
# create conda virtual env
RUN conda env create -f=conda_env.yml

RUN echo 'Conda env is built. please relunch this terminal by using this command "docker restart <container name>"'
RUN echo 'You can check <container name> by using this command "docker ps"'