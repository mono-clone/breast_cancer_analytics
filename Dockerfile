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
RUN /install.sh
# prepend the new path
ENV PATH /root/miniconda3/bin:$PATH
# install pip package
RUN pip3 install --upgrade pip
