#FROM ubuntu:latest
FROM --platform=linux/amd64 ubuntu:latest

WORKDIR /home/breast-cancer-analytics

COPY ./install.sh ./
RUN chmod u+x ./install.sh &&\ 
    ./install.sh

ENV PATH /root/miniconda3/bin:$PATH