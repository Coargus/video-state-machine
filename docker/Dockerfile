ARG BASE_IMG=ubuntu:20.04

FROM $BASE_IMG as dev

USER root
#----------------------------------------------#
# Install Default System Dependencies for Ubuntu
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg git vim libglew-dev libglfw3-dev libglm-dev \
    tmux wget unzip unrar curl \ 
    x11-xserver-utils xvfb \
    make cmake 

#----------------------------------------------#
# Install project dependent system dependencies

ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential libboost-all-dev libcln-dev \
    libgmp-dev libginac-dev automake libglpk-dev \
    libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev \
    curl software-properties-common 

# Set up Python 3.10
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python
# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py
RUN rm get-pip.py
# Update PATH to include pip
ENV PATH="/usr/local/bin:${PATH}"

# >>> FOR GROUNDING DINO >>>
# RUN apt-get install -y \
#     rustc cargo \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# <<< FOR GROUNDING DINO <<<
    
#---------------------DO NOT CHANGE BELOW---------------------#
# STORMPY PACKAGE INSTALLATION
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt install -y g++-11 \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-11
# GIT CLONE
WORKDIR /opt
RUN git clone -b stable https://github.com/moves-rwth/carl-storm
RUN git clone -b stable https://github.com/moves-rwth/storm.git
RUN git clone -b stable https://github.com/moves-rwth/stormpy.git
# INSTALL CARL STORM
WORKDIR /opt/carl-storm
RUN mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CLN_NUMBERS=ON -DUSE_GINAC=ON -DTHREAD_SAFE=ON 
WORKDIR /opt/carl-storm/build
RUN make lib_carl -j 1
# INSTALL STORM
WORKDIR /opt/storm
RUN mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DSTORM_DEVELOPER=OFF -DSTORM_LOG_DISABLE_DEBUG=ON -DSTORM_PORTABLE=ON -DSTORM_USE_SPOT_SHIPPED=ON 
WORKDIR /opt/storm/build
RUN make resources -j 1 && make storm -j 1 && make binaries -j 1
RUN export PATH=$PATH:/opt/storm/build/bin
# INSTALL STORMPY
WORKDIR /opt/stormpy
RUN python3 -m pip install -ve .
#---------------------DO NOT CHANGE ABOVE---------------------#

# PYTHON TORCH
RUN python3 -m pip install --upgrade pip build
# You may need to update PyTorch installation command here if needed for CUDA 12.2
#----------------------------------------------#
WORKDIR /opt/vsm