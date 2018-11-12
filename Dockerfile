FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER nejumi <dr_jingles@mac.com>

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y git build-essential cmake && \
    git --version
           
##############################################################################
# anaconda python
##############################################################################
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.continuum.io/archive/Anaconda3-5.3.0-Linux-x86_64.sh && \
    /bin/bash Anaconda3-5.3.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda3-5.3.0-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip
RUN conda install python=3.6.6

RUN apt-get update && \
    # Anaconda's build of gcc is way out of date; monkey-patch some linking problems that affect
    # packages like xgboost and Shapely
    rm /opt/conda/lib/libstdc++* && rm /opt/conda/lib/libgomp.* && \
    ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 /opt/conda/lib/libgomp.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

##############################################################################
# XGBoost-GPU
##############################################################################
RUN cd /usr/local/src && git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && mkdir build && cd build && cmake .. -DPLUGIN_UPDATER_GPU=ON && make -j4 && \
    cd ../python-package && python3 setup.py install

##############################################################################
# keras
##############################################################################
RUN cd /usr/local/src && pip --no-cache-dir install -I -U tensorflow-gpu
RUN pip install keras

##############################################################################
# other libraries
##############################################################################
RUN cd /usr/local/src && pip install catboost gmail lightgbm kaggle umap-learn tqdm hdbscan #TO DO: LightGBM-GPU is to be installed.
RUN cd /usr/local/src && pip install torch torchvision
RUN cd /usr/local/src && pip install git+https://github.com/hyperopt/hyperopt.git
