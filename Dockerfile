FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER nejumi <dr_jingles@mac.com>

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y git build-essential cmake && \
    git --version

RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    vim \
    mercurial \
    subversion \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    gcc \
    g++

RUN apt update && \
    apt install unzip

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

##############################################################################
# TINI
##############################################################################

# Install tini
ENV TINI_VERSION v0.14.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
           
##############################################################################
# Miniconda python
##############################################################################
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    /bin/bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py37_4.8.3-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip

RUN apt-get update && \
    # Miniconda's build of gcc is way out of date; monkey-patch some linking problems that affect
    # packages like xgboost and Shapely
    rm /opt/conda/lib/libstdc++* && rm /opt/conda/lib/libgomp.* && \
    ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 /opt/conda/lib/libgomp.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6
    
RUN cd /usr/local/src && pip install scikit-learn tables
RUN cd /usr/local/src && conda install lxml h5py hdf5 html5lib beautifulsoup4

##############################################################################
# LightGBM-GPU
##############################################################################

RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
git clone -b v2.3.1 https://github.com/microsoft/LightGBM && \
cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \ 
    make OPENCL_HEADERS=/usr/local/cuda/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda/targets/x86_64-linux/lib

ENV PATH /usr/local/src/lightgbm/LightGBM:${PATH}

RUN /bin/bash -c "cd /usr/local/src/lightgbm/LightGBM/python-package && python setup.py install --precompile"

##############################################################################
# XGBoost-GPU
##############################################################################
RUN cd /usr/local/src && pip install xgboost

##############################################################################
# tensorflow
##############################################################################
RUN cd /usr/local/src && pip --no-cache-dir install -I -U tensorflow==2.2.0
RUN cd /usr/local/src && pip install keras

##############################################################################
# rapidsai
##############################################################################
RUN cd /usr/local/src && conda install -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.14 python=3.7 cudatoolkit=10.1
RUN conda install -y -c conda-forge ipywidgets && jupyter nbextension enable --py widgetsnbextension

##############################################################################
# xfeat
##############################################################################
RUN cd /usr/local/src && git clone --recursive https://github.com/pfnet-research/xfeat && \
cd xfeat && python setup.py install

##############################################################################
# other libraries
##############################################################################
RUN cd /usr/local/src && pip install albumentations seaborn pyarrow fastparquet catboost kaggle \
    category_encoders optuna opencv-python image-classifiers tsfresh librosa gsutil
RUN cd /usr/local/src && conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN cd /usr/local/src && pip install git+https://github.com/hyperopt/hyperopt.git
