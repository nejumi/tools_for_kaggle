FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
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
# anaconda python
##############################################################################
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    /bin/bash Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda3-2019.10-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip

RUN apt-get update && \
    # Anaconda's build of gcc is way out of date; monkey-patch some linking problems that affect
    # packages like xgboost and Shapely
    rm /opt/conda/lib/libstdc++* && rm /opt/conda/lib/libgomp.* && \
    ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 /opt/conda/lib/libgomp.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

##############################################################################
# LightGBM-GPU
##############################################################################

RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
git clone --recursive https://github.com/Microsoft/LightGBM && \
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
# keras
##############################################################################
RUN cd /usr/local/src && pip --no-cache-dir install -I -U tensorflow-gpu==1.13.1
RUN pip install keras

##############################################################################
# other libraries
##############################################################################
RUN cd /usr/local/src && pip install albumentations pyarrow fastparquet catboost kaggle umap-learn tqdm category_encoders optuna cupy opencv-python image-classifiers #TO DO: catboost-GPU is to be installed.
RUN cd /usr/local/src && pip install torch torchvision
RUN cd /usr/local/src && pip install git+https://github.com/hyperopt/hyperopt.git
