FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update \
    && apt-get install -y \
	python3.7 \ 
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
	libjpeg8-dev \
        pkg-config \
        libswscale-dev \
        libtbb2 \
	libgtk2.0-dev \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
	libatlas-base-dev \
	gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy \ 
  && pip3 install --upgrade imutils

WORKDIR /

ARG OPENCV_VERSION="4.1.1"

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip 

# Get OpenCV contrib modules
RUN wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip opencv_contrib.zip

RUN mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DWITH_GTK=ON \
  -DINSTALL_PYTHON_EXAMPLES=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
  -DCMAKE_BUILD_TYPE=RELEASE \
  .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

WORKDIR /host/
