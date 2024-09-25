FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

RUN apt update -y && apt-get install libcurl4-openssl-dev wget unzip git nasm python3 python3-pip pkg-config cmake libatomic1 -y
RUN apt purge ffmpeg -y
WORKDIR /tensorrt 

# torch
RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip -d /tensorrt/pytorch

# tensorrt
RUN wget "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz" -O /tmp/TensorRT.tar
RUN tar -xf /tmp/TensorRT.tar -C /usr/local/
RUN mv /usr/local/TensorRT-10.3.0.26 /usr/local/tensorrt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
RUN ldconfig

# g++13
RUN apt install build-essential manpages-dev software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt update -y && apt install gcc-13 g++-13 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13

# nv-codec-headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers && cd nv-codec-headers && make -j$(nproc) && make install

# ffmpeg
RUN git clone https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && git switch release/7.0 && \
  CFLAGS=-fPIC ./configure --enable-cuvid --enable-nvdec --enable-nvenc --disable-zlib --disable-libdav1d --enable-cuda --enable-nonfree --disable-shared \
    --enable-static --enable-gpl --enable-version3 --disable-programs --disable-doc --disable-avdevice --disable-swresample --disable-postproc --disable-avfilter \
    --disable-debug --enable-pic --extra-ldflags="-static" --extra-cflags="-march=native" && \
  make -j$(nproc) && make install -j$(nproc)
  
# compile
RUN git clone https://github.com/styler00dollar/FrameSmith
RUN cd FrameSmith/FrameSmith && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=/tensorrt/pytorch/libtorch/share/cmake/Torch \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DTENSORRT_DIR=/usr/local/tensorrt/ -DFFMPEG_DIR=/usr/local/bin && make -j$(nproc)
