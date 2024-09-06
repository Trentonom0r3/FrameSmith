# RifeTensorRT

## Overview

RifeTensorRT is a high-performance C++ application for video frame interpolation using TensorRT and CUDA. The project leverages TensorRT for accelerated deep learning-based frame interpolation and utilizes FFmpeg for handling video input/output, including support for CUDA/NVENC acceleration. The application supports both Windows and Linux platforms and can be easily built using CMake and vcpkg for dependency management.

## Features

- **TensorRT Integration**: Uses TensorRT for accelerated deep learning-based frame interpolation.
- **Parallel Processing**: Supports asynchronous frame processing and writing using multiple CUDA streams for higher performance.
- **CUDA/NVENC Support**: Utilizes FFmpeg with CUDA/NVENC for fast video encoding and decoding.
- **Cross-Platform**: Can be built on both Windows and Linux using CMake.
- **Model Support**: Works with a wide range of interpolation models, including RIFE and ShuffleSpan, with options for TensorRT, DirectML, and NCNN backends.

## Requirements

### General Dependencies
- **CMake 3.18+**
- **C++17** compatible compiler
- **CUDA Toolkit** (Tested with CUDA 12.1)
- **TensorRT** (Tested with TensorRT 10.3)
- **libtorch** (Tested with libtorch 2.4.0+cu121)
- **FFmpeg** (with CUDA/NVENC support, compiled via vcpkg)
- **Protobuf** (libprotobuf, libprotobuf-lite, protoc)
- **libcurl** (for downloading models)

### Optional Dependencies
- **vcpkg**: For managing dependencies like `libcurl`, `FFmpeg`, and `Protobuf`.

## vcpkg Dependencies

The project relies on the following specific packages from vcpkg for the x64 Windows build:

- **FFmpeg (7.0.2)**: Includes the `avcodec`, `avdevice`, `avformat`, `avfilter`, `swresample`, `swscale`, `nvcodec` features.
  - `ffmpeg[avcodec]`
  - `ffmpeg[avdevice]`
  - `ffmpeg[avfilter]`
  - `ffmpeg[avformat]`
  - `ffmpeg[swresample]`
  - `ffmpeg[swscale]`
  - `ffmpeg[nvcodec]`: CUDA/NVENC support for video acceleration.

- **libcurl (8.9.1)**: Provides HTTP/HTTPS support for downloading models.
  - `curl[ssl]`, `curl[schannel]`, `curl[sspi]`.

- **Protobuf (3.21.12)**: Required for handling serialized data.
  - `protobuf`.

- **CUDA (10.1+)**: Required for GPU acceleration.

### Installing vcpkg Dependencies

To install the necessary dependencies via vcpkg, use the following commands:

```bash
vcpkg install ffmpeg[avcodec,avdevice,avfilter,avformat,swresample,swscale,nvcodec]:x64-windows
vcpkg install curl[ssl,schannel,sspi]:x64-windows
vcpkg install protobuf:x64-windows
```

Ensure that the vcpkg toolchain file is correctly set in your CMake configuration to use the installed packages.	

```bash
set(CMAKE_TOOLCHAIN_FILE "C:/Users/tjerf/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
```

## Building on Windows

### Prerequisites
- **Visual Studio 2019/2022** with C++ and CMake support.
- **vcpkg** (optional, for managing dependencies like `libcurl`).
- **CUDA Toolkit**: Installed in `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1`.
- **TensorRT**: Installed in a directory of your choice.
- **libtorch**: Download and extract the Windows package from the [PyTorch website](https://pytorch.org/get-started/locally/).

### Steps

1. **Open the Project in Visual Studio**:
   - Open the project folder in Visual Studio. Ensure that CMake is detected and configured automatically by Visual Studio.

2. **Configure the Project**:
   - Set the necessary paths in your CMake configuration, either via CMakeSettings.json or by manually editing the CMake configuration.

3. **Build the Project**:
   - Use Visual Studio to build the project by selecting the `Build` option. The output binary will be placed in the `out/build/x64-debug` directory.

4. **Running the Application**:
   - The executable will be located in your build directory. You can run it directly or through Visual Studio.

5. **Ensure DLLs are in the Same Directory**:
   - The build process should automatically copy the necessary DLLs to the output directory. If any DLLs are missing, make sure they are in the same directory as the `.exe`.

## Building on Linux

### Prerequisites
- **CMake 3.18+**
- **GCC or Clang** (C++17 compatible)
- **CUDA Toolkit**: Install using the package manager or from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
- **TensorRT**: Download and install from [NVIDIA's website](https://developer.nvidia.com/tensorrt).
- **libtorch**: Download and extract the Linux package from the [PyTorch website](https://pytorch.org/get-started/locally/).
- **FFmpeg**: Ensure FFmpeg is compiled with CUDA/NVENC support or install it via the package manager:

  **FFMPEG**
  ```bash
  sudo apt-get install ffmpeg
  ```
- **Protobuf**: Install via the package manager:
  ```bash
  sudo apt-get install libprotobuf-dev protobuf-compiler
  ```
- **libcurl**: Install via the package manager:
  ```bash
  sudo apt-get install libcurl4-openssl-dev
  ```

### Steps

1. **Clone the Project**:
   ```bash
   git clone https://github.com/your-repository/RifeTensorRT.git
   cd RifeTensorRT
   ```

2. **Create a Build Directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the Project with CMake**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release             -DTorch_DIR=/path/to/libtorch/share/cmake/Torch             -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda             -DTENSORRT_DIR=/path/to/TensorRT
   ```

4. **Build the Project**:
   ```bash
   make -j$(nproc)
   ```

5. **Running the Application**:
   - The executable will be located in the `build` directory. You can run it using:
   ```bash
   ./RifeTensorRT /path/to/input_video.mp4 /path/to/output_video.mp4 /model_name
   ```

6. **Ensure SO Files are in the Same Directory**:
   - The build process should automatically copy the necessary `.so` files to the output directory. If any `.so` files are missing, ensure they are in the same directory as the executable or set the `LD_LIBRARY_PATH` accordingly.

## Dockerfile

1. **Build the docker**:
   ```bash
   DOCKER_BUILDKIT=1 sudo docker build -t rife_cpp:latest . 
   ```
   
2. **Run the docker**:
   ```bash
   sudo docker run --privileged --gpus all -it --rm -v /path/to/RifeTensorRT/:/tensorrt/mount rife_cpp:latest
   ```
   
## Model Names

You need to specify a model name when running the application. Available models include: (Note that not all are set up to work with the .exe yet)


- `rife4.6-tensorrt`
- `rife4.15-lite-tensorrt`
- `rife4.17-tensorrt`
- `rife4.18-tensorrt`
- `rife4.20-tensorrt`
- `rife4.21-tensorrt`
- `rife4.22-tensorrt`
- `rife4.22-lite-tensorrt`

Refer to the source code or documentation for a full list of supported models.

## Usage
```bash
./RifeTensorRT /path/to/input_video.mp4 /path/to/output_video.mp4 model_name interpolation_factor [--benchmark]
```

## Troubleshooting

### Common Issues
- **Missing DLLs/SOs**: Ensure all necessary dependencies are in the same directory as the executable.
- **CMake Errors**: Check that all paths to dependencies are correctly set in the CMake configuration.
- **Runtime Errors**: Verify that your CUDA and TensorRT installations are correctly configured and that your GPU is supported.

### Additional Help
If you encounter any issues, consider checking the CMake logs (`CMakeError.log` and `CMakeOutput.log`) in the `build` directory, or consult the official documentation for each dependency.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
