
# RifeTensorRT

## Overview

RifeTensorRT is a C++ application that uses TensorRT for fast, high-performance video frame interpolation. The project is cross-platform and can be built on both Windows and Linux using CMake.

## Requirements

### General Dependencies
- **CMake 3.18+**
- **C++17** compatible compiler
- **CUDA Toolkit** (Tested with CUDA 12.1)
- **TensorRT** (Tested with TensorRT 10.3)
- **libtorch** (Tested with libtorch 2.4.0+cu121)
- **OpenCV** (Tested with OpenCV 4.x)
- **Protobuf** (libprotobuf, libprotobuf-lite, protoc)
- **libcurl** (optional, for downloading models)

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
- **OpenCV**: Install via the package manager:
  ```bash
  sudo apt-get install libopencv-dev
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

## Model Names

You need to specify a model name when running the application. Available models include: (Note that not all are set up to work with the .exe yet)

- `shufflespan`
- `shufflespan-directml`
- `shufflespan-tensorrt`
- `aniscale2`
- `aniscale2-directml`
- `aniscale2-tensorrt`
- `open-proteus`
- `compact`
- `ultracompact`
- `superultracompact`
- `span`
- `shufflecugan`
- `segment`
- `segment-tensorrt`
- `segment-directml`
- `scunet`
- `dpir`
- `real-plksr`
- `nafnet`
- `rife`
- `rife4.6`
- `rife4.15-lite`
- `rife4.16-lite`
- `rife4.17`
- `rife4.18`
- `rife4.20`
- `rife4.21`
- `rife4.22`
- `rife4.22-lite`
- `shufflecugan-directml`
- `open-proteus-directml`
- `compact-directml`
- `ultracompact-directml`
- `superultracompact-directml`
- `span-directml`
- `open-proteus-tensorrt`
- `shufflecugan-tensorrt`
- `compact-tensorrt`
- `ultracompact-tensorrt`
- `superultracompact-tensorrt`
- `span-tensorrt`
- `rife4.6-tensorrt`
- `rife4.15-lite-tensorrt`
- `rife4.17-tensorrt`
- `rife4.18-tensorrt`
- `rife4.20-tensorrt`
- `rife4.21-tensorrt`
- `rife4.22-tensorrt`
- `rife4.22-lite-tensorrt`
- `rife-v4.6-ncnn`
- `rife-v4.15-lite-ncnn`
- `rife-v4.16-lite-ncnn`
- `rife-v4.17-ncnn`
- `rife-v4.18-ncnn`
- `span-ncnn`
- `shufflecugan-ncnn`
- `small_v2`
- `base_v2`
- `large_v2`
- `small_v2-directml`
- `base_v2-directml`
- `large_v2-directml`
- `small_v2-tensorrt`
- `base_v2-tensorrt`
- `large_v2-tensorrt`
- `maxxvit-tensorrt`
- `maxxvit-directml`
- `shift_lpips-tensorrt`
- `shift_lpips-directml`
- `differential-tensorrt`
- `rife4.19`
- `rife4.19-lite`
- `rife4.23`
- `rife4.23-lite`
- `rife4.24`
- `rife4.24-lite`
- `rife-v4.19-ncnn`
- `rife-v4.19-lite-ncnn`
- `rife-v4.23-ncnn`
- `rife-v4.23-lite-ncnn`
- `rife-v4.24-ncnn`
- `rife-v4.24-lite-ncnn`


Refer to the source code or documentation for a full list of supported models.

## Troubleshooting

### Common Issues
- **Missing DLLs/SOs**: Ensure all necessary dependencies are in the same directory as the executable.
- **CMake Errors**: Check that all paths to dependencies are correctly set in the CMake configuration.
- **Runtime Errors**: Verify that your CUDA and TensorRT installations are correctly configured and that your GPU is supported.

### Additional Help
If you encounter any issues, consider checking the CMake logs (`CMakeError.log` and `CMakeOutput.log`) in the `build` directory, or consult the official documentation for each dependency.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
