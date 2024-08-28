cmake_minimum_required(VERSION 3.18)
project(RifeTensorRT)

set(CMAKE_CXX_STANDARD 17)

# Set paths to your dependencies
set(Torch_DIR "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cu121/libtorch/share/cmake/Torch")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
set(TENSORRT_DIR "C:/Users/tjerf/Downloads/TensorRT-10.3.0.26.Windows.win10.cuda-12.5/TensorRT-10.3.0.26")

# Add vcpkg to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "C:/Users/tjerf/vcpkg/installed/x64-windows;${Torch_DIR};${TENSORRT_DIR}")

# Find CUDA
find_package(CUDA REQUIRED)

# Find Protobuf
find_package(protobuf CONFIG REQUIRED)

# Find Torch
find_package(Torch REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include/ src/ ${TENSORRT_DIR}/include)
#print the opencv include directories
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
# Source files location
set(SOURCE_FILES src/main.cpp src/RifeTensorRT.cpp)

# Create the executable
add_executable(RifeTensorRT ${SOURCE_FILES})

# Link libraries
target_link_libraries(RifeTensorRT 
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    protobuf::libprotobuf
    protobuf::libprotobuf-lite
    protobuf::libprotoc
    ${TENSORRT_DIR}/lib/nvinfer_10.lib  # Link TensorRT library
    ${TENSORRT_DIR}/lib/nvinfer_plugin_10.lib  # Link TensorRT plugin library
)

# If Torch uses C++17, ensure compatibility
set_property(TARGET RifeTensorRT PROPERTY CXX_STANDARD 17)

# Required by PyTorch
if (MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc ${CMAKE_CXX_FLAGS}")
endif()

# Add CUDA flags if necessary
if(CUDA_FOUND)
    set_target_properties(RifeTensorRT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;60;61;70;75;86"  # Set according to your GPU architecture
    )
endif()

# Link Torch directly for Windows
if(WIN32)
  target_link_libraries(RifeTensorRT "${TORCH_LIBRARIES}")
endif()

#link curl
target_link_libraries(RifeTensorRT "C:/Users/tjerf/vcpkg/installed/x64-windows/lib/libcurl.lib")
#link tensorrt libs
target_link_libraries(RifeTensorRT "${TENSORRT_DIR}/lib/nvonnxparser_10.lib")
# Set RPATH for Linux or MacOS if needed
set(CMAKE_BUILD_RPATH "$ORIGIN")

# Enable verbose output for troubleshooting
set(CMAKE_VERBOSE_MAKEFILE ON)
