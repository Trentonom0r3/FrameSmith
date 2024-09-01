cmake_minimum_required(VERSION 3.18)
project(RifeTensorRT)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-w)
# Set paths to your dependencies

if(WIN32)
    set(Torch_DIR "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cu121/libtorch/share/cmake/Torch" CACHE PATH "Path to Torch directory")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" CACHE PATH "Path to CUDA Toolkit directory")
    set(TENSORRT_DIR "C:/Users/tjerf/Downloads/TensorRT-10.3.0.26.Windows.win10.cuda-12.5/TensorRT-10.3.0.26" CACHE PATH "Path to TensorRT directory")
    set(VCPKG_DIR "C:/Users/tjerf/vcpkg/installed/x64-windows" CACHE PATH "Path to VCPKG_DIR installation directory")

    # Add vcpkg to CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${Torch_DIR};${TENSORRT_DIR};${VCPKG_DIR}")
endif()

# Platform-specific options
if(WIN32)
    set(LIB_SUFFIX ".lib")
    set(DLL_SUFFIX ".dll")
elseif(UNIX)
    set(LIB_SUFFIX ".so")
    set(DLL_SUFFIX ".so")
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Find Protobuf
find_package(protobuf CONFIG)
if(protobuf_FOUND)
    set(PROTOBUF_FOUND ${protobuf_FOUND})
    set(PROTOBUF_VERSION ${protobuf_VERSION})
else()
    # fallback to system
    find_package(protobuf)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    set(PROTOBUF_FOUND ${Protobuf_FOUND})
    set(PROTOBUF_VERSION ${Protobuf_VERSION})
endif()

# Find Torch
find_package(Torch REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include/ src/ ${TENSORRT_DIR}/include)

# Source files location
set(SOURCE_FILES src/main.cpp src/RifeTensorRT.cpp)

# Create the executable
add_executable(RifeTensorRT ${SOURCE_FILES})

# Link libraries
if(WIN32)
    target_link_libraries(RifeTensorRT 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        protobuf::libprotobuf
        protobuf::libprotobuf-lite
        protobuf::libprotoc
        ${TENSORRT_DIR}/lib/nvinfer_10${LIB_SUFFIX}  # Link TensorRT library
        ${TENSORRT_DIR}/lib/nvinfer_plugin_10${LIB_SUFFIX}  # Link TensorRT plugin library
        ${TENSORRT_DIR}/lib/nvonnxparser_10${LIB_SUFFIX}
    )
elseif(UNIX)
    target_link_libraries(RifeTensorRT 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        protobuf::libprotobuf
        ${TENSORRT_DIR}/lib/libnvinfer${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvinfer_plugin${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvonnxparser${LIB_SUFFIX}
    )
endif()

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
        CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89"  # Set according to your GPU architecture
    )
endif()

# Link Torch directly for Windows
if(WIN32)
  target_link_libraries(RifeTensorRT "${TORCH_LIBRARIES}")
endif()

# Link curl
if(WIN32)
    target_link_libraries(RifeTensorRT "C:/Users/tjerf/vcpkg/installed/x64-windows/lib/libcurl${LIB_SUFFIX}")
elseif(UNIX)
    find_package(CURL REQUIRED)
    target_link_libraries(RifeTensorRT ${CURL_LIBRARIES})
endif()

# Set RPATH for Linux or MacOS if needed
if(UNIX)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

# Enable verbose output for troubleshooting
set(CMAKE_VERBOSE_MAKEFILE ON)
