cmake_minimum_required(VERSION 3.18)
project(RifeTensorRT LANGUAGES CXX CUDA)
add_definitions(-DTORCH_USE_CUDA_DSA=1)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-w)
# Source files
set(SOURCE_FILES src/main.cpp src/RifeTensorRT.cpp src/Writer.cpp src/conversion.cu)

# Create the executable
add_executable(RifeTensorRT ${SOURCE_FILES})

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd")  # Use dynamic debug runtime for Debug builds
    set(CMAKE_CXX_FLAGS_RELEASE "/MD") # Use dynamic release runtime for Release builds
endif()


# Enable verbose output for troubleshooting
set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/tjerf/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    set(Protobuf_PROTOC_EXECUTABLE "C:/Users/tjerf/vcpkg/installed/x64-windows/tools/protobuf/protoc.exe")

    set(Torch_DIR "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cu121/libtorch/share/cmake/Torch" CACHE PATH "Path to Torch directory")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" CACHE PATH "Path to CUDA Toolkit directory")
    set(TENSORRT_DIR "C:/Users/tjerf/Downloads/TensorRT-10.3.0.26.Windows.win10.cuda-12.5/TensorRT-10.3.0.26" CACHE PATH "Path to TensorRT directory")
    set(VCPKG_DIR "C:/Users/tjerf/vcpkg/installed/x64-windows" CACHE PATH "Path to VCPKG_DIR installation directory")

    # Add vcpkg and other paths
    list(APPEND CMAKE_MODULE_PATH "${VCPKG_DIR}/share/ffmpeg" "${VCPKG_DIR}/share/protobuf")
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

find_package(Torch REQUIRED)
#find_package(OpenCV REQUIRED)  # Add this line to find OpenCV
# Find CURL
find_package(CURL REQUIRED)

# Link curl
if(WIN32)
    target_link_libraries(RifeTensorRT ${CURL_LIBRARIES})
elseif(UNIX)
    target_link_libraries(RifeTensorRT ${CURL_LIBRARIES})
endif()

include_directories(${FFMPEG_DIR}/include)
link_directories(${FFMPEG_DIR}/lib)

find_library(AVCODEC_LIBRARY avcodec HINTS "C:/Users/tjerf/vcpkg/installed/x64-windows/lib")
find_library(AVFORMAT_LIBRARY avformat HINTS "C:/Users/tjerf/vcpkg/installed/x64-windows/lib")
find_library(AVUTIL_LIBRARY avutil HINTS "C:/Users/tjerf/vcpkg/installed/x64-windows/lib")
find_library(SWSCALE_LIBRARY swscale HINTS "C:/Users/tjerf/vcpkg/installed/x64-windows/lib")
find_library(SWRESAMPLE_LIBRARY swresample HINTS "C:/Users/tjerf/vcpkg/installed/x64-windows/lib")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} include/ src/ concurrentqueue/ ${TENSORRT_DIR}/include ${VCPKG_DIR}/include ${OpenCV_INCLUDE_DIRS})

# Link libraries
if(WIN32)
    target_link_libraries(RifeTensorRT 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWSCALE_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
        # ${OpenCV_LIBS}  # Link OpenCV libraries
        CURL::libcurl
        protobuf::libprotobuf
        "${TENSORRT_DIR}/lib/nvinfer_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_plugin_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvonnxparser_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_dispatch_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_lean_10${LIB_SUFFIX}"

        # NPP libraries
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppisu.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppitc.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/npps.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppc.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppial.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppicc.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppidei.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppif.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppig.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppim.lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/nppist.lib"
    )
elseif(UNIX)
    target_link_libraries(RifeTensorRT 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        avcodec avformat avutil swscale
        protobuf::libprotobuf
        ${TENSORRT_DIR}/lib/libnvinfer${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvinfer_plugin${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvonnxparser${LIB_SUFFIX}
        #${OpenCV_LIBS}  # Link OpenCV libraries for UNIX
    )
endif()

# Set C++ standard
set_property(TARGET RifeTensorRT PROPERTY CXX_STANDARD 17)
set_property(TARGET RifeTensorRT PROPERTY CUDA_STANDARD 17)

# Required by PyTorch
if (MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# Add CUDA flags if necessary
if(CUDA_FOUND)
    set_target_properties(RifeTensorRT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89"
    )
endif()

