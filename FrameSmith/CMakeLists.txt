cmake_minimum_required(VERSION 3.18)
project(FrameSmith LANGUAGES CXX CUDA)
add_definitions(-DTORCH_USE_CUDA_DSA=1)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-w)
# Source files
set(SOURCE_FILES src/main.cpp src/TrTBase.cpp src/UpscaleTrt.cpp src/RifeTrt.cpp src/Writer.cpp src/nv12_to_rgb.cu src/rgb_to_nv12.cu src/Reader.cpp)

add_executable(FrameSmith ${SOURCE_FILES})

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd")  # Use dynamic debug runtime for Debug builds
    set(CMAKE_CXX_FLAGS_RELEASE "/MD") # Use dynamic release runtime for Release builds
endif()


# Enable verbose output for troubleshooting
set(CMAKE_VERBOSE_MAKEFILE ON)
if(WIN32)
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not defined. Please specify it as a CMake option or environment variable.")
    endif()

    if(NOT DEFINED Protobuf_PROTOC_EXECUTABLE)
        find_program(Protobuf_PROTOC_EXECUTABLE protoc)
        if(NOT Protobuf_PROTOC_EXECUTABLE)
            message(FATAL_ERROR "Protobuf_PROTOC_EXECUTABLE is not found. Please ensure Protobuf is installed or provide the path as an option.")
        endif()
    endif()

    # Torch
    if(NOT DEFINED Torch_DIR)
        find_path(Torch_DIR TorchConfig.cmake HINTS ENV TORCH_DIR PATHS "$ENV{TORCH_DIR}")
        if(NOT Torch_DIR)
            message(FATAL_ERROR "Torch_DIR is not defined and could not be found. Please set it via environment variable or CMake option.")
        endif()
    endif()

    # CUDA Toolkit
    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
        find_package(CUDA REQUIRED)
        if(NOT CUDA_TOOLKIT_ROOT_DIR)
            message(FATAL_ERROR "CUDA_TOOLKIT_ROOT_DIR is not defined. Please set it via environment variable or CMake option.")
        endif()
    endif()

    # TensorRT
    if(NOT DEFINED TENSORRT_DIR)
        find_path(TENSORRT_DIR nvinfer HINTS ENV TENSORRT_DIR PATHS "$ENV{TENSORRT_DIR}")
        if(NOT TENSORRT_DIR)
            message(FATAL_ERROR "TENSORRT_DIR is not defined and could not be found. Please set it via environment variable or CMake option.")
        endif()
    endif()

    # Vcpkg
    if(NOT DEFINED VCPKG_DIR)
        find_path(VCPKG_DIR HINTS ENV VCPKG_DIR PATHS "$ENV{VCPKG_DIR}")
        if(NOT VCPKG_DIR)
            message(FATAL_ERROR "VCPKG_DIR is not defined and could not be found. Please set it via environment variable or CMake option.")
        endif()
    endif()

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
    target_link_libraries(FrameSmith ${CURL_LIBRARIES})
elseif(UNIX)
    target_link_libraries(FrameSmith ${CURL_LIBRARIES})
endif()

if(NOT DEFINED FFMPEG_INCLUDE_DIR)
    find_path(FFMPEG_INCLUDE_DIR HINTS ENV FFMPEG_INCLUDE_DIR PATHS "$ENV{FFMPEG_INCLUDE_DIR}")
    if(NOT FFMPEG_INCLUDE_DIR)
        message(FATAL_ERROR "FFMPEG_INCLUDE_DIR is not defined. Please set it via environment variable or CMake option.")
    endif()
endif()

include_directories(${FFMPEG_INCLUDE_DIR})

if(NOT DEFINED FFMPEG_LIB_DIR)
    find_path(FFMPEG_LIB_DIR HINTS ENV FFMPEG_LIB_DIR PATHS "$ENV{FFMPEG_LIB_DIR}")
    if(NOT FFMPEG_LIB_DIR)
        message(FATAL_ERROR "FFMPEG_LIB_DIR is not defined. Please set it via environment variable or CMake option.")
    endif()
endif()

find_library(AVCODEC_LIBRARY avcodec HINTS "${FFMPEG_LIB_DIR}")
find_library(AVFORMAT_LIBRARY avformat HINTS "${FFMPEG_LIB_DIR}")
find_library(AVUTIL_LIBRARY avutil HINTS "${FFMPEG_LIB_DIR}")
find_library(SWSCALE_LIBRARY swscale HINTS "${FFMPEG_LIB_DIR}")
find_library(SWRESAMPLE_LIBRARY swresample HINTS "${FFMPEG_LIB_DIR}")


# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS} include/ src/ ${TENSORRT_DIR}/include ${VCPKG_DIR}/include ${OpenCV_INCLUDE_DIRS})

# Link libraries
if(WIN32)
    target_link_libraries(FrameSmith 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWSCALE_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
        CURL::libcurl
        protobuf::libprotobuf
        "${TENSORRT_DIR}/lib/nvinfer_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_plugin_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvonnxparser_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_dispatch_10${LIB_SUFFIX}"
        "${TENSORRT_DIR}/lib/nvinfer_lean_10${LIB_SUFFIX}"
    )
elseif(UNIX)
    target_link_libraries(FrameSmith 
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        avcodec avformat avutil swscale
        protobuf::libprotobuf
        ${TENSORRT_DIR}/lib/libnvinfer${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvinfer_plugin${LIB_SUFFIX}
        ${TENSORRT_DIR}/lib/libnvonnxparser${LIB_SUFFIX}
        
        atomic
    )
endif()

# Set C++ standard
set_property(TARGET FrameSmith PROPERTY CXX_STANDARD 17)
set_property(TARGET FrameSmith PROPERTY CUDA_STANDARD 17)

# Required by PyTorch
if (MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# Add CUDA flags if necessary

if(CUDA_FOUND)
    set_target_properties(FrameSmith PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "53;60;61;70;75;80;86;89"
    )
endif()


