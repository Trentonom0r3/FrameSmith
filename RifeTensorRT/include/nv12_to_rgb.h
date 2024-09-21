// nv12_to_rgb_dynamic.h

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declare the CUDA functions
extern "C" {
    void nv12_to_rgb_normalized_float(
        const unsigned char* yPlane,
        const unsigned char* uvPlane,
        int width,
        int height,
        int yStride,
        int uvStride,
        float* tensorOutput, // {1, 3, H, W}
        int tensorStride, // C * H * W
        cudaStream_t stream
    );

    void nv12_to_rgb_normalized_half(
        const unsigned char* yPlane,
        const unsigned char* uvPlane,
        int width,
        int height,
        int yStride,
        int uvStride,
        __half* tensorOutput, // {1, 3, H, W}
        int tensorStride, // C * H * W
        cudaStream_t stream
    );
}

// Dispatch function
inline void launch_nv12_to_rgb_normalized_dispatch(
    const unsigned char* yPlane,
    const unsigned char* uvPlane,
    int width,
    int height,
    int yStride,
    int uvStride,
    void* tensorOutput, // void pointer; actual type depends on tensor
    int tensorStride, // C * H * W
    cudaDataType_t dataType, // Torch tensor dtype
    cudaStream_t stream = 0
)
{
    if (dataType == CUDA_R_32F) { // float
        nv12_to_rgb_normalized_float(
            yPlane,
            uvPlane,
            width,
            height,
            yStride,
            uvStride,
            static_cast<float*>(tensorOutput),
            tensorStride,
            stream
        );
    }
    else if (dataType == CUDA_R_16F) { // half
        nv12_to_rgb_normalized_half(
            yPlane,
            uvPlane,
            width,
            height,
            yStride,
            uvStride,
            static_cast<__half*>(tensorOutput),
            tensorStride,
            stream
        );
    }
    else {
        throw std::runtime_error("Unsupported tensor data type for NV12 to RGB conversion.");
    }
}
