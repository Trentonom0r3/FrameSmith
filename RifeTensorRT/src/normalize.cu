// normalize.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <normalize.h>
#include <iostream>


__global__ void normalize_cuda_kernel(const uint8_t* rgb, float* normalized, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        normalized[idx] = min(max(static_cast<float>(rgb[idx]) / 255.0f, 0.0f), 1.0f);
    }
}

void launch_normalize_kernel(const uint8_t* rgb, float* normalized, int total) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_cuda_kernel << <blocks, threads >> > (rgb, normalized, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error in normalize_cuda_kernel: " << cudaGetErrorString(err) << std::endl;
    }
    // Optionally synchronize if necessary
    // cudaDeviceSynchronize();
}
