// RifeKernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

extern "C" {

    __global__ void prepareDummyInputKernelFloat(
        float* dummyInput,
        float* source,
        float* destination,
        float* timestep,
        int height,
        int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int totalPixels = height * width;
        if (idx < totalPixels) {
            // Copy source channels
            dummyInput[idx] = source[idx]; // Channel 0
            dummyInput[idx + totalPixels] = source[idx + totalPixels]; // Channel 1
            dummyInput[idx + 2 * totalPixels] = source[idx + 2 * totalPixels]; // Channel 2

            // Copy destination channels
            dummyInput[idx + 3 * totalPixels] = destination[idx];
            dummyInput[idx + 4 * totalPixels] = destination[idx + totalPixels];
            dummyInput[idx + 5 * totalPixels] = destination[idx + 2 * totalPixels];

            // Copy timestep
            dummyInput[idx + 6 * totalPixels] = timestep[idx];
        }
    }

    __global__ void prepareDummyInputKernelHalf(
        __half* dummyInput,
        __half* source,
        __half* destination,
        __half* timestep,
        int height,
        int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int totalPixels = height * width;
        if (idx < totalPixels) {
            // Copy source channels
            dummyInput[idx] = source[idx]; // Channel 0
            dummyInput[idx + totalPixels] = source[idx + totalPixels]; // Channel 1
            dummyInput[idx + 2 * totalPixels] = source[idx + 2 * totalPixels]; // Channel 2

            // Copy destination channels
            dummyInput[idx + 3 * totalPixels] = destination[idx];
            dummyInput[idx + 4 * totalPixels] = destination[idx + totalPixels];
            dummyInput[idx + 5 * totalPixels] = destination[idx + 2 * totalPixels];

            // Copy timestep
            dummyInput[idx + 6 * totalPixels] = timestep[idx];
        }
    }

    // Function to initialize tensor with a scalar value (float)
    __global__ void initializeTensorKernelFloat(float* tensor, float value, int height, int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int totalElements = height * width;
        if (idx < totalElements) {
            tensor[idx] = value;
        }
    }

    // Function to initialize tensor with a scalar value (half)
    __global__ void initializeTensorKernelHalf(__half* tensor, __half value, int height, int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int totalElements = height * width;
        if (idx < totalElements) {
            tensor[idx] = value;
        }
    }

    // Launch functions
    void launch_prepare_dummy_input_float(
        float* dummyInput,
        float* source,
        float* destination,
        float* timestep,
        int height,
        int width,
        cudaStream_t stream) {
        int totalPixels = height * width;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
        prepareDummyInputKernelFloat << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
            dummyInput, source, destination, timestep, height, width);
    }

    void launch_prepare_dummy_input_half(
        __half* dummyInput,
        __half* source,
        __half* destination,
        __half* timestep,
        int height,
        int width,
        cudaStream_t stream) {
        int totalPixels = height * width;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
        prepareDummyInputKernelHalf << <blocksPerGrid, threadsPerBlock, 0, stream >> > (
            dummyInput, source, destination, timestep, height, width);
    }

    void initializeTensorWithScalarFloat(float* tensor, float value, int height, int width) {
        int totalElements = height * width;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
        initializeTensorKernelFloat << <blocksPerGrid, threadsPerBlock >> > (tensor, value, height, width);
      //  cudaDeviceSynchronize();
    }

    void initializeTensorWithScalarHalf(__half* tensor, float value, int height, int width) {
        __half halfValue = __float2half(value);
        int totalElements = height * width;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
        initializeTensorKernelHalf << <blocksPerGrid, threadsPerBlock >> > (tensor, halfValue, height, width);
      //  cudaDeviceSynchronize();
    }

} // extern "C"
