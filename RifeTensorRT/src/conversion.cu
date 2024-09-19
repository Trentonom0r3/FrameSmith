// color_conversion.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Define a macro for CUDA error checking.
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(err); \
    }

extern "C" {

    // CUDA kernel for converting RGB (with squeeze and permute) to NV12 format (YUV420).
    __global__ void rgb_to_nv12(const float* __restrict__ rgb, uint8_t* y, uint8_t* uv, int width, int height, int y_linesize, int uv_linesize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y_pos = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure we don't go out of bounds.
        if (x >= width || y_pos >= height) return;

        // Calculate the index for RGB data in channels-first format (C, H, W)
        int idx = y_pos * width + x;

        // Load RGB values in channels-first order (C=3, H, W)
        int r_idx = idx;                 // Red channel (C=0)
        int g_idx = width * height + idx; // Green channel (C=1)
        int b_idx = 2 * width * height + idx; // Blue channel (C=2)

        // Load RGB values directly from the float data.
        float r_f = rgb[r_idx];         // Red
        float g_f = rgb[g_idx];         // Green
        float b_f = rgb[b_idx];         // Blue

        // Normalize by multiplying by 255.
        r_f *= 255.0f;
        g_f *= 255.0f;
        b_f *= 255.0f;

        // Clamp values to [0, 255].
        r_f = fminf(fmaxf(r_f, 0.0f), 255.0f);
        g_f = fminf(fmaxf(g_f, 0.0f), 255.0f);
        b_f = fminf(fmaxf(b_f, 0.0f), 255.0f);

        // Convert to uint8 for output.
        uint8_t r = static_cast<uint8_t>(r_f);
        uint8_t g = static_cast<uint8_t>(g_f);
        uint8_t b = static_cast<uint8_t>(b_f);

        // Convert RGB to Y (luminance), using the Y linesize for stride addressing.
        y[y_pos * y_linesize + x] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);

        // For U and V, we subsample and interleave them into NV12 format.
        if (y_pos % 2 == 0 && x % 2 == 0) {
            int sub_idx = (y_pos / 2) * (width / 2) + (x / 2);

            // Calculate U and V (chrominance).
            float u_f = (-0.14713f * r - 0.28886f * g + 0.436f * b + 128.0f);
            float v_f = (0.615f * r - 0.51499f * g - 0.10001f * b + 128.0f);

            // Clamp U and V to [0, 255].
            u_f = fminf(fmaxf(u_f, 0.0f), 255.0f);
            v_f = fminf(fmaxf(v_f, 0.0f), 255.0f);

            uint8_t u = static_cast<uint8_t>(u_f);
            uint8_t v = static_cast<uint8_t>(v_f);

            // Interleave U and V in NV12 format, using the UV linesize for stride addressing.
            uv[(y_pos / 2) * uv_linesize + x] = u;
            uv[(y_pos / 2) * uv_linesize + x + 1] = v;
        }
    }


    // Kernel launcher for the combined RGB to NV12 conversion.
    void launch_rgb_to_nv12(const float* rgb, uint8_t* y, uint8_t* uv, int width, int height, int y_linesize, int uv_linesize, cudaStream_t stream) {
        dim3 threads(16, 16);  // Define the number of threads per block.
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);  // Define the number of blocks.

        // Launch the RGB to NV12 kernel with channels-first data layout.
        rgb_to_nv12 << <blocks, threads, 0, stream >> > (rgb, y, uv, width, height, y_linesize, uv_linesize);

        // Check for kernel launch errors.
        cudaError_t err = cudaGetLastError();
        CUDA_CHECK(err);
    }


}  // extern "C"
