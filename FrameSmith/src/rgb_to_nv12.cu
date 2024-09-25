// rgb_to_nv12_optimized.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(err); \
    }

extern "C" {

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

	// Optimized FP32 Kernel for RGB to NV12 conversion in channels-first format
	__global__ void rgb_to_nv12_fp32_optimized(const float* __restrict__ rgb, uint8_t* __restrict__ y_plane, uint8_t* __restrict__ uv_plane,
		int width, int height, int y_pitch, int uv_pitch) {

		int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
		int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row index

		if (x >= width || y >= height) return;

		int idx = y * width + x;

		// Load and clamp RGB values, scale to [0, 255]
		float r = __saturatef(rgb[idx]) * 255.0f;
		float g = __saturatef(rgb[width * height + idx]) * 255.0f;
		float b = __saturatef(rgb[2 * width * height + idx]) * 255.0f;

		int r_int = __float2int_rn(r);
		int g_int = __float2int_rn(g);
		int b_int = __float2int_rn(b);

		// Compute Y component
		int y_int = (77 * r_int + 150 * g_int + 29 * b_int + 128) >> 8;
		y_int = min(max(y_int, 0), 255);
		y_plane[y * y_pitch + x] = static_cast<uint8_t>(y_int);

		// Compute U and V components for even pixels
		if ((x % 2 == 0) && (y % 2 == 0)) {
			// Handle edge cases
			int x1 = min(x + 1, width - 1);
			int y1 = min(y + 1, height - 1);

			int idx_right = y * width + x1;
			int idx_down = y1 * width + x;
			int idx_diag = y1 * width + x1;

			// Sum R, G, B values of 2x2 block
			int r_sum = r_int
				+ __float2int_rn(__saturatef(rgb[idx_right]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[idx_down]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[idx_diag]) * 255.0f);

			int g_sum = g_int
				+ __float2int_rn(__saturatef(rgb[width * height + idx_right]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[width * height + idx_down]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[width * height + idx_diag]) * 255.0f);

			int b_sum = b_int
				+ __float2int_rn(__saturatef(rgb[2 * width * height + idx_right]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[2 * width * height + idx_down]) * 255.0f)
				+ __float2int_rn(__saturatef(rgb[2 * width * height + idx_diag]) * 255.0f);

			// Average the R, G, B values
			int r_avg = (r_sum + 2) >> 2; // Divide by 4 with rounding
			int g_avg = (g_sum + 2) >> 2;
			int b_avg = (b_sum + 2) >> 2;

			// Calculate U and V components using standard BT.601 coefficients
			int u_int = ((-38 * r_avg - 74 * g_avg + 112 * b_avg + 128 * 256 + 128) >> 8);
			int v_int = ((112 * r_avg - 94 * g_avg - 18 * b_avg + 128 * 256 + 128) >> 8);

			u_int = min(max(u_int, 0), 255);
			v_int = min(max(v_int, 0), 255);

			// UV plane indices
			int uv_x = x;
			int uv_y = y / 2;
			int uv_idx = uv_y * uv_pitch + uv_x;

			uv_plane[uv_idx] = static_cast<uint8_t>(u_int);
			uv_plane[uv_idx + 1] = static_cast<uint8_t>(v_int);
		}
	}

	// Optimized FP32 Kernel Launcher
	void launch_rgb_to_nv12_fp32(const float* rgb, uint8_t* y_plane, uint8_t* uv_plane,
		int width, int height, int y_pitch, int uv_pitch,
		cudaStream_t stream) {

		dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

		// Launch the optimized FP32 kernel
		rgb_to_nv12_fp32_optimized << <blocks, threads, 0, stream >> > (rgb, y_plane, uv_plane, width, height, y_pitch, uv_pitch);

		// Check for kernel launch errors
		cudaError_t err = cudaGetLastError();
		CUDA_CHECK(err);
	}

	// Optimized FP16 Kernel for RGB to NV12 conversion in channels-first format
	__global__ void rgb_to_nv12_fp16_optimized(const __half* __restrict__ rgb, uint8_t* __restrict__ y_plane, uint8_t* __restrict__ uv_plane,
		int width, int height, int y_pitch, int uv_pitch) {

		int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
		int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row index

		if (x >= width || y >= height) return;

		int idx = y * width + x;

		// Load and clamp RGB values, scale to [0, 255]
		__half r_h = __hmul(__hmin(__hmax(rgb[idx], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f));
		__half g_h = __hmul(__hmin(__hmax(rgb[width * height + idx], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f));
		__half b_h = __hmul(__hmin(__hmax(rgb[2 * width * height + idx], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f));

		int r_int = __half2int_rn(r_h);
		int g_int = __half2int_rn(g_h);
		int b_int = __half2int_rn(b_h);

		// Compute Y component
		int y_int = (77 * r_int + 150 * g_int + 29 * b_int + 128) >> 8;
		y_int = min(max(y_int, 0), 255);
		y_plane[y * y_pitch + x] = static_cast<uint8_t>(y_int);

		// Compute U and V components for even pixels
		if ((x % 2 == 0) && (y % 2 == 0)) {
			// Handle edge cases
			int x1 = min(x + 1, width - 1);
			int y1 = min(y + 1, height - 1);

			int idx_right = y * width + x1;
			int idx_down = y1 * width + x;
			int idx_diag = y1 * width + x1;

			// Sum R, G, B values of 2x2 block
			int r_sum = r_int
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[idx_right], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[idx_down], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[idx_diag], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)));

			int g_sum = g_int
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[width * height + idx_right], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[width * height + idx_down], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[width * height + idx_diag], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)));

			int b_sum = b_int
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[2 * width * height + idx_right], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[2 * width * height + idx_down], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)))
				+ __half2int_rn(__hmul(__hmin(__hmax(rgb[2 * width * height + idx_diag], __float2half(0.0f)), __float2half(1.0f)), __float2half(255.0f)));

			// Average the R, G, B values
			int r_avg = (r_sum + 2) >> 2; // Divide by 4 with rounding
			int g_avg = (g_sum + 2) >> 2;
			int b_avg = (b_sum + 2) >> 2;

			// Calculate U and V components using standard BT.601 coefficients
			int u_int = ((-38 * r_avg - 74 * g_avg + 112 * b_avg + 128 * 256 + 128) >> 8);
			int v_int = ((112 * r_avg - 94 * g_avg - 18 * b_avg + 128 * 256 + 128) >> 8);

			u_int = min(max(u_int, 0), 255);
			v_int = min(max(v_int, 0), 255);

			// UV plane indices
			int uv_x = x;
			int uv_y = y / 2;
			int uv_idx = uv_y * uv_pitch + uv_x;

			uv_plane[uv_idx] = static_cast<uint8_t>(u_int);
			uv_plane[uv_idx + 1] = static_cast<uint8_t>(v_int);
		}
	}

	// Optimized FP16 Kernel Launcher
	void launch_rgb_to_nv12_fp16(const __half* rgb, uint8_t* y_plane, uint8_t* uv_plane,
		int width, int height, int y_pitch, int uv_pitch,
		cudaStream_t stream) {

		dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

		// Launch the optimized FP16 kernel
		rgb_to_nv12_fp16_optimized << <blocks, threads, 0, stream >> > (rgb, y_plane, uv_plane, width, height, y_pitch, uv_pitch);

		// Check for kernel launch errors
		cudaError_t err = cudaGetLastError();
		CUDA_CHECK(err);
	}

}  // extern "C"
