#pragma once
#include <string>
#include <atomic>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <iostream>
#include <cuda_fp16.h> // Include CUDA half-precision support

// Forward declarations of custom CUDA functions
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>

	void launch_rgb_to_nv12_fp32(const float* rgb, uint8_t* y, uint8_t* uv,
		int width, int height, int y_linesize, int uv_linesize,
		cudaStream_t stream);

	void launch_rgb_to_nv12_fp16(const __half* rgb, uint8_t* y, uint8_t* uv,
		int width, int height, int y_linesize, int uv_linesize,
		cudaStream_t stream);
}

// Custom deleter for AVFrame
struct AVFrameDeleter {
	void operator()(AVFrame* frame) const noexcept {
		av_frame_free(&frame);
	}
};

// Error handling macro
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1); \
        } \
    }

// Structure representing a node in the lock-free stack
struct FrameNode {
	AVFrame* frame;
	FrameNode* next;

	FrameNode(AVFrame* f) : frame(f), next(nullptr) {}
};

// Structure combining a pointer with a version tag to mitigate the ABA problem
struct TaggedPointer {
	FrameNode* ptr;
	uintptr_t tag;

	TaggedPointer(FrameNode* p = nullptr, uintptr_t t = 0) : ptr(p), tag(t) {}
};

// Ensure TaggedPointer is trivially copyable for atomic operations
static_assert(std::is_trivially_copyable<TaggedPointer>::value, "TaggedPointer must be trivially copyable");

// Helper function to set up CUDA device context
inline int init_cuda_context(AVBufferRef** hw_device_ctx) {
	int err = av_hwdevice_ctx_create(hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
	if (err < 0) {
		char err_buf[AV_ERROR_MAX_STRING_SIZE];
		av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, err);
		std::cerr << "Failed to create CUDA device context: " << err_buf << std::endl;
		return err;
	}
	return 0;
}

// Helper function to set up CUDA frames context
inline AVBufferRef* init_cuda_frames_ctx(AVBufferRef* hw_device_ctx, int width, int height, AVPixelFormat sw_format) {
	AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
	if (!hw_frames_ref) {
		std::cerr << "Failed to create hardware frames context" << std::endl;
		return nullptr;
	}

	AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(hw_frames_ref->data);
	frames_ctx->format = AV_PIX_FMT_CUDA;
	frames_ctx->sw_format = sw_format;
	frames_ctx->width = width;
	frames_ctx->height = height;
	frames_ctx->initial_pool_size = 20;

	int err = av_hwframe_ctx_init(hw_frames_ref);
	if (err < 0) {
		char err_buf[AV_ERROR_MAX_STRING_SIZE];
		av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, err);
		std::cerr << "Failed to initialize hardware frame context: " << err_buf << std::endl;
		av_buffer_unref(&hw_frames_ref);
		return nullptr;
	}

	return hw_frames_ref;
}


class FFmpegWriter {
public:
	FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps, bool benchmark);
	~FFmpegWriter();
	void finalize();  // Handle any finalization if necessary
	void setStream(cudaStream_t stream) { writestream = stream; }
	inline cudaStream_t getStream() const { return writestream; }
	inline cudaStream_t getConvertStream() const { return convertstream; }
	void writeFrame(AVFrame* inputFrame);
	// Template addFrame method
	template <typename T>
	void addFrameTemplate(const T* rgb_ptr, bool benchmark);

	// Non-template addFrame methods that call the template
	void addFrame(const float* rgb_ptr, bool benchmark) {
		addFrameTemplate<float>(rgb_ptr, benchmark);
	}

	void addFrame(const __half* rgb_ptr, bool benchmark) {
		addFrameTemplate<__half>(rgb_ptr, benchmark);
	}

private:
	// FFmpeg components
	AVFormatContext* formatCtx = nullptr;
	AVCodecContext* codecCtx = nullptr;
	AVStream* stream = nullptr;
	AVPacket* packet = nullptr;
	int width, height, fps;
	std::atomic<int64_t> pts{ 0 };

	cudaStream_t writestream, convertstream;

	AVBufferRef* hw_frames_ctx = nullptr;
	AVBufferRef* hw_device_ctx = nullptr;

	// Benchmark mode flag
	bool isBenchmark;

	// Lock-Free Frame Pool
	std::atomic<TaggedPointer> head;

	// NV12 buffer
	uint8_t* nv12_buffer = nullptr; // Device pointer for NV12 data

	uint8_t* y_buffer = nullptr;
	uint8_t* uv_buffer = nullptr;

	// Internal methods
	AVFrame* acquireFrame();  // Acquire a frame from the pool
	void releaseFrame(AVFrame* frame);  // Release a frame back to the pool

	// Lock-Free Stack operations
	void pushFrame(FrameNode* node);
	AVFrame* popFrame();
};
