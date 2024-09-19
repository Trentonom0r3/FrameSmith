#pragma once
#include <string>
#include <mutex>
#include <atomic>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include "concurrentqueue.h" // Include moodycamel's ConcurrentQueue

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <iostream>

// Forward declarations of custom CUDA functions
extern "C" {
    void launch_rgb_to_nv12(const float* rgb, uint8_t* y, uint8_t* uv, int width, int height, int y_linesize, int uv_linesize, cudaStream_t stream);
}

struct AVFrameDeleter {
    void operator()(AVFrame* frame) const noexcept {
        av_frame_free(&frame);
    }
};

// Error handling macros
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1); \
        } \
    }

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
    void addFrame(const float* rgb_ptr, bool benchmark);
    void finalize();  // Handle any finalization if necessary
    void setStream(cudaStream_t stream) { writestream = stream; }
    inline cudaStream_t getStream() const { return writestream; }
    inline cudaStream_t getConvertStream() const { return convertstream; }

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

    // Frame pool with custom deleter
    std::vector<std::unique_ptr<AVFrame, AVFrameDeleter>> framePool;
    std::mutex poolMutex; // Protect access to the pool

    // NV12 buffer
    uint8_t* nv12_buffer = nullptr; // Device pointer for NV12 data

    uint8_t* y_buffer = nullptr;
    uint8_t* uv_buffer = nullptr;
    // Internal methods
    AVFrame* acquireFrame();  // Acquire a frame from the pool
    void releaseFrame(AVFrame* frame);  // Release a frame back to the pool
    void writeFrame(AVFrame* inputFrame);  // Encode and write a frame
};
