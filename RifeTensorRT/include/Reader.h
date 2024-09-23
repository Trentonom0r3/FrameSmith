#pragma once

#include <string>
#include <iostream>
#include <thread>
#include <algorithm>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>

    void nv12_to_rgb(
        const unsigned char* yPlane,
        const unsigned char* uvPlane,
        int width,
        int height,
        int yStride,
        int uvStride,
        unsigned char* rgbOutput,
        int rgbStride,
        cudaStream_t stream);
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath, torch::Device device, bool halfPrecision = false);
    ~FFmpegReader();

    /**
     * @brief Reads the next frame and fills the provided tensor with RGB data.
     *
     * @param tensor The tensor to fill with the processed frame. Must be preallocated with shape {1, 3, height, width}.
     * @return true if a frame was successfully read and processed.
     * @return false if no more frames are available or an error occurred.
     */
    bool readFrame(torch::Tensor& tensor);

    // Accessor methods
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    double getFPS() const {
        if (formatCtx->streams[0]->avg_frame_rate.den != 0)
            return static_cast<double>(formatCtx->streams[0]->avg_frame_rate.num) / formatCtx->streams[0]->avg_frame_rate.den;
        else
            return fps;
    }
    int getTotalFrames() const {
        double duration = getDuration();
        double fps_val = getFPS();
        return static_cast<int>(duration * fps_val);
    }

    // Getters for CUDA stream
    cudaStream_t getStream() const { return stream; }

private:
    // FFmpeg components
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int videoStreamIndex = -1;
    int width = 0, height = 0;
    double fps = 0.0;

    // Torch and CUDA components
    torch::Device device;
    bool halfPrecision;

    // Intermediate tensors
    torch::Tensor rgb_tensor;          // For NV12 to RGB conversion
    torch::Tensor intermediate_tensor; // For reshaping and normalization
    // Conversion and normalization
    void avframe_nv12_to_rgb_npp(AVFrame* gpu_frame);
    void normalizeFrame();
    cudaStream_t stream;
    double getDuration() const {
        return static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    // Static callback for hardware format
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);
};

