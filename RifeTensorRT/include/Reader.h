#pragma once

#include <string>
#include <iostream>
#include <thread>
#include <algorithm>
#include <torch/torch.h>
#include <cuda_runtime.h>

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

/*
TODO
How to get rid of torch dependency in FFmpegReader class?
Would have to take either __half* or float* as input to nv12_to_rgb function
*/
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
    // double getFPS() const { return fps; }
    double getFPS() const {
        // Implement this method to return the FPS of the video
        // Example implementation:
        if (formatCtx->streams[0]->avg_frame_rate.den != 0)
            return static_cast<double>(formatCtx->streams[0]->avg_frame_rate.num) / formatCtx->streams[0]->avg_frame_rate.den;
        else
            return fps;
    }
    int getTotalFrames() const {
        // Assuming FFmpegReader has a method to get duration in seconds
        double duration = getDuration(); // Implement getDuration() if not available
        double fps = getFPS();
        return static_cast<int>(duration * fps);
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
    int frameCount = 0;
    // Torch and CUDA components
    torch::Device device;
    bool halfPrecision;
    int framecount = 0;
    // Intermediate tensors
    torch::Tensor rgb_tensor;          // For NV12 to RGB conversion
    torch::Tensor intermediate_tensor; // For reshaping and normalization
    // Conversion and normalization
    void avframe_nv12_to_rgb_npp(AVFrame* gpu_frame);
    void normalizeFrame();
    cudaStream_t stream;
    double getDuration() const {
        // Implement this method to return the duration of the video in seconds
        // Example implementation:
        return static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    // Static callback for hardware format
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);
};
