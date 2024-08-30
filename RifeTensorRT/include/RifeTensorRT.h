#ifndef RIFETENSORRT_H
#define RIFETENSORRT_H
#pragma once
#include <torch/torch.h>
#include <torch/cuda.h>
#include <NvInfer.h>
#include <fstream>
#include <c10/cuda/CUDAStream.h> // Ensure correct include for CUDAStream
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

class RifeTensorRT {
public:
    RifeTensorRT(
        std::string interpolateMethod = "rife4.20-tensorrt",
        int interpolateFactor = 2,
        int width = 0,
        int height = 0,
        bool half = true,
        bool ensemble = false
    );

    ~RifeTensorRT();  // Destructor to clean up resources
    void run(const at::Tensor& frame, bool benchmark, AVCodecContext* enc_ctx, AVFrame* outputFrame, AVFormatContext* fmt_ctx, AVStream* video_stream, int64_t pts, int64_t pts_step);


private:
    void handleModel();
    at::Tensor processFrame(const at::Tensor& frame) const;
    void cacheFrame();
    void cacheFrameReset(const at::Tensor& frame);
    void allocateResources(AVCodecContext* enc_ctx);  // Resource allocation method
    void freeResources();  // Resource cleanup method

    std::string interpolateMethod;
    int interpolateFactor;
    int width;
    int height;
    bool half;
    bool ensemble;
    bool firstRun;
    bool useI0AsSource;
    int64_t last_dts = AV_NOPTS_VALUE;
    torch::Device device;
    torch::Tensor I0, I1, dummyInput, dummyOutput;
    std::vector<void*> bindings;
    torch::ScalarType dType;
    c10::cuda::CUDAStream stream;

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::string enginePath;

    // New member variables for AVFrame and SwsContext
    AVFrame* frame_yuv;
    SwsContext* sws_ctx;
};

#endif
