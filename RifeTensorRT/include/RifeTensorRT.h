#pragma once
#include <torch/torch.h>
#include <torch/cuda.h>
#include <NvInfer.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
extern "C" {
#include <libavformat/avformat.h>
}
class FFmpegWriter;

class RifeTensorRT {
public:
    RifeTensorRT(std::string interpolateMethod, int interpolateFactor, int width, int height, bool half, bool ensemble, bool benchmark);
    void handleModel();
    at::Tensor processFrame(const at::Tensor& frame) const;
    void cacheFrame(at::Tensor& frame);
    void RifeTensorRT::run(AVFrame* inputFrame, FFmpegWriter& writer);
    int getInterpolateFactor() const { return interpolateFactor; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    cudaStream_t getStream() const { return stream; }
    cudaStream_t getWriteStream() const { return writestream; }
    AVBufferRef* hw_frames_ctx;
    AVBufferRef* hw_device_ctx;
    ~RifeTensorRT();
    bool benchmarkMode;
protected:
    std::string interpolateMethod;
    int interpolateFactor;
    int width;
    int height;
    bool half;
    bool ensemble;
    bool firstRun;
    bool useI0AsSource;
    torch::Device device;
    cudaStream_t stream, writestream;
    // Tensors
    torch::Tensor I0, I1, dummyInput, dummyOutput;
    AVFrame* interpolatedFrame, preAllocatedInputFrame;
    // TensorRT engine and context
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::string enginePath;
    std::vector<void*> bindings;
    torch::ScalarType dType;

};
