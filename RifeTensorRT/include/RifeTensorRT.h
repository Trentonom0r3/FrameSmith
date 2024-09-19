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
    RifeTensorRT(std::string interpolateMethod, int interpolateFactor,
        int width, int height, bool half, bool ensemble, bool benchmark, FFmpegWriter& writer);
    void handleModel();
    void run(at::Tensor input); // Corrected method signature
    // Allocate tensors for Y, U, and V planes on GPU
    int getInterpolateFactor() const { return interpolateFactor; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    FFmpegWriter& getWriter() { return writer; }

    // Separate getters for each stream
    cudaStream_t getInferenceStream() const { return inferenceStream; }

    ~RifeTensorRT();
    bool benchmarkMode;

    std::string interpolateMethod;
    int interpolateFactor;
    int width;
    int height;
    bool half;
    bool ensemble;
    bool firstRun;
    bool useI0AsSource;
    torch::Device device;
    cudaStream_t inferenceStream;
    // Tensors
    std::vector<at::Tensor> timestep_tensors;
    torch::Tensor I0, I1, dummyInput, dummyOutput, rgb_tensor;

    // TensorRT engine and context
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::string enginePath;
    std::vector<void*> bindings;
    torch::ScalarType dType;

    FFmpegWriter& writer; // Moved to the end for better organization
};
