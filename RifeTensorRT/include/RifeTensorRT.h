#ifndef RIFETENSORRT_H
#define RIFETENSORRT_H
#pragma once
#include <torch/torch.h>
#include <torch/cuda.h>
#include <NvInfer.h>
#include <fstream>
#include <c10/cuda/CUDAStream.h> // Ensure correct include for CUDAStream

extern "C" {
#include <libavutil/frame.h>
}

class RifeTensorRT {
public:
    // Constructor
    RifeTensorRT(
        std::string interpolateMethod = "rife4.20-tensorrt",
        int interpolateFactor = 2,
        int width = 0,
        int height = 0,
        bool half = true,
        bool ensemble = false
    );

    // Main method to run interpolation on AVFrame
    AVFrame* RifeTensorRT::run(AVFrame* rgbFrame, AVFrame* interpolatedFrame, cudaEvent_t& inferenceFinishedEvent);

    // Caching of frames for performance optimization
    at::Tensor  cachedFrameI0;  // Cached source frame (AVFrame*)
    at::Tensor  cachedFrameI1;  // Cached destination frame (AVFrame*)
    at::Tensor  cachedInterpolatedFrame; // Cached interpolated frame (AVFrame*)
    bool isFrameCached;  // Flag to indicate if frames are cached
    // Helper function to convert AVFrame to Tensor
    inline at::Tensor RifeTensorRT::AVFrameToTensor(AVFrame* frame) {
        // Ensure frame is in RGB format and is interleaved (planar format would need conversion).
        // This assumes an interleaved RGB24 format for the conversion.

        if (frame->format != AV_PIX_FMT_RGB24) {
            std::cerr << "AVFrame is not in RGB24 format!" << std::endl;
            return torch::Tensor();
        }

        // Create tensor from the AVFrame data
        return torch::from_blob(
            frame->data[0],  // AVFrame data pointer
            { 1, frame->height, frame->width, 3 },  // Dimensions: Batch size, Height, Width, Channels
            torch::kByte  // Data type: Byte (8-bit per channel)
        ).to(torch::kCUDA);  // Move tensor to CUDA (GPU)
    }


    // Helper function to convert Tensor back to AVFrame
    inline AVFrame* RifeTensorRT::TensorToAVFrame(const at::Tensor& tensor, int width, int height, AVPixelFormat format) {
        AVFrame* frame = av_frame_alloc();
        frame->format = format;
        frame->width = width;
        frame->height = height;

        // Allocate frame buffer for storing the image data
        if (av_frame_get_buffer(frame, 0) < 0) {
            std::cerr << "Could not allocate AVFrame buffer!" << std::endl;
            av_frame_free(&frame);
            return nullptr;
        }

        // Ensure the tensor is on CPU before copying to the AVFrame buffer
        at::Tensor cpuTensor = tensor.to(torch::kCPU).contiguous();

        // Copy data from tensor to the AVFrame buffer
        std::memcpy(frame->data[0], cpuTensor.data_ptr(), cpuTensor.numel());

        return frame;
    }

    // Internal helper methods
    void handleModel();
    at::Tensor processFrame(const at::Tensor& frame) const;
    void cacheFrame();  // Adjusted to cache AVFrame*
    void cacheFrameReset(AVFrame* frame);  // Adjusted to reset cache with AVFrame*

    // Model parameters and configuration
    std::string interpolateMethod;
    int interpolateFactor;
    int width;
    int height;
    bool half;
    bool ensemble;
    bool firstRun;
    bool useI0AsSource;
    torch::Device device;

    // Internal buffers and Tensors
    torch::Tensor I0, I1, dummyInput, dummyOutput;
    std::vector<void*> bindings;
    torch::ScalarType dType;
    c10::cuda::CUDAStream stream;

    // TensorRT engine and context for inference
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::string enginePath;
    AVFrame* cachedInterpolatedAVFrame;  // Cached interpolated frame


};

#endif