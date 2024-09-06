#include "RifeTensorRT.h"
#include "downloadModels.h"
#include "coloredPrints.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <c10/cuda/CUDAStream.h>
#include <trtHandler.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];

    }
    return dims;
}

// Constructor: Initializes the TensorRT RIFE model
RifeTensorRT::RifeTensorRT(
    std::string interpolateMethod,
    int interpolateFactor,
    int width,
    int height,
    bool half,
    bool ensemble
) : interpolateMethod(interpolateMethod),
interpolateFactor(interpolateFactor),
width(width),
height(height),
half(half),
ensemble(ensemble),
firstRun(true),
useI0AsSource(true),
isFrameCached(false), // Initialize caching flag
device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
stream(c10::cuda::getStreamFromPool(false, device.index())),
cachedInterpolatedAVFrame(nullptr)
{
    if (width > 1920 && height > 1080 && half) {
        std::cout << "UHD and fp16 are not compatible with RIFE, defaulting to fp32" << std::endl;
        this->half = false;
    }

    handleModel();
    // Create a tensor for the interpolation timestep
    timestep = torch::full({ 1, 1, height, width }, 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();

}

// Method to handle model loading and initialization
void RifeTensorRT::handleModel() {
    std::string filename = modelsMap(interpolateMethod, "onnx", half, ensemble);
    std::string folderName = interpolateMethod;
    folderName.replace(folderName.find("-tensorrt"), 9, "-onnx");

    std::filesystem::path modelPath = std::filesystem::path(getWeightsDir()) / folderName / filename;

    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Model not found, downloading it..." << std::endl;
        modelPath = downloadModels(interpolateMethod, "onnx", half, ensemble);
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Failed to download or locate the model: " << modelPath << std::endl;
            return;
        }
    }

    enginePath = TensorRTEngineNameHandler(modelPath.string(), half, { 1, 7, height, width });
    std::tie(engine, context) = TensorRTEngineLoader(enginePath);

    if (!engine || !context || !std::filesystem::exists(enginePath)) {
        std::cout << "Loading engine failed, creating a new one" << std::endl;
        std::tie(engine, context) = TensorRTEngineCreator(
            modelPath.string(), enginePath, half, { 1, 7, height, width }, { 1, 7, height, width }, { 1, 7, height, width }
        );
    }

    // Setup Torch tensors for input/output
    dType = half ? torch::kFloat16 : torch::kFloat32;
    I0 = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    I1 = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyInput = torch::zeros({ 1, 7, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyOutput = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();

    bindings = { dummyInput.data_ptr(), dummyOutput.data_ptr() };

    // Set Tensor Addresses and Input Shapes
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        void* bindingPtr = (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) ?
            static_cast<void*>(dummyInput.data_ptr()) :
            static_cast<void*>(dummyOutput.data_ptr());
        bool setaddyinfo = context->setTensorAddress(tensorName, bindingPtr);

        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            bool success = context->setInputShape(tensorName, toDims(dummyInput.sizes()));
            if (!success) {
                std::cerr << "Failed to set input shape for " << tensorName << std::endl;
                return;
            }
        }
    }

    firstRun = true;
    useI0AsSource = true;
    // Set the input and output tensor addresses in the TensorRT context
    context->setTensorAddress("input", dummyInput.data_ptr());
    context->setTensorAddress("output", dummyOutput.data_ptr());

}

// Preprocess frame and convert it to Torch tensor
at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    try {
        auto processed = frame.to(dType, /*non_blocking=*/true, /*copy=*/true)
            .permute({ 0, 3, 1, 2 })  // NHWC to NCHW
            .div(255.0)               // Normalize to [0, 1]
            .contiguous();
        return processed;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during processFrame: " << e.what() << std::endl;
        throw; // Re-throw after logging
    }
}

// Run inference on an input frame and return the interpolated result asynchronously
AVFrame* RifeTensorRT::run(AVFrame* rgbFrame, AVFrame* interpolatedFrame, cudaEvent_t& inferenceFinishedEvent) {
    // Get raw CUDA stream from torch::cuda::CUDAStream for asynchronous execution.
    c10::cuda::CUDAStreamGuard guard(stream);
    cudaStream_t raw_stream = stream.stream();

    // Ensure the AVFrame is writable before modifying it.
    if (av_frame_make_writable(rgbFrame) < 0) {
        std::cerr << "Cannot make AVFrame writable!" << std::endl;
        return nullptr;
    }

    // Convert AVFrame to Torch tensor (assuming interleaved RGB format).
    at::Tensor inputTensor = AVFrameToTensor(rgbFrame);

    if (firstRun) {
        // On the first run, initialize the source frame.
        I0.copy_(processFrame(inputTensor), true);
        firstRun = false;
        interpolatedFrame = rgbFrame;
        return interpolatedFrame;
    }

    // Use the source and destination buffers for double-buffering.
    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;

    // Copy the input frame to the destination buffer.
    destination.copy_(processFrame(inputTensor), true);

    // Create a tensor for the interpolation timestep
    // Prepare the input tensor for the interpolation (concatenating source, destination, and timestep)
    dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true);
    // Enqueue the inference on the raw CUDA stream (asynchronously)
    if (!context->enqueueV3(raw_stream)) {
        std::cerr << "Error during TensorRT inference!" << std::endl;
        return nullptr;
    }

    // Record an event to notify when inference has completed
    cudaEventRecord(inferenceFinishedEvent, raw_stream);

    // Flip the source buffer flag for the next run.
    useI0AsSource = !useI0AsSource;
    // Return the interpolated frame
    return interpolatedFrame;
}

