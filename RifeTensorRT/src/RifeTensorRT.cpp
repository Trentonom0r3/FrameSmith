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
device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
stream(c10::cuda::getStreamFromPool(false, device.index()))
{
    if (width > 1920 && height > 1080 && half) {
        std::cout << "UHD and fp16 are not compatible with RIFE, defaulting to fp32" << std::endl;
        this->half = false;
    }

    handleModel();
}

void RifeTensorRT::cacheFrame() {
    I0.copy_(I1, true);
}

void RifeTensorRT::cacheFrameReset(const at::Tensor& frame) {
    I0.copy_(processFrame(frame), true);
    useI0AsSource = true;
}

nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];

    }
    return dims;
}

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

    bool isCudnnEnabled = torch::cuda::cudnn_is_available();

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
    dummyInput = torch::empty({ 1, 7, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
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
}

at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    try {
        // Ensure the frame is properly normalized
        auto processed = frame.to(device, dType, /*non_blocking=*/false, /*copy=*/true)
            .permute({ 0, 3, 1, 2 })  // Change the order of the dimensions: from NHWC to NCHW
            .div(255.0)               // Normalize to [0, 1]
            .contiguous();

        return processed;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during processFrame: " << e.what() << std::endl;
        std::cerr << "Frame dimensions: " << frame.sizes() << " Frame dtype: " << frame.dtype() << std::endl;
        throw; // Re-throw the error after logging
    }
}


cv::Mat RifeTensorRT::run(const cv::Mat& frame) {
    // Ensure the CUDA stream guard is in place.
    c10::cuda::CUDAStreamGuard guard(stream);

    // Convert OpenCV's BGR image format to RGB for processing.
    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

    // Convert the RGB image to a tensor.
    at::Tensor inputTensor = torch::from_blob(rgbFrame.data, { 1, rgbFrame.rows, rgbFrame.cols, 3 }, torch::kByte).to(torch::kCUDA);

    if (firstRun) {
        // On the first run, initialize the source frame.
        I0.copy_(processFrame(inputTensor), true);
        firstRun = false;
        return cv::Mat();
    }

    // Toggle between source and destination buffers.
    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(processFrame(inputTensor), true);

    // Perform the interpolation.
    at::Tensor timestep = torch::full({ 1, 1, height, width }, 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true);

    context->setTensorAddress("input", dummyInput.data_ptr());
    context->setTensorAddress("output", dummyOutput.data_ptr());

    if (!context->enqueueV3(static_cast<cudaStream_t>(stream))) {
        std::cerr << "Error during TensorRT inference!" << std::endl;
        return cv::Mat();
    }

    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

    // Convert the output tensor to an OpenCV image format (BGR).
    at::Tensor outputTensor = dummyOutput.squeeze(0).permute({ 1, 2, 0 }).mul(255.0).clamp(0, 255).to(torch::kU8).to(torch::kCPU).contiguous();
    cv::Mat interpolatedFrame(outputTensor.size(0), outputTensor.size(1), CV_8UC3, outputTensor.data_ptr());

    // Convert back to BGR format for OpenCV.
    cv::cvtColor(interpolatedFrame, interpolatedFrame, cv::COLOR_RGB2BGR);

    // Flip the source buffer flag for the next run.
    useI0AsSource = !useI0AsSource;

    return interpolatedFrame;
}
