#include "RifeTensorRT.h"
#include "downloadModels.h"
#include "coloredPrints.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <c10/cuda/CUDAStream.h> // Ensure correct include for CUDAStream
#include <trtHandler.h>
#include <c10/core/ScalarType.h>
#include <fstream>
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
    // Properly initialize the CUDA stream within the constructor body
    //stream = c10::cuda::getStreamFromPool(false, device.index());
    std::cout << "Initializing RIFE with TensorRT" << std::endl;
    if (width > 1920 && height > 1080 && half) {
        std::cout << yellow("UHD and fp16 are not compatible with RIFE, defaulting to fp32") << std::endl;
        this->half = false;
    }
    std::cout << "Interpolation method: " << interpolateMethod << std::endl;
    handleModel();
    std::cout << "RIFE with TensorRT initialized" << std::endl;
}


void RifeTensorRT::handleModel() {
    std::string filename = modelsMap(interpolateMethod, "onnx", half, ensemble);
    std::string folderName = interpolateMethod;
    folderName.replace(folderName.find("-tensorrt"), 9, "-onnx");

    std::cout << "Model: " << filename << std::endl;
    std::filesystem::path modelPath = std::filesystem::path(getWeightsDir()) / folderName / filename;

    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Model not found, downloading it..." << std::endl;
        modelPath = downloadModels(interpolateMethod, "onnx", half, ensemble);
    }

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "Failed to download or locate the model: " << modelPath << std::endl;
        return;
    }

    bool isCudnnEnabled = torch::cuda::cudnn_is_available();
    std::cout << "cuDNN is " << (isCudnnEnabled ? "enabled" : "disabled") << std::endl;

    // Initialize TensorRT engine
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

    // Additional setup for TensorRT API usage
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = std::to_string(i).c_str();
        if (context->setInputShape(name, nvinfer1::Dims4{ 1, 7, height, width })) {
            // Successfully set the input shape
        }
        else {
            std::cerr << "Error: setInputShape or equivalent method is not supported in this TensorRT version." << std::endl;
        }
    }
};

at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    return frame.to(device, dType, /*non_blocking=*/false, /*copy=*/false)
        .permute({ 2, 0, 1 })
        .unsqueeze(0)
        .mul(1.0 / 255.0)
        .contiguous();
}

void RifeTensorRT::cacheFrame() {
    I0.copy_(I1, true);
}

void RifeTensorRT::cacheFrameReset(const at::Tensor& frame) {
    I0.copy_(processFrame(frame), true);
    useI0AsSource = true;
}


void RifeTensorRT::run(const at::Tensor& frame, bool benchmark, std::ofstream& writeBuffer) {
    c10::cuda::CUDAStreamGuard guard(stream);

    if (firstRun) {
        I0.copy_(processFrame(frame), true);
        firstRun = false;
        return;
    }

    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(processFrame(frame), true);

    // Process in the background with multiple streams for overlapping tasks
    for (int i = 0; i < interpolateFactor - 1; ++i) {
        at::Tensor timestep = torch::full({ 1, 1, height, width },
            (i + 1) * 1.0 / interpolateFactor,
            torch::TensorOptions().dtype(dType).device(device)).contiguous();

        dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true).contiguous();

        // Bind input and output tensors
        context->setTensorAddress("input", dummyInput.data_ptr());
        context->setTensorAddress("output", dummyOutput.data_ptr());

        context->enqueueV3(static_cast<cudaStream_t>(stream));
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

        at::Tensor output = dummyOutput.squeeze(0).permute({ 1, 2, 0 }).mul(255.0);
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

        if (!benchmark) {
            // Asynchronously write output to disk
            writeBuffer.write(reinterpret_cast<const char*>(output.data_ptr()), output.nbytes());
        }
    }
}



