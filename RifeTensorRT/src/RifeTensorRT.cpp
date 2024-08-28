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
    if (width > 1920 && height > 1080 && half) {
        std::cout << yellow("UHD and fp16 are not compatible with RIFE, defaulting to fp32") << std::endl;
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

    // Set Tensor Addresses and Input Shapes
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        void* bindingPtr = (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) ? 
                            static_cast<void*>(dummyInput.data_ptr()) : 
                            static_cast<void*>(dummyOutput.data_ptr());
        context->setTensorAddress(tensorName, bindingPtr);

        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims dims = toDims(dummyInput.sizes());
            context->setInputShape(tensorName, dims);
        }
    }

    firstRun = true;
    useI0AsSource = true;
}

at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {

    // Ensure the frame is properly normalized
    auto processed = frame.to(device, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true)
        .permute({ 2, 0, 1 })  // Change the order of the dimensions: from HWC to CHW
        .unsqueeze(0)           // Add a batch dimension
        .div(255.0)             // Normalize to [0, 1]
        .contiguous();

    return processed;
}

void RifeTensorRT::run(const at::Tensor& frame, bool benchmark, cv::VideoWriter& writer) {
    c10::cuda::CUDAStreamGuard guard(stream);

    if (firstRun) {
        I0.copy_(processFrame(frame), true); // No need to normalize again here
        firstRun = false;
        return;
    }

    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(processFrame(frame), true); // Normalize the frame by dividing by 255.0

    for (int i = 0; i < interpolateFactor - 1; ++i) {
        at::Tensor timestep = torch::full({ 1, 1, height, width },
            (i + 1) * 1.0 / interpolateFactor,
            torch::TensorOptions().dtype(dType).device(device)).contiguous();

        dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true).contiguous();

        // Bind input and output tensors to the context
        context->setTensorAddress("input", dummyInput.data_ptr());
        context->setTensorAddress("output", dummyOutput.data_ptr());

        // Execute TensorRT inference
        if (!context->enqueueV3(static_cast<cudaStream_t>(stream))) {
            std::cerr << "Error during TensorRT inference!" << std::endl;
            return;
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

        // Convert output tensor to CPU and ensure it has the correct dimensions
        at::Tensor output = dummyOutput.squeeze(0)
            .permute({ 1, 2, 0 }) // CHW to HWC
            .mul(255.0)
            .clamp(0, 255)
            .to(torch::kU8)
            .to(torch::kCPU); // Multiply back to the range [0, 255] and convert to 8-bit

        // Make sure the output tensor is contiguous
        output = output.contiguous();

        // Initialize OpenCV matrix as a 3-channel image
        cv::Mat outputFrame(height, width, CV_8UC3);
        std::memcpy(outputFrame.data, output.data_ptr(), output.nbytes());

        // Check if the frame has valid data
        if (!outputFrame.empty()) {
            writer.write(outputFrame);
        }
        else {
            std::cerr << "Output frame is empty!" << std::endl;
        }
    }

    useI0AsSource = !useI0AsSource;
}

