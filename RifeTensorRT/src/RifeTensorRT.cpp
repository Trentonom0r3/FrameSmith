#include "RifeTensorRT.h"
#include "downloadModels.h"
#include "coloredPrints.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <trtHandler.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <Writer.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>
#include <future>


nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];
    }
    return dims;
}

// Constructor implementation with CUDA context setup
RifeTensorRT::RifeTensorRT(std::string interpolateMethod, int interpolateFactor, int width, int height, bool half, bool ensemble, bool benchmark, FFmpegWriter& writer)
    : interpolateMethod(interpolateMethod),
    interpolateFactor(interpolateFactor),
    width(width),
    height(height),
    half(half),
    ensemble(ensemble),
    firstRun(true),
    useI0AsSource(true),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    benchmarkMode(benchmark),
    writer(writer)
{
    // Initialize model and tensors
    handleModel();

    // Initialize CUDA inferenceStreams
    cudaStreamCreate(&inferenceStream);
   // cudaStreamCreate(&writeinferenceStream);
    //writer.setinferenceStream(writeinferenceStream);
    for (int i = 0; i < interpolateFactor - 1; ++i) {
        auto timestep = torch::full({ 1, 1, height, width }, (i + 1) * 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();
        timestep_tensors.push_back(timestep);
    }
}

RifeTensorRT::~RifeTensorRT() {
    cudaStreamDestroy(inferenceStream);
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
    rgb_tensor = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    // Bindings
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

void RifeTensorRT::run(at::Tensor input) {
    // Use the CUDA inferenceStream for all operations to enable asynchronous execution

    if (firstRun) {
        // Asynchronously copy the input to I0 using the provided CUDA inferenceStream
        cudaMemcpyAsync(I0.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
        firstRun = false;
        // Add frame asynchronously (ensure writer supports async or synchronize as needed)
        writer.addFrame(input.to(torch::kFloat32).contiguous().data_ptr<float>(), benchmarkMode);
        return;
    }

    // Alternate between I0 and I1 for source and destination
    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;

    // Asynchronously copy input data
    cudaMemcpyAsync(destination.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);

    // Prepare input tensors
    cudaMemcpyAsync(dummyInput.slice(1, 0, 3).data_ptr(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
    cudaMemcpyAsync(dummyInput.slice(1, 3, 6).data_ptr(), destination.data_ptr(), destination.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);

    // Perform interpolation for the required number of frames
    for (int i = 0; i < interpolateFactor - 1; ++i) {
        // Update timestep asynchronously
        cudaMemcpyAsync(dummyInput.slice(1, 6, 7).data_ptr(), timestep_tensors[i].data_ptr(), timestep_tensors[i].nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);

        // Enqueue inference using the inferenceStream
        context->enqueueV3(inferenceStream);

        // Assuming the inference output is stored in dummyOutput, pass the result to the writer
        rgb_tensor = dummyOutput;
        // Add the interpolated frame asynchronously
        writer.addFrame(rgb_tensor.to(torch::kFloat32).contiguous().data_ptr<float>(), benchmarkMode);
    }

    // Add the original frame after processing asynchronously
    writer.addFrame(input.to(torch::kFloat32).contiguous().data_ptr<float>(), benchmarkMode);

    // Flip the source flag for the next run
    useI0AsSource = !useI0AsSource;
}
