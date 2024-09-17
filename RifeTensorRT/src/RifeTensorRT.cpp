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

    // Initialize CUDA streams
    cudaStreamCreate(&stream);
   // cudaStreamCreate(&writestream);
    //writer.setStream(writestream);
    for (int i = 0; i < interpolateFactor - 1; ++i) {
        auto timestep = torch::full({ 1, 1, height, width }, (i + 1) * 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();
        timestep_tensors.push_back(timestep);
    }
}

RifeTensorRT::~RifeTensorRT() {
    cudaStreamDestroy(stream);
   // cudaStreamDestroy(writestream);
  //  av_frame_free(&interpolatedFrame);
   // av_buffer_unref(&hw_frames_ctx);
   // av_buffer_unref(&hw_device_ctx);
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
    rgb_tensor = torch::empty({ height, width, 3 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).contiguous();
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

// Preprocess the frame: normalize pixel values
void RifeTensorRT::processFrame(at::Tensor& frame) const {
    frame = frame.to(half ? torch::kFloat16 : torch::kFloat32)
        .div_(255.0)
        .clamp_(0.0, 1.0)
        .contiguous();
}


void RifeTensorRT::run(at::Tensor input) {
    // [Line 1] Static variables for graph state
    static bool graphInitialized = false;
    static cudaGraph_t graph;
    static cudaGraphExec_t graphExec;

    if (firstRun) {
        I0.copy_(input, true);
        firstRun = false;
        if (!benchmarkMode) {
            writer.addFrame(input);
        }
        return;
    }

    // [Line 15] Alternate between I0 and I1
    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(input, true);

    // [Line 20] Prepare input tensors
    dummyInput.slice(1, 0, 3).copy_(source, true);
    dummyInput.slice(1, 3, 6).copy_(destination, true);

    for (int i = 0; i < interpolateFactor - 1; ++i) {
        // [Line 25] Update timestep in dummyInput
        dummyInput.slice(1, 6, 7).copy_(timestep_tensors[i], true);

        // [Line 28] Enqueue inference
        if (!graphInitialized) {
            // [Line 30] Begin graph capture
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // [Line 32] Enqueue inference
            context->enqueueV3(stream);

            // [Line 35] End graph capture
            cudaStreamEndCapture(stream, &graph);

            // [Line 37] Instantiate the graph
            cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

            graphInitialized = true;
        }

        // [Line 42] Launch the graph
        cudaGraphLaunch(graphExec, stream);
        rgb_tensor = dummyOutput;
           // cudaStreamSynchronize(stream);

            if (!benchmarkMode) {
				writer.addFrame(rgb_tensor);
			}
    }

    if (!benchmarkMode) {
     writer.addFrame(input);
    }
   
   // cudaStreamSynchronize(getWriteStream());  // Synchronize write stream

    useI0AsSource = !useI0AsSource;
}
