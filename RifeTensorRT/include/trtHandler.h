#pragma once

#include "coloredPrints.h"
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvonnxparser;
namespace fs = std::filesystem;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            std::cerr << yellow("[TensorRT] ") << msg << std::endl;
        }
    }
};

static Logger gLogger;
std::pair<ICudaEngine*, IExecutionContext*> TensorRTEngineCreator(
    const std::string& modelPath,
    const std::string& enginePath,
    bool fp16,
    const std::vector<int>& inputsMin,
    const std::vector<int>& inputsOpt,
    const std::vector<int>& inputsMax,
    const std::string& inputName = "input",
    size_t maxWorkspaceSize = (1 << 30),
    int optimizationLevel = 3
) {
    std::string toPrint = "Model engine not found, creating engine for model: " + modelPath + ", this may take a while...";
    std::cout << yellow(toPrint) << std::endl;

    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT Builder" << std::endl;
        return { nullptr, nullptr };
    }

    // Create the network definition
    INetworkDefinition* network = builder->createNetworkV2(0U);
    if (!network) {
        std::cerr << "Failed to create TensorRT Network Definition" << std::endl;
        return { nullptr, nullptr };
    }

    // Parse the ONNX model
    IParser* parser = createParser(*network, gLogger);
    if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Model path: " << modelPath << std::endl;  // Added for debugging
        std::cerr << "Failed to parse ONNX file" << std::endl;
        std::cerr << "Check that the ONNX model is valid and compatible with TensorRT" << std::endl;
        return { nullptr, nullptr };
    }

    // Create optimization profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();

    // Note: Correct the dimension at axis 1 (number of channels) from 1 to 7.
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims4{ 1, 7, inputsMin[1], inputsMin[2] });
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims4{ 1, 7, inputsOpt[1], inputsOpt[2] });
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims4{ 1, 7, inputsMax[1], inputsMax[2] });

    // Configure builder
    IBuilderConfig* config = builder->createBuilderConfig();
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, maxWorkspaceSize);  // Updated for newer TensorRT

    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Build the engine (updated method)
    IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "Failed to serialize TensorRT Engine" << std::endl;
        return { nullptr, nullptr };
    }

    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Could not open engine file for writing" << std::endl;
        return { nullptr, nullptr };
    }
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT Runtime" << std::endl;
        return { nullptr, nullptr };
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT Engine" << std::endl;
        return { nullptr, nullptr };
    }

    // Create execution context
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << red("Failed to create TensorRT Execution Context") << std::endl;
        return { nullptr, nullptr };
    }


std::pair<ICudaEngine*, IExecutionContext*> TensorRTEngineLoader(const std::string& enginePath) {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return { nullptr, nullptr };
    }

    engineFile.seekg(0, engineFile.end);
    const size_t fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT Runtime" << std::endl;
        return { nullptr, nullptr };
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT Engine" << std::endl;
        return { nullptr, nullptr };
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create TensorRT Execution Context" << std::endl;
        return { nullptr, nullptr };
    }

    return { engine, context };
}

std::string TensorRTEngineNameHandler(const std::string& modelPath, bool fp16, const std::vector<int>& optInputShape) {
    std::string enginePrecision = fp16 ? "fp16" : "fp32";
    int height = optInputShape[2];
    int width = optInputShape[3];
    std::string engineName = modelPath;
    size_t pos = engineName.find(".onnx");
    if (pos != std::string::npos) {
        engineName.replace(pos, 5, "_" + enginePrecision + "_" + std::to_string(height) + "x" + std::to_string(width) + ".engine");
    }
    return engineName;
}
