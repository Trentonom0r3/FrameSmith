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

using namespace nvinfer1;
using namespace nvonnxparser;
namespace fs = std::filesystem;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
     
         if (severity == Severity::kINFO) {
            std::cerr << cyan("[TensorRT INFO] ") << msg << std::endl;
        }
        else if (severity == Severity::kWARNING) {
            std::cerr << yellow("[TensorRT WARNING] ") << msg << std::endl;
        }
        else if (severity == Severity::kERROR) {
            std::cerr << red("[TensorRT ERROR] ") << msg << std::endl;
        }
        else if (severity == Severity::kINTERNAL_ERROR) {
            std::cerr << red("[TensorRT INTERNAL ERROR] ") << msg << std::endl;
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
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << red("Failed to create TensorRT Builder") << std::endl;
        return { nullptr, nullptr };
    }

    // Create the network definition
    INetworkDefinition* network = builder->createNetworkV2(0U);
    if (!network) {
        std::cerr << red("Failed to create TensorRT Network Definition") << std::endl;
        return { nullptr, nullptr };
    }
    // Parse the ONNX model
    IParser* parser = createParser(*network, gLogger);
    if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        std::cerr << red("Failed to parse ONNX file: ") << modelPath << std::endl;
        std::cerr << red("Check that the ONNX model is valid and compatible with TensorRT") << std::endl;
        return { nullptr, nullptr };
    }
    // Create optimization profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << red("Failed to create TensorRT Optimization Profile") << std::endl;
        return { nullptr, nullptr };
    }

    bool b1 = profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims4{ 1, 7, inputsMin[2], inputsMin[3] });
    bool b2 = profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims4{ 1, 7, inputsOpt[2], inputsOpt[3] });
    bool b3 = profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims4{ 1, 7, inputsMax[2], inputsMax[3] });
    
    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << red("Failed to create TensorRT Builder Config") << std::endl;
        return { nullptr, nullptr };
    }
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, maxWorkspaceSize);

    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    
    IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << red("Failed to serialize TensorRT Engine") << std::endl;
        return { nullptr, nullptr };
    }
  

    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << red("Could not open engine file for writing: ") << enginePath << std::endl;
        return { nullptr, nullptr };
    }
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
   

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << red("Failed to create TensorRT Runtime") << std::endl;
        return { nullptr, nullptr };
    }
    

    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << red("Failed to deserialize TensorRT Engine") << std::endl;
        return { nullptr, nullptr };
    }


    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << red("Failed to create TensorRT Execution Context") << std::endl;
        return { nullptr, nullptr };
    }

    return { engine, context };
}

std::pair<ICudaEngine*, IExecutionContext*> TensorRTEngineLoader(const std::string& enginePath) {
    std::cout << yellow("Loading TensorRT engine from file: ") << enginePath << std::endl;

    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << red("Failed to open engine file: ") << enginePath << std::endl;
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
        std::cerr << red("Failed to create TensorRT Runtime") << std::endl;
        return { nullptr, nullptr };
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine) {
        std::cerr << red("Failed to deserialize TensorRT Engine") << std::endl;
        return { nullptr, nullptr };
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << red("Failed to create TensorRT Execution Context") << std::endl;
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
