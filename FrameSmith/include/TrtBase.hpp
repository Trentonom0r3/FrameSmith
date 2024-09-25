#pragma once
#include <torch/torch.h>
#include <torch/cuda.h>
#include <NvInfer.h>
#include <iostream>
#include <string>
#include <chrono>
#include <queue>
#include <fstream>
#include <filesystem>
#include "downloadModels.h"
#include "coloredPrints.h"
#include <trtHandler.h>
#include <Writer.h>
#include <Reader.h>

class TRTBase {
public:
	TRTBase(std::string modelName, int factor, int width, int height, bool half,
		bool benchmark, torch::Device device, std::string outputPath, int fps);
	virtual ~TRTBase();

	void handleModel(std::vector<int> minDims, std::vector<int> optDims,
		std::vector<int> maxDims);

	virtual void run(at::Tensor input) = 0; // user must implement this method

	void synchronizeStreams(FFmpegReader& reader);
	cudaStream_t getInferenceStream() const { return inferenceStream; }
	int getFactor() const { return factor; } //Returns the factor of the operation (e.g. upscaleFactor, interpolateFactor)
	int getWidth() const { return width; }
	int getHeight() const { return height; }
	torch::ScalarType getDtype() const { return dType; }
	torch::Device getDevice() const { return device; }
	 inline void addToWriter(FFmpegWriter* writer, torch::Tensor& rgb_tensor, bool half, bool benchmarkMode) {
		 if (half) {
			 auto input_fp16 = rgb_tensor.to(torch::kFloat16).contiguous();
			 // Cast c10::Half* to __half*
			 const __half* data_ptr = reinterpret_cast<const __half*>(input_fp16.data_ptr<c10::Half>());
			 writer->addFrame(data_ptr, benchmarkMode);
		 }
		 else {
			 auto input_fp32 = rgb_tensor.to(torch::kFloat32).contiguous();
			 const float* data_ptr = input_fp32.data_ptr<float>();
			 writer->addFrame(data_ptr, benchmarkMode);
		 }
	 }

	std::string modelName;
	std::string outputPath;
	int factor;
	int fps;
	int width;
	int height;
	bool half;
	bool benchmarkMode;
	// Members used for inference
	torch::Device device;
	torch::ScalarType dType;
	cudaStream_t inferenceStream;
	torch::Tensor dummyInput, dummyOutput;

	// TensorRT engine and context
	std::string enginePath;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	std::vector<void*> bindings;

	FFmpegWriter* writer;
};

