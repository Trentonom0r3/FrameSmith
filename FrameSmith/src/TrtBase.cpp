#include <TrtBase.hpp>

TRTBase::TRTBase(std::string modelName, int factor, int width, int height, bool half, bool benchmark,
	torch::Device device, std::string outputPath, int fps)

	: modelName(modelName), factor(factor), width(width), height(height),
	half(half), benchmarkMode(benchmark), device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
	writer(nullptr), outputPath(outputPath), fps(fps)
{
	// Set device
	//device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	dType = half ? torch::kHalf : torch::kFloat32;

	// Create CUDA stream for inference
	CUDA_CHECK(cudaStreamCreate(&inferenceStream));
} //handlemodel is not called here, left for the user to implement and call explicitly

TRTBase::~TRTBase()
{
    CUDA_CHECK(cudaStreamDestroy(inferenceStream));
	if (writer) {
		delete writer;
	}
}

void TRTBase::handleModel(std::vector<int> minDims, std::vector<int> optDims, std::vector<int> maxDims)
{
	std::string filename = modelsMap(modelName, "onnx", half);
	std::string folderName = modelName;
	folderName.replace(folderName.find("-tensorrt"), 9, "-onnx");

	std::filesystem::path modelPath = std::filesystem::path(getWeightsDir()) / folderName / filename;

	if (!std::filesystem::exists(modelPath)) {
		std::cout << "Model not found, downloading it..." << std::endl;
		modelPath = downloadModels(modelName, "onnx", half);
		if (!std::filesystem::exists(modelPath)) {
			std::cerr << "Failed to download or locate the model: " << modelPath << std::endl;
			return;
		}
	}

	enginePath = TensorRTEngineNameHandler(modelPath.string(), half, optDims);
	std::tie(engine, context) = TensorRTEngineLoader(enginePath);

	if (!engine || !context || !std::filesystem::exists(enginePath)) {
		std::cout << "Loading engine failed, creating a new one" << std::endl;
		std::tie(engine, context) = TensorRTEngineCreator(
			modelPath.string(), enginePath, half, minDims, optDims, maxDims
		);
	}

	bindings = { dummyInput.data_ptr(), dummyOutput.data_ptr() };

	// Set Tensor Addresses and Input Shapes
	for (int i = 0; i < engine->getNbIOTensors(); ++i) {
		const char* tensorName = engine->getIOTensorName(i);
		void* bindingPtr = (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) ?
			static_cast<void*>(dummyInput.data_ptr()) :
			static_cast<void*>(dummyOutput.data_ptr());

		bool setaddyinfo = context->setTensorAddress(tensorName, bindingPtr);

		if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
			bool success = context->setInputShape(tensorName, nvinfer1::Dims4{ optDims[0], optDims[1], optDims[2], optDims[3] });
			if (!success) {
				std::cerr << "Failed to set input shape for " << tensorName << std::endl;
				return;
			}
		}
	}
}

void TRTBase::synchronizeStreams(FFmpegReader& reader)
{
	CUDA_CHECK(cudaStreamSynchronize(reader.getStream()));
	CUDA_CHECK(cudaStreamSynchronize(getInferenceStream()));
	CUDA_CHECK(cudaStreamSynchronize(writer->getStream()));
}

