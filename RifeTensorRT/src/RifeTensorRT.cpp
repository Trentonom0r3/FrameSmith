#include "RifeTensorRT.h"

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
	cudaStreamCreate(&writeinferenceStream);
	//writer.setinferenceStream(writeinferenceStream);
	for (int i = 0; i < interpolateFactor - 1; ++i) {
		auto timestep = torch::full({ 1, 1, height, width }, (i + 1) * 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();
		timestep_tensors.push_back(timestep);
	}
}

RifeTensorRT::~RifeTensorRT() {
	cudaStreamDestroy(inferenceStream);
	cudaStreamDestroy(writeinferenceStream);
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
			bool success = context->setInputShape(tensorName, nvinfer1::Dims4{ 1, 7, height, width });
			if (!success) {
				std::cerr << "Failed to set input shape for " << tensorName << std::endl;
				return;
			}
		}
	}

	firstRun = true;
	useI0AsSource = true;
}

void addToWriter(FFmpegWriter& writer, torch::Tensor& rgb_tensor, bool half, bool benchmarkMode) {
	if (half) {
		auto input_fp16 = rgb_tensor.to(torch::kFloat16).contiguous();
		// Cast c10::Half* to __half*
		const __half* data_ptr = reinterpret_cast<const __half*>(input_fp16.data_ptr<c10::Half>());
		writer.addFrame(data_ptr, benchmarkMode);
	}
	else {
		auto input_fp32 = rgb_tensor.to(torch::kFloat32).contiguous();
		const float* data_ptr = input_fp32.data_ptr<float>();
		writer.addFrame(data_ptr, benchmarkMode);
	}
}

void RifeTensorRT::run(at::Tensor input) {
	// Use the CUDA inferenceStream for all operations to enable asynchronous execution

	if (firstRun) {
		// Asynchronously copy the input to I0 using the provided CUDA inferenceStream
		cudaMemcpyAsync(I0.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
		firstRun = false;
		//addToWriter(writer, I0, half, benchmarkMode);
	}

	// Alternate between I0 and I1 for source and destination
	auto& source = useI0AsSource ? I0 : I1;
	auto& destination = useI0AsSource ? I1 : I0;

	// Asynchronously copy input data
	cudaMemcpyAsync(destination.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
	cudaMemcpyAsync(dummyInput.slice(1, 0, 3).data_ptr(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
	cudaMemcpyAsync(dummyInput.slice(1, 3, 6).data_ptr(), destination.data_ptr(), destination.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);

	// Prepare input tensor
	// Perform interpolation for the required number of frames
	for (int i = 0; i < interpolateFactor - 1; ++i) {
		cudaMemcpyAsync(dummyInput.slice(1, 6, 7).data_ptr(), timestep_tensors[i].data_ptr(), timestep_tensors[i].nbytes(),
			cudaMemcpyDeviceToDevice, inferenceStream);

		// Set the input and output tensor addresses in the TensorRT context
		context->setTensorAddress("input", dummyInput.data_ptr());
		context->setTensorAddress("output", dummyOutput.data_ptr());

		// Enqueue inference using the correct stream
		context->enqueueV3(inferenceStream);

		// Synchronize to ensure inference is complete
		cudaStreamSynchronize(inferenceStream);

		// Update the source for the next interpolation step
		rgb_tensor = dummyOutput;

		// Add the interpolated frame asynchronously
		addToWriter(writer, rgb_tensor, half, benchmarkMode);
	}


	addToWriter(writer, destination, half, benchmarkMode);


	// Flip the source flag for the next run
	useI0AsSource = !useI0AsSource;
}
