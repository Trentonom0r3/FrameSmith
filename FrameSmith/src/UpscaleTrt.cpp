#include <UpscaleTrt.hpp>

UpscaleTrt::UpscaleTrt(std::string upscaleMethod, int upscaleFactor, int width,
	int height, bool half, bool benchmark, std::string outputPath, int fps)
	: TRTBase(upscaleMethod, upscaleFactor, width, height, half, benchmark, torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, outputPath, fps)
{

	writer = new FFmpegWriter(outputPath, width * upscaleFactor, height * upscaleFactor, fps, benchmark);

	dummyInput = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
	dummyOutput = torch::zeros({ 1, 3, height * upscaleFactor, width * upscaleFactor }, torch::TensorOptions().dtype(dType).device(device)).contiguous();

	handleModel({ 1, 3, 8, 8 }, { 1, 3, height, width }, { 1, 3, height, width });

	// Set the input and output tensor addresses in the TensorRT context
	context->setTensorAddress("input", dummyInput.data_ptr());
	context->setTensorAddress("output", dummyOutput.data_ptr());
}

UpscaleTrt::~UpscaleTrt()
{
}

void UpscaleTrt::run(at::Tensor input)
{
	dummyInput.copy_(input);
	context->enqueueV3(inferenceStream);
	cudaStreamSynchronize(inferenceStream);
	addToWriter(writer, dummyOutput, half, benchmarkMode);
}
