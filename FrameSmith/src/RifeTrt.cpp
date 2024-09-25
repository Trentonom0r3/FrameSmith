#include "RifeTrt.hpp"

RifeTrt::RifeTrt(std::string modelName, int interpFactor, int width, int height,
	bool half, bool benchmark, std::string outputPath, int fps)
	: TRTBase(modelName, interpFactor, width, height, half, benchmark,
		torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, outputPath, fps), firstRun(true), useI0AsSource(true)
{
	writer = new FFmpegWriter(outputPath, width, height, fps * interpFactor, benchmark);

	I0 = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
	I1 = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
	dummyInput = torch::empty({ 1, 7, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
	dummyOutput = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();

	for (int i = 0; i < interpFactor - 1; ++i) {
		auto timestep = torch::full({ 1, 1, height, width }, (i + 1) * 1.0 / interpFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();
		timestep_tensors.push_back(timestep);
	}

	handleModel({ 1, 7, height, width }, { 1, 7, height, width }, { 1, 7, height, width });

	// Set the input and output tensor addresses in the TensorRT context
	context->setTensorAddress("input", dummyInput.data_ptr());
	context->setTensorAddress("output", dummyOutput.data_ptr());
}

RifeTrt::~RifeTrt()
{
}

void RifeTrt::run(at::Tensor input)
{
	if (firstRun) {
		// Asynchronously copy the input to I0 using the provided CUDA inferenceStream
		cudaMemcpyAsync(I0.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
		firstRun = false;
	}

	// Alternate between I0 and I1 for source and destination
	auto& source = useI0AsSource ? I0 : I1;
	auto& destination = useI0AsSource ? I1 : I0;
	cudaMemcpyAsync(destination.data_ptr(), input.data_ptr(), input.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
	cudaMemcpyAsync(dummyInput.slice(1, 0, 3).data_ptr(), source.data_ptr(), source.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);
	cudaMemcpyAsync(dummyInput.slice(1, 3, 6).data_ptr(), destination.data_ptr(), destination.nbytes(), cudaMemcpyDeviceToDevice, inferenceStream);

	// Perform interpolation for the required number of frames
	for (int i = 0; i < factor - 1; ++i) {

		cudaMemcpyAsync(dummyInput.slice(1, 6, 7).data_ptr(), timestep_tensors[i].data_ptr(), timestep_tensors[i].nbytes(),
			cudaMemcpyDeviceToDevice, inferenceStream);

		// Enqueue inference using the correct stream
		context->enqueueV3(inferenceStream);

		// Synchronize to ensure inference is complete
		cudaStreamSynchronize(inferenceStream);

		// Update the source for the next interpolation step
		rgb_tensor = dummyOutput;
		addToWriter(writer, rgb_tensor, half, benchmarkMode);
	}
	// Flip the source flag for the next run
	useI0AsSource = !useI0AsSource;
	addToWriter(writer, destination, half, benchmarkMode);
}
