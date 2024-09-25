#include <main.h>

//UNIQUE PTR OF TRT BASE TO PASS IN THE RUN FUNCTION
//define custom name for the unique ptr
using TrtApp = std::unique_ptr<TRTBase>;

void readAndProcessFrames(FFmpegReader& reader, TrtApp& trtApp, int& frameCount) {
	torch::Tensor frameTensor = torch::zeros({ 1, 3, trtApp->getHeight(), trtApp->getWidth() },
		torch::TensorOptions().dtype(trtApp->getDtype()).device(trtApp->getDevice()).requires_grad(false));

	size_t freeMem = 0;
	size_t totalMem = 0;
	const float memoryThreshold = 0.75f; // 75% threshold for memory usage
	std::cout << "Processing frames..." << std::endl;
	while (reader.readFrame(frameTensor)) {
		//but instead do   while (reader.readFrame(frameTensor.data_ptr())) { to pass void* data_ptr
		trtApp->run(frameTensor);

		frameCount++;

		// Check memory usage every 10 frames to avoid excessive checking
		if (frameCount % 100 == 0) {
			cudaMemGetInfo(&freeMem, &totalMem);

			// Calculate memory usage
			float memoryUsed = 1.0f - (static_cast<float>(freeMem) / static_cast<float>(totalMem));

			// If memory usage exceeds 75%, synchronize the streams
			if (memoryUsed >= memoryThreshold) {
				std::cout << "Memory usage exceeds 75%. Synchronizing streams..." << std::endl;
				trtApp->synchronizeStreams(reader);
			}
		}
	}
}

int main(int argc, char** argv) {
	printASCII();
	if (argc < 7) { // Minimum required arguments
		printUsage(argv[0]);
		return -1;
	}

	std::string inputVideoPath = argv[1];
	std::string outputVideoPath = argv[2];

	// Initialize default values
	std::string mode;
	std::string modelName;
	int factor = 1;
	bool halfPrecision = false;
	bool benchmarkMode = false;

	// Parse arguments
	int argIndex = 3;
	while (argIndex < argc) {
		std::string arg = argv[argIndex];

		if (arg == "--mode") {
			if (argIndex + 3 > argc) { // Ensure enough arguments for mode
				std::cerr << "Error: --mode requires <upscale|interpolate> <model_name> <factor>" << std::endl;
				printUsage(argv[0]);
				return -1;
			}
			mode = argv[argIndex + 1];
			std::cout << "Running in " << mode << " mode." << std::endl;
			modelName = argv[argIndex + 2];
			factor = std::stoi(argv[argIndex + 3]);

			// Validate mode
			if (mode != "upscale" && mode != "interpolate") {
				std::cerr << "Error: --mode must be either 'upscale' or 'interpolate'." << std::endl;
				printUsage(argv[0]);
				return -1;
			}

			// Validate factor
			if (factor <= 0) {
				std::cerr << "Error: <factor> must be a positive integer." << std::endl;
				printUsage(argv[0]);
				return -1;
			}

			argIndex += 4; // Move past mode, modelName, and factor
		}
		else if (arg == "--half") {
			halfPrecision = true;
			//std::cout << "Half precision enabled." << std::endl;
			argIndex++;
		}
		else if (arg == "--benchmark") {
			benchmarkMode = true;
			std::cout << "Benchmark mode enabled." << std::endl;
			argIndex++;
		}
		else {
			std::cerr << "Unknown argument: " << arg << std::endl;
			printUsage(argv[0]);
			return -1;
		}
	}

	// Validate that mode was provided
	if (mode.empty()) {
		std::cerr << "Error: --mode <upscale|interpolate> must be specified." << std::endl;
		printUsage(argv[0]);
		return -1;
	}

	// Determine device
	torch::Device device = torch::kCUDA;

	  // Initialize FFmpeg-based video reader
	FFmpegReader reader(inputVideoPath, device, halfPrecision);

	int width = reader.getWidth();
	int height = reader.getHeight();
	double fps = reader.getFPS();
	int frameCount = 0;
	auto startTime = std::chrono::high_resolution_clock::now();

	TrtApp trtApp;
	if (mode == "upscale") {
		trtApp = std::make_unique<UpscaleTrt>(modelName, factor, width, height, halfPrecision, benchmarkMode, outputVideoPath, fps);
	}
	else if (mode == "interpolate") {
		trtApp = std::make_unique<RifeTrt>(modelName, factor, width, height, halfPrecision, benchmarkMode, outputVideoPath, fps);
	}
		startTime = std::chrono::high_resolution_clock::now();

		// Directly call read and process function with CUDA stream-based concurrency
		readAndProcessFrames(reader, trtApp, frameCount);
		trtApp->synchronizeStreams(reader);

		auto endTime = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> duration = endTime - startTime;

		double processingFPS = 0;

		if (mode == "upscale") {
			std::cout << "Upscaling Complete. ";
			processingFPS = frameCount / duration.count();

			std::cout << "Processed " << frameCount << " frames in "
				<< duration.count() << " seconds." << std::endl;

			std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

		}
		else {
			std::cout << "Interpolation Complete. ";
			processingFPS = frameCount * factor / duration.count();

			std::cout << "Processed " << frameCount * factor << " frames in "
				<< duration.count() << " seconds." << std::endl;
			std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;
		}

	return 0;
}
