#include <main.h>

//UNIQUE PTR OF TRT BASE TO PASS IN THE RUN FUNCTION
//define custom name for the unique ptr
using TrtApp = std::unique_ptr<TRTBase>;

void progressBar(const std::atomic<int>& frameCount, int totalFrames,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& startTime,
    const std::atomic<bool>& done) {
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Update every 0.5 seconds

        int current = frameCount.load();
        double progress = (totalFrames > 0) ? static_cast<double>(current) / totalFrames : 0.0;
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - startTime;
        double fps = (elapsed.count() > 0) ? current / elapsed.count() : 0.0;

        // Display colored progress bar
        int barWidth = 50;
        std::cout << "\r[";

        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) {
                std::cout << green("=");
            }
            else if (i == pos) {
                std::cout << yellow(">");
            }
            else {
                std::cout << blue(" ");
            }
        }

        std::cout << "] " << cyan(std::to_string(static_cast<int>(progress * 100.0)) + "% ")
            << "FPS: " << cyan(std::to_string(fps)) << "    ";
        std::cout.flush();
    }

    // Ensure the progress bar completes to 100% with colored indicators
    std::cout << "\r[";
    for (int i = 0; i < 50; ++i) {
        std::cout << green("=");
    }
    std::cout << "] " << magenta("100.00% ")
        << "FPS: " << cyan(std::to_string(
            frameCount.load() /
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count()
        )) << "    " << std::endl;
}


void readAndProcessFrames(FFmpegReader& reader, TrtApp& trtApp, std::atomic<int>& frameCount) {
	torch::Tensor frameTensor = torch::zeros({ 1, 3, trtApp->getHeight(), trtApp->getWidth() },
		torch::TensorOptions().dtype(trtApp->getDtype()).device(trtApp->getDevice()).requires_grad(false));

	size_t freeMem = 0;
	size_t totalMem = 0;
	const float memoryThreshold = 0.75f; // 75% threshold for memory usage
	
	while (reader.readFrame(frameTensor)) {
		// Pass the data pointer if required
		// while (reader.readFrame(frameTensor.data_ptr())) { 
		trtApp->run(frameTensor);

		frameCount++;

		// Check memory usage every 100 frames to avoid excessive checking
		if (frameCount % 100 == 0) {
			cudaMemGetInfo(&freeMem, &totalMem);

			// Calculate memory usage
			float memoryUsed = 1.0f - (static_cast<float>(freeMem) / static_cast<float>(totalMem));

			// If memory usage exceeds 75%, synchronize the streams
			if (memoryUsed >= memoryThreshold) {
				std::cout << "\nMemory usage exceeds 75%. Synchronizing streams..." << std::endl;
				trtApp->synchronizeStreams(reader);
			}
		}
	}
}


int main(int argc, char** argv) {
    auto startTime = std::chrono::high_resolution_clock::now();
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
            argIndex++;
        }
        else if (arg == "--benchmark") {
            benchmarkMode = true;
            std::cout << yellow("Benchmark mode enabled.") << std::endl;
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
    std::atomic<int> frameCount(0);
    int totalFrames = reader.getTotalFrames(); // Ensure this is called before processing starts

    TrtApp trtApp;
    if (mode == "upscale") {
        std::cout << yellow("Upscaling video to ") << cyan(std::to_string(width* factor) + "x" + std::to_string(height * factor)) << yellow(" using ") << yellow(modelName) << std::endl;
            trtApp = std::make_unique<UpscaleTrt>(modelName, factor, width, height, halfPrecision, benchmarkMode, outputVideoPath, fps);
    }
    else if (mode == "interpolate") {
        std::cout << yellow("Interpolating video to ") << cyan(std::to_string(fps * factor) + " FPS") << yellow(" using ") << yellow(modelName) << std::endl;
            trtApp = std::make_unique<RifeTrt>(modelName, factor, width, height, halfPrecision, benchmarkMode, outputVideoPath, fps);
    }

    // Initialize the 'done' flag
    std::atomic<bool> done(false);

    // Start the progress bar thread
    std::thread progressThread(progressBar, std::cref(frameCount), totalFrames, startTime, std::cref(done));

    // Directly call read and process function with CUDA stream-based concurrency
    readAndProcessFrames(reader, trtApp, frameCount);
    trtApp->synchronizeStreams(reader);

    // Signal the progress bar to finish
    done.store(true);

    // Wait for the progress bar thread to finish
    if (progressThread.joinable()) {
        progressThread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = endTime - startTime;

    double processingFPS = 0;


    if (mode == "upscale") {
        std::cout << green("Upscaling Complete. ");
        processingFPS = frameCount.load() / duration.count();

        std::cout << "Processed " << frameCount.load() << " frames in "
            << duration.count() << " seconds." << std::endl;

        std::cout << yellow("Processing FPS: ") << cyan(std::to_string(processingFPS))
            << " frames per second." << std::endl;

    }
    else {
        std::cout << green("Interpolation Complete. ");
        processingFPS = (frameCount.load() * factor) / duration.count();

        std::cout << "Processed " << frameCount.load() * factor << " frames in "
            << duration.count() << " seconds." << std::endl;
        std::cout << yellow("Processing FPS: ") << cyan(std::to_string(processingFPS))
            << " frames per second." << std::endl;
    }

    return 0;
}
