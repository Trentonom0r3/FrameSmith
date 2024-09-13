#include "RifeTensorRT.h"
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include "Reader.h"
#include "Writer.h"  // Your FFmpegWriter class
#include <cuda_runtime.h>

// Helper to synchronize CUDA stream after batchSize frames
void synchronizeStreams(RifeTensorRT& rifeTensorRT) {
    cudaStreamSynchronize(rifeTensorRT.getStream());  // Synchronize inference stream
    cudaStreamSynchronize(rifeTensorRT.getWriteStream());  // Synchronize write stream
}

void readAndProcessFrames(FFmpegReader& reader, RifeTensorRT& rifeTensorRT, int batchSize, bool benchmarkMode, int& frameCount) {
    AVFrame* preAllocatedInputFrame = av_frame_alloc();
    int batchCounter = 0;  // Counter for batching synchronization

    while (reader.readFrame(preAllocatedInputFrame)) {
        // Asynchronously run TensorRT inference on the frame
        rifeTensorRT.run(preAllocatedInputFrame);

        frameCount++;
        batchCounter++;


    }
    synchronizeStreams(rifeTensorRT);  // Synchronize the last batch
    av_frame_free(&preAllocatedInputFrame);  // Free the frame memory after done
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
            << " <input_video_path> <output_video_path> <model_name> <interpolation_factor>"
            << "[--benchmark]" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelName = argv[3];
    int interpolationFactor = std::stoi(argv[4]);

    int batchSize = 25;  // Default batch size
    bool benchmarkMode = false;

    // Parse optional arguments
    int argIndex = 5;
    while (argIndex < argc) {
        std::string arg = argv[argIndex];
        if (arg == "--benchmark") {
            benchmarkMode = true;
            std::cout << "Benchmark mode enabled." << std::endl;
            argIndex++;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return -1;
        }
    }

    // Initialize FFmpeg-based video reader
    FFmpegReader reader(inputVideoPath);
    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    FFmpegWriter* writer = new FFmpegWriter(outputVideoPath, width, height, fps * interpolationFactor);

    // Initialize RifeTensorRT with the model name and interpolation factor
    RifeTensorRT rifeTensorRT(modelName,interpolationFactor, width, height, true, false, benchmarkMode, *writer);

    // Initialize FFmpeg-based video writer if not in benchmark mode
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Directly call read and process function with CUDA stream-based concurrency
    readAndProcessFrames(reader, rifeTensorRT, batchSize, benchmarkMode, frameCount);

    // Finalize the writer if not in benchmark mode
    if (!benchmarkMode && writer != nullptr) {
        writer->finalize();
        delete writer;
    }

    // Calculate total processing time and FPS
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount * interpolationFactor / duration.count();

    std::cout << "Processed " << frameCount * interpolationFactor << " frames in "
        << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    return 0;
}