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

void readAndProcessFrames(FFmpegReader& reader, FFmpegWriter* writer, RifeTensorRT& rifeTensorRT, int batchSize, bool benchmarkMode, int& frameCount) {
    AVFrame* preAllocatedInputFrame = av_frame_alloc();
    int batchCounter = 0;  // Counter for batching synchronization

    while (reader.readFrame(preAllocatedInputFrame)) {
        // Asynchronously run TensorRT inference on the frame
        rifeTensorRT.run(preAllocatedInputFrame, *writer);

        frameCount++;
        batchCounter++;

        // After processing batchSize frames, synchronize CUDA streams
        if (batchCounter >= batchSize) {
            synchronizeStreams(rifeTensorRT);  // Ensure all previous work is complete
            batchCounter = 0;  // Reset batch counter
        }

        if (benchmarkMode) {
            continue;
        }
    }

    // Ensure any remaining frames are processed after exiting the loop
    if (batchCounter > 0) {
        synchronizeStreams(rifeTensorRT);
    }

    av_frame_free(&preAllocatedInputFrame);  // Free the frame memory after done
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path> <model_name> <interpolation_factor> [--benchmark]" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelName = argv[3];
    int interpolationFactor = std::stoi(argv[4]);

    bool benchmarkMode = false;
    if (argc > 5 && std::string(argv[5]) == "--benchmark") {
        benchmarkMode = true;
        std::cout << "Benchmark mode enabled." << std::endl;
    }

    // Initialize FFmpeg-based video reader
    FFmpegReader reader(inputVideoPath);
    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    // Initialize RifeTensorRT with the model name and interpolation factor
    RifeTensorRT rifeTensorRT(modelName, interpolationFactor, width, height, true, false, benchmarkMode);

    // Initialize FFmpeg-based video writer if not in benchmark mode
    FFmpegWriter* writer = nullptr;
    if (!benchmarkMode) {
        writer = new FFmpegWriter(outputVideoPath, width, height, fps * interpolationFactor);
    }

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Directly call read and process function with CUDA stream-based concurrency
    readAndProcessFrames(reader, writer, rifeTensorRT, 25, benchmarkMode, frameCount);

    // Finalize the writer if not in benchmark mode
    if (!benchmarkMode && writer != nullptr) {
        writer->finalize();
        delete writer;
    }

    // Calculate total processing time and FPS
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount * interpolationFactor / duration.count();

    std::cout << "Processed " << frameCount * interpolationFactor << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    return 0;
}
