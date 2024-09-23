#include "RifeTensorRT.h"
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include "Reader.h"
#include "Writer.h"  // Your FFmpegWriter class
#include <cuda_runtime.h>
#include <iomanip>        // For std::fixed and std::setprecision

void synchronizeStreams(RifeTensorRT& rifeTensorRT, FFmpegReader& reader){
    CUDA_CHECK(cudaStreamSynchronize(reader.getStream()));
    CUDA_CHECK(cudaStreamSynchronize(rifeTensorRT.getInferenceStream()));
  
    CUDA_CHECK(cudaStreamSynchronize(rifeTensorRT.writer.getStream()));
}

void readAndProcessFrames(FFmpegReader& reader, RifeTensorRT& rifeTensorRT, bool benchmarkMode, int& frameCount) {
    bool halfPrecision = true;
    torch::Device device(torch::kCUDA);
    torch::Dtype dtype = halfPrecision ? torch::kFloat16 : torch::kFloat32;
    torch::Tensor frameTensor = torch::zeros({ 1, 3, rifeTensorRT.height, rifeTensorRT.width },
        torch::TensorOptions().dtype(dtype).device(device).requires_grad(false));
 
    std::cout << "Processing frames..." << std::endl;
    while (reader.readFrame(frameTensor)) {
    
        rifeTensorRT.run(frameTensor);
        frameCount++;

      
    }
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

    torch::Device device(torch::kCUDA);
    bool halfPrecision = true;

    // Initialize FFmpeg-based video reader
    FFmpegReader reader(inputVideoPath, device, halfPrecision);

    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    std::cout << R"(

     _______   __   ______                  _______   __                            _______   __                     
    |       \ |  \ /      \                |       \ |  \                          |       \ |  \                    
    | $$$$$$$\ \$$|  $$$$$$\ ______        | $$$$$$$\| $$ __    __   _______       | $$$$$$$\| $$ __    __   _______ 
    | $$__| $$|  \| $$_  \$$/      \       | $$__/ $$| $$|  \  |  \ /       \      | $$__/ $$| $$|  \  |  \ /       \
    | $$    $$| $$| $$ \   |  $$$$$$\      | $$    $$| $$| $$  | $$|  $$$$$$$      | $$    $$| $$| $$  | $$|  $$$$$$$
    | $$$$$$$\| $$| $$$$   | $$    $$      | $$$$$$$ | $$| $$  | $$ \$$    \       | $$$$$$$ | $$| $$  | $$ \$$    \ 
    | $$  | $$| $$| $$     | $$$$$$$$      | $$      | $$| $$__/ $$ _\$$$$$$\      | $$      | $$| $$__/ $$ _\$$$$$$\
    | $$  | $$| $$| $$      \$$     \      | $$      | $$ \$$    $$|       $$      | $$      | $$ \$$    $$|       $$
     \$$   \$$ \$$ \$$       \$$$$$$$       \$$       \$$  \$$$$$$  \$$$$$$$        \$$       \$$  \$$$$$$  \$$$$$$$                  
                RIFE FOR C++ WITH TENSORRT. Created by: @Trentonom0r3 - https://github.com/Trentonom0r3
                           Source: https://github.com/Trentonom0r3/RifeTensorRT                                                                                                                                                                                                             
        )" << std::endl;

    FFmpegWriter* writer = new FFmpegWriter(outputVideoPath, width, height, fps * interpolationFactor, benchmarkMode);

    // Initialize RifeTensorRT with the model name and interpolation factor
    RifeTensorRT rifeTensorRT(modelName, interpolationFactor, width, height, true, false, benchmarkMode, *writer);

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Directly call read and process function with CUDA stream-based concurrency
    readAndProcessFrames(reader, rifeTensorRT, benchmarkMode, frameCount);
    // Final synchronization after processing all frames
    synchronizeStreams(rifeTensorRT, reader);

    // Finalize the writer
    writer->finalize();
    delete writer;
    // Calculate total processing time and FPS

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = endTime - startTime;

    double processingFPS = frameCount * interpolationFactor / duration.count();

    std::cout << "Processed " << frameCount * interpolationFactor << " frames in "
        << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    return 0;
}