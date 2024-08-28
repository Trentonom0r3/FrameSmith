#include "RifeTensorRT.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <downloadmodels.h>

// Function to check if the model is in the list
bool isModelValid(const std::string& model) {
    const auto& models = modelsList();
    return std::find(models.begin(), models.end(), model) != models.end();
}

int main(int argc, char** argv) {
    // Check for proper command-line argument usage
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path> <model_name>" << std::endl;
        return -1;
    }

    // Load video using OpenCV
    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelName = argv[3]; // Model name is now passed as the third argument

    // Check if the model is valid
    if (!isModelValid(modelName)) {
        std::cerr << "Invalid model name: " << modelName << std::endl;
        std::cerr << "Please choose from the following models:" << std::endl;
        const auto& models = modelsList();
        for (const auto& model : models) {
            std::cerr << " - " << model << std::endl;
        }
        return -1;
    }

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << inputVideoPath << std::endl;
        return -1;
    }
    std::cout << "Video file opened successfully." << std::endl;

    // Create RifeTensorRT instance with the model name
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    RifeTensorRT rife(modelName, 2, width, height, true, false);

    int codec = cv::VideoWriter::fourcc('H', '2', '6', '4');
    cv::VideoWriter writer(outputVideoPath, codec, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer!" << std::endl;
        return 1;
    }

    cv::Mat frame;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << "Processing video..." << std::endl;
    while (cap.read(frame)) {
        at::Tensor tensorFrame = torch::from_blob(frame.data, { frame.rows, frame.cols, 3 }, torch::kByte);
        rife.run(tensorFrame, false, writer);

        frameCount++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount / duration.count();

    std::cout << "Processing completed." << std::endl;
    std::cout << "Processed " << frameCount << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    writer.release();
    cap.release();
    return 0;
}
