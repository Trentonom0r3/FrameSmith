#include "RifeTensorRT.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    // Check for proper command-line argument usage
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path>" << std::endl;
        return -1;
    }

    // Load video using OpenCV
    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << inputVideoPath << std::endl;
        return -1;
    }
    std::cout << "Video file opened successfully." << std::endl;

    // Create RifeTensorRT instance
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    RifeTensorRT rife("rife4.20-tensorrt", 2, width, height, true, false);

    // Retrieve FPS of the video
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Original Video FPS: " << fps << std::endl;

    cv::Mat frame;
    std::ofstream outputStream(outputVideoPath, std::ios::binary);

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << "Processing video..." << std::endl;
    while (cap.read(frame)) {
        // Convert frame to tensor and run interpolation
        at::Tensor tensorFrame = torch::from_blob(frame.data, { frame.rows, frame.cols, 3 }, torch::kByte);
        rife.run(tensorFrame, false, outputStream);

        frameCount++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount / duration.count();

    std::cout << "Processing completed." << std::endl;
    std::cout << "Processed " << frameCount << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    outputStream.close();
    cap.release();
    return 0;
}
