#include "RifeTensorRT.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path> <model_name>" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelName = argv[3];

    // Open the input video file using OpenCV.
    cv::VideoCapture capture(inputVideoPath);
    if (!capture.isOpened()) {
        std::cerr << "Could not open input file: " << inputVideoPath << std::endl;
        return -1;
    }

    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));

    // Initialize the RIFE TensorRT model.
    RifeTensorRT rife(modelName, 2, width, height, true, false);

    // Open the output video file using OpenCV.
    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps * 2, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Could not open output file: " << outputVideoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Process the current frame with RIFE.
        cv::Mat interpolatedFrame = rife.run(frame);

        // Write the original and interpolated frames to the output.
        writer.write(frame);
        writer.write(interpolatedFrame);

        frameCount++;
    }

    // Calculate the total processing time and FPS.
    // Calculate the total processing time and FPS.
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    // Adjust frame count to include both original and interpolated frames.
    int totalProcessedFrames = frameCount * 2; // Original frames + interpolated frames
    double processingFPS = totalProcessedFrames / duration.count();

    std::cout << "Processing completed." << std::endl;
    std::cout << "Processed " << totalProcessedFrames << " frames (including interpolated frames) in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    // Release the resources.
    capture.release();
    writer.release();

    return 0;
}
