#include <iostream>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Reader.h"
#include "Writer.h"

std::queue<AVFrame*> frameQueue;
std::mutex mtx;
std::condition_variable cv;
std::atomic<bool> doneReading(false);

void readFrames(FFmpegReader& reader) {
    AVFrame* preAllocatedInputFrame = av_frame_alloc();
    int frameIndex = 0;  // Keep track of frames being read
    while (reader.readFrame(preAllocatedInputFrame)) {
        if (preAllocatedInputFrame && preAllocatedInputFrame->data[0]) {
            std::cout << "Read and transferred frame " << frameIndex << " - pushing to queue" << std::endl;
            std::unique_lock<std::mutex> lock(mtx);
            frameQueue.push(av_frame_clone(preAllocatedInputFrame));  // Clone the input frame to preserve original
            lock.unlock();
            cv.notify_one();
            frameIndex++;
        }
        else {
            std::cerr << "Error reading or transferring frame " << frameIndex << std::endl;
        }
    }
    std::cout << "Finished reading all frames. Total frames read: " << frameIndex << std::endl;
    doneReading = true;
    cv.notify_all();
}


void processFrames(int& frameCount, FFmpegWriter& writer) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !frameQueue.empty() || doneReading; });

        if (frameQueue.empty() && doneReading) {
            std::cout << "No more frames to process, exiting..." << std::endl;
            break;  // Exit when no more frames
        }

        AVFrame* frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        if (!frame || !frame->data[0]) {
            std::cerr << "Invalid input frame or frame has no data." << std::endl;
            continue;
        }

        std::cout << "Processing frame " << frameCount << std::endl;
        writer.addFrame(frame);  // Write the frame directly to the writer
        frameCount++;  // Increment the frame count

        av_frame_free(&frame);  // Free the frame after processing
    }

    std::cout << "Processed " << frameCount << " frames." << std::endl;
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

    FFmpegReader reader(inputVideoPath);
    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    FFmpegWriter writer(outputVideoPath, width, height, fps);  // No need for doubled FPS

    cv::Mat frame;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Thread for reading frames
    std::thread readerThread(readFrames, std::ref(reader));

    // Thread for processing (just writing frames here)
    std::thread processorThread(processFrames, std::ref(frameCount), std::ref(writer));

    readerThread.join();
    processorThread.join();

    writer.finalize();  // Finalize the writer to close the file

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount / duration.count();

    std::cout << "Processed " << frameCount << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    writer.release();
    cap.release();
    return 0;
}
