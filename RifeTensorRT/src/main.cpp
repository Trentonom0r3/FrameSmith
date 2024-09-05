#include <iostream>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <torch/torch.h>  // Include PyTorch for tensor handling
#include "Reader.h"  // Updated reader header using tensors
#include "Writer.h"  // Updated writer header using tensors

std::queue<torch::Tensor> frameQueue;
std::mutex mtx;
std::condition_variable cv;
std::atomic<bool> doneReading(false);

// Function for reading frames into a tensor queue
void readFrames(FFmpegReader& reader) {
    torch::Tensor preAllocatedInputTensor;  // GPU tensor
    while (reader.readFrame(preAllocatedInputTensor)) {  // Updated to read into tensor
        std::unique_lock<std::mutex> lock(mtx);
        frameQueue.push(preAllocatedInputTensor.clone());  // Clone tensor to preserve original
        lock.unlock();
        cv.notify_one();
    }
    doneReading = true;
    cv.notify_all();
}

// Function for processing (writing) frames from the tensor queue
void processFrames(int& frameCount, FFmpegWriter& writer) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !frameQueue.empty() || doneReading; });

        if (frameQueue.empty() && doneReading) break;  // Exit when no more frames

        torch::Tensor tensor = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        writer.addFrame(tensor);  // Write tensor directly to the writer
        frameCount++;  // Increment the frame count
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path>" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];

    // Initialize FFmpeg reader and writer
    FFmpegReader reader(inputVideoPath);
    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    FFmpegWriter writer(outputVideoPath, width, height, fps);

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Thread for reading frames into tensors
    std::thread readerThread(readFrames, std::ref(reader));

    // Thread for processing (writing frames from tensors)
    std::thread processorThread(processFrames, std::ref(frameCount), std::ref(writer));

    // Wait for both threads to finish
    readerThread.join();
    processorThread.join();

    writer.finalize();  // Finalize the writer to close the output file

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount / duration.count();

    std::cout << "Processed " << frameCount << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    return 0;
}
