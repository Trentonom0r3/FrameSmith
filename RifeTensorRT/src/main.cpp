#include "RifeTensorRT.h"
#include <iostream>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Reader.h"
#include "Writer.h"  // Your FFmpegWriter class

std::queue<AVFrame*> frameQueue;
std::mutex mtx;
std::condition_variable cv;
std::atomic<bool> doneReading(false);

void readFrames(FFmpegReader& reader) {
    AVFrame* preAllocatedInputFrame = av_frame_alloc();
    while (reader.readFrame(preAllocatedInputFrame)) {
        std::unique_lock<std::mutex> lock(mtx);
        frameQueue.push(av_frame_clone(preAllocatedInputFrame));  // Clone frame
        lock.unlock();
        cv.notify_one();
    }
    doneReading = true;
    cv.notify_all();
}

void processFrames(int& frameCount, FFmpegWriter& writer, RifeTensorRT& rifeTensorRT) {
    AVFrame* interpolatedFrame = av_frame_alloc();  // Reuse this frame
    cudaEvent_t inferenceFinishedEvent;
    cudaEventCreate(&inferenceFinishedEvent);  // Create a CUDA event

    c10::cuda::CUDAStream processStream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStream writeStream = c10::cuda::getStreamFromPool();

    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !frameQueue.empty() || doneReading; });

        if (frameQueue.empty() && doneReading) break;

        AVFrame* frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        if (!frame || !frame->data[0]) {
            std::cerr << "Invalid input frame." << std::endl;
            continue;
        }

        // Perform RIFE interpolation asynchronously
        rifeTensorRT.run(frame, interpolatedFrame, inferenceFinishedEvent);

        // Synchronize the event to ensure inference has completed before copying the data
        cudaEventSynchronize(inferenceFinishedEvent);

        // Convert the output tensor to an AVFrame format (after inference has finished)
        at::Tensor outputTensor = rifeTensorRT.dummyOutput.squeeze(0).permute({ 1, 2, 0 }).mul(255.0).clamp(0, 255).to(torch::kU8).to(torch::kCPU).contiguous();

        // Ensure interpolatedFrame is allocated properly
        interpolatedFrame->format = frame->format;
        interpolatedFrame->width = outputTensor.size(1);
        interpolatedFrame->height = outputTensor.size(0);

        // Allocate memory for interpolatedFrame data
        if (av_frame_get_buffer(interpolatedFrame, 0) < 0) {
            std::cerr << "Error allocating buffer for interpolated frame!" << std::endl;
        }

        // Copy the output tensor data back to AVFrame (this is crucial and was missing).
        memcpy(interpolatedFrame->data[0], outputTensor.data_ptr(), outputTensor.numel());

        // Add to writer queue (optional)
        //writer.addFrame(frame);
        //writer.addFrame(interpolatedFrame);

        frameCount++;
        av_frame_free(&frame);  // Free input frame
    }

    av_frame_free(&interpolatedFrame);  // Free interpolated frame
    cudaEventDestroy(inferenceFinishedEvent);  // Cleanup
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path>" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];

    // Initialize FFmpeg-based video reader
    FFmpegReader reader(inputVideoPath);
    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();

    // Initialize RifeTensorRT
    RifeTensorRT rifeTensorRT("rife4.20-tensorrt", 2, width, height, true, false);

    // Initialize FFmpeg-based video writer
    FFmpegWriter writer(outputVideoPath, width, height, fps * 2);

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Create threads for reading and processing frames
    std::thread readerThread(readFrames, std::ref(reader));
    std::thread processorThread(processFrames, std::ref(frameCount), std::ref(writer), std::ref(rifeTensorRT));

    // Join the threads after work is done
    readerThread.join();
    processorThread.join();

    // Finalize the writer
    writer.finalize();

    // Calculate total processing time and FPS
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount * 2 / duration.count();

    std::cout << "Processed " << frameCount * 2 << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    return 0;
}
