
#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}
#include <torch/torch.h>

class FFmpegWriter {
public:
    FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps);
    ~FFmpegWriter();
    void addFrame(const at::Tensor& inputTensor);  // Add tensor frame (in GPU memory) to the queue
    void writeThread();  // Thread for writing frames
    void finalize();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* stream = nullptr;
    AVPacket* packet = nullptr;
    int width, height, fps;
    int64_t pts = 0;

    // Threading components
    std::queue<at::Tensor> frameQueue;  // Use tensor queue instead of AVFrame*
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> doneReadingFrames{ false };
    std::thread writerThread;

    void writeFrame(const at::Tensor& inputTensor);  // Internal method for writing tensor frames
};

inline FFmpegWriter::FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps)
    : width(width), height(height), fps(fps) {

    avformat_alloc_output_context2(&formatCtx, nullptr, "mp4", outputFilePath.c_str());

    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");  // Use NVENC encoder
    if (!codec) {
        std::cerr << "Error finding NVENC codec." << std::endl;
        return;
    }
    codecCtx = avcodec_alloc_context3(codec);
    codecCtx->codec_id = codec->id;
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = { 1, fps };
    codecCtx->framerate = { fps, 1 };
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;
    codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;  // NVENC can encode YUV420P directly from GPU

    // Use CRF for variable bitrate
    av_opt_set(codecCtx->priv_data, "crf", "23", 0);  // Adjust CRF value (18-28 is typical)

    // Use multiple threads for encoding
    codecCtx->thread_count = std::thread::hardware_concurrency();
    codecCtx->thread_type = FF_THREAD_FRAME;  // Frame-based threading

    avcodec_open2(codecCtx, codec, nullptr);

    avio_open(&formatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE);
    stream = avformat_new_stream(formatCtx, codec);
    stream->time_base = { 1, fps };
    avcodec_parameters_from_context(stream->codecpar, codecCtx);
    avformat_write_header(formatCtx, nullptr);

    packet = av_packet_alloc();

    writerThread = std::thread(&FFmpegWriter::writeThread, this);
}

inline void FFmpegWriter::addFrame(const at::Tensor& inputTensor) {
    std::unique_lock<std::mutex> lock(mtx);

    // Add the tensor directly to the frame queue
    frameQueue.push(inputTensor.clone());
    lock.unlock();
    cv.notify_one();  // Notify the writer thread that a new frame is available
}

// Thread method for writing frames from the queue
inline void FFmpegWriter::writeThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !frameQueue.empty() || doneReadingFrames; });

        if (frameQueue.empty() && doneReadingFrames) break;

        at::Tensor tensorFrame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        // Write the tensor frame
        writeFrame(tensorFrame);
    }
}

// Internal method for writing tensor frames
#include <cuda_runtime.h>

inline void FFmpegWriter::writeFrame(const at::Tensor& inputTensor) {
    if (!inputTensor.is_cuda()) {
        std::cerr << "Input tensor must be on the GPU." << std::endl;
        return;
    }

    // Set up the NVENC frame for encoding from the GPU tensor
    AVFrame* frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_NV12;  // Assuming the tensor is in NV12 format
    frame->width = width;
    frame->height = height;

    // Ensure the input tensor has the correct layout (Y plane followed by UV plane)
    int y_size = width * height;
    int uv_size = (width / 2) * (height / 2) * 2;  // UV is half resolution, so half width and half height

    // Debug print the input tensor sizes
    std::cout << "Input tensor sizes: " << inputTensor.sizes() << std::endl;

    // Check if the tensor has the expected size and layout (assuming NV12 is represented with 2 channels)
    if (inputTensor.sizes() != at::IntArrayRef({ 1, 3, height, width })) {
        std::cerr << "Error: Tensor must be in [1, 3, H, W] format (3 channels for NV12)." << std::endl;
        return;
    }

    // Split the tensor into Y and UV planes
    at::Tensor y_plane = inputTensor[0][0].contiguous();
    at::Tensor uv_plane = inputTensor[0][1].contiguous();

    // Debug print the sizes of Y and UV planes
    std::cout << "Y plane size: " << y_plane.sizes() << std::endl;
    std::cout << "UV plane size: " << uv_plane.sizes() << std::endl;

    // Check if the tensor planes are contiguous
    if (!y_plane.is_contiguous() || !uv_plane.is_contiguous()) {
        std::cerr << "Error: Tensor planes are not contiguous!" << std::endl;
        return;
    }

    // Allocate GPU memory for the NV12 format (Y and UV planes)
    uint8_t* d_nv12_y;
    uint8_t* d_nv12_uv;
    cudaMalloc(&d_nv12_y, y_size);
    cudaMalloc(&d_nv12_uv, uv_size);

    // Explicitly copy data from the input tensor to the allocated GPU memory (for both Y and UV planes)
    std::cout << "Copying Y plane to GPU..." << std::endl;
    cudaMemcpy(d_nv12_y, y_plane.data_ptr(), y_size, cudaMemcpyDeviceToDevice);

    std::cout << "Copying UV plane to GPU..." << std::endl;
    cudaMemcpy(d_nv12_uv, uv_plane.data_ptr(), uv_size, cudaMemcpyDeviceToDevice);

    // Map the copied data into the AVFrame's data pointers
    frame->data[0] = d_nv12_y;   // Luma (Y plane)
    frame->data[1] = d_nv12_uv;  // Chroma (UV plane)

    // Set the PTS value for the frame
    frame->pts = pts++;

    // Send the frame to NVENC for encoding
    std::cout << "Sending frame to NVENC..." << std::endl;
    int ret = avcodec_send_frame(codecCtx, frame);
    if (ret < 0) {
        std::cerr << "Error sending frame for encoding: " << ret << std::endl;
        return;
    }

    // Receive and write the encoded packet
    while (ret >= 0) {
        ret = avcodec_receive_packet(codecCtx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        else if (ret < 0) {
            std::cerr << "Error encoding frame: " << ret << std::endl;
            return;
        }

        // Rescale PTS and DTS to match the stream's time base
        packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
        packet->dts = packet->pts;
        packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);

        // Write the encoded packet to the output file
        packet->stream_index = stream->index;
        av_interleaved_write_frame(formatCtx, packet);
        av_packet_unref(packet);
    }

    // Free the frame and the allocated GPU memory
    av_frame_free(&frame);
    cudaFree(d_nv12_y);
    cudaFree(d_nv12_uv);

    std::cout << "Frame written successfully!" << std::endl;
}


inline void FFmpegWriter::finalize() {
    doneReadingFrames = true;  // Indicate that no more frames will be added
    cv.notify_all();  // Wake up the writer thread to finish processing

    if (writerThread.joinable()) {
        writerThread.join();  // Wait for the writer thread to finish
    }

    std::cout << "Finalizing the writer..." << std::endl;

    // Flush the encoder
    if (codecCtx) {
        int ret = avcodec_send_frame(codecCtx, nullptr);  // Flush the encoder
        while (ret >= 0) {
            ret = avcodec_receive_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            else if (ret < 0) {
                std::cerr << "Error receiving packet during finalization: " << ret << std::endl;
                return;
            }

            packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
            packet->dts = packet->pts;
            packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;

            av_interleaved_write_frame(formatCtx, packet);
            av_packet_unref(packet);
        }
    }

    // Write the trailer to finalize the file
    if (formatCtx && formatCtx->pb) {
        av_write_trailer(formatCtx);
        avio_closep(&formatCtx->pb);
    }

    avformat_free_context(formatCtx);
    avcodec_free_context(&codecCtx);
    av_packet_free(&packet);

    std::cout << "Finalization complete." << std::endl;
}

inline FFmpegWriter::~FFmpegWriter() {
    if (writerThread.joinable()) {
        writerThread.join();
    }
}
