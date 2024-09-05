#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <torch/torch.h>  // Include PyTorch for tensor handling

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
}

class FFmpegWriter {
public:
    FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps);
    ~FFmpegWriter();
    void addFrame(torch::Tensor& tensor);  // Add frame from a tensor
    void writeThread();  // Thread for writing frames
    void finalize();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* stream = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    int width, height, fps;
    int64_t pts = 0;

    // Threading components
    std::queue<torch::Tensor> frameQueue;  // Queue for tensor frames
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> doneReadingFrames{ false };
    std::thread writerThread;

    void writeFrame(torch::Tensor& tensor);  // Internal method for writing tensors
};

FFmpegWriter::FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps)
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
    codecCtx->pix_fmt = AV_PIX_FMT_NV12;  // Set to NV12 format

    avcodec_open2(codecCtx, codec, nullptr);

    avio_open(&formatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE);
    stream = avformat_new_stream(formatCtx, codec);
    stream->time_base = { 1, fps };
    avcodec_parameters_from_context(stream->codecpar, codecCtx);
    avformat_write_header(formatCtx, nullptr);

    frame = av_frame_alloc();
    frame->format = codecCtx->pix_fmt;
    frame->width = codecCtx->width;
    frame->height = codecCtx->height;
    av_frame_get_buffer(frame, 32);

    packet = av_packet_alloc();

    writerThread = std::thread(&FFmpegWriter::writeThread, this);
}

// Thread method for writing frames from the queue
void FFmpegWriter::writeThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !frameQueue.empty() || doneReadingFrames; });

        if (frameQueue.empty() && doneReadingFrames) break;

        torch::Tensor tensor = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        // Write the frame
        writeFrame(tensor);
    }
}

// Add tensor frame to the queue
void FFmpegWriter::addFrame(torch::Tensor& tensor) {
    std::unique_lock<std::mutex> lock(mtx);
    frameQueue.push(tensor.clone());  // Clone the tensor to add to queue
    lock.unlock();
    cv.notify_one();  // Notify the writer thread that a new frame is available
}

// Internal method for writing tensor frames
void FFmpegWriter::writeFrame(torch::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        std::cerr << "Tensor is not on GPU, expecting CUDA tensor." << std::endl;
        return;
    }

    // Access the raw CUDA pointer from the tensor
    CUdeviceptr dev_ptr = (CUdeviceptr)tensor.data_ptr();

    // NV12 format has 2 planes: Y (width * height) and UV (width * height / 2)
    frame->data[0] = (uint8_t*)dev_ptr;                  // Y plane
    frame->data[1] = (uint8_t*)(dev_ptr + width * height);  // UV plane
    frame->linesize[0] = width;
    frame->linesize[1] = width;

    // Set the correct PTS value for the frame
    frame->pts = pts++;  // Monotonically increasing PTS

    // Encode the frame
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
        packet->dts = packet->pts;  // Ensure DTS is monotonic by setting it to PTS
        packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);

        // Write the encoded packet to the output file
        packet->stream_index = stream->index;
        av_interleaved_write_frame(formatCtx, packet);
        av_packet_unref(packet);  // Free the packet
    }
}

void FFmpegWriter::finalize() {
    doneReadingFrames = true;  // Indicate that no more frames will be added
    cv.notify_all();  // Wake up the writer thread to finish processing

    if (writerThread.joinable()) {
        writerThread.join();  // Wait for the writer thread to finish
    }

    std::cout << "Finalizing the writer..." << std::endl;

    if (!formatCtx || !codecCtx) {
        std::cerr << "Error: Invalid format or codec context during finalization." << std::endl;
        return;
    }

    // Flush remaining packets by sending a NULL frame to the encoder
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
        int ret = av_write_trailer(formatCtx);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
            std::cerr << "Error writing trailer: " << errbuf << std::endl;
        }
        else {
            std::cout << "Trailer written successfully." << std::endl;
        }
    }

    if (formatCtx && formatCtx->pb) {
        avio_closep(&formatCtx->pb);
    }

    if (formatCtx) {
        avformat_free_context(formatCtx);
        formatCtx = nullptr;
    }

    if (codecCtx) {
        avcodec_free_context(&codecCtx);
        codecCtx = nullptr;
    }

    if (frame) {
        av_frame_free(&frame);
        frame = nullptr;
    }

    if (packet) {
        av_packet_free(&packet);
        packet = nullptr;
    }

    std::cout << "Finalization complete." << std::endl;
}

FFmpegWriter::~FFmpegWriter() {
    if (frame) {
        av_frame_free(&frame);
    }
    if (packet) {
        av_packet_free(&packet);
    }
}
