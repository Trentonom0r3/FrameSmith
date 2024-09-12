#pragma once

#include <string>
#include <iostream>
#include <thread>     // For hardware concurrency
#include <algorithm>  // For std::min
#include <torch/torch.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath);
    ~FFmpegReader();
    bool readFrame(AVFrame*& frameOut);
    int getWidth();
    int getHeight();
    double getFPS();
    torch::Tensor avframe_to_tensor(AVFrame* gpu_frame);

    //private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;   // Reuse AVPacket across calls
    AVBufferRef* hw_device_ctx = nullptr; // Hardware device context (CUDA)
    int videoStreamIndex = -1;
    int width, height;
    double fps;
};

FFmpegReader::FFmpegReader(const std::string& inputFilePath) {
    // Initialize FFmpeg

    // Initialize the hardware device (CUDA)
    int err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err < 0) {
        std::cerr << "Failed to create CUDA device context" << std::endl;
        throw std::runtime_error("Failed to create CUDA device context.");
    }

    // Open the input file and check for errors
    int ret = avformat_open_input(&formatCtx, inputFilePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Could not open input file: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open input file.");
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(formatCtx, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to retrieve input stream information." << std::endl;
        throw std::runtime_error("Failed to find stream info.");
    }

    // Find the video stream index
    for (int i = 0; i < formatCtx->nb_streams; i++) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Could not find a video stream in the input file." << std::endl;
        throw std::runtime_error("Failed to find video stream.");
    }

    AVCodecParameters* codecPar = formatCtx->streams[videoStreamIndex]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecPar->codec_id);
    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, codecPar);

    // Set the hardware device context for CUDA decoding
    codecCtx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    // Enable multithreading
    codecCtx->thread_count = std::min(16, static_cast<int>(std::thread::hardware_concurrency()));
    codecCtx->thread_type = FF_THREAD_FRAME;  // Use frame threading for better performance

    avcodec_open2(codecCtx, codec, nullptr);

    // Allocate frame and reusable packet
    frame = av_frame_alloc();
    packet = av_packet_alloc();

    // Set the video dimensions
    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
}

bool FFmpegReader::readFrame(AVFrame*& frameOut) {
    av_frame_unref(frame);
    av_frame_unref(frameOut);
    bool endOfStream = false;

    while (!endOfStream) {
        int ret = av_read_frame(formatCtx, packet);
        if (ret >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                ret = avcodec_send_packet(codecCtx, packet);
                av_packet_unref(packet);  // Unref the packet but reuse its buffer
                if (ret < 0) {
                    std::cerr << "Error sending packet: " << ret << std::endl;
                    return false;
                }

                ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == 0) {
                    if (frame->format == AV_PIX_FMT_CUDA) {
                        // Frame is already on the GPU in CUDA memory
                        frameOut = av_frame_alloc();
                        av_frame_ref(frameOut, frame);
                    }
                    else {
                        std::cerr << "Frame is not in CUDA format." << std::endl;
                        return false;
                    }
                    return true;
                }
                else if (ret == AVERROR(EAGAIN)) {
                    continue;
                }
                else if (ret == AVERROR_EOF) {
                    endOfStream = true;
                }
            }
        }
        else if (ret == AVERROR_EOF) {
            avcodec_send_packet(codecCtx, nullptr);  // Send a NULL packet to flush
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == 0) {
                frameOut = av_frame_alloc();
                av_frame_ref(frameOut, frame);
                return true;
            }
            endOfStream = true;
        }
        else {
            std::cerr << "Error reading frame: " << ret << std::endl;
            return false;
        }
    }

    return false;
}


int FFmpegReader::getWidth() { return width; }
int FFmpegReader::getHeight() { return height; }
double FFmpegReader::getFPS() { return fps; }

FFmpegReader::~FFmpegReader() {
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);
    av_buffer_unref(&hw_device_ctx);
}