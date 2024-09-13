#pragma once

#include <string>
#include <iostream>
#include <thread>
#include <algorithm>
#include <torch/torch.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath);
    ~FFmpegReader();
    bool readFrame(AVFrame* frameOut);
    int getWidth();
    int getHeight();
    double getFPS();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* frameOut = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int videoStreamIndex = -1;
    int width = 0, height = 0;
    double fps = 0.0;

    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);
};

FFmpegReader::FFmpegReader(const std::string& inputFilePath) {
    // Initialize FFmpeg and hardware device
    int err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(err, errBuf, sizeof(errBuf));
        std::cerr << "Failed to create CUDA device context: " << errBuf << std::endl;
        throw std::runtime_error("Failed to create CUDA device context.");
    }

    // Open input file
    int ret = avformat_open_input(&formatCtx, inputFilePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
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

    // Find the video stream
    for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
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
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid"); // Use appropriate decoder
    if (!codec) {
        std::cerr << "Hardware decoder not found." << std::endl;
        throw std::runtime_error("Failed to find hardware decoder.");
    }

    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, codecPar);

    // Set hardware device context
    codecCtx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    // Set get_format callback
    codecCtx->get_format = get_hw_format;

    // Open codec
    ret = avcodec_open2(codecCtx, codec, nullptr);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Failed to open codec: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open codec.");
    }

    // Allocate frames and packet
    frame = av_frame_alloc();
    frameOut = av_frame_alloc();
    packet = av_packet_alloc();

    // Get video properties
    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
}

enum AVPixelFormat FFmpegReader::get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    std::cerr << "Failed to get HW surface format." << std::endl;
    return AV_PIX_FMT_NONE;
}

bool FFmpegReader::readFrame(AVFrame* frameOut) {
    av_frame_unref(frame);
    av_frame_unref(frameOut);
    int ret;

    while (true) {
        ret = av_read_frame(formatCtx, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // Flush decoder
                ret = avcodec_send_packet(codecCtx, nullptr);
                if (ret < 0) {
                    char errBuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    std::cerr << "Error sending flush packet: " << errBuf << std::endl;
                    return false;
                }
            }
            else {
                char errBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errBuf, sizeof(errBuf));
                std::cerr << "Error reading frame: " << errBuf << std::endl;
                return false;
            }
        }
        else {
            if (packet->stream_index == videoStreamIndex) {
                ret = avcodec_send_packet(codecCtx, packet);
                if (ret < 0) {
                    char errBuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    std::cerr << "Error sending packet: " << errBuf << std::endl;
                    return false;
                }
            }
            av_packet_unref(packet);
        }

        // Receive frames from decoder
        while (true) {
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == AVERROR(EAGAIN)) {
                break; // Need more packets
            }
            else if (ret == AVERROR_EOF) {
                return false; // End of stream
            }
            else if (ret < 0) {
                char errBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errBuf, sizeof(errBuf));
                std::cerr << "Error receiving frame: " << errBuf << std::endl;
                return false;
            }

            if (frame->format == AV_PIX_FMT_CUDA) {
                av_frame_ref(frameOut, frame);
                return true;
            }
            else {
                std::cerr << "Frame is not in CUDA format." << std::endl;
                return false;
            }
        }

        if (ret == AVERROR_EOF) {
            return false; // No more frames
        }
    }

    return false;
}

int FFmpegReader::getWidth() { return width; }
int FFmpegReader::getHeight() { return height; }
double FFmpegReader::getFPS() { return fps; }

FFmpegReader::~FFmpegReader() {
    av_frame_free(&frame);
    av_frame_free(&frameOut);
    av_packet_free(&packet);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);
    av_buffer_unref(&hw_device_ctx);
}
