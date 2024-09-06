#pragma once
#include <string>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath);
    ~FFmpegReader();
    bool readFrame(AVFrame*& frameOut);
    int getWidth();
    int getHeight();
    double getFPS();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsCtx = nullptr;
    int videoStreamIndex = -1;
    int width, height;
    double fps;
    AVPixelFormat hw_pix_fmt;
};


FFmpegReader::FFmpegReader(const std::string& inputFilePath) {
    // Allocate the AVFormatContext
    formatCtx = avformat_alloc_context();

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

    // Enable multithreading
    codecCtx->thread_count = std::thread::hardware_concurrency(); // Utilize available CPU cores
    codecCtx->thread_type = FF_THREAD_FRAME;  // Use frame threading for better performance

    avcodec_open2(codecCtx, codec, nullptr);

    // Allocate packet, frame, and scaling context
    frame = av_frame_alloc();
    packet = av_packet_alloc();
    swsCtx = sws_getContext(codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
        codecCtx->width, codecCtx->height, AV_PIX_FMT_RGB24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
}

bool FFmpegReader::readFrame(AVFrame*& frameOut) {
    static AVFrame* rgbFrame = av_frame_alloc();
    rgbFrame->format = AV_PIX_FMT_RGB24;
    rgbFrame->width = codecCtx->width;
    rgbFrame->height = codecCtx->height;

    av_frame_get_buffer(rgbFrame, 0);  // Ensure the frame buffer is properly initialized

    bool endOfStream = false;

    while (!endOfStream) {
        int ret = av_read_frame(formatCtx, packet);
        if (ret >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                ret = avcodec_send_packet(codecCtx, packet);
                av_packet_unref(packet);
                if (ret < 0) {
                    std::cerr << "Error sending packet: " << ret << std::endl;
                    return false;
                }

                ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == 0) {
                    if (av_frame_make_writable(rgbFrame) < 0) {
                        std::cerr << "Error making frame writable" << std::endl;
                        return false;
                    }

                    sws_scale(swsCtx, frame->data, frame->linesize, 0, codecCtx->height,
                        rgbFrame->data, rgbFrame->linesize);

                    frameOut = av_frame_clone(rgbFrame);
                    return true;
                }
                else if (ret == AVERROR(EAGAIN)) {
                    // The decoder needs more packets, continue reading
                    continue;
                }
                else if (ret == AVERROR_EOF) {
                    endOfStream = true;
                }
            }
        }
        else if (ret == AVERROR_EOF) {
            // End of file reached, now flush the decoder
            avcodec_send_packet(codecCtx, nullptr);  // Send a NULL packet to flush
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == 0) {
                if (av_frame_make_writable(rgbFrame) < 0) {
                    std::cerr << "Error making frame writable" << std::endl;
                    return false;
                }

                sws_scale(swsCtx, frame->data, frame->linesize, 0, codecCtx->height,
                    rgbFrame->data, rgbFrame->linesize);

                frameOut = av_frame_clone(rgbFrame);
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
    sws_freeContext(swsCtx);
}