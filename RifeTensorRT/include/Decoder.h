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
#include <libavutil/imgutils.h>
}

#include <npp.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath);
    ~FFmpegReader();
    bool readFrame(at::Tensor& tensorOut);  // Read a frame and return it as a PyTorch CUDA tensor
    int getWidth();
    int getHeight();
    double getFPS();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    int videoStreamIndex = -1;
    int width, height;
    double fps;

    cudaStream_t cudaStream;  // CUDA stream for asynchronous operations
};

// Constructor
FFmpegReader::FFmpegReader(const std::string& inputFilePath) {
    formatCtx = avformat_alloc_context();

    int ret = avformat_open_input(&formatCtx, inputFilePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Could not open input file: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open input file.");
    }

    ret = avformat_find_stream_info(formatCtx, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to retrieve input stream information." << std::endl;
        throw std::runtime_error("Failed to find stream info.");
    }

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
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        std::cerr << "Error: Could not find h264_cuvid decoder." << std::endl;
        throw std::runtime_error("Failed to find decoder.");
    }

    codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, codecPar);
    codecCtx->thread_count = std::thread::hardware_concurrency();
    codecCtx->thread_type = FF_THREAD_FRAME;

    ret = avcodec_open2(codecCtx, codec, nullptr);
    if (ret < 0) {
        std::cerr << "Error opening codec: " << ret << std::endl;
        throw std::runtime_error("Failed to open codec.");
    }

    frame = av_frame_alloc();
    packet = av_packet_alloc();
    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);

    // Initialize CUDA stream
    cudaStreamCreate(&cudaStream);
    std::cout << "Opened video stream, resolution: " << width << "x" << height << ", FPS: " << fps << std::endl;
}

// Destructor
FFmpegReader::~FFmpegReader() {
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);
    cudaStreamDestroy(cudaStream);
}

bool FFmpegReader::readFrame(at::Tensor& tensorOut) {
    av_frame_unref(frame);

    while (true) {
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
                    std::cout << "Y plane address: " << static_cast<void*>(frame->data[0]) << std::endl;
                    std::cout << "UV plane address: " << static_cast<void*>(frame->data[1]) << std::endl;

                    // Create a PyTorch tensor for the output RGB frame
                    tensorOut = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
                    if (!tensorOut.defined()) {
                        std::cerr << "Error: Tensor allocation failed!" << std::endl;
                        return false;
                    }

                    // Define the NPP pointers for the Y and UV planes
                    const Npp8u* pSrc[2] = { frame->data[0], frame->data[1] };
                    Npp8u* pDst = tensorOut.data_ptr<Npp8u>();

                    // Set up the region of interest (ROI)
                    NppiSize oSizeROI = { width, height };

                    // Perform the NV12 to RGB conversion using NPP
                    NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, frame->linesize[0], pDst, width * 3, oSizeROI);
                    if (status != NPP_SUCCESS) {
                        std::cerr << "Error during NV12 to RGB conversion: " << status << std::endl;
                        return false;
                    }

                    std::cout << "NV12 to RGB conversion completed successfully!" << std::endl;
                    return true;
                }
                else if (ret == AVERROR(EAGAIN)) {
                    continue;
                }
                else if (ret == AVERROR_EOF) {
                    return false;
                }
                else {
                    std::cerr << "Error receiving frame: " << ret << std::endl;
                    return false;
                }
            }
        }
        else if (ret == AVERROR_EOF) {
            return false;
        }
        else {
            std::cerr << "Error reading frame: " << ret << std::endl;
            return false;
        }
    }
}

int FFmpegReader::getWidth() { return width; }
int FFmpegReader::getHeight() { return height; }
double FFmpegReader::getFPS() { return fps; }
