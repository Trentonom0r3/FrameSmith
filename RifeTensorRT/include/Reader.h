#pragma once
#include <string>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <torch/torch.h>  // For creating tensors

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>  // For hardware context
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath);
    ~FFmpegReader();
    bool readFrame(torch::Tensor& tensorOut);
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
    AVBufferRef* hw_device_ctx = nullptr;  // CUDA hardware context

    int initializeHWDecoder(AVCodecContext* ctx, AVHWDeviceType type);
};

FFmpegReader::FFmpegReader(const std::string& inputFilePath) {
    // Allocate the AVFormatContext
    formatCtx = avformat_alloc_context();

    // Open the input file and check for errors
    int ret = avformat_open_input(&formatCtx, inputFilePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Could not open input file: " << inputFilePath << " - Error: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open input file.");
    }
    std::cout << "Opened input file successfully: " << inputFilePath << std::endl;

    // Retrieve stream information
    ret = avformat_find_stream_info(formatCtx, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to retrieve input stream information." << std::endl;
        throw std::runtime_error("Failed to find stream info.");
    }
    std::cout << "Retrieved stream information successfully." << std::endl;

    // Find the video stream index
    for (int i = 0; i < formatCtx->nb_streams; i++) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            std::cout << "Found video stream at index: " << i << std::endl;
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
        std::cerr << "Could not find the h264_cuvid codec." << std::endl;
        throw std::runtime_error("Failed to find the h264_cuvid codec.");
    }
    std::cout << "Found the h264_cuvid codec." << std::endl;

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        std::cerr << "Could not allocate codec context." << std::endl;
        throw std::runtime_error("Failed to allocate codec context.");
    }

    ret = avcodec_parameters_to_context(codecCtx, codecPar);
    if (ret < 0) {
        std::cerr << "Could not copy codec parameters to context." << std::endl;
        throw std::runtime_error("Failed to copy codec parameters.");
    }

    // Initialize CUDA hardware decoder
    ret = initializeHWDecoder(codecCtx, AV_HWDEVICE_TYPE_CUDA);
    if (ret < 0) {
        std::cerr << "Failed to initialize hardware decoder for CUDA." << std::endl;
        throw std::runtime_error("Hardware initialization failed.");
    }

    ret = avcodec_open2(codecCtx, codec, nullptr);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Could not open codec: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open codec.");
    }

    std::cout << "Opened codec successfully." << std::endl;

    // Allocate the frame and packet
    frame = av_frame_alloc();
    packet = av_packet_alloc();  // <--- Allocate the packet

    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    std::cout << "Video stream info - Width: " << width << ", Height: " << height << ", FPS: " << fps << std::endl;
}

int FFmpegReader::initializeHWDecoder(AVCodecContext* ctx, AVHWDeviceType type) {
    // Create hardware device context for CUDA
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, type, nullptr, nullptr, 0);
    if (ret < 0) {
        char errBuf[256];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Failed to create hardware device context: " << errBuf << std::endl;
        return ret;
    }

    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    return 0;
}

bool FFmpegReader::readFrame(torch::Tensor& tensorOut) {
    bool endOfStream = false;

    while (!endOfStream) {
        int ret = av_read_frame(formatCtx, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                std::cout << "End of file reached." << std::endl;
                return false;
            }
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            std::cerr << "Error reading frame: " << errBuf << std::endl;
            return false;
        }

        if (packet->stream_index == videoStreamIndex) {
            ret = avcodec_send_packet(codecCtx, packet);
            av_packet_unref(packet);  // Unref the packet regardless of send success
            if (ret < 0) {
                std::cerr << "Error sending packet to decoder: " << ret << std::endl;
                return false;
            }

            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == 0) {
                std::cout << "Successfully decoded a frame with format: " << frame->format << std::endl;

                // Directly create a tensor from the GPU buffer (no CPU transfer)
                if (frame->format == AV_PIX_FMT_CUDA) {
                    std::cout << "Frame is in CUDA format, creating tensor directly." << std::endl;

                    // Access GPU memory directly for tensor creation
                    CUdeviceptr cu_frame_ptr = (CUdeviceptr)frame->data[0];
                    int frame_width = frame->width;
                    int frame_height = frame->height;

                    // Create tensor from GPU buffer (assuming RGB24-like data)
                    tensorOut = torch::from_blob((void*)cu_frame_ptr, { frame_height, frame_width, 3 }, torch::kCUDA);
                }
                else {
                    std::cerr << "Unexpected frame format: " << frame->format << std::endl;
                    return false;
                }

                return true;
            }
            else if (ret == AVERROR(EAGAIN)) {
                std::cout << "Decoder needs more packets, continuing." << std::endl;
                continue;  // Need more packets
            }
            else if (ret == AVERROR_EOF) {
                std::cout << "Decoder reached EOF." << std::endl;
                endOfStream = true;
            }
            else {
                std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
                return false;
            }
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
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
}
