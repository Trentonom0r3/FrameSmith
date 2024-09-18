#pragma once

#include <string>
#include <iostream>
#include <thread>
#include <algorithm>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

class FFmpegReader {
public:
    FFmpegReader(const std::string& inputFilePath, torch::Device device, bool halfPrecision = false);
    ~FFmpegReader();

    /**
     * @brief Reads the next frame and fills the provided tensor with RGB data.
     *
     * @param tensor The tensor to fill with the processed frame. Must be preallocated with shape {1, 3, height, width}.
     * @return true if a frame was successfully read and processed.
     * @return false if no more frames are available or an error occurred.
     */
    bool readFrame(torch::Tensor& tensor);

    // Accessor methods
    int getWidth() const { return width; }
    int getHeight() const { return height; }
   // double getFPS() const { return fps; }
    double getFPS() const {
        // Implement this method to return the FPS of the video
        // Example implementation:
        if (formatCtx->streams[0]->avg_frame_rate.den != 0)
            return static_cast<double>(formatCtx->streams[0]->avg_frame_rate.num) / formatCtx->streams[0]->avg_frame_rate.den;
        else
            return fps;
    }
    int getTotalFrames() const {
        // Assuming FFmpegReader has a method to get duration in seconds
        double duration = getDuration(); // Implement getDuration() if not available
        double fps = getFPS();
        return static_cast<int>(duration * fps);
    }

private:
    // FFmpeg components
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int videoStreamIndex = -1;
    int width = 0, height = 0;
    double fps = 0.0;

    // Torch and CUDA components
    torch::Device device;
    bool halfPrecision;

    // Intermediate tensors
    torch::Tensor rgb_tensor;          // For NV12 to RGB conversion
    torch::Tensor intermediate_tensor; // For reshaping and normalization

    // NPP error handling
    void checkNPP(NppStatus status, const std::string& msg);

    // Conversion and normalization
    void avframe_nv12_to_rgb_npp(AVFrame* gpu_frame);
    void normalizeFrame();

    double getDuration() const {
        // Implement this method to return the duration of the video in seconds
        // Example implementation:
        return static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    // Static callback for hardware format
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);
};


// Define NPP error handling
void FFmpegReader::checkNPP(NppStatus status, const std::string& msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error (" << msg << "): " << status << std::endl;
        throw std::runtime_error("NPP operation failed.");
    }
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


FFmpegReader::FFmpegReader(const std::string& inputFilePath, torch::Device device, bool halfPrecision)
    : device(device), halfPrecision(halfPrecision)
{
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
    codecCtx->pkt_timebase = formatCtx->streams[videoStreamIndex]->time_base;

    // Open codec
    ret = avcodec_open2(codecCtx, codec, nullptr);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errBuf, sizeof(errBuf));
        std::cerr << "Failed to open codec: " << errBuf << std::endl;
        throw std::runtime_error("Failed to open codec.");
    }

    // Allocate packet
    packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Failed to allocate AVPacket." << std::endl;
        throw std::runtime_error("Failed to allocate AVPacket.");
    }

    // Get video properties
    width = codecCtx->width;
    height = codecCtx->height;
    fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);

    // Initialize intermediate tensors
    // RGB tensor with shape {height, width, 3}, dtype uint8, device CUDA
    rgb_tensor = torch::empty({ height, width, 3 }, torch::TensorOptions().dtype(torch::kUInt8).device(device).requires_grad(false));

    // Intermediate tensor with shape {1, 3, height, width}, dtype float32 or float16
    intermediate_tensor = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(halfPrecision ? torch::kFloat16 : torch::kFloat32).device(device).requires_grad(false));
}

FFmpegReader::~FFmpegReader() {
    if (packet) {
        av_packet_free(&packet);
    }
    if (codecCtx) {
        avcodec_free_context(&codecCtx);
    }
    if (formatCtx) {
        avformat_close_input(&formatCtx);
    }
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
}

void FFmpegReader::avframe_nv12_to_rgb_npp(AVFrame* gpu_frame) {
    int nYUVPitch = gpu_frame->linesize[0];
    // Assuming planar NV12, where U and V are interleaved in data[1]
    // NPP expects separate U and V planes for NV12, but since NV12 has interleaved UV, we pass data[1] as U
    // NPP handles NV12 as a two-plane format: Y and interleaved UV

    NppiSize oSizeROI = { width, height };
    const Npp8u* pSrc[2] = { static_cast<Npp8u*>(gpu_frame->data[0]), static_cast<Npp8u*>(gpu_frame->data[1]) };

    // Perform NV12 to RGB conversion
    NppStatus status = nppiNV12ToRGB_8u_P2C3R(
        pSrc,
        gpu_frame->linesize[0],
        static_cast<Npp8u*>(rgb_tensor.data_ptr()),
        rgb_tensor.stride(0),
        oSizeROI
    );

    checkNPP(status, "nppiNV12ToRGB_8u_P2C3R");
}

void FFmpegReader::normalizeFrame() {
    // Convert RGB tensor to {1, 3, H, W} and normalize
    // Assuming rgb_tensor is already on CUDA and contiguous

    // Reshape and permute to {3, H, W}
    torch::Tensor reshaped = rgb_tensor.view({ height, width, 3 }).permute({ 2, 0, 1 });

    // Add batch dimension to make it {1, 3, H, W}
    torch::Tensor batched = reshaped.unsqueeze(0);

    // Convert to float and normalize
    intermediate_tensor = batched.to(intermediate_tensor.dtype()) // Convert to float16 or float32
        .div_(255.0)                                     // Normalize to [0,1]
        .clamp_(0.0, 1.0)                                // Clamp values
        .contiguous();                                  // Ensure contiguous memory
}

bool FFmpegReader::readFrame(torch::Tensor& tensor) {
    AVFrame* frameOut = av_frame_alloc();
    if (!frameOut) {
        std::cerr << "Failed to allocate AVFrame." << std::endl;
        return false;
    }

    bool success = false;

    while (true) {
        int ret = av_read_frame(formatCtx, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // Flush decoder
                ret = avcodec_send_packet(codecCtx, nullptr);
                if (ret < 0) {
                    break;
                }
            }
            else {
                char errBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errBuf, sizeof(errBuf));
                std::cerr << "Error reading frame: " << errBuf << std::endl;
                break;
            }
        }
        else {
            if (packet->stream_index == videoStreamIndex) {
                ret = avcodec_send_packet(codecCtx, packet);
                if (ret < 0) {
                    char errBuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    std::cerr << "Error sending packet: " << errBuf << std::endl;
                    break;
                }
            }
            av_packet_unref(packet);
        }

        // Receive frames from decoder
        ret = avcodec_receive_frame(codecCtx, frameOut);
        if (ret == AVERROR(EAGAIN)) {
            continue; // Need more packets
        }
        else if (ret == AVERROR_EOF) {
            break; // End of stream
        }

        if (frameOut->format == AV_PIX_FMT_CUDA) {
                avframe_nv12_to_rgb_npp(frameOut);
                normalizeFrame();
             
                // Copy the intermediate tensor to the passed tensor
                tensor.copy_(intermediate_tensor);
                success = true;
            break;
        }
        else {
            std::cerr << "Frame is not in CUDA format." << std::endl;
            success = false;
            break;
        }
    }

    av_frame_free(&frameOut);
    return success;
}
