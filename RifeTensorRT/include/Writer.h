#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>  // std::min
#include <thread>     // std::thread::hardware_concurrency()
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

class FFmpegWriter {
public:
    FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps);
    ~FFmpegWriter();
    void addFrame(AVFrame* inputFrame);  // Add frame to the queue
    void writeThread();  // Thread for writing frames
    void finalize();

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* stream = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsCtx = nullptr;
    int width, height, fps;
    int64_t pts = 0;

    // Threading components
    std::queue<AVFrame*> frameQueue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> doneReadingFrames{ false };
    std::thread writerThread;

    void writeFrame(AVFrame* inputFrame);  // Internal method for writing frames
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
    codecCtx->pix_fmt = AV_PIX_FMT_NV12;  // Use NV12 for hardware encoding

    // Use CRF for variable bitrate
    av_opt_set(codecCtx->priv_data, "crf", "23", 0);  // Adjust CRF value
    av_opt_set(codecCtx->priv_data, "preset", "p1", 0);  // Use p1-p7 preset for NVENC
    av_opt_set(codecCtx->priv_data, "tune", "ll", 0);  // Tune for low-latency

    // Use multiple threads for encoding (limited to 16)
    codecCtx->thread_count = (std::min)(static_cast<int>(std::thread::hardware_concurrency()), 16);
    codecCtx->thread_type = FF_THREAD_FRAME;  // Frame-based threading

    avcodec_open2(codecCtx, codec, nullptr);

    avio_open(&formatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE);
    stream = avformat_new_stream(formatCtx, codec);
    stream->time_base = { 1, fps };
    avcodec_parameters_from_context(stream->codecpar, codecCtx);
    avformat_write_header(formatCtx, nullptr);

    frame = av_frame_alloc();
    frame->format = codecCtx->pix_fmt;  // Ensure NV12 format
    frame->width = codecCtx->width;
    frame->height = codecCtx->height;
    av_frame_get_buffer(frame, 32);

    packet = av_packet_alloc();

    // Use faster scaling algorithm for RGB to NV12 conversion
    swsCtx = sws_getContext(width, height, AV_PIX_FMT_RGB24,  // Source format
        width, height, AV_PIX_FMT_NV12,  // Destination format (ensure YUV conversion)
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);  // Fast scaling algorithm

    writerThread = std::thread(&FFmpegWriter::writeThread, this);
    writerThread.detach();  // Detach the writer thread
}

// Thread method for writing frames from the queue
inline void FFmpegWriter::writeThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !frameQueue.empty() || doneReadingFrames; });

        if (frameQueue.empty() && doneReadingFrames) break;

        AVFrame* frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        // Write the frame immediately
        writeFrame(frame);
        av_frame_free(&frame);
    }
}

inline void FFmpegWriter::addFrame(AVFrame* inputFrame) {
    if (!inputFrame || !inputFrame->data[0]) {
        std::cerr << "Invalid input frame or uninitialized data pointers." << std::endl;
        return;
    }

    AVFrame* frameToEncode = nullptr;

    // Assign a PTS to the input frame if it's not already set
    inputFrame->pts = pts++;

    if (inputFrame->format == AV_PIX_FMT_CUDA) {
        // Convert CUDA frame to NV12
        AVFrame* nv12Frame = av_frame_alloc();
        nv12Frame->format = AV_PIX_FMT_NV12;
        nv12Frame->width = inputFrame->width;
        nv12Frame->height = inputFrame->height;

        // Transfer CUDA frame to NV12
        av_hwframe_transfer_data(nv12Frame, inputFrame, 0);
        nv12Frame->pts = inputFrame->pts;  // Carry over the PTS
        frameToEncode = nv12Frame;
    }
    else if (inputFrame->format == AV_PIX_FMT_RGB24) {
        // Convert RGB frame to NV12
        frameToEncode = av_frame_alloc();
        frameToEncode->format = AV_PIX_FMT_NV12;
        frameToEncode->width = inputFrame->width;
        frameToEncode->height = inputFrame->height;
        av_frame_get_buffer(frameToEncode, 32);

        // Convert RGB24 to NV12
        sws_scale(swsCtx, inputFrame->data, inputFrame->linesize, 0, height,
            frameToEncode->data, frameToEncode->linesize);
        frameToEncode->pts = inputFrame->pts;  // Carry over the PTS
    }
    else if (inputFrame->format == AV_PIX_FMT_NV12) {
        frameToEncode = av_frame_clone(inputFrame);  // Directly use NV12 frame
        frameToEncode->pts = inputFrame->pts;  // Carry over the PTS
    }
    else {
        std::cerr << "Error: Unsupported pixel format: " << av_get_pix_fmt_name((AVPixelFormat)inputFrame->format) << std::endl;
        return;
    }

    std::unique_lock<std::mutex> lock(mtx);
    frameQueue.push(frameToEncode);  // Add the frame to the queue
    lock.unlock();
    cv.notify_one();  // Notify the writer thread that a new frame is available
}

inline void FFmpegWriter::writeFrame(AVFrame* inputFrame) {
    if (!inputFrame || !inputFrame->data[0]) {
        std::cerr << "Error: Invalid input frame or uninitialized data pointers." << std::endl;
        return;
    }

    // Ensure the frame resolution matches the encoder
    if (inputFrame->width != codecCtx->width || inputFrame->height != codecCtx->height) {
        std::cerr << "Error: Frame resolution does not match codec context." << std::endl;
        return;
    }

    // Ensure the output frame is writable
    if (av_frame_make_writable(frame) < 0) {
        std::cerr << "Error: Could not make frame writable." << std::endl;
        return;
    }

    // Ensure the correct PTS is set
    if (inputFrame->pts == AV_NOPTS_VALUE) {
        inputFrame->pts = pts++;
    }

    // Send the frame to the encoder
    int ret = avcodec_send_frame(codecCtx, inputFrame);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error sending frame for encoding: " << errbuf << std::endl;
        return;
    }

    // Receive and write the encoded packet
    while (ret >= 0) {
        ret = avcodec_receive_packet(codecCtx, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        else if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
            std::cerr << "Error encoding frame: " << errbuf << std::endl;
            return;
        }

        // Rescale PTS and DTS to match the stream's time base
        packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
        packet->dts = packet->pts;
        packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);

        // Write the encoded packet to the output file
        packet->stream_index = stream->index;
        av_interleaved_write_frame(formatCtx, packet);
        av_packet_unref(packet);  // Free the packet after writing

        // Flush the format context to force writing to disk
        avio_flush(formatCtx->pb);
    }
}

inline void FFmpegWriter::finalize() {
    doneReadingFrames = true;  // Indicate that no more frames will be added
    cv.notify_all();  // Wake up the writer thread to finish processing

    if (writerThread.joinable()) {
        writerThread.join();  // Wait for the writer thread to finish
    }

    // Flush remaining packets by sending a NULL frame to the encoder
    avcodec_send_frame(codecCtx, nullptr);
    while (avcodec_receive_packet(codecCtx, packet) >= 0) {
        packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
        packet->dts = packet->pts;
        packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);
        packet->stream_index = stream->index;
        av_interleaved_write_frame(formatCtx, packet);
        av_packet_unref(packet);
    }

    av_write_trailer(formatCtx);  // Write the trailer to finalize the file

    // Close output
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

inline FFmpegWriter::~FFmpegWriter() {
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
}
