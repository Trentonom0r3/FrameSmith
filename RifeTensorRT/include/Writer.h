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
    codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

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

    frame = av_frame_alloc();
    frame->format = codecCtx->pix_fmt;
    frame->width = codecCtx->width;
    frame->height = codecCtx->height;
    av_frame_get_buffer(frame, 32);

    packet = av_packet_alloc();

    // Use faster scaling algorithm
    swsCtx = sws_getContext(width, height, AV_PIX_FMT_RGB24,  // Source format
        width, height, AV_PIX_FMT_YUV420P,  // Destination format
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);  // Fast scaling algorithm

    writerThread = std::thread(&FFmpegWriter::writeThread, this);
}



// Thread method for writing frames from the queue
void FFmpegWriter::writeThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !frameQueue.empty() || doneReadingFrames; });

        if (frameQueue.empty() && doneReadingFrames) break;

        AVFrame* frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        // Write the frame
        writeFrame(frame);
        av_frame_free(&frame);
    }
}

// Add frame to the queue
void FFmpegWriter::addFrame(AVFrame* inputFrame) {
    if (!frame || !frame->data[0]) {
        std::cerr << "Invalid input frame or uninitialized data pointers." << std::endl;
        return;
    }
    std::unique_lock<std::mutex> lock(mtx);
    frameQueue.push(av_frame_clone(inputFrame));  // Clone the input frame
    lock.unlock();
    cv.notify_one();  // Notify the writer thread that a new frame is available
}

// Internal method for writing frames
void FFmpegWriter::writeFrame(AVFrame* inputFrame) {
    if (!inputFrame || !inputFrame->data[0]) {
        std::cerr << "Invalid input frame or uninitialized data pointers." << std::endl;
        return;
    }

    // Ensure the frame is writable (for YUV conversion)
    if (av_frame_make_writable(frame) < 0) {
        std::cerr << "Error making frame writable" << std::endl;
        return;
    }

    // Convert the input frame (RGB) to YUV420P
    sws_scale(swsCtx, inputFrame->data, inputFrame->linesize, 0, height,
        frame->data, frame->linesize);

    // Set the correct PTS value for the frame
    frame->pts = pts;  // Monotonically increasing PTS
    pts++;  // Increment the PTS for the next frame

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

            // Ensure the remaining frames are properly rescaled and written
            packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
            packet->dts = packet->pts;  // Force DTS to be the same as PTS for the remaining packets
            packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;

            // Write the final packets to the output file
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
    //finalize();
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
}