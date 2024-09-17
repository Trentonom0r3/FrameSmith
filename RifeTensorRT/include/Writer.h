#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>  // std::min
#include <thread>     // std::thread::hardware_concurrency()
#include <atomic>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

// Helper function to set up CUDA device context
inline int init_cuda_context(AVBufferRef** hw_device_ctx) {
    int err = av_hwdevice_ctx_create(hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (err < 0) {
        char err_buf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, err);
        std::cerr << "Failed to create CUDA device context: " << err_buf << std::endl;
        return err;
    }
    return 0;
}

// Helper function to set up CUDA frames context
inline AVBufferRef* init_cuda_frames_ctx(AVBufferRef* hw_device_ctx, int width, int height, AVPixelFormat sw_format) {
    AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ref) {
        std::cerr << "Failed to create hardware frames context" << std::endl;
        return nullptr;
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(hw_frames_ref->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = sw_format;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    int err = av_hwframe_ctx_init(hw_frames_ref);
    if (err < 0) {
        char err_buf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, err);
        std::cerr << "Failed to initialize hardware frame context: " << err_buf << std::endl;
        av_buffer_unref(&hw_frames_ref);
        return nullptr;
    }

    return hw_frames_ref;
}

class FFmpegWriter {
public:
    FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps);
    ~FFmpegWriter();
    void addFrame(at::Tensor inputTensor);  // Add frame to the queue
    void writeThread();  // Thread for writing frames
    void finalize();
    void FFmpegWriter::avframe_rgb_to_nv12_npp(at::Tensor output);
    void setStream(cudaStream_t stream) { writestream = stream; }
    inline cudaStream_t getStream() const { return writestream; }
private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* stream = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsCtx = nullptr;
    int width, height, fps;
    int64_t pts = 0;
    AVFrame* interpolatedFrame = nullptr;
    cudaStream_t writestream;
    AVBufferRef* hw_frames_ctx;
    AVBufferRef* hw_device_ctx;
    // Threading components
    std::queue<AVFrame*> frameQueue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> doneReadingFrames{ false };
    std::thread writerThread;
    torch::Tensor y_plane, u_plane, v_plane, uv_flat,
        uv_plane, u_flat, v_flat, nv12_tensor;

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
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;


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

    y_plane = torch::empty({ height, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    u_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    v_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_flat = torch::empty({ u_plane.numel() * 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_plane = torch::empty({ height / 2, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    u_flat = u_plane.view(-1);
    v_flat = v_plane.view(-1);

   // cudaStreamCreate(&writestream);
    // Initialize CUDA device context
    if (init_cuda_context(&hw_device_ctx) < 0) {
        std::cerr << "Failed to initialize CUDA context" << std::endl;
        return;
    }

    // Initialize CUDA frames context
    hw_frames_ctx = init_cuda_frames_ctx(hw_device_ctx, width, height, AV_PIX_FMT_NV12); // Change this based on your source format
    if (!hw_frames_ctx) {
        std::cerr << "Failed to initialize CUDA frames context" << std::endl;
        return;
    }

    // Allocate interpolatedFrame and get buffer
    interpolatedFrame = av_frame_alloc();
    interpolatedFrame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx); // Set the CUDA frames context
    interpolatedFrame->format = AV_PIX_FMT_NV12;
    interpolatedFrame->width = width;
    interpolatedFrame->height = height;

    int err = av_hwframe_get_buffer(hw_frames_ctx, interpolatedFrame, 0);
    if (err < 0) {
        std::cerr << "Failed to allocate hardware frame buffer" << std::endl;
        return;
    }

    cudaStreamCreate(&writestream);

    writerThread = std::thread(&FFmpegWriter::writeThread, this);
}

// Define error handling macros
#define NPP_CHECK_NPP(func) { \
    NppStatus status = (func); \
    if (status != NPP_SUCCESS) { \
        std::cerr << "NPP Error: " << status << std::endl; \
        exit(-1); \
    } \
}

inline void FFmpegWriter::avframe_rgb_to_nv12_npp(at::Tensor output) {
    // Check if output tensor is on CUDA
    if (!output.is_cuda()) {
        throw std::runtime_error("Input tensor 'output' is not on CUDA.");
    }

    // Check if output tensor has the correct dtype
    if (output.dtype() != torch::kUInt8) {
        throw std::runtime_error("Input tensor 'output' does not have dtype torch::kUInt8.");
    }

    // Check if output tensor is contiguous
    if (!output.is_contiguous()) {
        output = output.contiguous();
        std::cout << "Converted 'output' tensor to contiguous." << std::endl;
    }

    // Check dimensions
    if (output.dim() != 3 || output.size(2) != 3) { // Assuming RGB channels last
        throw std::runtime_error("Input tensor 'output' does not have the expected shape (C, H, W).");
    }

    // Set up destination pointers and strides
    Npp8u* pDst[3] = { y_plane.data_ptr<Npp8u>(), u_plane.data_ptr<Npp8u>(), v_plane.data_ptr<Npp8u>() };
    int rDstStep[3] = { static_cast<int>(y_plane.stride(0)), static_cast<int>(u_plane.stride(0)), static_cast<int>(v_plane.stride(0)) };

    // Source RGB data and step
    Npp8u* pSrc = output.data_ptr<Npp8u>();
    int nSrcStep = output.stride(0);

    // Define ROI
    NppiSize oSizeROI = { width, height };

    // Perform RGB to YUV420 conversion (planar)
    NPP_CHECK_NPP(nppiRGBToYUV420_8u_C3P3R(
        pSrc,
        nSrcStep,
        pDst,
        rDstStep,
        oSizeROI
    ));

    // Perform tensor operations
    try {
        uv_flat.index_put_({ torch::indexing::Slice(0, torch::indexing::None, 2) }, u_flat);
        uv_flat.index_put_({ torch::indexing::Slice(1, torch::indexing::None, 2) }, v_flat);
    }
    catch (const c10::Error& e) {
        std::cerr << "c10::Error during index_put_: " << e.what() << std::endl;
        throw; // Re-throw after logging
    }

    // Reshape UV data back to 2D
    try {
        uv_plane = uv_flat.view({ height / 2, width });
    }
    catch (const c10::Error& e) {
        std::cerr << "c10::Error during view(): " << e.what() << std::endl;
        throw;
    }

    // Concatenate Y and UV planes
    try {
        nv12_tensor = torch::cat({ y_plane.view(-1), uv_plane.view(-1) }, 0);
    }
    catch (const c10::Error& e) {
        std::cerr << "c10::Error during cat(): " << e.what() << std::endl;
        throw;
    }

    // Final check on nv12_tensor
    if (nv12_tensor.numel() != static_cast<int64_t>(width) * height * 3 / 2) {
        throw std::runtime_error("nv12_tensor has an unexpected size.");
    }

    //std::cout << "avframe_rgb_to_nv12_npp completed successfully." << std::endl;
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


inline void FFmpegWriter::addFrame(at::Tensor inputTensor) {
    //if size of tensor is not 3 chn, 
    /*input.squeeze(0).permute({ 1, 2, 0 })
                .mul_(255.0).clamp_(0, 255).to(torch::kU8).contiguous()
                */
    if (inputTensor.dim() != 3) {
		inputTensor = inputTensor.squeeze(0).permute({ 1, 2, 0 })
			.mul_(255.0).clamp_(0, 255).to(torch::kU8).contiguous();
	}
    else {
		inputTensor = inputTensor.permute({ 1, 2, 0 })
			.mul_(255.0).clamp_(0, 255).to(torch::kU8).contiguous();
	}
    avframe_rgb_to_nv12_npp(inputTensor);
    // [Line 54] Copy NV12 data into interpolated frame

    cudaMemcpy2DAsync(
        interpolatedFrame->data[0],
        interpolatedFrame->linesize[0],
        nv12_tensor.data_ptr<uint8_t>(),
        width * sizeof(uint8_t),
        width * sizeof(uint8_t),
        height,
        cudaMemcpyDeviceToDevice,
        writestream
    );

    cudaMemcpy2DAsync(
        interpolatedFrame->data[1],
        interpolatedFrame->linesize[1],
        nv12_tensor.data_ptr<uint8_t>() + (width * height),
        width * sizeof(uint8_t),
        width * sizeof(uint8_t),
        height / 2,
        cudaMemcpyDeviceToDevice,
        writestream
    );

    AVFrame* frameToEncode = nullptr;

    if (interpolatedFrame->format == AV_PIX_FMT_NV12) {
        frameToEncode = av_frame_clone(interpolatedFrame);  // Directly use NV12 frame
        frameToEncode->pts = interpolatedFrame->pts;  // Carry over the PTS
    }
    else if (interpolatedFrame->format == AV_PIX_FMT_CUDA) {
        // Convert CUDA frame to NV12
        AVFrame* nv12Frame = av_frame_alloc();
        nv12Frame->format = AV_PIX_FMT_NV12;
        nv12Frame->width = interpolatedFrame->width;
        nv12Frame->height = interpolatedFrame->height;

        // Transfer CUDA frame to NV12
        av_hwframe_transfer_data(nv12Frame, interpolatedFrame, 0);
        nv12Frame->pts = interpolatedFrame->pts;  // Carry over the PTS
        frameToEncode = nv12Frame;
    }
    else {
        std::cerr << "Error: Unsupported pixel format: " << av_get_pix_fmt_name((AVPixelFormat)interpolatedFrame->format) << std::endl;
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
    cudaStreamDestroy(writestream);
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
}
