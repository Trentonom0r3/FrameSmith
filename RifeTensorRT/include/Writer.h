// Writer.h
#pragma once
#include <string>
#include <mutex>
#include <atomic>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <nppi_support_functions.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include "concurrentqueue.h" // Include moodycamel's ConcurrentQueue

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <chrono>
#include <vector>
#include <memory>
#include <thread>
#include <iostream>

struct AVFrameDeleter {
    void operator()(AVFrame* frame) const noexcept {
        av_frame_free(&frame);
    }
};

// Error handling macros
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(-1); \
        } \
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

// Error handling macro for NPP
#define NPP_CHECK_NPP(func) { \
    NppStatus status = (func); \
    if (status != NPP_SUCCESS) { \
        std::cerr << "NPP Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
}

class FFmpegWriter {
public:
    FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps, bool benchmark);
    ~FFmpegWriter();
    void addFrame(at::Tensor inputTensor, bool benchmark);  // Add frame to the queue
    void finalize();  // Handle any finalization if necessary
    void avframe_rgb_to_nv12_npp(at::Tensor output);
    void setStream(cudaStream_t stream) { writestream = stream; }
    inline cudaStream_t getStream() const { return writestream; }
    inline cudaStream_t getConvertStream() const { return convertstream; }
    inline cudaStream_t getUStream() const { return uStream; }

private:
    // FFmpeg components
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* stream = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsCtx = nullptr;
    int width, height, fps;
    int64_t pts = 0;
    AVFrame* interpolatedFrame = nullptr;
    cudaStream_t writestream, convertstream, uStream;
    AVBufferRef* hw_frames_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;

    // NPP Stream Context
    NppStreamContext nppStreamCtx;

    // Torch tensors for YUV planes
    torch::Tensor y_plane, u_plane, v_plane, uv_flat,
        uv_plane, u_flat, v_flat, nv12_tensor;

    // Benchmark mode flag
    bool isBenchmark;

    // Frame queue using moodycamel's ConcurrentQueue
    moodycamel::ConcurrentQueue<AVFrame*> frameQueue;

    // Frame pool with custom deleter
    std::vector<std::unique_ptr<AVFrame, AVFrameDeleter>> framePool;
    std::mutex poolMutex; // Protect access to the pool

    // Writer thread
    std::thread writerThread;
    std::atomic<bool> doneReadingFrames{ false };

    // Internal methods
    void writeThread();  // Thread function for encoding
    AVFrame* acquireFrame();  // Acquire a frame from the pool
    void releaseFrame(AVFrame* frame);  // Release a frame back to the pool
    void writeFrame(AVFrame* inputFrame);  // Encode and write a frame
};

// Implementation

inline FFmpegWriter::FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps, bool benchmark)
    : width(width), height(height), fps(fps), isBenchmark(benchmark) {

    // Allocate output context
    avformat_alloc_output_context2(&formatCtx, nullptr, "mp4", outputFilePath.c_str());
    if (!formatCtx) {
        std::cerr << "Could not allocate output context" << std::endl;
        throw std::runtime_error("Failed to allocate output context");
    }

    // Find encoder
    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");  // Use NVENC encoder
    if (!codec) {
        std::cerr << "Error finding NVENC codec." << std::endl;
        throw std::runtime_error("NVENC codec not found");
    }

    // Allocate codec context
    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        std::cerr << "Could not allocate codec context" << std::endl;
        throw std::runtime_error("Failed to allocate codec context");
    }

    // Configure codec context
    codecCtx->codec_id = codec->id;
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = { 1, fps };
    codecCtx->framerate = { fps, 1 };
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;
    codecCtx->pix_fmt = AV_PIX_FMT_NV12;  // Use NV12 for hardware encoding

    // Set encoder options
    av_opt_set(codecCtx->priv_data, "crf", "23", 0);      // Adjust CRF value
    av_opt_set(codecCtx->priv_data, "preset", "p7", 0);   // Use p1-p7 preset for NVENC
    av_opt_set(codecCtx->priv_data, "tune", "ll", 0);     // Tune for low-latency

    // Multi-threaded encoding
    codecCtx->thread_count = (std::min)(static_cast<int>(std::thread::hardware_concurrency()), 16);
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    // Open codec
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        throw std::runtime_error("Failed to open codec");
    }

    // Create new stream
    stream = avformat_new_stream(formatCtx, codec);
    if (!stream) {
        std::cerr << "Failed allocating output stream" << std::endl;
        throw std::runtime_error("Failed to allocate output stream");
    }
    stream->time_base = { 1, fps };
    avcodec_parameters_from_context(stream->codecpar, codecCtx);

    // Open output file
    if (!(codecCtx->flags & AV_CODEC_FLAG_GLOBAL_HEADER)) {
        formatCtx->flags |= AVFMT_GLOBALHEADER;
    }
    if (avio_open(&formatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE) < 0) {
        std::cerr << "Could not open output file: " << outputFilePath << std::endl;
        throw std::runtime_error("Failed to open output file");
    }

    // Write header
    if (avformat_write_header(formatCtx, nullptr) < 0) {
        std::cerr << "Error occurred when writing header to output file" << std::endl;
        throw std::runtime_error("Failed to write header");
    }

    // Allocate frame
    frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Could not allocate frame" << std::endl;
        throw std::runtime_error("Failed to allocate frame");
    }
    frame->format = codecCtx->pix_fmt;
    frame->width = codecCtx->width;
    frame->height = codecCtx->height;
    if (av_frame_get_buffer(frame, 32) < 0) {
        std::cerr << "Could not allocate frame data" << std::endl;
        throw std::runtime_error("Failed to allocate frame data");
    }

    // Allocate packet
    packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Could not allocate packet" << std::endl;
        throw std::runtime_error("Failed to allocate packet");
    }

    // Initialize Torch tensors on CUDA
    y_plane = torch::empty({ height, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    u_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    v_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_flat = torch::empty({ u_plane.numel() * 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_plane = torch::empty({ height / 2, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    u_flat = u_plane.view(-1);
    v_flat = v_plane.view(-1);

    // Initialize CUDA device context
    if (init_cuda_context(&hw_device_ctx) < 0) {
        std::cerr << "Failed to initialize CUDA context" << std::endl;
        throw std::runtime_error("Failed to initialize CUDA context");
    }

    // Initialize CUDA frames context
    hw_frames_ctx = init_cuda_frames_ctx(hw_device_ctx, width, height, AV_PIX_FMT_NV12); // Adjust based on source format
    if (!hw_frames_ctx) {
        std::cerr << "Failed to initialize CUDA frames context" << std::endl;
        throw std::runtime_error("Failed to initialize CUDA frames context");
    }

    // Allocate interpolatedFrame and get buffer
    interpolatedFrame = av_frame_alloc();
    if (!interpolatedFrame) {
        std::cerr << "Could not allocate interpolated frame" << std::endl;
        throw std::runtime_error("Failed to allocate interpolated frame");
    }
    interpolatedFrame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx); // Set the CUDA frames context
    interpolatedFrame->format = AV_PIX_FMT_NV12;
    interpolatedFrame->width = width;
    interpolatedFrame->height = height;

    if (av_hwframe_get_buffer(hw_frames_ctx, interpolatedFrame, 0) < 0) {
        std::cerr << "Failed to allocate hardware frame buffer" << std::endl;
        throw std::runtime_error("Failed to allocate hardware frame buffer");
    }

    // Create CUDA streams
    CUDA_CHECK(cudaStreamCreate(&writestream));
    CUDA_CHECK(cudaStreamCreate(&convertstream));
    CUDA_CHECK(cudaStreamCreate(&uStream));

    // Initialize NPP Stream Context
    nppStreamCtx.hStream = convertstream;
    nppStreamCtx.nCudaDeviceId = 0; // Assuming device 0; adjust if necessary

    // Retrieve device properties to fill in other fields
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0)); // Assuming device 0

    nppStreamCtx.nMultiProcessorCount = deviceProp.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = deviceProp.sharedMemPerBlock;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMajor = deviceProp.major;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMinor = deviceProp.minor;
    nppStreamCtx.nStreamFlags = 0; // Default flags
    nppStreamCtx.nReserved0 = 0;    // Reserved field should be set to 0

    // Initialize Frame Pool with custom deleter
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        for (int i = 0; i < 20; ++i) { // Pre-allocate 20 frames
            AVFrame* poolFrame = av_frame_alloc();
            if (!poolFrame) {
                std::cerr << "Failed to allocate frame for pool." << std::endl;
                continue;
            }
            poolFrame->format = codecCtx->pix_fmt;
            poolFrame->width = codecCtx->width;
            poolFrame->height = codecCtx->height;
            if (av_frame_get_buffer(poolFrame, 32) < 0) {
                std::cerr << "Could not allocate frame data for pool." << std::endl;
                av_frame_free(&poolFrame);
                continue;
            }
            framePool.emplace_back(poolFrame, AVFrameDeleter()); // Correctly initialize unique_ptr with custom deleter
        }
    }

    // Start the writer thread only if not in benchmark mode
    if (!isBenchmark) {
        writerThread = std::thread(&FFmpegWriter::writeThread, this);
    }
}

inline FFmpegWriter::~FFmpegWriter() {
    finalize();  // Ensure finalization is done before destruction

    // Clean up CUDA streams
    CUDA_CHECK(cudaStreamSynchronize(convertstream));
    CUDA_CHECK(cudaStreamSynchronize(uStream));
    CUDA_CHECK(cudaStreamSynchronize(writestream));
    CUDA_CHECK(cudaStreamDestroy(convertstream));
    CUDA_CHECK(cudaStreamDestroy(uStream));
    CUDA_CHECK(cudaStreamDestroy(writestream));

    // Free SwsContext if allocated
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }

    // Free AVFrame and AVPacket
    if (frame) {
        av_frame_free(&frame);
    }
    if (interpolatedFrame) {
        av_frame_free(&interpolatedFrame);
    }
    if (packet) {
        av_packet_free(&packet);
    }

    // Free hardware contexts
    if (hw_frames_ctx) {
        av_buffer_unref(&hw_frames_ctx);
    }
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }

    // Free format context
    if (formatCtx && !(formatCtx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&formatCtx->pb);
    }
    if (formatCtx) {
        avformat_free_context(formatCtx);
    }

  //  std::cout << "FFmpegWriter destroyed successfully." << std::endl;
}

inline AVFrame* FFmpegWriter::acquireFrame() {
    std::lock_guard<std::mutex> lock(poolMutex);
    if (!framePool.empty()) {
        std::unique_ptr<AVFrame, AVFrameDeleter> framePtr = std::move(framePool.back());
        framePool.pop_back();
        return framePtr.release(); // Release ownership and return raw pointer
    }
    // If pool is empty, allocate a new frame
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Failed to allocate frame." << std::endl;
        return nullptr;
    }
    frame->format = codecCtx->pix_fmt;
    frame->width = codecCtx->width;
    frame->height = codecCtx->height;
    if (av_frame_get_buffer(frame, 32) < 0) {
        std::cerr << "Could not allocate frame data" << std::endl;
        av_frame_free(&frame);
        return nullptr;
    }
    return frame;
}

inline void FFmpegWriter::releaseFrame(AVFrame* frame) {
    std::lock_guard<std::mutex> lock(poolMutex);
    framePool.emplace_back(frame, AVFrameDeleter()); // Re-wrap the raw pointer into a unique_ptr
}

inline void FFmpegWriter::addFrame(at::Tensor inputTensor, bool benchmark) {
   

    // Prepare tensor
    if (inputTensor.dim() != 3) {
        inputTensor = inputTensor.squeeze(0).permute({ 1, 2, 0 })
            .mul_(255.0).clamp_(0, 255).to(torch::kUInt8).contiguous();
    }
    else {
        inputTensor = inputTensor.permute({ 1, 2, 0 })
            .mul_(255.0).clamp_(0, 255).to(torch::kUInt8).contiguous();
    }

    // Convert RGB to NV12 using NPP on the CUDA stream
    avframe_rgb_to_nv12_npp(inputTensor);

    // Asynchronously copy NV12 data into interpolated frame on the same CUDA stream
    CUDA_CHECK(cudaMemcpy2DAsync(
        interpolatedFrame->data[0],
        interpolatedFrame->linesize[0],
        nv12_tensor.data_ptr<uint8_t>(),
        width * sizeof(uint8_t),
        width * sizeof(uint8_t),
        height,
        cudaMemcpyDeviceToDevice,
        uStream
    ));

    CUDA_CHECK(cudaMemcpy2DAsync(
        interpolatedFrame->data[1],
        interpolatedFrame->linesize[1],
        nv12_tensor.data_ptr<uint8_t>() + (width * height),
        width * sizeof(uint8_t),
        width * sizeof(uint8_t),
        height / 2,
        cudaMemcpyDeviceToDevice,
        writestream
    ));

    if (benchmark) {
        return;  // Skip encoding in benchmark mode
    }

    AVFrame* frameToEncode = acquireFrame();
    if (!frameToEncode) {
        std::cerr << "Failed to acquire frame for encoding." << std::endl;
        return;
    }

    if (interpolatedFrame->format == AV_PIX_FMT_NV12) {
        av_frame_copy(frameToEncode, interpolatedFrame);
        frameToEncode->pts = pts++;
    }
    else if (interpolatedFrame->format == AV_PIX_FMT_CUDA) {
        // Transfer CUDA frame to NV12
        if (av_hwframe_transfer_data(frameToEncode, interpolatedFrame, 0) < 0) {
            std::cerr << "Failed to transfer CUDA frame to NV12" << std::endl;
            releaseFrame(frameToEncode);
            return;
        }
        frameToEncode->pts = pts++;
    }
    else {
        std::cerr << "Error: Unsupported pixel format: " << av_get_pix_fmt_name(static_cast<AVPixelFormat>(interpolatedFrame->format)) << std::endl;
        releaseFrame(frameToEncode);
        return;
    }

    // Enqueue the frame for encoding
    frameQueue.enqueue(frameToEncode);
}

inline void FFmpegWriter::writeThread() {

    while (true) {
        AVFrame* frame = nullptr;
        if (!frameQueue.try_dequeue(frame)) { // Non-blocking dequeue
            if (doneReadingFrames.load()) break; // Exit if no more frames will be added
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Avoid busy waiting
            continue;
        }

        if (frame) {
            writeFrame(frame);
            releaseFrame(frame);
        }
    }
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

    // Set PTS if necessary
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

    // Receive and write encoded packets
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
        if (av_interleaved_write_frame(formatCtx, packet) < 0) {
            std::cerr << "Error writing packet to output file." << std::endl;
            av_packet_unref(packet);
            return;
        }
        av_packet_unref(packet);  // Free the packet after writing
    }
}

inline void FFmpegWriter::finalize() {
    if (!isBenchmark) {
        doneReadingFrames.store(true);  // Indicate no more frames will be added

        if (writerThread.joinable()) {
            writerThread.join();  // Wait for writer thread to finish
        }

        // Flush encoder
        avcodec_send_frame(codecCtx, nullptr);
        while (avcodec_receive_packet(codecCtx, packet) >= 0) {
            packet->pts = av_rescale_q(packet->pts, codecCtx->time_base, stream->time_base);
            packet->dts = packet->pts;
            packet->duration = av_rescale_q(packet->duration, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            av_interleaved_write_frame(formatCtx, packet);
            av_packet_unref(packet);
        }

        // Write trailer
        av_write_trailer(formatCtx);

        // Close output
        if (formatCtx && !(formatCtx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&formatCtx->pb);
        }
    }

   // std::cout << "Finalization complete." << std::endl;
}

inline void FFmpegWriter::avframe_rgb_to_nv12_npp(at::Tensor output) {
    // Set up destination pointers and strides
    Npp8u* pDst[3] = { y_plane.data_ptr<Npp8u>(), u_plane.data_ptr<Npp8u>(), v_plane.data_ptr<Npp8u>() };
    int rDstStep[3] = { static_cast<int>(y_plane.stride(0)), static_cast<int>(u_plane.stride(0)), static_cast<int>(v_plane.stride(0)) };

    // Source RGB data and step
    Npp8u* pSrc = output.data_ptr<Npp8u>();
    int nSrcStep = output.stride(0);

    // Define ROI
    NppiSize oSizeROI = { width, height };

    // Perform RGB to YUV420 conversion (planar) using NPP and the specified stream
    NPP_CHECK_NPP(nppiRGBToYUV420_8u_C3P3R_Ctx(
        pSrc,
        nSrcStep,
        pDst,
        rDstStep,
        oSizeROI,
        nppStreamCtx // Pass by value, not by pointer
    ));

    // Interleave U and V planes to create UV plane for NV12
    uv_flat.index_put_({ torch::indexing::Slice(0, torch::indexing::None, 2) }, u_flat);
    uv_flat.index_put_({ torch::indexing::Slice(1, torch::indexing::None, 2) }, v_flat);

    uv_plane = uv_flat.view({ height / 2, width });

    nv12_tensor = torch::cat({ y_plane.view(-1), uv_plane.view(-1) }, 0);

    // No need to synchronize here; operations are enqueued on the stream
}
