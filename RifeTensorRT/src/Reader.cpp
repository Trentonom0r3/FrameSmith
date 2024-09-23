#include <Reader.h>

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
    fps = static_cast<double>(formatCtx->streams[0]->avg_frame_rate.num) / formatCtx->streams[0]->avg_frame_rate.den;

    // Initialize intermediate tensors
    // RGB tensor with shape {height, width, 3}, dtype uint8, device CUDA
    rgb_tensor = torch::empty({ height, width, 3 }, torch::TensorOptions().dtype(torch::kUInt8).device(device).requires_grad(false));

    // Intermediate tensor with shape {1, 3, height, width}, dtype float32 or float16
    intermediate_tensor = torch::empty({ 1, 3, height, width }, torch::TensorOptions().dtype(halfPrecision ? torch::kFloat16 : torch::kFloat32).device(device).requires_grad(false));
    cudaStreamCreate(&stream);
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
    cudaStreamDestroy(stream);
}

void FFmpegReader::avframe_nv12_to_rgb_npp(AVFrame* gpu_frame) {
    // Y and UV planes
    const unsigned char* yPlane = gpu_frame->data[0];
    const unsigned char* uvPlane = gpu_frame->data[1];

    // Strides
    int yStride = gpu_frame->linesize[0];
    int uvStride = gpu_frame->linesize[1];

    // RGB output pointer
    unsigned char* rgbOutput = static_cast<unsigned char*>(rgb_tensor.data_ptr());

    // Assuming rgb_tensor is allocated with stride = width * 3
    int rgbStride = width * 3;

    // Launch the custom CUDA kernel
    nv12_to_rgb(
        yPlane,
        uvPlane,
        width,
        height,
        yStride,
        uvStride,
        rgbOutput,
        rgbStride,
        stream
    );
}

void FFmpegReader::normalizeFrame() {
    // Convert RGB tensor to {1, 3, H, W} and normalize
    // Assuming rgb_tensor is already on CUDA and contiguous

    // Reshape and permute to {3, H, W}
    torch::Tensor reshaped = rgb_tensor.view({ height, width, 3 }).permute({ 2, 0, 1 });

    // Add batch dimension to make it {1, 3, H, W}
    torch::Tensor batched = reshaped.unsqueeze(0);

    // Convert to float and normalize shape of {1, 3, H, W}
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
        // Read a packet from the input file
        int ret = av_read_frame(formatCtx, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                // Flush decoder at EOF
                avcodec_send_packet(codecCtx, nullptr);
            }
            else {
                // Handle other errors
                char errBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errBuf, sizeof(errBuf));
                std::cerr << "Error reading frame: " << errBuf << std::endl;
                break;
            }
        }
        else {
            // If the packet belongs to the video stream, send it to the decoder
            if (packet->stream_index == videoStreamIndex) {
                ret = avcodec_send_packet(codecCtx, packet);
                if (ret < 0) {
                    char errBuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    std::cerr << "Error sending packet: " << errBuf << std::endl;
                    break;
                }
            }
            // Unref packet to free its data
            av_packet_unref(packet);
        }

        // Receive decoded frames
        while (true) {
            ret = avcodec_receive_frame(codecCtx, frameOut);
            if (ret == AVERROR(EAGAIN)) {
                // Decoder needs more data, exit inner loop to read next packet
                break;
            }
            else if (ret == AVERROR_EOF) {
                // End of stream
                success = false;
                break;
            }
            else if (ret < 0) {
                char errBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errBuf, sizeof(errBuf));
                std::cerr << "Error receiving frame: " << errBuf << std::endl;
                break;
            }

            // If the frame is in CUDA format, process it
            if (frameOut->format == AV_PIX_FMT_CUDA) {
                avframe_nv12_to_rgb_npp(frameOut);  // Convert NV12 to RGB
                normalizeFrame();                   // Normalize the frame
                // Copy the processed frame to the output tensor
                tensor.copy_(intermediate_tensor, true);

                success = true;
                break;  // Break out of inner loop after successfully processing one frame
            }
            else {
                std::cerr << "Frame is not in CUDA format." << std::endl;
                success = false;
                break;
            }
        }

        if (success || ret == AVERROR_EOF) {
            break;  // Exit outer loop if frame was successfully read or end of stream
        }
    }

    av_frame_free(&frameOut);
    return success;
}
