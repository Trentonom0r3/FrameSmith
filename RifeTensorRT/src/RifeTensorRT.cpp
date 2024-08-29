#include "RifeTensorRT.h"
#include "downloadModels.h"
#include "coloredPrints.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <c10/cuda/CUDAStream.h> 
#include <trtHandler.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

RifeTensorRT::RifeTensorRT(
    std::string interpolateMethod,
    int interpolateFactor,
    int width,
    int height,
    bool half,
    bool ensemble
) : interpolateMethod(interpolateMethod),
interpolateFactor(interpolateFactor),
width(width),
height(height),
half(half),
ensemble(ensemble),
firstRun(true),
useI0AsSource(true),
device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
stream(c10::cuda::getStreamFromPool(false, device.index()))
{
    if (width > 1920 && height > 1080 && half) {
        std::cout << yellow("UHD and fp16 are not compatible with RIFE, defaulting to fp32") << std::endl;
        this->half = false;
    }
    handleModel();
}

void RifeTensorRT::cacheFrame() {
    I0.copy_(I1, true);
}

void RifeTensorRT::cacheFrameReset(const at::Tensor& frame) {
    I0.copy_(processFrame(frame), true);
    useI0AsSource = true;
}

nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];
        std::cout << sizes[i] << " ";

    }
    return dims;
}

void RifeTensorRT::handleModel() {
    std::string filename = modelsMap(interpolateMethod, "onnx", half, ensemble);
    std::string folderName = interpolateMethod;
    folderName.replace(folderName.find("-tensorrt"), 9, "-onnx");

    std::cout << "Model: " << filename << std::endl;
    std::filesystem::path modelPath = std::filesystem::path(getWeightsDir()) / folderName / filename;

    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Model not found, downloading it..." << std::endl;
        modelPath = downloadModels(interpolateMethod, "onnx", half, ensemble);
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Failed to download or locate the model: " << modelPath << std::endl;
            return;
        }
    }

    bool isCudnnEnabled = torch::cuda::cudnn_is_available();
    std::cout << "cuDNN is " << (isCudnnEnabled ? "enabled" : "disabled") << std::endl;

    // Initialize TensorRT engine
    std::cout << "Setting up TensorRT engine with Arguments: " << std::endl;
    std::cout << "Model Path: " << modelPath << std::endl;
    std::cout << "Half Precision: " << half << std::endl;
    std::cout << "Input Shape: " << "{ 1, 7, " << height << ", " << width << " }" << std::endl;

    enginePath = TensorRTEngineNameHandler(modelPath.string(), half, { 1, 7, height, width });
    std::tie(engine, context) = TensorRTEngineLoader(enginePath);

    if (!engine || !context || !std::filesystem::exists(enginePath)) {
        std::cout << "Loading engine failed, creating a new one" << std::endl;
        std::tie(engine, context) = TensorRTEngineCreator(
            modelPath.string(), enginePath, half, { 1, 7, height, width }, { 1, 7, height, width }, { 1, 7, height, width }
        );
    }

    // Setup Torch tensors for input/output
    dType = half ? torch::kFloat16 : torch::kFloat32;
    I0 = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    I1 = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyInput = torch::empty({ 1, 7, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyOutput = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();

    bindings = { dummyInput.data_ptr(), dummyOutput.data_ptr() };

    // Set Tensor Addresses and Input Shapes
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        void* bindingPtr = (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) ?
            static_cast<void*>(dummyInput.data_ptr()) :
            static_cast<void*>(dummyOutput.data_ptr());
        bool setaddyinfo = context->setTensorAddress(tensorName, bindingPtr);

        if (!setaddyinfo) {
			std::cerr << "Failed to set tensor address for " << tensorName << std::endl;
			///return;
		}

        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            auto expectedDims = engine->getTensorShape(tensorName);
            std::cout << "TensorRT expected dimensions: ";
            for (int j = 0; j < expectedDims.nbDims; ++j) {
                std::cout << expectedDims.d[j] << " ";
            }
            std::cout << std::endl;

            // Adjust input dimensions to match what TensorRT expects
            nvinfer1::Dims dims = toDims(dummyInput.sizes());
            dims.d[1] = expectedDims.d[1];  // Ensure the channel/temporal dimension matches
            dims.d[2] = expectedDims.d[2];  // Ensure the height/other dimension matches
            dims.d[3] = expectedDims.d[3];  // Ensure the width/other dimension matches

            bool success = context->setInputShape(tensorName, dims);
            nvinfer1::Dims dims2  = context->getTensorShape(tensorName);
            std::cout << "TensorRT input dimensions: ";
            for (int j = 0; j < dims2.nbDims; ++j) {
				std::cout << dims2.d[j] << " ";
			}
            std::cout << std::endl;
            if (!success) {
                std::cerr << "Failed to set input shape for " << tensorName << std::endl;
                return;
            }
        }
    }

    firstRun = true;
    useI0AsSource = true;
}

at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    try {
        // Ensure the frame is properly normalized
        auto processed = frame.to(device, dType, /*non_blocking=*/false, /*copy=*/true)
            .permute({ 2, 0, 1 })  // Change the order of the dimensions: from HWC to CHW
            .unsqueeze(0)           // Add a batch dimension
            .div(255.0)             // Normalize to [0, 1]
            .contiguous();

        return processed;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during processFrame: " << e.what() << std::endl;
        std::cerr << "Frame dimensions: " << frame.sizes() << " Frame dtype: " << frame.dtype() << std::endl;
        throw; // Re-throw the error after logging
    }
}

void RifeTensorRT::run(const at::Tensor& frame, bool benchmark, AVCodecContext* enc_ctx, AVFrame* outputFrame, AVFormatContext* fmt_ctx, AVStream* video_stream, int64_t pts) {
    c10::cuda::CUDAStreamGuard guard(stream);

    if (firstRun) {
        I0.copy_(processFrame(frame), true);
        firstRun = false;
        return;
    }

    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(processFrame(frame), true);

    for (int i = 0; i < interpolateFactor - 1; ++i) {
        at::Tensor timestep = torch::full({ 1, 1, height, width },
            (i + 1) * 1.0 / interpolateFactor,
            torch::TensorOptions().dtype(dType).device(device)).contiguous();

        dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true).contiguous();

        context->setTensorAddress("input", dummyInput.data_ptr());
        context->setTensorAddress("output", dummyOutput.data_ptr());

        if (!context->enqueueV3(static_cast<cudaStream_t>(stream))) {
            std::cerr << "Error during TensorRT inference!" << std::endl;
            return;
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

        at::Tensor output = dummyOutput.squeeze(0)
            .permute({ 1, 2, 0 })
            .mul(255.0)
            .clamp(0, 255)
            .to(torch::kU8)
            .to(torch::kCPU); // Convert to CPU for sws_scale

        // Convert RGB frame to YUV420P using sws_scale (inside RifeTensorRT::run)
        output = output.contiguous();
        AVFrame* frame_yuv = av_frame_alloc();
        frame_yuv->format = AV_PIX_FMT_YUV420P;
        frame_yuv->width = enc_ctx->width;
        frame_yuv->height = enc_ctx->height;
        av_frame_get_buffer(frame_yuv, 0);

        SwsContext* sws_ctx = sws_getContext(
            output.size(1), // width
            output.size(0), // height
            AV_PIX_FMT_RGB24,
            enc_ctx->width, enc_ctx->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);

        uint8_t* src_slices[1] = { output.data_ptr<uint8_t>() };
        int src_stride[1] = { static_cast<int>(output.size(1) * 3) }; // width * 3 channels

        int res = sws_scale(sws_ctx, src_slices, src_stride, 0, output.size(0), frame_yuv->data, frame_yuv->linesize);
        if (res <= 0) {
            std::cerr << "Error: sws_scale in RifeTensorRT failed with result " << res << std::endl;
        }

        // Debug: Print some YUV pixel values to check if they make sense
        std::cout << "YUV frame first pixel: Y=" << static_cast<int>(frame_yuv->data[0][0])
            << " U=" << static_cast<int>(frame_yuv->data[1][0])
            << " V=" << static_cast<int>(frame_yuv->data[2][0]) << std::endl;


        sws_freeContext(sws_ctx);

        // Set the timestamp for the YUV frame
        frame_yuv->pts = pts;

        // Send the YUV frame to the encoder
        int ret = avcodec_send_frame(enc_ctx, frame_yuv);
        if (ret == AVERROR(EAGAIN)) {
            AVPacket pkt;
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;

            while (avcodec_receive_packet(enc_ctx, &pkt) == 0) {
                pkt.stream_index = video_stream->index;
                av_packet_rescale_ts(&pkt, enc_ctx->time_base, video_stream->time_base);
                ret = av_write_frame(fmt_ctx, &pkt);
                if (ret < 0) {
                    std::cerr << "Error writing frame to output file! Error Code: " << ret << std::endl;
                    av_packet_unref(&pkt);
                    av_frame_free(&frame_yuv);
                    return;
                }
                av_packet_unref(&pkt);
            }

            ret = avcodec_send_frame(enc_ctx, frame_yuv);
        }

        if (ret < 0) {
            std::cerr << "Error sending frame to encoder! Error Code: " << ret << std::endl;
            av_frame_free(&frame_yuv);
            return;
        }

        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = nullptr;
        pkt.size = 0;

        while (avcodec_receive_packet(enc_ctx, &pkt) == 0) {
            pkt.stream_index = video_stream->index;
            av_packet_rescale_ts(&pkt, enc_ctx->time_base, video_stream->time_base);
            ret = av_write_frame(fmt_ctx, &pkt);
            if (ret < 0) {
                std::cerr << "Error writing frame to output file! Error Code: " << ret << std::endl;
                av_packet_unref(&pkt);
                av_frame_free(&frame_yuv);
                return;
            }
            av_packet_unref(&pkt);
        }

        av_frame_free(&frame_yuv);
    }

    useI0AsSource = !useI0AsSource;
}
