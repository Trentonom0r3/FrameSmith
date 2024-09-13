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
#include <Writer.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h> // Add this line
#include <nppi_support_functions.h>

nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];

    }
    return dims;
}

// Helper function to set up CUDA device context
int init_cuda_context(AVBufferRef** hw_device_ctx) {
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
AVBufferRef* init_cuda_frames_ctx(AVBufferRef* hw_device_ctx, int width, int height, AVPixelFormat sw_format) {
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

// Constructor implementation with CUDA context setup
RifeTensorRT::RifeTensorRT(std::string interpolateMethod, int interpolateFactor, int width, int height, bool half, bool ensemble, bool benchmark, FFmpegWriter& writer)
    : interpolateMethod(interpolateMethod),
    interpolateFactor(interpolateFactor),
    width(width),
    height(height),
    half(half),
    ensemble(ensemble),
    firstRun(true),
    useI0AsSource(true),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    benchmarkMode(benchmark),
    writer(writer)
{
    // Initialize model and tensors
    handleModel();

    // Initialize CUDA streams
    cudaStreamCreate(&stream);
    cudaStreamCreate(&writestream);

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

    interpolatedFrame = av_frame_alloc();
    interpolatedFrame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx); // Set the CUDA frames context

}

RifeTensorRT::~RifeTensorRT() {
    cudaStreamDestroy(stream);
    cudaStreamDestroy(writestream);
    av_frame_free(&interpolatedFrame);
    av_buffer_unref(&hw_frames_ctx);
    av_buffer_unref(&hw_device_ctx);
}


// Method to handle model loading and initialization
void RifeTensorRT::handleModel() {
    std::string filename = modelsMap(interpolateMethod, "onnx", half, ensemble);
    std::string folderName = interpolateMethod;
    folderName.replace(folderName.find("-tensorrt"), 9, "-onnx");

    std::filesystem::path modelPath = std::filesystem::path(getWeightsDir()) / folderName / filename;

    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Model not found, downloading it..." << std::endl;
        modelPath = downloadModels(interpolateMethod, "onnx", half, ensemble);
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "Failed to download or locate the model: " << modelPath << std::endl;
            return;
        }
    }

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
    dummyInput = torch::zeros({ 1, 7, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    dummyOutput = torch::zeros({ 1, 3, height, width }, torch::TensorOptions().dtype(dType).device(device)).contiguous();
    rgb_tensor = torch::empty({ height, width, 3 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).contiguous();
    y_plane = torch::empty({ height, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    u_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    v_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_flat = torch::empty({ u_plane.numel() * 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    uv_plane = torch::empty({ height / 2, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));


    bindings = { dummyInput.data_ptr(), dummyOutput.data_ptr() };

    // Set Tensor Addresses and Input Shapes
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        void* bindingPtr = (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) ?
            static_cast<void*>(dummyInput.data_ptr()) :
            static_cast<void*>(dummyOutput.data_ptr());
        bool setaddyinfo = context->setTensorAddress(tensorName, bindingPtr);

        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            bool success = context->setInputShape(tensorName, toDims(dummyInput.sizes()));
            if (!success) {
                std::cerr << "Failed to set input shape for " << tensorName << std::endl;
                return;
            }
        }
    }

    firstRun = true;
    useI0AsSource = true;
    // Set the input and output tensor addresses in the TensorRT context
    context->setTensorAddress("input", dummyInput.data_ptr());
    context->setTensorAddress("output", dummyOutput.data_ptr());

}

#include <npp.h>
#include <nppi.h>


// Define error handling macros
#define NPP_CHECK_NPP(func) { \
    NppStatus status = (func); \
    if (status != NPP_SUCCESS) { \
        std::cerr << "NPP Error: " << status << std::endl; \
        exit(-1); \
    } \
}


torch::Tensor RifeTensorRT::avframe_nv12_to_rgb_npp(AVFrame* gpu_frame) {
    int width = gpu_frame->width;
    int height = gpu_frame->height;

    if (gpu_frame->format != AV_PIX_FMT_CUDA) {
        std::cerr << "Frame is not in CUDA format." << std::endl;
        return torch::Tensor();
    }

    int nYUVPitch = gpu_frame->linesize[0];
    int nUVPitch = gpu_frame->linesize[1];

    NppiSize oSizeROI = { width, height };
    const Npp8u* pSrc[2] = { gpu_frame->data[0], gpu_frame->data[1] };

    NPP_CHECK_NPP(nppiNV12ToRGB_8u_P2C3R(
        pSrc, nYUVPitch,
        rgb_tensor.data_ptr<Npp8u>(), rgb_tensor.stride(0),
        oSizeROI
    ));

    return rgb_tensor.unsqueeze(0).permute({ 0, 3, 1, 2 }).contiguous();
}



// Preprocess the frame: normalize and permute dimensions (NHWC to NCHW)
// Preprocess the frame: normalize pixel values
at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    // Since the input tensor is already [1, 3, H, W] (NCHW), no need to permute
   // showTensorAsImage(frame.squeeze(0));
    return frame.to(half ? torch::kFloat16 : torch::kFloat32)
        .div_(255.0)              // Normalize pixel values
        .clamp_(0.0, 1.0)         // Clamp to [0, 1]
        .contiguous();           // Ensure memory is contiguous
}

torch::Tensor RifeTensorRT::avframe_rgb_to_nv12_npp(at::Tensor rgb_tensor) {
    // Ensure the input tensor is contiguous
    // Set up destination pointers and strides
    Npp8u* pDst[3] = { y_plane.data_ptr<Npp8u>(), u_plane.data_ptr<Npp8u>(), v_plane.data_ptr<Npp8u>() };
    int rDstStep[3] = { static_cast<int>(y_plane.stride(0)), static_cast<int>(u_plane.stride(0)), static_cast<int>(v_plane.stride(0)) };

    // Source RGB data and step
    Npp8u* pSrc = rgb_tensor.data_ptr<Npp8u>();
    int nSrcStep = rgb_tensor.stride(0);

    // Define ROI
    NppiSize oSizeROI = { width, height };

    // Perform RGB to YUV420 conversion (planar)
 // Perform RGB to YUV420 conversion (planar)
    NPP_CHECK_NPP(nppiRGBToYUV420_8u_C3P3R(
        pSrc,
        nSrcStep,
        pDst,
        rDstStep,
        oSizeROI
    ));

    // Interleave U and V planes on GPU
    // Flatten U and V planes
    torch::Tensor u_flat = u_plane.view(-1);
    torch::Tensor v_flat = v_plane.view(-1);

    // Reuse uv_flat
    // Interleave U and V values
    uv_flat.index_put_({ torch::indexing::Slice(0, torch::indexing::None, 2) }, u_flat);
    uv_flat.index_put_({ torch::indexing::Slice(1, torch::indexing::None, 2) }, v_flat);

    // Reshape UV data back to 2D
    uv_plane = uv_flat.view({ height / 2, width });

    // Concatenate Y plane and UV plane into NV12 tensor
    torch::Tensor nv12_tensor = torch::cat({ y_plane.view(-1), uv_plane.view(-1) }, 0);

    return nv12_tensor;
}

void RifeTensorRT::run(AVFrame* inputFrame) {

    // Convert the input AVFrame (NV12) to a Torch tensor (RGB) using NPP
    at::Tensor processedTensor = processFrame(avframe_nv12_to_rgb_npp(inputFrame));

    if (firstRun) {
        I0.copy_(processedTensor, true);
        firstRun = false;
        if (!benchmarkMode) {
            writer.addFrame(inputFrame);
        }
        return;
    }

    // Alternate between I0 and I1 for double-buffering
    auto& source = useI0AsSource ? I0 : I1;
    auto& destination = useI0AsSource ? I1 : I0;
    destination.copy_(processedTensor, true);

    for (int i = 0; i < interpolateFactor - 1; ++i) {
        auto timestep = torch::full({ 1, 1, height, width }, (i + 1) * 1.0 / interpolateFactor, torch::TensorOptions().dtype(dType).device(device)).contiguous();

        // New code using slicing and copying
        dummyInput.slice(1, 0, 3).copy_(source);
        dummyInput.slice(1, 3, 6).copy_(destination);
        dummyInput.select(1, 6).fill_((i + 1) * 1.0 / interpolateFactor);

        if (!context->enqueueV3(stream)) {
            std::cerr << "Error during TensorRT inference!" << std::endl;
            return;
        }

        // Convert the result tensor back to NV12 format
        at::Tensor output = dummyOutput.squeeze(0).permute({ 1, 2, 0 })
            .mul(255.0).clamp(0, 255).to(torch::kU8).contiguous();

        // Convert the output tensor (RGB) to NV12
        at::Tensor nv12_tensor = avframe_rgb_to_nv12_npp(output);

        // Check if nv12_tensor is valid
        if (!nv12_tensor.defined()) {
            std::cerr << "Failed to convert RGB to NV12." << std::endl;
            return;
        }

        // Allocate the frame
        av_frame_unref(interpolatedFrame); // Unreference the frame before reusing
        interpolatedFrame->format = AV_PIX_FMT_NV12;
        interpolatedFrame->width = width;
        interpolatedFrame->height = height;

        if (av_hwframe_get_buffer(hw_frames_ctx, interpolatedFrame, 0) < 0) {
            std::cerr << "Error allocating frame" << std::endl;
            return;
        }

        // Copy the NV12 tensor data back into the interpolated frame

        // Copy Y plane
        cudaMemcpy2DAsync(
            interpolatedFrame->data[0],             // Destination pointer
            interpolatedFrame->linesize[0],         // Destination pitch (line size)
            nv12_tensor.data_ptr<uint8_t>(),        // Source pointer (Y plane)
            width * sizeof(uint8_t),                // Source pitch
            width * sizeof(uint8_t),                // Width of the copied region
            height,                                 // Height of the copied region
            cudaMemcpyDeviceToDevice ,               // Copy kind
            writestream							 // Stream
        );

        // Copy UV plane
        cudaMemcpy2DAsync(
            interpolatedFrame->data[1],                                     // Destination pointer
            interpolatedFrame->linesize[1],                                 // Destination pitch (line size)
            nv12_tensor.data_ptr<uint8_t>() + (width * height),             // Source pointer (UV plane)
            width * sizeof(uint8_t),                                        // Source pitch
            width * sizeof(uint8_t),                                        // Width of the copied region
            height / 2,                                                     // Height of the copied region (UV plane is half the height)
            cudaMemcpyDeviceToDevice,               // Copy kind
            writestream							 // Stream
        );


        if (!benchmarkMode) {
            writer.addFrame(interpolatedFrame);
        }
    }

    if (!benchmarkMode) {
        writer.addFrame(inputFrame);
    }

    useI0AsSource = !useI0AsSource;
}
