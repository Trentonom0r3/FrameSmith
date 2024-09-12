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

nvinfer1::Dims toDims(const c10::IntArrayRef& sizes) {
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = sizes[i];

    }
    return dims;
}

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
RifeTensorRT::RifeTensorRT(std::string interpolateMethod, int interpolateFactor, int width, int height, bool half, bool ensemble, bool benchmark)
    : interpolateMethod(interpolateMethod),
    interpolateFactor(interpolateFactor),
    width(width),
    height(height),
    half(half),
    ensemble(ensemble),
    firstRun(true),
    useI0AsSource(true),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    benchmarkMode(benchmark)
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
    hw_frames_ctx = init_cuda_frames_ctx(hw_device_ctx, width, height, AV_PIX_FMT_YUV420P); // Change this based on your source format
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


torch::Tensor avframe_nv12_to_rgb_npp(AVFrame* gpu_frame, uint8_t*& d_rgb, size_t& rgb_pitch) {
    int width = gpu_frame->width;
    int height = gpu_frame->height;

    // NPP uses pitch (line size) for memory alignment
    int nYUVPitch = gpu_frame->linesize[0];  // NV12 Y plane pitch
    int nUVPitch = gpu_frame->linesize[1];   // NV12 UV plane pitch

    if (!d_rgb) {
        // Allocate device memory for the RGB image if it's not already allocated
        cudaMallocPitch(&d_rgb, &rgb_pitch, width * 3 * sizeof(uint8_t), height);
    }

    // NPP requires ROI (Region of Interest) to process the image
    NppiSize oSizeROI = { width, height };

    // Set up the array of pointers to Y and UV data
    const Npp8u* pSrc[2] = { gpu_frame->data[0], gpu_frame->data[1] };  // Y plane, UV plane (interleaved)

    // Perform NV12 to RGB conversion using NPP
    NPP_CHECK_NPP(nppiNV12ToRGB_8u_P2C3R(
        pSrc, nYUVPitch,   // Source Y and UV data and pitch
        d_rgb, rgb_pitch,  // Destination RGB buffer and pitch
        oSizeROI          // Size of the image (Region of Interest)
    ));

    // Step 1: Allocate a Torch tensor on the GPU (with exact size)
    torch::Tensor tensor = torch::empty({ 1, 3, height, width }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));

    // Step 2: Copy the GPU buffer (d_rgb) into the Torch tensor memory using cudaMemcpy2D
    cudaMemcpy2D(
		tensor.data_ptr(), width * 3,  // Destination pointer and pitch
		d_rgb, rgb_pitch,              // Source pointer and pitch
		width * 3, height,              // Width, height
		cudaMemcpyDeviceToDevice        // Direction of the copy
	);
   //showTensorAsImage(tensor);
    // Step 3: Return the tensor
    return tensor;
}


// Preprocess the frame: normalize and permute dimensions (NHWC to NCHW)
// Preprocess the frame: normalize pixel values
at::Tensor RifeTensorRT::processFrame(const at::Tensor& frame) const {
    // Since the input tensor is already [1, 3, H, W] (NCHW), no need to permute
   // showTensorAsImage(frame.squeeze(0));
    return frame.to(half ? torch::kFloat16 : torch::kFloat32)
        .div(255.0)              // Normalize pixel values
        .clamp(0.0, 1.0)         // Clamp to [0, 1]
        .contiguous();           // Ensure memory is contiguous
}

torch::Tensor avframe_to_tensor(AVFrame* gpu_frame) {
    int width = gpu_frame->width;
    int height = gpu_frame->height;
    int channels = 3;  // Assuming RGB or YUV format

    // Wrap the GPU memory (gpu_frame->data[0]) into a torch::Tensor
    torch::Tensor tensor = torch::from_blob(
        gpu_frame->data[0],  // pointer to GPU data
        { height, width, channels },  // frame dimensions
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8)
    );

    return tensor;
}
void RifeTensorRT::run(AVFrame* inputFrame, FFmpegWriter& writer) {
    static uint8_t* d_rgb = nullptr;  // Reusable device buffer for RGB frames
    static size_t rgb_pitch = 0;      // Reusable pitch for RGB buffer

    // Convert the input AVFrame (NV12) to a Torch tensor (RGB) using NPP
    at::Tensor processedTensor = processFrame(avframe_nv12_to_rgb_npp(inputFrame, d_rgb, rgb_pitch));
   // showTensorAsImage(processedTensor.squeeze(0));
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
        dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), true);

        if (!context->enqueueV3(stream)) {
            std::cerr << "Error during TensorRT inference!" << std::endl;
            return;
        }

        at::Tensor output = dummyOutput.squeeze(0).permute({ 1, 2, 0 }).mul(255.0).clamp(0, 255).to(torch::kU8);
     
        int tensorHeight = output.size(0);
        int tensorWidth = output.size(1);
        int tensorChannels = output.size(2);

       // std::cout << "Showing output" << std::endl;
       // showTensorAsImage(output);
      //  std::cout << "Frame Format: " << av_get_pix_fmt_name((AVPixelFormat)inputFrame->format) << std::endl;
        interpolatedFrame->format = AV_PIX_FMT_CUDA;
        interpolatedFrame->width = tensorWidth;
        interpolatedFrame->height = tensorHeight;

        if (av_hwframe_get_buffer(hw_frames_ctx, interpolatedFrame, 32) < 0) {
            std::cerr << "Error allocating CUDA frame buffer!" << std::endl;
            return;
        }

        void* dstPtr = interpolatedFrame->data[0];
        void* srcPtr = output.data_ptr();
        cudaMemcpyAsync(dstPtr, srcPtr, tensorWidth * tensorHeight * tensorChannels, cudaMemcpyDeviceToDevice, writestream);
       // debugShowAVFrameYUV(interpolatedFrame);
        if (!benchmarkMode) {
			writer.addFrame(interpolatedFrame);
		}
       // writer.addFrame(interpolatedFrame);
    }
    if (!benchmarkMode) {
        writer.addFrame(inputFrame);
    }
    useI0AsSource = !useI0AsSource;
}
