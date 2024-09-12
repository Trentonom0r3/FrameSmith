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
#include <opencv2/opencv.hpp>

void debugShowAVFrameYUV(AVFrame* gpu_yuv_frame) {
    // Step 1: Transfer the CUDA frame to a CPU-based frame (still in YUV format)
    cv::setUseOptimized(false);
    AVFrame* cpu_yuv_frame = av_frame_alloc();
    if (av_hwframe_transfer_data(cpu_yuv_frame, gpu_yuv_frame, 0) < 0) {
        std::cerr << "Error transferring frame from CUDA to CPU" << std::endl;
        av_frame_free(&cpu_yuv_frame);
        return;
    }

    // Print the format to verify it's correct
    std::cout << "Transferred YUV Frame Format: " << cpu_yuv_frame->format << std::endl;

    // Step 2: Create a SwsContext for converting YUV (CPU) to RGB
    // Force the YUV format to NV12 or YUV420p if necessary
    SwsContext* sws_ctx = sws_getContext(
        cpu_yuv_frame->width, cpu_yuv_frame->height, AV_PIX_FMT_NV12,  // Force format to NV12 (adjust if needed)
        cpu_yuv_frame->width, cpu_yuv_frame->height, AV_PIX_FMT_RGB24, // Destination format (RGB)
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    if (!sws_ctx) {
        std::cerr << "Failed to create SwsContext!" << std::endl;
        av_frame_free(&cpu_yuv_frame);
        return;
    }

    // Step 3: Allocate an AVFrame for the RGB image
    AVFrame* rgb_frame = av_frame_alloc();
    rgb_frame->format = AV_PIX_FMT_RGB24;
    rgb_frame->width = cpu_yuv_frame->width;
    rgb_frame->height = cpu_yuv_frame->height;
    av_frame_get_buffer(rgb_frame, 32);  // Allocate RGB frame buffer

    // Step 4: Convert YUV (CPU) to RGB
    sws_scale(
        sws_ctx,
        cpu_yuv_frame->data, cpu_yuv_frame->linesize, 0, cpu_yuv_frame->height,  // Source YUV data
        rgb_frame->data, rgb_frame->linesize                                      // Destination RGB data
    );

    // Step 5: Convert the RGB AVFrame to OpenCV Mat
    cv::Mat rgb_image(
        rgb_frame->height, rgb_frame->width, CV_8UC3,               // OpenCV expects CV_8UC3 for RGB
        rgb_frame->data[0], rgb_frame->linesize[0]                  // Data pointer and line size from AVFrame
    );

    // Step 6: Display the image using OpenCV
    cv::imshow("YUV to RGB Image", rgb_image);
    cv::waitKey(0);  // Wait for a key press to close the window

    // Step 7: Clean up and free the frames
    sws_freeContext(sws_ctx);
    av_frame_free(&rgb_frame);
    av_frame_free(&cpu_yuv_frame);
}


// Function to convert a Torch tensor to OpenCV Mat and display the image
void showTensorAsImage(torch::Tensor tensor) {
    // Step 1: Move the tensor to CPU if it's on the GPU (CUDA)
    std::cout << "DEBUG PRINTS OF ALL TENSOR INFO: " << std::endl;
    std::cout << "Tensor Device: " << tensor.device() << std::endl;
    std::cout << "Tensor Data Type: " << tensor.dtype() << std::endl;
    std::cout << "Tensor Shape: " << tensor.sizes() << std::endl;
    std::cout << "Tensor Layout: " << tensor.layout() << std::endl;
    std::cout << "Tensor Stride: " << tensor.strides() << std::endl;
    std::cout << "Tensor Storage Offset: " << tensor.storage_offset() << std::endl;

    if (tensor.device().is_cuda()) {
        std::cout << "Moving tensor to CPU..." << std::endl;
        tensor = tensor.to(torch::kCPU);
    }
    cv::setUseOptimized(false);

    // Step 2: Convert the tensor to uint8 (expected by OpenCV)
    tensor = tensor // Remove the batch dimension
        // Ensure contiguous memory layout
        .to(torch::kU8).contiguous();       // Convert to unsigned 8-bit (0-255)

    // Step 3: Get tensor dimensions
    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

    // Step 4: Convert Torch tensor to OpenCV Mat
    cv::Mat image(height, width, (channels == 3 ? CV_8UC3 : CV_8UC1), tensor.data_ptr<uint8_t>());

    // Step 5: Display the image using OpenCV
    cv::imshow("Tensor Image", image);
    cv::waitKey(0);  // Wait for a key press to close the window
}

torch::Tensor avframe_nv12_to_rgb_npp(AVFrame* gpu_frame, uint8_t*& d_rgb, size_t& rgb_pitch) {
    int width = gpu_frame->width;
    int height = gpu_frame->height;

    // Ensure the frame format is CUDA (GPU memory)
    if (gpu_frame->format != AV_PIX_FMT_CUDA) {
        std::cerr << "Frame is not in CUDA format." << std::endl;
        return torch::Tensor();
    }

    // Source pitch for Y and UV planes
    int nYUVPitch = gpu_frame->linesize[0];  // Y plane pitch
    int nUVPitch = gpu_frame->linesize[1];   // UV plane pitch

    // Allocate device memory for the RGB image if not already allocated
    if (!d_rgb) {
        cudaMallocPitch(&d_rgb, &rgb_pitch, width * 3 * sizeof(uint8_t), height);
    }

    // NPP requires ROI (Region of Interest) to process the image
    NppiSize oSizeROI = { width, height };

    // Set up the array of pointers to Y and UV data
    const Npp8u* pSrc[2] = { gpu_frame->data[0], gpu_frame->data[1] };  // Y plane, UV plane (interleaved)

    // Perform NV12 to RGB conversion using NPP
    NppStatus status = nppiNV12ToRGB_8u_P2C3R(
        pSrc, nYUVPitch,           // Source YUV planes and pitch
        d_rgb, rgb_pitch,          // Destination RGB buffer and pitch
        oSizeROI                   // ROI specifying the image size
    );

    // Check for errors in NPP function
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error during NV12 to RGB conversion: " << status << std::endl;
        return torch::Tensor();
    }

    // Allocate a Torch tensor on the GPU with the shape {height, width, 3}
    torch::Tensor tensor = torch::empty({ height, width, 3 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).contiguous();

    // Copy the RGB data from the NPP buffer to the Torch tensor
    cudaError_t err = cudaMemcpy2D(
        tensor.data_ptr(), width * 3 * sizeof(uint8_t),  // Destination tensor pointer and pitch
        d_rgb, rgb_pitch,                               // Source RGB buffer and pitch
        width * 3 * sizeof(uint8_t), height,            // Width, height
        cudaMemcpyDeviceToDevice                        // Copy direction
    );

    // Check for CUDA errors
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during memcpy2D: " << cudaGetErrorString(err) << std::endl;
        return torch::Tensor();  // Return an empty tensor on error
    }

    // Display the tensor as an image for debugging
  //  showTensorAsImage(tensor);

    // Return the tensor for further processing
    return tensor.unsqueeze(0) // Add a batch dimension
		.permute({ 0, 3, 1, 2 }) // Permute dimensions to NCHW
		.contiguous();           // Ensure memory is contiguous
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
    ).contiguous();  // Ensure memory is contiguous

    return tensor;
}

torch::Tensor avframe_rgb_to_nv12_npp(at::Tensor rgb_tensor, int width, int height) {
    // Ensure the input tensor is contiguous and in HWC format
    rgb_tensor = rgb_tensor.contiguous();

    // Allocate tensors for Y, U, and V planes on GPU
    torch::Tensor y_plane = torch::empty({ height, width }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    torch::Tensor u_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    torch::Tensor v_plane = torch::empty({ height / 2, width / 2 }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    // Set up destination pointers and strides
    Npp8u* pDst[3] = { y_plane.data_ptr<Npp8u>(), u_plane.data_ptr<Npp8u>(), v_plane.data_ptr<Npp8u>() };
    int rDstStep[3] = { static_cast<int>(width), static_cast<int>(width / 2), static_cast<int>(width / 2) };

    // Source RGB data and step
    Npp8u* pSrc = rgb_tensor.data_ptr<Npp8u>();
    int nSrcStep = width * 3;

    // Define ROI
    NppiSize oSizeROI = { width, height };

    // Perform RGB to YUV420 conversion (planar)
    NppStatus status = nppiRGBToYUV420_8u_C3P3R(
        pSrc,
        nSrcStep,
        pDst,
        rDstStep,
        oSizeROI
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error during RGB to YUV420 conversion: " << status << std::endl;
        return torch::Tensor();
    }

    // Move U and V planes to CPU
    torch::Tensor u_plane_cpu = u_plane.cpu();
    torch::Tensor v_plane_cpu = v_plane.cpu();

    // Interleave U and V planes on CPU to create the UV plane for NV12
    // Allocate UV plane on CPU
    torch::Tensor uv_plane_cpu = torch::empty({ height / 2, width }, torch::TensorOptions().dtype(torch::kUInt8));

    // Interleave U and V planes
    for (int i = 0; i < height / 2; ++i) {
        uint8_t* u_row = u_plane_cpu.data_ptr<uint8_t>() + i * (width / 2);
        uint8_t* v_row = v_plane_cpu.data_ptr<uint8_t>() + i * (width / 2);
        uint8_t* uv_row = uv_plane_cpu.data_ptr<uint8_t>() + i * width;
        for (int j = 0; j < width / 2; ++j) {
            uv_row[j * 2] = u_row[j];
            uv_row[j * 2 + 1] = v_row[j];
        }
    }

    // Move UV plane back to GPU
    torch::Tensor uv_plane = uv_plane_cpu.to(torch::kCUDA);

    // Concatenate Y plane and UV plane into NV12 tensor
    torch::Tensor nv12_tensor = torch::cat({ y_plane.flatten(), uv_plane.flatten() }, 0);

    return nv12_tensor;
}

void RifeTensorRT::run(AVFrame* inputFrame, FFmpegWriter& writer) {
    static uint8_t* d_rgb = nullptr;  // Reusable device buffer for RGB frames
    static size_t rgb_pitch = 0;      // Reusable pitch for RGB buffer

    // Convert the input AVFrame (NV12) to a Torch tensor (RGB) using NPP
    at::Tensor processedTensor = processFrame(avframe_nv12_to_rgb_npp(inputFrame, d_rgb, rgb_pitch));

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
        dummyInput.copy_(torch::cat({ source, destination, timestep }, 1), false);

        if (!context->enqueueV3(stream)) {
            std::cerr << "Error during TensorRT inference!" << std::endl;
            return;
        }

        // Convert the result tensor back to NV12 format
        at::Tensor output = dummyOutput.squeeze(0).permute({ 1, 2, 0 })
            .mul(255.0).clamp(0, 255).to(torch::kU8).contiguous();

        // Convert the output tensor (RGB) to NV12
        at::Tensor nv12_tensor = avframe_rgb_to_nv12_npp(output, width, height);

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
            cudaMemcpyDeviceToDevice,               // Copy kind
            writestream													  // Stream for asynchronous copy
        );

        // Copy UV plane
        cudaMemcpy2DAsync(
            interpolatedFrame->data[1],                                     // Destination pointer
            interpolatedFrame->linesize[1],                                 // Destination pitch (line size)
            nv12_tensor.data_ptr<uint8_t>() + (width * height),             // Source pointer (UV plane)
            width * sizeof(uint8_t),                                        // Source pitch
            width * sizeof(uint8_t),                                        // Width of the copied region
            height / 2,                                                     // Height of the copied region (UV plane is half the height)
            cudaMemcpyDeviceToDevice,                                        // Copy kind
            writestream													  // Stream for asynchronous copy
        );

        // Add the frame to the writer
        writer.addFrame(interpolatedFrame);
    }

    if (!benchmarkMode) {
        writer.addFrame(inputFrame);
    }

    useI0AsSource = !useI0AsSource;
}
