#include "Writer.h"

// Implementation

// Constructor
FFmpegWriter::FFmpegWriter(const std::string& outputFilePath, int width, int height, int fps, bool benchmark)
	: width(width), height(height), fps(fps), isBenchmark(benchmark), head(TaggedPointer(nullptr, 0)) {

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
	codecCtx->pix_fmt = AV_PIX_FMT_CUDA;  // Use NV12 for hardware encoding
	codecCtx->hw_device_ctx = av_buffer_ref(hw_device_ctx);  // Set the CUDA device context
	codecCtx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);  // Set the CUDA frames context

	// Set encoder options
	av_opt_set(codecCtx->priv_data, "crf", "23", 0);      // Adjust CRF value
	av_opt_set(codecCtx->priv_data, "preset", "p5", 0);   // Use p1-p7 preset for NVENC

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

	// Allocate packet
	packet = av_packet_alloc();
	if (!packet) {
		std::cerr << "Could not allocate packet" << std::endl;
		throw std::runtime_error("Failed to allocate packet");
	}

	// Create CUDA streams
	CUDA_CHECK(cudaStreamCreate(&writestream));

	// Initialize Frame Pool with lock-free stack
	for (int i = 0; i < 20; ++i) { // Pre-allocate 20 frames
		AVFrame* poolFrame = av_frame_alloc();
		if (!poolFrame) {
			std::cerr << "Failed to allocate frame for pool." << std::endl;
			continue;
		}
		poolFrame->format = codecCtx->pix_fmt;
		poolFrame->width = codecCtx->width;
		poolFrame->height = codecCtx->height;
		poolFrame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx); // Set the CUDA frames context
		poolFrame->format = AV_PIX_FMT_CUDA;
		if (av_hwframe_get_buffer(hw_frames_ctx, poolFrame, 0) < 0) {
			std::cerr << "Failed to allocate hardware frame buffer for pool." << std::endl;
			av_frame_free(&poolFrame);
			continue;
		}
		// Wrap the frame in a FrameNode and push to the stack
		FrameNode* node = new FrameNode(poolFrame);
		pushFrame(node);
	}

	// Allocate NV12 buffer
	size_t nv12_size = width * height + 2 * (width / 2) * (height / 2);
	CUDA_CHECK(cudaMalloc(&nv12_buffer, nv12_size));

	// Allocate Y, U, V buffers
	size_t y_size = width * height;               // Y plane size
	size_t uv_size = (width * height) / 2;        // UV plane size for NV12

	// Allocate Y buffer
	CUDA_CHECK(cudaMalloc(&y_buffer, y_size));

	// Allocate UV buffer
	CUDA_CHECK(cudaMalloc(&uv_buffer, uv_size));
}

// Destructor
FFmpegWriter::~FFmpegWriter() {
	finalize();  // Ensure finalization is done before destruction

	 // Clean up all frames in the pool
	while (true) {
		AVFrame* frame = popFrame();
		if (!frame) break;
		av_frame_free(&frame);
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

	// Free NV12 buffer
	if (nv12_buffer) {
		CUDA_CHECK(cudaFree(nv12_buffer));
	}

	if (y_buffer) {
		cudaFree(y_buffer);
		y_buffer = nullptr;
	}
	if (uv_buffer) {
		cudaFree(uv_buffer);
		uv_buffer = nullptr;
	}

	// Free format context
	if (formatCtx && !(formatCtx->oformat->flags & AVFMT_NOFILE)) {
		avio_closep(&formatCtx->pb);
	}
	if (formatCtx) {
		avformat_free_context(formatCtx);
	}
	// Clean up CUDA streams
	CUDA_CHECK(cudaStreamDestroy(writestream));
	//  std::cout << "FFmpegWriter destroyed successfully." << std::endl;
}

// Push a frame onto the stack
void FFmpegWriter::pushFrame(FrameNode* node) {
	TaggedPointer oldHead = head.load(std::memory_order_relaxed);
	while (true) {
		node->next = oldHead.ptr;
		TaggedPointer newHead(node, oldHead.tag + 1);
		if (head.compare_exchange_weak(oldHead, newHead, std::memory_order_release, std::memory_order_relaxed)) {
			break;
		}
		// If CAS failed, oldHead is updated with the current head
	}
}

// Pop a frame from the stack
AVFrame* FFmpegWriter::popFrame() {
	TaggedPointer oldHead = head.load(std::memory_order_relaxed);
	while (oldHead.ptr != nullptr) {
		FrameNode* node = oldHead.ptr;
		TaggedPointer newHead(node->next, oldHead.tag + 1);
		if (head.compare_exchange_weak(oldHead, newHead, std::memory_order_acquire, std::memory_order_relaxed)) {
			AVFrame* frame = node->frame;
			delete node;
			return frame;
		}
		// If CAS failed, oldHead is updated with the current head
	}
	return nullptr; // Stack is empty
}

AVFrame* FFmpegWriter::acquireFrame() {
	AVFrame* frame = popFrame();
	if (frame) {
		return frame;
	}
	// If pool is empty, allocate a new frame
	frame = av_frame_alloc();
	if (!frame) {
		std::cerr << "Failed to allocate frame." << std::endl;
		return nullptr;
	}
	frame->format = codecCtx->pix_fmt;
	frame->width = codecCtx->width;
	frame->height = codecCtx->height;
	if (av_hwframe_get_buffer(hw_frames_ctx, frame, 0) < 0) {
		std::cerr << "Failed to allocate hardware frame buffer." << std::endl;
		av_frame_free(&frame);
		return nullptr;
	}
	return frame;
}

void FFmpegWriter::releaseFrame(AVFrame* frame) {
	// Wrap the frame in a FrameNode and push to the stack
	FrameNode* node = new FrameNode(frame);
	pushFrame(node);
}


template <typename T>
void FFmpegWriter::addFrameTemplate(const T* rgb_ptr, bool benchmark) {
	AVFrame* frameToEncode = acquireFrame();
	if (!frameToEncode) {
		std::cerr << "Failed to acquire frame for encoding." << std::endl;
		return;
	}

	if constexpr (std::is_same<T, float>::value) {
		// FP32 path
		launch_rgb_to_nv12_fp32(rgb_ptr,
			frameToEncode->data[0],  // Destination Y plane.
			frameToEncode->data[1],  // Destination UV plane.
			width,
			height,
			frameToEncode->linesize[0],  // Y plane stride.
			frameToEncode->linesize[1],  // UV plane stride.
			writestream);
	}
	else if constexpr (std::is_same<T, __half>::value) {
		// FP16 path
		launch_rgb_to_nv12_fp16(rgb_ptr,
			frameToEncode->data[0],  // Destination Y plane.
			frameToEncode->data[1],  // Destination UV plane.
			width,
			height,
			frameToEncode->linesize[0],  // Y plane stride.
			frameToEncode->linesize[1],  // UV plane stride.
			writestream);
	}
	else {
		static_assert(always_false<T>::value, "Unsupported data type for addFrameTemplate");
	}

	if (benchmark) {
		releaseFrame(frameToEncode);
		return;  // Skip encoding in benchmark mode
	}

	frameToEncode->pts = pts++;

	// Enqueue the frame for encoding
	writeFrame(frameToEncode);
	releaseFrame(frameToEncode);
}

// Explicit template instantiation
template void FFmpegWriter::addFrameTemplate<float>(const float*, bool);
template void FFmpegWriter::addFrameTemplate<__half>(const __half*, bool);

void FFmpegWriter::writeFrame(AVFrame* inputFrame) {
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
	if (av_frame_make_writable(inputFrame) < 0) {
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

void FFmpegWriter::finalize() {
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
	// std::cout << "Finalization complete." << std::endl;
}
