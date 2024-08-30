#include "RifeTensorRT.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_video_path> <output_video_path> <model_name>" << std::endl;
        return -1;
    }

    // Initialize network protocols.
    avformat_network_init();

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelName = argv[3];

    // Open input video file.
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, inputVideoPath.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file: " << inputVideoPath << std::endl;
        return -1;
    }

    // Retrieve stream information.
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        return -1;
    }

    const AVCodec* decoder = nullptr;
    AVCodecParameters* codec_params = nullptr;
    int video_stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (video_stream_index < 0) {
        std::cerr << "Could not find video stream" << std::endl;
        return -1;
    }

    codec_params = fmt_ctx->streams[video_stream_index]->codecpar;
    decoder = avcodec_find_decoder(codec_params->codec_id);
    if (!decoder) {
        std::cerr << "Unsupported codec" << std::endl;
        return -1;
    }

    // Initialize the decoder context.
    AVCodecContext* dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, codec_params);
    avcodec_open2(dec_ctx, decoder, nullptr);

    // Initialize the encoder context.
    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    AVCodecContext* enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->bit_rate = 400000;
    enc_ctx->width = dec_ctx->width;
    enc_ctx->height = dec_ctx->height;

    // Set time_base and framerate for 48 fps.
    enc_ctx->time_base.num = 1;
    enc_ctx->time_base.den = 48;
    enc_ctx->framerate.num = 48;
    enc_ctx->framerate.den = 1;

    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    avcodec_open2(enc_ctx, encoder, nullptr);

    // Create output format context.
    AVFormatContext* out_fmt_ctx = nullptr;
    avformat_alloc_output_context2(&out_fmt_ctx, nullptr, nullptr, outputVideoPath.c_str());
    if (!out_fmt_ctx) {
        std::cerr << "Could not create output context" << std::endl;
        return -1;
    }

    // Create and initialize a new stream for the output file.
    AVStream* out_stream = avformat_new_stream(out_fmt_ctx, nullptr);
    if (!out_stream) {
        std::cerr << "Failed to allocate stream" << std::endl;
        return -1;
    }

    avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    out_stream->time_base = enc_ctx->time_base;

    if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&out_fmt_ctx->pb, outputVideoPath.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Could not open output file" << std::endl;
            return -1;
        }
    }

    // Write the header for the output file.
    if (avformat_write_header(out_fmt_ctx, nullptr) < 0) {
        std::cerr << "Error occurred when opening output file" << std::endl;
        return -1;
    }

    // Initialize RIFE with the specified model name.
    RifeTensorRT rife(modelName, 2, enc_ctx->width, enc_ctx->height, true, false);

    // Allocate memory for frames and buffers.
    AVFrame* frame_rgb = av_frame_alloc();
    frame_rgb->width = dec_ctx->width;
    frame_rgb->height = dec_ctx->height;
    frame_rgb->format = AV_PIX_FMT_RGB24;

    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame_rgb->width, frame_rgb->height, 32);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, buffer, AV_PIX_FMT_RGB24, frame_rgb->width, frame_rgb->height, 32);

    // Initialize SwsContext for color space conversion.
    SwsContext* sws_ctx = sws_getContext(
        dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        dec_ctx->width, dec_ctx->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* outputFrame = av_frame_alloc();
    outputFrame->width = enc_ctx->width;
    outputFrame->height = enc_ctx->height;
    outputFrame->format = enc_ctx->pix_fmt;
    av_frame_get_buffer(outputFrame, 32);

    // Initialize target_time_base for PTS calculations.
    AVRational target_time_base = { 1, 48 };

    // Calculate pts_step based on input stream's time base and interpolation factor.
    int64_t pts_step = av_rescale_q(1, target_time_base, enc_ctx->time_base);
    std::cout << "Calculated pts_step: " << pts_step << std::endl;

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    int64_t pts = frameCount * pts_step;

    // Read frames from the input video.
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(dec_ctx, pkt) >= 0) {
                while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
                    if (frame->width == 0 || frame->height == 0) {
                        std::cerr << "Error: Decoded frame has zero dimensions." << std::endl;
                        continue;
                    }
                    std::cout << "Frame PTS: " << frame->pts << std::endl;
                    std::cout << "Frame DTS: " << frame->pkt_dts << std::endl;
                    // Perform color conversion from YUV to RGB.
                    int result = sws_scale(sws_ctx, (uint8_t const* const*)frame->data, frame->linesize, 0, frame->height, frame_rgb->data, frame_rgb->linesize);
                    if (result <= 0) {
                        std::cerr << "Error: sws_scale failed with result " << result << std::endl;
                        continue;
                    }

                    // Convert the RGB frame to a tensor.
                    at::Tensor tensorFrame = torch::from_blob(frame_rgb->data[0], { frame_rgb->height, frame_rgb->width, 3 }, torch::kByte).to(torch::kCUDA);

                    // Log the PTS before running the RIFE interpolation.
                    std::cout << "Main PTS before run: " << pts << std::endl;
                    std::cout << "Frame PTS: within main " << frame->pts << std::endl;
                    // Run the RIFE interpolation.
                    rife.run(tensorFrame, false, enc_ctx, outputFrame, out_fmt_ctx, out_stream, pts, pts_step);

                    // Log the PTS after running the RIFE interpolation.
                    std::cout << "Main PTS after run: " << pts << std::endl;

                    // Update pts for the next original frame.
                    pts += pts_step * 2;
                    std::cout << "Updated main PTS for the next frame: " << pts << std::endl;

                    frameCount++;
                }
            }
        }
        av_packet_unref(pkt);
    }

    // Flush the encoder to ensure all frames are written.
    std::cout << "Flushing encoder..." << std::endl;
    avcodec_send_frame(enc_ctx, nullptr);
    while (avcodec_receive_packet(enc_ctx, pkt) == 0) {
        pkt->stream_index = out_stream->index;
        av_packet_rescale_ts(pkt, enc_ctx->time_base, out_stream->time_base);
        std::cout << "Flushing packet with PTS: " << pkt->pts << ", DTS: " << pkt->dts << std::endl;
        if (av_write_frame(out_fmt_ctx, pkt) < 0) {
            std::cerr << "Error writing packet during flushing" << std::endl;
        }
        av_packet_unref(pkt);
    }

    // Write the trailer for the output file.
    if (av_write_trailer(out_fmt_ctx) < 0) {
        std::cerr << "Error writing output trailer" << std::endl;
    }

    // Calculate the total processing time and FPS.
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    double processingFPS = frameCount / duration.count();

    std::cout << "Processing completed." << std::endl;
    std::cout << "Processed " << frameCount << " frames in " << duration.count() << " seconds." << std::endl;
    std::cout << "Processing FPS: " << processingFPS << " frames per second." << std::endl;

    // Cleanup allocated resources.
    sws_freeContext(sws_ctx);
    av_free(buffer);
    av_frame_free(&frame_rgb);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    av_frame_free(&outputFrame);
    av_packet_free(&pkt);

    // Close the output file.
    avio_close(out_fmt_ctx->pb);
    avformat_free_context(out_fmt_ctx);

    return 0;
}

