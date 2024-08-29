#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <curl/curl.h>
#include <stdexcept> // For exceptions

namespace fs = std::filesystem;


inline std::vector<std::string> modelsList() {
    return {
        // Add all the models listed in the Python code
        "shufflespan", "shufflespan-directml", "shufflespan-tensorrt",
        "aniscale2", "aniscale2-directml", "aniscale2-tensorrt",
        "open-proteus", "compact", "ultracompact", "superultracompact", "span",
        "shufflecugan", "segment", "segment-tensorrt", "segment-directml",
        "scunet", "dpir", "real-plksr", "nafnet", "rife", "rife4.6",
        "rife4.15-lite", "rife4.16-lite", "rife4.17", "rife4.18",
        "rife4.20", "rife4.21", "rife4.22", "rife4.22-lite",
        "shufflecugan-directml", "open-proteus-directml", "compact-directml",
        "ultracompact-directml", "superultracompact-directml", "span-directml",
        "open-proteus-tensorrt", "shufflecugan-tensorrt", "compact-tensorrt",
        "ultracompact-tensorrt", "superultracompact-tensorrt", "span-tensorrt",
        "rife4.6-tensorrt", "rife4.15-lite-tensorrt", "rife4.17-tensorrt",
        "rife4.18-tensorrt", "rife4.20-tensorrt", "rife4.21-tensorrt",
        "rife4.22-tensorrt", "rife4.22-lite-tensorrt", "rife-v4.6-ncnn",
        "rife-v4.15-lite-ncnn", "rife-v4.16-lite-ncnn", "rife-v4.17-ncnn",
        "rife-v4.18-ncnn", "span-ncnn", "shufflecugan-ncnn", "small_v2",
        "base_v2", "large_v2", "small_v2-directml", "base_v2-directml",
        "large_v2-directml", "small_v2-tensorrt", "base_v2-tensorrt",
        "large_v2-tensorrt", "maxxvit-tensorrt", "maxxvit-directml",
        "shift_lpips-tensorrt", "shift_lpips-directml", "differential-tensorrt"
    };
}

inline std::string modelsMap(const std::string& model, const std::string& modelType = "pth", bool half = true, bool ensemble = false) {
    if (model == "shufflespan" || model == "shufflespan-directml" || model == "shufflespan-tensorrt") {
        return (modelType == "pth") ? "sudo_shuffle_span_10.5m.pth" :
            (half ? "sudo_shuffle_span_op20_10.5m_1080p_fp16_op21_slim.onnx" : "sudo_shuffle_span_op20_10.5m_1080p_fp32_op21_slim.onnx");
    }
    else if (model == "aniscale2" || model == "aniscale2-directml" || model == "aniscale2-tensorrt") {
        return (modelType == "pth") ? "2x_AniScale2S_Compact_i8_60K.pth" :
            (half ? "2x_AniScale2S_Compact_i8_60K-fp16.onnx" : "2x_AniScale2S_Compact_i8_60K-fp32.onnx");
    }
    else if (model == "open-proteus" || model == "open-proteus-directml" || model == "open-proteus-tensorrt") {
        return (modelType == "pth") ? "2x_OpenProteus_Compact_i2_70K.pth" :
            (half ? "2x_OpenProteus_Compact_i2_70K-fp16.onnx" : "2x_OpenProteus_Compact_i2_70K-fp32.onnx");
    }
    else if (model == "compact" || model == "compact-directml" || model == "compact-tensorrt") {
        return (modelType == "pth") ? "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth" :
            (half ? "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_fp16_op18_onnxslim.onnx" : "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k_clamp_op18_onnxslim.onnx");
    }
    else if (model == "ultracompact" || model == "ultracompact-directml" || model == "ultracompact-tensorrt") {
        return (modelType == "pth") ? "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k.pth" :
            (half ? "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_fp16_op18_onnxslim.onnx" : "2x_AnimeJaNai_HD_V3_Sharp1_UltraCompact_425k_clamp_op18_onnxslim.onnx");
    }
    else if (model == "superultracompact" || model == "superultracompact-directml" || model == "superultracompact-tensorrt") {
        return (modelType == "pth") ? "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k.pth" :
            (half ? "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_fp16_op18_onnxslim.1.onnx" : "2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_op18_onnxslim.onnx");
    }
    else if (model == "span" || model == "span-directml" || model == "span-tensorrt" || model == "span-ncnn") {
        if (modelType == "pth") {
            return "2x_ModernSpanimationV1.5.pth";
        }
        else if (modelType == "onnx") {
            return half ? "2x_ModernSpanimationV1.5_clamp_fp16_op20_onnxslim.onnx" : "2x_ModernSpanimationV1.5_clamp_op20_onnxslim.onnx";
        }
        else if (modelType == "ncnn") {
            return "2x_modernspanimationv1.5-ncnn.zip";
        }
    }
    else if (model == "shufflecugan" || model == "shufflecugan-directml" || model == "shufflecugan-tensorrt" || model == "shufflecugan-ncnn") {
        if (modelType == "pth") {
            return "sudo_shuffle_cugan_9.584.969.pth";
        }
        else if (modelType == "onnx") {
            return half ? "sudo_shuffle_cugan_fp16_op18_clamped.onnx" : "sudo_shuffle_cugan_op18_clamped.onnx";
        }
        else if (modelType == "ncnn") {
            return "2xsudo_shuffle_cugan-ncnn.zip";
        }
    }
    else if (model == "rife4.22-lite" || model == "rife4.22-lite-tensorrt") {
        return (modelType == "pth") ? "rife422_lite.pth" :
            (half ? "rife4.22_lite_fp16_op21_slim.onnx" : "rife4.22_lite_fp32_op21_slim.onnx");
    }
    else if (model == "rife4.20" || model == "rife4.20-tensorrt") {
        return (modelType == "pth") ? "rife420.pth" :
            (half ? (ensemble ? "rife420_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx" : "rife420_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx")
                : (ensemble ? "rife420_v2_ensembleTrue_op20_clamp_onnxslim.onnx" : "rife420_v2_ensembleFalse_op20_clamp_onnxslim.onnx"));
    }
    else if (model == "rife" || model == "rife4.22" || model == "rife4.22-tensorrt") {
        return (modelType == "pth") ? "rife422.pth" :
            (half ? "rife422_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx" : "rife422_v2_ensembleFalse_op20_clamp_onnxslim.onnx");
    }
    else if (model == "rife4.21" || model == "rife4.21-tensorrt") {
        return (modelType == "pth") ? "rife421.pth" :
            (half ? "rife421_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx" : "rife421_v2_ensembleFalse_op20_clamp_onnxslim.onnx");
    }
    else if (model == "rife4.18" || model == "rife4.18-tensorrt" || model == "rife-v4.18-ncnn") {
        return (modelType == "pth") ? "rife418.pth" :
            (half ? (ensemble ? "rife418_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx" : "rife418_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx")
                : (ensemble ? "rife418_v2_ensembleTrue_op20_clamp_onnxslim.onnx" : "rife418_v2_ensembleFalse_op20_clamp_onnxslim.onnx"));
    }
    else if (model == "rife4.17" || model == "rife4.17-tensorrt" || model == "rife-v4.17-ncnn") {
        return (modelType == "pth") ? "rife417.pth" :
            (half ? (ensemble ? "rife417_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx" : "rife417_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx")
                : (ensemble ? "rife417_v2_ensembleTrue_op20_clamp_onnxslim.onnx" : "rife417_v2_ensembleFalse_op20_clamp_onnxslim.onnx"));
    }
    else if (model == "rife4.15-lite" || model == "rife4.15-lite-tensorrt" || model == "rife-v4.15-lite-ncnn") {
        return (modelType == "pth") ? "rife415_lite.pth" :
            (half ? (ensemble ? "rife_v4.15_lite_ensemble_fp16_op20_sim.onnx" : "rife_v4.15_lite_fp16_op20_sim.onnx")
                : (ensemble ? "rife_v4.15_lite_ensemble_fp32_op20_sim.onnx" : "rife_v4.15_lite_fp32_op20_sim.onnx"));
    }
    else if (model == "rife4.6" || model == "rife4.6-tensorrt" || model == "rife-v4.6-ncnn") {
        return (modelType == "pth") ? "rife46.pth" :
            (half ? (ensemble ? "rife46_v2_ensembleTrue_op16_fp16_mlrt_sim.onnx" : "rife46_v2_ensembleFalse_op16_fp16_mlrt_sim.onnx")
                : (ensemble ? "rife46_v2_ensembleTrue_op16_mlrt_sim.onnx" : "rife46_v2_ensembleFalse_op16_mlrt_sim.onnx"));
    }
    else if (model == "rife4.16-lite" || model == "rife-v4.16-lite-ncnn") {
        return (modelType == "pth") ? "rife416_lite.pth" :
            (ensemble ? "rife-v4.16-lite-ensemble-ncnn.zip" : "rife-v4.16-lite-ncnn.zip");
    }
    else if (model == "segment-tensorrt" || model == "segment-directml") {
        return "isnet_is.onnx";
    }
    else if (model == "maxxvit-tensorrt" || model == "maxxvit-directml") {
        return half ? "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx" :
            "maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_op17_onnxslim.onnx";
    }
    else if (model == "shift_lpips-tensorrt" || model == "shift_lpips-directml") {
        return half ? "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx" :
            "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_onnxslim.onnx";
    }
    else if (model == "differential-tensorrt") {
        return "scene_change_nilas.onnx";
    }
    else if (model == "small_v2") {
        return "depth_anything_v2_vits.pth";
    }
    else if (model == "base_v2") {
        return "depth_anything_v2_vitb.pth";
    }
    else if (model == "large_v2") {
        return "depth_anything_v2_vitl.pth";
    }
    else if (model == "small_v2-directml" || model == "small_v2-tensorrt") {
        return half ? "depth_anything_v2_vits14_float16_slim.onnx" : "depth_anything_v2_vits14_float32_slim.onnx";
    }
    else if (model == "base_v2-directml" || model == "base_v2-tensorrt") {
        return half ? "depth_anything_v2_vitb14_float16_slim.onnx" : "depth_anything_v2_vitb14_float32_slim.onnx";
    }
    else if (model == "large_v2-directml" || model == "large_v2-tensorrt") {
        return half ? "depth_anything_v2_vitl14_float16_slim.onnx" : "depth_anything_v2_vitl14_float32_slim.onnx";
    }
    else {
        std::cerr << "Model not found: " << model << std::endl;
        return "";
    }
}// Define the URLs
inline const std::string TASURL = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/";
inline const std::string DEPTHURL = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/";
inline const std::string SUDOURL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/";

inline const std::string DEPTHV2URLSMALL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/";
inline const std::string DEPTHV2URLBASE = "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/";
inline const std::string DEPTHV2URLLARGE = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/";

// Helper function to get the download path and create directories if needed
inline std::string getWeightsDir() {
    std::string weightsDir;
#ifdef _WIN32
    char* appdata = getenv("APPDATA");
    if (appdata) {
        std::string mainPath = std::string(appdata) + "\\RifeCpp";
        weightsDir = mainPath + "\\weights";
        fs::create_directories(weightsDir);
        std::cout << "Weights directory: " << weightsDir << std::endl;
    }
    else {
        throw std::runtime_error("APPDATA environment variable not found.");
    }
#else
    char* xdgConfigHome = getenv("XDG_CONFIG_HOME");
    if (xdgConfigHome) {
        weightsDir = std::string(xdgConfigHome) + "/RifeCpp/weights";
    }
    else {
        weightsDir = std::string(getenv("HOME")) + "/.config/RifeCpp/weights";
    }
    fs::create_directories(weightsDir);
#endif
    return weightsDir;
}

void printProgress(double percentage, double speed) {
    int barWidth = 50;  // Width of the progress bar
    std::cout << "[";
    int pos = static_cast<int>(barWidth * percentage);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << "% (" << speed << " KB/s)\r";
    std::cout.flush();
}

size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    fflush(stream);  // Ensure all data is flushed to disk
    return written;
}

int progress_callback(void* ptr, curl_off_t total_to_download, curl_off_t now_downloaded, curl_off_t, curl_off_t) {
    if (total_to_download > 0) {
        double progress = static_cast<double>(now_downloaded) / total_to_download;
        double speed = *(static_cast<double*>(ptr));  // Speed in KB/s
        printProgress(progress, speed);
    }
    return 0;
}

std::string downloadAndLog(const std::string& model, const std::string& filename, const std::string& downloadUrl, const std::string& folderPath, int retries = 3) {
    std::string filePath = folderPath + "\\" + filename;

    for (int attempt = 0; attempt < retries; ++attempt) {
        try {
            if (std::filesystem::exists(filePath)) {
                std::cout << model << " model already exists at: " << filePath << std::endl;
                return filePath;
            }

            CURL* curl = curl_easy_init();
            if (curl) {
                FILE* fp = fopen(filePath.c_str(), "wb");
                if (!fp) {
                    throw std::runtime_error("Failed to open file for writing: " + filePath);
                }

                double speed = 0.0;

                curl_easy_setopt(curl, CURLOPT_URL, downloadUrl.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);  // Enable progress meter
                curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_callback);
                curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &speed);
                curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
                curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &speed);
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
                curl_easy_setopt(curl, CURLOPT_MAX_RECV_SPEED_LARGE, 1048576L); // 1MB/s limit

                CURLcode res = curl_easy_perform(curl);
                if (res != CURLE_OK) {
                    std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
                    fclose(fp);
                    curl_easy_cleanup(curl);
                    std::filesystem::remove(filePath);  // Clean up partially downloaded files
                    if (attempt == retries - 1) {
                        throw std::runtime_error("Failed to download after multiple attempts");
                    }
                }
                else {
                    fclose(fp);
                    curl_easy_cleanup(curl);
                    size_t downloaded_size = std::filesystem::file_size(filePath);
                    std::cout << "\nDownloaded: " << filePath << " (Size: " << downloaded_size << " bytes)" << std::endl;
                    return filePath;
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));  // Sleep before retrying
        }
    }

    throw std::runtime_error("Failed to download model after multiple attempts.");
}


// Function to download models
inline std::string downloadModels(const std::string& model, const std::string& modelType = "pth", bool half = true, bool ensemble = false) {
    std::string weightsDir = getWeightsDir();

    std::string filename = modelsMap(model, modelType, half, ensemble);

    std::string folderName = model;
    if (model.find("-tensorrt") != std::string::npos || model.find("-directml") != std::string::npos) {
        folderName = model.substr(0, model.find('-')) + "-onnx";
    }

    std::string folderPath = weightsDir + "\\" + folderName;
    fs::create_directories(folderPath);
    std::cout << "Downloading model to: " << folderPath << std::endl;

    std::string fullUrl;
    if (model == "rife4.22-tensorrt" || model == "rife4.21-tensorrt" ||
        model == "rife4.20-tensorrt" || model == "rife4.18-tensorrt" ||
        model == "rife4.17-tensorrt" || model == "rife4.6-tensorrt" ||
        model == "span-tensorrt" || model == "span-directml" ||
        model == "shift_lpips-tensorrt" || model == "shift_lpips-directml") {

        fullUrl = SUDOURL + filename;
        bool isSudoUrl = true;
        if (!downloadAndLog(model, filename, fullUrl, folderPath).empty()) {
            //output full url, modle name, filename for debugging
            std::cout << "Downloaded from SUDOURL: " << fullUrl << std::endl;
            std::cout << "Model: " << model << std::endl;
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Folderpath: " << folderPath << std::endl;
            std::cout << "WeightsDir: " << weightsDir << std::endl;
            std::cout << "FolderName: " << folderName << std::endl;
            std::cout << "FullUrl: " << fullUrl << std::endl;
            return folderPath + "\\" + filename;
        }
        else {
            std::cerr << "Failed to download from SUDOURL, trying TASURL..." << std::endl;
            fullUrl = TASURL + filename;
            if (!downloadAndLog(model, filename, fullUrl, folderPath).empty()) {
                std::cout << "Downloaded from TASURL: " << fullUrl << std::endl;
                std::cout << "Model: " << model << std::endl;
                std::cout << "Filename: " << filename << std::endl;
                std::cout << "Folderpath: " << folderPath << std::endl;
                std::cout << "WeightsDir: " << weightsDir << std::endl;
                std::cout << "FolderName: " << folderName << std::endl;
                std::cout << "FullUrl: " << fullUrl << std::endl;
                return folderPath + "\\" + filename;
            }
            throw std::runtime_error("Failed to download model from both SUDOURL and TASURL.");
        }
    }
    else if (model == "small_v2") {
        fullUrl = DEPTHV2URLSMALL + filename;
    }
    else if (model == "base_v2") {
        fullUrl = DEPTHV2URLBASE + filename;
    }
    else if (model == "large_v2") {
        fullUrl = DEPTHV2URLLARGE + filename;
    }
    else {
        fullUrl = TASURL + filename;
        std::cout << "Downloading from TASURL: " << fullUrl << std::endl;
    }

    if (!downloadAndLog(model, filename, fullUrl, folderPath).empty()) {
        std::cout << "Downloaded from TASURL: " << fullUrl << std::endl;
        return fullUrl;
    }

    throw std::runtime_error("Failed to download model.");
}