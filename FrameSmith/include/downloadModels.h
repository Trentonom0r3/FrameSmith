#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <curl/curl.h>
#include <stdexcept> // For exceptions
#include <thread>

namespace fs = std::filesystem;

inline const std::string TASURL = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/";
inline const std::string DEPTHURL = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/";
inline const std::string SUDOURL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/";

inline const std::string DEPTHV2URLSMALL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/";
inline const std::string DEPTHV2URLBASE = "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/";
inline const std::string DEPTHV2URLLARGE = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/";


inline std::vector<std::string> modelsList() {
    return {
        // Add all the models listed in the Python code
        "shufflespan-tensorrt",
         "aniscale2-tensorrt",
        "open-proteus-tensorrt", "shufflecugan-tensorrt", "compact-tensorrt",
        "ultracompact-tensorrt", "superultracompact-tensorrt", "span-tensorrt",
        "rife4.6-tensorrt", "rife4.15-lite-tensorrt", "rife4.17-tensorrt",
        "rife4.18-tensorrt", "rife4.20-tensorrt", "rife4.21-tensorrt",
        "rife4.22-tensorrt", "rife4.22-lite-tensorrt"
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
    }
    else if (model == "shufflecugan" || model == "shufflecugan-directml" || model == "shufflecugan-tensorrt" || model == "shufflecugan-ncnn") {
        if (modelType == "pth") {
            return "sudo_shuffle_cugan_9.584.969.pth";
        }
        else if (modelType == "onnx") {
            return half ? "sudo_shuffle_cugan_fp16_op18_clamped_9.584.969.onnx" : "sudo_shuffle_cugan_op18_clamped_9.584.969.onnx";
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
    else if (model == "rife4.15" || model == "rife4.15-tensorrt" || model == "rife-v4.15-ncnn") {
		return (modelType == "pth") ? "rife415.pth" :
			(half ? (ensemble ? "rife415_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx" : "rife415_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx")
				: (ensemble ? "rife415_v2_ensembleTrue_op20_clamp_onnxslim.onnx" : "rife415_v2_ensembleFalse_op20_clamp_onnxslim.onnx"));
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
    else if (model == "rife4.22" || model == "rife4.22-tensorrt") {
		return (modelType == "pth") ? "rife422.pth" :
			(half ? (ensemble ? "rife422_v2_ensembleTrue_op20_fp16_clamp_onnxslim.onnx" : "rife422_v2_ensembleFalse_op20_fp16_clamp_onnxslim.onnx")
				: (ensemble ? "rife422_v2_ensembleTrue_op20_clamp_onnxslim.onnx" : "rife422_v2_ensembleFalse_op20_clamp_onnxslim.onnx"));
	}
    else {
        std::cerr << "Model not found: " << model << std::endl;
        return "";
    }
}// Define the URLs

// Helper function to get the download path and create directories if needed
inline std::string getWeightsDir() {
    std::string weightsDir;
#ifdef _WIN32
    char* appdata = getenv("APPDATA");
    if (appdata) {
        std::string mainPath = std::string(appdata) + "\\FrameSmith";
        weightsDir = mainPath + "\\weights";
        fs::create_directories(weightsDir);
    }
    else {
        throw std::runtime_error("APPDATA environment variable not found.");
    }
#else
    char* xdgConfigHome = getenv("XDG_CONFIG_HOME");
    if (xdgConfigHome) {
        weightsDir = std::string(xdgConfigHome) + "/FrameSmith/weights";
    }
    else {
        weightsDir = std::string(getenv("HOME")) + "/.config/FrameSmith/weights";
    }
    fs::create_directories(weightsDir);
#endif
    return weightsDir;
}

inline void printProgress(double percentage, double speed) {
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

inline size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    fflush(stream);  // Ensure all data is flushed to disk
    return written;
}

inline int progress_callback(void* ptr, curl_off_t total_to_download, curl_off_t now_downloaded, curl_off_t, curl_off_t) {
    if (total_to_download > 0) {
        double progress = static_cast<double>(now_downloaded) / total_to_download;
        double speed = *(static_cast<double*>(ptr));  // Speed in KB/s
        printProgress(progress, speed);
    }
    return 0;
}

inline std::string downloadAndLog(const std::string& model, const std::string& filename, const std::string& downloadUrl, const std::string& folderPath, int retries = 3) {
    std::string filePath = folderPath + "\\" + filename;

    for (int attempt = 0; attempt < retries; ++attempt) {
        try {
            if (std::filesystem::exists(filePath)) {
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
        model == "rife4.15-lite-tensorrt" || model == "rife4.16-lite-tensorrt" ||
        model == "rife4.15-tensorrt" || model == "rife-v4.18-ncnn" ||
        model == "span-tensorrt" || model == "span-directml" ||
        model == "shift_lpips-tensorrt" || model == "shift_lpips-directml" ||
        model == "shufflecugan-tensorrt") {

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
    if (!downloadAndLog(model, filename, fullUrl, folderPath).empty()) {
        std::cout << "Downloaded from TASURL: " << fullUrl << std::endl;
        return fullUrl;
    }

    throw std::runtime_error("Failed to download model.");
}
