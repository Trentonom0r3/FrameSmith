#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include "Reader.h"
#include "Writer.h"  // Your FFmpegWriter class
#include <cuda_runtime.h>
#include <iomanip>        // For std::fixed and std::setprecision
#include <variant>
#include <memory>
#include "RifeTrt.hpp"
#include "UpscaleTrt.hpp"



inline void printASCII() {
    std::cout << R"(



         ________                                            ______                 __    __      __       
        |        \                                          /      \               |  \  |  \    |  \      
        | $$$$$$$$______   ______   ______ ____    ______  |  $$$$$$\ ______ ____   \$$ _| $$_   | $$____  
        | $$__   /      \ |      \ |      \    \  /      \ | $$___\$$|      \    \ |  \|   $$ \  | $$    \ 
        | $$  \ |  $$$$$$\ \$$$$$$\| $$$$$$\$$$$\|  $$$$$$\ \$$    \ | $$$$$$\$$$$\| $$ \$$$$$$  | $$$$$$$\
        | $$$$$ | $$   \$$/      $$| $$ | $$ | $$| $$    $$ _\$$$$$$\| $$ | $$ | $$| $$  | $$ __ | $$  | $$
        | $$    | $$     |  $$$$$$$| $$ | $$ | $$| $$$$$$$$|  \__| $$| $$ | $$ | $$| $$  | $$|  \| $$  | $$
        | $$    | $$      \$$    $$| $$ | $$ | $$ \$$     \ \$$    $$| $$ | $$ | $$| $$   \$$  $$| $$  | $$
         \$$     \$$       \$$$$$$$ \$$  \$$  \$$  \$$$$$$$  \$$$$$$  \$$  \$$  \$$ \$$    \$$$$  \$$   \$$
                                                      STUDIO                                                                                                                                                                           
                                 Interpolation and Upscaling for C++ with TensorRT.
                             Created by: @Trentonom0r3 - https://github.com/Trentonom0r3
                               Source: https://github.com/Trentonom0r3/RifeTensorRT                                                                                                                                                                                                             
        )" << std::endl;
}

// Function to display usage instructions
inline void printUsage(const std::string& programName) {
    std::cerr << "Usage: " << programName
        << " <input_video_path> <output_video_path> --mode <upscale|interpolate> <model_name> <factor> [--half] [--benchmark]"
        << std::endl;
    std::cerr << "Example: " << programName
        << " input.mp4 output.mp4 --mode upscale shufflecugan-tensorrt 2 --half --benchmark"
        << std::endl;
}
