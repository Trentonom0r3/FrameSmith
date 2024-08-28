#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

inline std::string green(const std::string& text) {
    return "\033[32m" + text + "\033[0m";
}

inline std::string red(const std::string& text) {
    return "\033[31m" + text + "\033[0m";
}

inline std::string yellow(const std::string& text) {
    return "\033[33m" + text + "\033[0m";
}

inline std::string blue(const std::string& text) {
    return "\033[34m" + text + "\033[0m";
}

inline std::string magenta(const std::string& text) {
    return "\033[35m" + text + "\033[0m";
}

inline std::string cyan(const std::string& text) {
    return "\033[36m" + text + "\033[0m";
}

inline std::string rainbow(const std::string& text) {
    std::vector<std::string> colors = {"\033[31m", "\033[33m", "\033[32m", "\033[34m", "\033[35m", "\033[36m"};
    std::string coloredText;
    for (size_t i = 0; i < text.size(); ++i) {
        coloredText += colors[i % colors.size()] + text[i] + "\033[0m";
    }
    return coloredText;
}

inline std::string gradient(const std::string& text) {
    std::vector<std::string> colors = {"\033[97m", "\033[91m", "\033[31m"};  // white, light red, red
    std::string coloredText;
    srand(time(nullptr));  // Seed the random number generator
    for (size_t i = 0; i < text.size(); ++i) {
        int baseIndex = static_cast<int>((static_cast<float>(i) / text.size()) * (colors.size() - 1));
        int randomOffset = rand() % 3 - 1;  // Random number between -1 and 1
        int colorIndex = std::clamp(baseIndex + randomOffset, 0, static_cast<int>(colors.size() - 1));
        coloredText += colors[colorIndex] + text[i] + "\033[0m";
    }
    return coloredText;
}
