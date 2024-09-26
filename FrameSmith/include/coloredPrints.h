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

inline std::string orange(const std::string& text) {
	return "\033[38;5;208m" + text + "\033[0m";
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
    // Blacksmith colors: Charcoal, Dark Grey, Steel Grey, Blue Grey, Gunmetal
    std::vector<std::string> colors = {
        "\033[38;5;238m",  // Charcoal
        "\033[38;5;239m",  // Slightly lighter Charcoal
        "\033[38;5;240m",  // Dark Grey
        "\033[38;5;241m",  // Lighter Dark Grey
        "\033[38;5;244m",  // Steel Grey
        "\033[38;5;245m",  // Lighter Steel Grey
        "\033[38;5;62m",   // Blue Grey
        "\033[38;5;59m"    // Gunmetal
    };

    std::string coloredText;
    size_t numColors = colors.size();
    size_t textLength = text.size();

    for (size_t i = 0; i < textLength; ++i) {
        // Calculate the color index based on the position in the text
        double ratio = static_cast<double>(i) / (textLength - 1);
        size_t colorIndex = static_cast<size_t>(ratio * (numColors - 1));
        colorIndex = (std::min)(colorIndex, numColors - 1); // Ensure index is within bounds
        coloredText += colors[colorIndex] + std::string(1, text[i]) + "\033[0m";
    }

    return coloredText;
}


