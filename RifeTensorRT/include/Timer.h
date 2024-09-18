// Timer.h
#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer {
public:
    Timer(const std::string& msg)
        : message(msg), start(std::chrono::high_resolution_clock::now()) {
        std::cout << message << " - Start" << std::endl;
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << message << " - Duration: " << duration << " ms" << std::endl;
    }

private:
    std::string message;
    std::chrono::high_resolution_clock::time_point start;
};
