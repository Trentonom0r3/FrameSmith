// normalize.h
#pragma once

#include <cstdint>

void launch_normalize_kernel(const uint8_t* rgb, float* normalized, int total);
