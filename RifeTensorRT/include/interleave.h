// normalize.h
#pragma once

#include <cstdint>


//void interleave_u_v(const uint8_t* u_buffer, const uint8_t* v_buffer, uint8_t* uv_buffer, int width, int height, cudaStream_t stream) {
void interleave_u_v(const uint8_t* u_buffer, const uint8_t* v_buffer, uint8_t* uv_buffer, int width, int height);
