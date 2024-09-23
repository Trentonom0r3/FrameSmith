#pragma once
#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <stdexcept>

class BufferPool {
public:
    BufferPool(size_t bufferSize, int poolSize, torch::Device device, torch::ScalarType dtype)
        : bufferSize(bufferSize), poolSize(poolSize), device(device), dtype(dtype) {
        // Preallocate buffers
        for (int i = 0; i < poolSize; ++i) {
            at::Tensor buffer = at::empty({ static_cast<long long>(bufferSize)}, at::TensorOptions().dtype(dtype).device(device));
            availableBuffers.push(buffer);
        }
    }

    // Acquire a buffer from the pool
    at::Tensor acquireBuffer() {
        std::lock_guard<std::mutex> lock(poolMutex);
        if (availableBuffers.empty()) {
            throw std::runtime_error("BufferPool: No available buffers to acquire.");
        }
        at::Tensor buffer = availableBuffers.front();
        availableBuffers.pop();
        return buffer;
    }

    // Release a buffer back to the pool
    void releaseBuffer(at::Tensor buffer) {
        std::lock_guard<std::mutex> lock(poolMutex);
        // Optional: Validate buffer size and type
        if (buffer.numel() != bufferSize) {
            throw std::runtime_error("BufferPool: Released buffer has incorrect size.");
        }
        if (buffer.scalar_type() != dtype || buffer.device() != device) {
            throw std::runtime_error("BufferPool: Released buffer has incorrect type or device.");
        }
        availableBuffers.push(buffer);
    }

private:
    size_t bufferSize;
    int poolSize;
    torch::Device device;
    torch::ScalarType dtype;
    std::queue<at::Tensor> availableBuffers;
    std::mutex poolMutex;
};
