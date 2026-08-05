#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpu_image {

// CUDA 错误异常类
class CudaException : public std::runtime_error {
public:
  CudaException(cudaError_t error, const char* file, int line);
  cudaError_t errorCode() const { return error_; }

private:
  cudaError_t error_;
};

// 错误检查宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw gpu_image::CudaException(error, __FILE__, __LINE__);               \
    }                                                                          \
  } while (0)

} // namespace gpu_image
