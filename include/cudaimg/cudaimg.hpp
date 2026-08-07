#pragma once

// cudaimg - CUDA 图像处理入门教学项目
// 通过图像处理算子学习 CUDA 编程

#include <cuda_runtime.h>
#include <string>

// Core components
#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/device_buffer.hpp"
#include "cudaimg/core/device_kernels.cuh"
#include "cudaimg/core/execution_context.hpp"
#include "cudaimg/core/image.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/core/kernel_helpers.hpp"

// Operators
#include "cudaimg/operators/color_space.hpp"
#include "cudaimg/operators/convolution_engine.hpp"
#include "cudaimg/operators/filters.hpp"
#include "cudaimg/operators/geometric.hpp"
#include "cudaimg/operators/histogram_calculator.hpp"
#include "cudaimg/operators/image_resizer.hpp"
#include "cudaimg/operators/morphology.hpp"
#include "cudaimg/operators/pixel_operator.hpp"
#include "cudaimg/operators/threshold.hpp"

// Processing
#include "cudaimg/processing/image_processor.hpp"
#include "cudaimg/processing/pipeline_processor.hpp"

// I/O
#include "cudaimg/io/image_io.hpp"

namespace cudaimg {

// 库版本信息
constexpr int VERSION_MAJOR = CUDAIMG_VERSION_MAJOR;
constexpr int VERSION_MINOR = CUDAIMG_VERSION_MINOR;
constexpr int VERSION_PATCH = CUDAIMG_VERSION_PATCH;

// 获取版本字符串
inline std::string getVersionString() {
  return std::to_string(VERSION_MAJOR) + "." + std::to_string(VERSION_MINOR) +
         "." + std::to_string(VERSION_PATCH);
}

// 检查 CUDA 是否可用
[[nodiscard]] inline bool isCudaAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return err == cudaSuccess && deviceCount > 0;
}

// 获取 CUDA 设备信息
[[nodiscard]] inline std::string getDeviceInfo() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    return "No CUDA devices available";
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::string info = "Device: ";
  info += prop.name;
  info += ", Compute Capability: ";
  info += std::to_string(prop.major) + "." + std::to_string(prop.minor);
  info += ", Memory: ";
  info += std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB";

  return info;
}

} // namespace cudaimg
