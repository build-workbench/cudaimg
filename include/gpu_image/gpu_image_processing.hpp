#pragma once

// GPU Image Processing Library
// 基于 CUDA 的高性能图像处理库

#include <cuda_runtime.h>
#include <string>

// Core components
#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/device_buffer.hpp"
#include "gpu_image/core/execution_context.hpp"
#include "gpu_image/core/gpu_image.hpp"
#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/core/kernel_helpers.hpp"
#include "gpu_image/core/memory_manager.hpp"

// Operators
#include "gpu_image/operators/color_space.hpp"
#include "gpu_image/operators/concrete_operators.hpp"
#include "gpu_image/operators/convolution_engine.hpp"
#include "gpu_image/operators/filters.hpp"
#include "gpu_image/operators/geometric.hpp"
#include "gpu_image/operators/histogram_calculator.hpp"
#include "gpu_image/operators/image_operator.hpp"
#include "gpu_image/operators/image_resizer.hpp"
#include "gpu_image/operators/morphology.hpp"
#include "gpu_image/operators/operator_pipeline.hpp"
#include "gpu_image/operators/pixel_operator.hpp"
#include "gpu_image/operators/threshold.hpp"

// Processing
#include "gpu_image/processing/image_processor.hpp"
#include "gpu_image/processing/pipeline_processor.hpp"

// I/O
#include "gpu_image/io/image_io.hpp"

namespace gpu_image {

// 库版本信息
constexpr int VERSION_MAJOR = GPU_IMAGE_VERSION_MAJOR;
constexpr int VERSION_MINOR = GPU_IMAGE_VERSION_MINOR;
constexpr int VERSION_PATCH = GPU_IMAGE_VERSION_PATCH;

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

} // namespace gpu_image
