#pragma once

// validateInput/validateSameSize 需要完整的 GpuImage 定义（调用其成员
// 函数），不能只依赖 image_utils.hpp 的前向声明，否则 nvcc/EDG 前端
// 会报 incomplete type。
#include "gpu_image/core/gpu_image.hpp"
#include "gpu_image/core/image_utils.hpp"
#include <stdexcept>

namespace gpu_image {

// 默认 2D block 大小 / Default 2D block size
inline constexpr int kBlockSize2D = 16;

// 默认 1D block 大小 / Default 1D block size
inline constexpr int kBlockSize1D = 256;

// 计算 2D kernel 的 grid 和 block 配置
// Calculate grid and block configuration for 2D kernels
inline void calcGridBlock2D(int width, int height, dim3& grid, dim3& block) {
  block = dim3(kBlockSize2D, kBlockSize2D);
  grid =
      dim3(static_cast<unsigned>((width + kBlockSize2D - 1) / kBlockSize2D),
           static_cast<unsigned>((height + kBlockSize2D - 1) / kBlockSize2D));
}

// 计算 1D kernel 的 grid 和 block 配置
// Calculate grid and block configuration for 1D kernels
inline void calcGridBlock1D(int totalElements, int& gridSize, int& blockSize) {
  blockSize = kBlockSize1D;
  gridSize = (totalElements + blockSize - 1) / blockSize;
}

// 验证输入图像有效，否则抛异常
// Validate input image, throw if invalid
inline void validateInput(const GpuImage& input,
                          const char* context = "Invalid input image") {
  if (!input.isValid()) {
    throw std::invalid_argument(context);
  }
}

// 验证两张图像尺寸匹配
// Validate two images have matching dimensions
inline void
validateSameSize(const GpuImage& a, const GpuImage& b,
                 const char* context = "Image dimensions must match") {
  if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
    throw std::invalid_argument(context);
  }
}

} // namespace gpu_image
