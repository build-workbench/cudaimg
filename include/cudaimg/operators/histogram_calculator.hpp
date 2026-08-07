#pragma once

#include "cudaimg/core/image.hpp"
#include <array>
#include <cuda_runtime.h>

namespace cudaimg {

// 直方图计算器类
class HistogramCalculator {
public:
  static constexpr int NUM_BINS = 256;

  // 计算灰度直方图
  // 输入可以是单通道灰度图像或多通道图像（会先转换为灰度）
  static std::array<int, NUM_BINS> calculate(const CudaImage& input,
                                             cudaStream_t stream = nullptr);

  // 计算 RGB 各通道直方图
  static std::array<std::array<int, NUM_BINS>, 3>
  calculateRGB(const CudaImage& input, cudaStream_t stream = nullptr);

  // 计算单通道直方图（不做灰度转换）
  static std::array<int, NUM_BINS>
  calculateChannel(const CudaImage& input, int channel,
                   cudaStream_t stream = nullptr);

  // 直方图均衡化
  static void equalize(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);
};

} // namespace cudaimg
