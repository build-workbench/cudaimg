#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>

namespace cudaimg {

// 像素级操作类
class PixelOperator {
public:
  // 图像反色
  // 对每个像素执行: output[i] = 255 - input[i]
  static void invert(const CudaImage& input, CudaImage& output,
                     cudaStream_t stream = nullptr);

  // RGB 转灰度
  // 使用公式: gray = 0.299*R + 0.587*G + 0.114*B
  static void toGrayscale(const CudaImage& input, CudaImage& output,
                          cudaStream_t stream = nullptr);

  // 亮度调整
  // 对每个像素执行: output[i] = clamp(input[i] + offset, 0, 255)
  static void adjustBrightness(const CudaImage& input, CudaImage& output,
                               int offset, cudaStream_t stream = nullptr);

  // 原地版本
  static void invertInPlace(CudaImage& image, cudaStream_t stream = nullptr);
  static void adjustBrightnessInPlace(CudaImage& image, int offset,
                                      cudaStream_t stream = nullptr);
};

} // namespace cudaimg
