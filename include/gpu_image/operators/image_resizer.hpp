#pragma once

#include "gpu_image/core/gpu_image.hpp"
#include <cuda_runtime.h>

namespace gpu_image {

// 插值模式
enum class InterpolationMode {
  NearestNeighbor, // 最近邻插值
  Bilinear,        // 双线性插值
  Bicubic          // 双三次插值（Catmull-Rom）
};

// 图像缩放器类
class ImageResizer {
public:
  // 双线性插值缩放
  static void resize(const GpuImage& input, GpuImage& output, int newWidth,
                     int newHeight,
                     InterpolationMode mode = InterpolationMode::Bilinear,
                     cudaStream_t stream = nullptr);

  // 按比例缩放
  static void
  resizeByScale(const GpuImage& input, GpuImage& output, float scaleX,
                float scaleY,
                InterpolationMode mode = InterpolationMode::Bilinear,
                cudaStream_t stream = nullptr);

  // 保持宽高比缩放（适应指定尺寸）
  static void resizeFit(const GpuImage& input, GpuImage& output, int maxWidth,
                        int maxHeight,
                        InterpolationMode mode = InterpolationMode::Bilinear,
                        cudaStream_t stream = nullptr);
};

} // namespace gpu_image
