#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>

namespace cudaimg {

// 翻转方向
enum class FlipDirection {
  Horizontal, // 水平翻转
  Vertical,   // 垂直翻转
  Both        // 水平+垂直翻转
};

// 几何变换类
class Geometric {
public:
  // 图像旋转（角度，顺时针）
  static void rotate(const CudaImage& input, CudaImage& output,
                     float angleDegrees, cudaStream_t stream = nullptr);

  // 90度旋转（更高效）
  static void rotate90(const CudaImage& input, CudaImage& output,
                       int times = 1, // 1=90°, 2=180°, 3=270°
                       cudaStream_t stream = nullptr);

  // 图像翻转
  static void flip(const CudaImage& input, CudaImage& output,
                   FlipDirection direction, cudaStream_t stream = nullptr);

  // 仿射变换
  // matrix: 2x3 变换矩阵 [a, b, tx, c, d, ty]
  static void affineTransform(const CudaImage& input, CudaImage& output,
                              const float* matrix, int outputWidth,
                              int outputHeight, cudaStream_t stream = nullptr);

  // 图像裁剪
  static void crop(const CudaImage& input, CudaImage& output, int x, int y,
                   int width, int height, cudaStream_t stream = nullptr);

  // 图像填充
  static void pad(const CudaImage& input, CudaImage& output, int top, int bottom,
                  int left, int right, unsigned char padValue = 0,
                  cudaStream_t stream = nullptr);
};

} // namespace cudaimg
