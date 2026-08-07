#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>

namespace cudaimg {

// 结构元素形状
enum class StructuringElement {
  Rectangle, // 矩形
  Cross,     // 十字形
  Ellipse    // 椭圆形
};

// 形态学操作类
class Morphology {
public:
  // 腐蚀操作
  static void erode(const CudaImage& input, CudaImage& output, int kernelSize = 3,
                    StructuringElement element = StructuringElement::Rectangle,
                    cudaStream_t stream = nullptr);

  // 膨胀操作
  static void dilate(const CudaImage& input, CudaImage& output,
                     int kernelSize = 3,
                     StructuringElement element = StructuringElement::Rectangle,
                     cudaStream_t stream = nullptr);

  // 开运算 (先腐蚀后膨胀)
  static void open(const CudaImage& input, CudaImage& output, int kernelSize = 3,
                   StructuringElement element = StructuringElement::Rectangle,
                   cudaStream_t stream = nullptr);

  // 闭运算 (先膨胀后腐蚀)
  static void close(const CudaImage& input, CudaImage& output, int kernelSize = 3,
                    StructuringElement element = StructuringElement::Rectangle,
                    cudaStream_t stream = nullptr);

  // 形态学梯度 (膨胀 - 腐蚀)
  static void
  gradient(const CudaImage& input, CudaImage& output, int kernelSize = 3,
           StructuringElement element = StructuringElement::Rectangle,
           cudaStream_t stream = nullptr);

  // 顶帽变换 (原图 - 开运算)
  static void topHat(const CudaImage& input, CudaImage& output,
                     int kernelSize = 3,
                     StructuringElement element = StructuringElement::Rectangle,
                     cudaStream_t stream = nullptr);

  // 黑帽变换 (闭运算 - 原图)
  static void
  blackHat(const CudaImage& input, CudaImage& output, int kernelSize = 3,
           StructuringElement element = StructuringElement::Rectangle,
           cudaStream_t stream = nullptr);
};

} // namespace cudaimg
