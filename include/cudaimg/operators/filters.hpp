#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>

namespace cudaimg {

// 滤波器类
class Filters {
public:
  // 中值滤波
  static void medianFilter(const CudaImage& input, CudaImage& output,
                           int kernelSize = 3, cudaStream_t stream = nullptr);

  // 双边滤波
  static void bilateralFilter(const CudaImage& input, CudaImage& output,
                              int kernelSize = 5, float sigmaSpace = 10.0f,
                              float sigmaColor = 50.0f,
                              cudaStream_t stream = nullptr);

  // 均值滤波（盒式滤波）
  static void boxFilter(const CudaImage& input, CudaImage& output,
                        int kernelSize = 3, cudaStream_t stream = nullptr);

  // 锐化滤波
  static void sharpen(const CudaImage& input, CudaImage& output,
                      float strength = 1.0f, cudaStream_t stream = nullptr);

  // 拉普拉斯滤波（边缘增强）
  static void laplacian(const CudaImage& input, CudaImage& output,
                        cudaStream_t stream = nullptr);
};

// 图像算术操作类
class ImageArithmetic {
public:
  // 图像加法
  static void add(const CudaImage& src1, const CudaImage& src2, CudaImage& output,
                  cudaStream_t stream = nullptr);

  // 图像混合 (alpha blending)
  static void blend(const CudaImage& src1, const CudaImage& src2,
                    CudaImage& output,
                    float alpha, // src1 权重
                    cudaStream_t stream = nullptr);

  // 标量加法
  static void addScalar(const CudaImage& input, CudaImage& output,
                        unsigned char value, cudaStream_t stream = nullptr);
};

} // namespace cudaimg
