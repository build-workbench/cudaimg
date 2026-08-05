#pragma once

#include "gpu_image/core/gpu_image.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace gpu_image {

// 边界处理模式
enum class BorderMode {
  Zero,     // 零填充
  Mirror,   // 镜像填充
  Replicate // 复制边界
};

// 卷积引擎类
class ConvolutionEngine {
public:
  // 通用卷积操作
  // 注意：实现为互相关（不翻转卷积核）。对称核（如高斯）结果与卷积一致；
  // 方向性核（如 Sobel）梯度符号相反，但幅值不受影响。
  static void convolve(const GpuImage& input, GpuImage& output,
                       const float* kernel, int kernelSize,
                       BorderMode borderMode = BorderMode::Zero,
                       cudaStream_t stream = nullptr);

  // 高斯模糊
  // kernelSize 必须为奇数，sigma 控制模糊程度
  static void gaussianBlur(const GpuImage& input, GpuImage& output,
                           int kernelSize, float sigma,
                           cudaStream_t stream = nullptr);

  // Sobel 边缘检测
  // 返回梯度幅值图像
  static void sobelEdgeDetection(const GpuImage& input, GpuImage& output,
                                 cudaStream_t stream = nullptr);

  // 可分离卷积（更高效的实现）
  static void separableConvolve(const GpuImage& input, GpuImage& output,
                                const float* rowKernel, const float* colKernel,
                                int kernelSize, cudaStream_t stream = nullptr);

  // 生成高斯核
  static std::vector<float> generateGaussianKernel(int size, float sigma);

  // 生成 1D 高斯核（用于可分离卷积）
  static std::vector<float> generateGaussianKernel1D(int size, float sigma);

private:
  // Sobel 核
  static constexpr float SOBEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static constexpr float SOBEL_Y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
};

} // namespace gpu_image
