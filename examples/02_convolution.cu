/**
 * Lv3 示例：Shared Memory 卷积
 *
 * 演示：高斯模糊 + Sobel 边缘检测
 * 对应源码：src/operators/convolution_engine.cu
 */
#include "cudaimg/cudaimg.hpp"
#include <iostream>

using namespace cudaimg;

int main() {
  if (!isCudaAvailable()) {
    std::cerr << "CUDA not available" << std::endl;
    return 1;
  }

  // 创建 64x64 RGB 渐变图像
  HostImage host = ImageUtils::createHostImage(64, 64, 3);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
      host.at(x, y, 0) = x * 4;  // R
      host.at(x, y, 1) = y * 4;  // G
      host.at(x, y, 2) = 128;    // B
    }

  ImageProcessor proc;
  CudaImage gpu = proc.loadFromHost(host);

  // 高斯模糊（5x5, sigma=1.5）
  CudaImage blurred = proc.gaussianBlur(gpu, 5, 1.5f);
  HostImage blurredHost = proc.download(blurred);
  std::cout << "Gaussian blur: center pixel R="
            << (int)blurredHost.at(32, 32, 0) << std::endl;

  // Sobel 边缘检测（输出单通道）
  CudaImage edges = proc.sobelEdgeDetection(gpu);
  HostImage edgesHost = proc.download(edges);
  std::cout << "Sobel: output channels=" << edgesHost.channels
            << ", center=" << (int)edgesHost.data[32 * 64 + 32] << std::endl;

  return 0;
}
