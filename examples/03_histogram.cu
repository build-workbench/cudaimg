/**
 * Lv5 示例：原子操作直方图
 *
 * 演示：直方图计算 + 均衡化
 * 对应源码：src/operators/histogram_calculator.cu
 */
#include "cudaimg/cudaimg.hpp"
#include <iostream>

using namespace cudaimg;

int main() {
  if (!isCudaAvailable()) {
    std::cerr << "CUDA not available" << std::endl;
    return 1;
  }

  // 创建 64x64 灰度图像（偏暗）
  HostImage host = ImageUtils::createHostImage(64, 64, 1);
  for (size_t i = 0; i < host.data.size(); ++i)
    host.data[i] = static_cast<unsigned char>(i % 100);  // 值集中在 0-99

  ImageProcessor proc;
  CudaImage gpu = proc.loadFromHost(host);

  // 计算直方图
  auto hist = proc.histogram(gpu);
  int nonZeroBins = 0;
  for (int i = 0; i < 256; ++i)
    if (hist[i] > 0) nonZeroBins++;
  std::cout << "Histogram: " << nonZeroBins << " non-zero bins (expected ~100)"
            << std::endl;

  // 直方图均衡化
  CudaImage equalized = proc.histogramEqualize(gpu);
  auto histEq = proc.histogram(equalized);
  int nonZeroBinsEq = 0;
  for (int i = 0; i < 256; ++i)
    if (histEq[i] > 0) nonZeroBinsEq++;
  std::cout << "After equalization: " << nonZeroBinsEq << " non-zero bins"
            << std::endl;

  return 0;
}
