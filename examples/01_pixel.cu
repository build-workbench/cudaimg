/**
 * Lv1 示例：最简单的 CUDA kernel
 *
 * 演示：上传图像 -> GPU 反色 -> 下载回主机
 * 对应源码：src/operators/pixel_operator.cu
 */
#include "cudaimg/cudaimg.hpp"
#include <iostream>

using namespace cudaimg;

int main() {
  if (!isCudaAvailable()) {
    std::cerr << "CUDA not available" << std::endl;
    return 1;
  }

  // 创建 8x8 灰度测试图像
  HostImage host = ImageUtils::createHostImage(8, 8, 1);
  for (size_t i = 0; i < host.data.size(); ++i)
    host.data[i] = static_cast<unsigned char>(i * 4);

  std::cout << "Before: pixel[0]=" << (int)host.data[0]
            << " pixel[63]=" << (int)host.data[63] << std::endl;

  // 上传 -> 反色 -> 下载
  ImageProcessor proc;
  CudaImage gpu = proc.loadFromHost(host);
  CudaImage inverted = proc.invert(gpu);
  HostImage result = proc.download(inverted);

  std::cout << "After:  pixel[0]=" << (int)result.data[0]
            << " pixel[63]=" << (int)result.data[63] << std::endl;
  // 预期：255 - 0 = 255, 255 - 252 = 3

  return 0;
}
