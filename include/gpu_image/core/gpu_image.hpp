#pragma once

#include "gpu_image/core/device_buffer.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gpu_image {

// GPU 上的图像表示
// GPU Image representation
struct GpuImage {
  DeviceBuffer buffer;
  int width = 0;
  int height = 0;
  int channels = 0; // 1 for grayscale, 3 for RGB, 4 for RGBA

  // 计算每行字节数 / Calculate row pitch in bytes
  size_t pitch() const { return static_cast<size_t>(width) * channels; }

  // 计算总字节数 / Calculate total bytes
  size_t totalBytes() const {
    return static_cast<size_t>(width) * height * channels;
  }

  // 检查是否有效 / Check if image is valid
  bool isValid() const {
    return buffer.isValid() && width > 0 && height > 0 &&
           (channels == 1 || channels == 3 || channels == 4);
  }

  // 获取像素数量 / Get total pixel count
  size_t pixelCount() const { return static_cast<size_t>(width) * height; }
};

// Host 端图像数据
// Host-side image data
struct HostImage {
  std::vector<unsigned char> data;
  int width = 0;
  int height = 0;
  int channels = 0;

  // 访问像素（带边界检查）/ Access pixel at (x, y, c) with bounds checking
  unsigned char& at(int x, int y, int c) {
    checkBounds(x, y, c);
    return data[(static_cast<size_t>(y) * width + x) * channels + c];
  }

  const unsigned char& at(int x, int y, int c) const {
    checkBounds(x, y, c);
    return data[(static_cast<size_t>(y) * width + x) * channels + c];
  }

  // 计算总字节数 / Calculate total bytes
  size_t totalBytes() const {
    return static_cast<size_t>(width) * height * channels;
  }

  // 检查是否有效 / Check if image is valid
  bool isValid() const {
    return !data.empty() && width > 0 && height > 0 &&
           (channels == 1 || channels == 3 || channels == 4) &&
           data.size() == totalBytes();
  }

private:
  // 越界抛 std::out_of_range（符合 STL at() 语义；此前负索引经 size_t 转换
  // 会变成巨值导致越界读写 UB）
  void checkBounds(int x, int y, int c) const {
    if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
      throw std::out_of_range("HostImage::at: index out of range");
    }
  }
};

} // namespace gpu_image
