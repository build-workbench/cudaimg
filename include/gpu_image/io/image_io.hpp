#pragma once

#include "gpu_image/core/gpu_image.hpp"
#include <string>

namespace gpu_image {

// 图像文件 I/O 类
class ImageIO {
public:
  // 从文件加载图像
  static HostImage loadFromFile(const std::string& filepath);

  // 保存图像到文件
  // 支持格式: PNG, JPG, BMP, TGA
  static bool saveToFile(const HostImage& image, const std::string& filepath);

  // 从内存加载图像
  static HostImage loadFromMemory(const unsigned char* data, size_t size);

  // 编码图像到内存
  static std::vector<unsigned char> encodeToMemory(const HostImage& image,
                                                   const std::string& format);

  // 获取支持解码（读取）的格式
  static std::vector<std::string> getSupportedFormats();

  // 获取支持编码（写入）的格式
  static std::vector<std::string> getWritableFormats();

  // 检查文件格式是否支持解码（读取）
  static bool isFormatSupported(const std::string& filepath);
};

} // namespace gpu_image
