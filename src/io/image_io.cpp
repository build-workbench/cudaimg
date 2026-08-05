#include "gpu_image/io/image_io.hpp"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

// stb_image 由 CMake FetchContent 提供
#ifdef GPU_IMAGE_HAS_STB

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace gpu_image {

namespace {

// JPEG 默认编码质量（0-100）
constexpr int kDefaultJpegQuality = 90;

// stbi 返回指针的 RAII 守卫：确保任何退出路径（含 data.assign 抛 bad_alloc）
// 都会释放 stbi 分配的内存，避免泄漏。
struct StbiDeleter {
  void operator()(unsigned char* p) const {
    if (p != nullptr) {
      stbi_image_free(p);
    }
  }
};
using StbiPtr = std::unique_ptr<unsigned char, StbiDeleter>;

// 将 stbi 解码结果组装为 HostImage。
// stbi_load/stbi_load_from_memory 是 stb 的 8-bit 接口：16-bit PNG/PSD/PNM
// 会被内部降级为 8-bit（stbi__convert_16_to_8），HDR 会转为 LDR
// （stbi__hdr_to_ldr），因此返回值始终是每通道 1 字节的 unsigned char。
// stbi 以 desired_channels=0 加载时会返回文件原始通道数（1/2/3/4），
// 而 HostImage 仅支持 1/3/4 通道。这里把 2 通道（灰度+Alpha）归一化为 RGBA，
// 其余合法通道数原样保留，非法通道数抛异常——避免返回一个 isValid()==false
// 的对象让调用者在后续 uploadToGpu 处得到莫名错误。
HostImage buildHostImage(StbiPtr pixels, int width, int height, int channels) {
  if (channels == 1 || channels == 3 || channels == 4) {
    HostImage image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.data.assign(pixels.get(), pixels.get() + static_cast<size_t>(width) *
                                                       height * channels);
    return image;
  }

  if (channels == 2) {
    // 灰度 + Alpha → RGBA（R=G=B=灰度，A=Alpha）
    HostImage image;
    image.width = width;
    image.height = height;
    image.channels = 4;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    image.data.resize(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i) {
      const unsigned char gray = pixels.get()[i * 2 + 0];
      const unsigned char alpha = pixels.get()[i * 2 + 1];
      image.data[i * 4 + 0] = gray;
      image.data[i * 4 + 1] = gray;
      image.data[i * 4 + 2] = gray;
      image.data[i * 4 + 3] = alpha;
    }
    return image;
  }

  throw std::runtime_error("Unsupported channel count in image: " +
                           std::to_string(channels));
}

// 从路径提取小写扩展名；无扩展名时返回空串（修复 find_last_of 返回 npos 时
// npos+1 回绕为 0、把整条路径当成扩展名的边界 bug）。
std::string extractExtension(const std::string& filepath) {
  const auto dotPos = filepath.find_last_of('.');
  if (dotPos == std::string::npos) {
    return {};
  }
  std::string ext = filepath.substr(dotPos + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext;
}

// 写入回调上下文：在回调内部捕获异常，避免 std::bad_alloc 穿越
// stb_image_write 的纯 C 栈帧（跨 C 边界抛异常属未定义行为）。
struct WriteContext {
  std::vector<unsigned char>* out;
  bool failed = false;
};

void stbiWriteCallback(void* context, void* data, int size) {
  auto* ctx = static_cast<WriteContext*>(context);
  if (ctx->failed || size <= 0) {
    return;
  }
  try {
    auto* bytes = static_cast<unsigned char*>(data);
    ctx->out->insert(ctx->out->end(), bytes, bytes + size);
  } catch (...) {
    ctx->failed = true; // 由 encodeToMemory 在 C++ 侧检查并抛异常
  }
}

} // namespace

HostImage ImageIO::loadFromFile(const std::string& filepath) {
  int width, height, channels;
  StbiPtr pixels(stbi_load(filepath.c_str(), &width, &height, &channels, 0));

  if (!pixels) {
    throw std::runtime_error("Failed to load image: " + filepath + " (" +
                             stbi_failure_reason() + ")");
  }

  return buildHostImage(std::move(pixels), width, height, channels);
}

bool ImageIO::saveToFile(const HostImage& image, const std::string& filepath) {
  if (!image.isValid()) {
    return false;
  }

  const std::string ext = extractExtension(filepath);
  const int stride = image.width * image.channels;

  int result = 0;
  if (ext == "png") {
    result = stbi_write_png(filepath.c_str(), image.width, image.height,
                            image.channels, image.data.data(), stride);
  } else if (ext == "jpg" || ext == "jpeg") {
    result =
        stbi_write_jpg(filepath.c_str(), image.width, image.height,
                       image.channels, image.data.data(), kDefaultJpegQuality);
  } else if (ext == "bmp") {
    result = stbi_write_bmp(filepath.c_str(), image.width, image.height,
                            image.channels, image.data.data());
  } else if (ext == "tga") {
    result = stbi_write_tga(filepath.c_str(), image.width, image.height,
                            image.channels, image.data.data());
  } else {
    return false;
  }

  return result != 0;
}

HostImage ImageIO::loadFromMemory(const unsigned char* data, size_t size) {
  if (data == nullptr || size == 0) {
    throw std::invalid_argument("Invalid memory buffer");
  }
  // stbi_load_from_memory 的长度参数是 int；超过 INT_MAX
  // 会截断为负值/实现定义值， 导致按错误长度解码。提前拒绝。
  if (size > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image buffer exceeds maximum supported size");
  }

  int width, height, channels;
  StbiPtr pixels(stbi_load_from_memory(data, static_cast<int>(size), &width,
                                       &height, &channels, 0));

  if (!pixels) {
    throw std::runtime_error(
        std::string("Failed to decode image from memory: ") +
        stbi_failure_reason());
  }

  return buildHostImage(std::move(pixels), width, height, channels);
}

std::vector<unsigned char> ImageIO::encodeToMemory(const HostImage& image,
                                                   const std::string& format) {
  if (!image.isValid()) {
    throw std::invalid_argument("Invalid image");
  }

  std::string fmt = format;
  std::transform(fmt.begin(), fmt.end(), fmt.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  std::vector<unsigned char> result;
  // 预分配以降低回调中重分配/抛异常的概率（未压缩格式输出约等于原始大小）。
  result.reserve(image.totalBytes() + 1024);

  WriteContext ctx{&result, false};
  const int stride = image.width * image.channels;

  int ok = 0;
  if (fmt == "png") {
    ok = stbi_write_png_to_func(stbiWriteCallback, &ctx, image.width,
                                image.height, image.channels, image.data.data(),
                                stride);
  } else if (fmt == "jpg" || fmt == "jpeg") {
    ok = stbi_write_jpg_to_func(stbiWriteCallback, &ctx, image.width,
                                image.height, image.channels, image.data.data(),
                                kDefaultJpegQuality);
  } else if (fmt == "bmp") {
    ok =
        stbi_write_bmp_to_func(stbiWriteCallback, &ctx, image.width,
                               image.height, image.channels, image.data.data());
  } else if (fmt == "tga") {
    ok =
        stbi_write_tga_to_func(stbiWriteCallback, &ctx, image.width,
                               image.height, image.channels, image.data.data());
  } else {
    throw std::invalid_argument("Unsupported format: " + format);
  }

  if (!ok || ctx.failed) {
    throw std::runtime_error("Failed to encode image to " + format);
  }

  return result;
}

std::vector<std::string> ImageIO::getSupportedFormats() {
  // 可解码（读取）的格式
  return {"png", "jpg", "jpeg", "bmp", "tga", "psd", "gif", "hdr", "pnm"};
}

std::vector<std::string> ImageIO::getWritableFormats() {
  // 可编码（写入）的格式
  return {"png", "jpg", "jpeg", "bmp", "tga"};
}

bool ImageIO::isFormatSupported(const std::string& filepath) {
  const std::string ext = extractExtension(filepath);
  if (ext.empty()) {
    return false;
  }

  auto formats = getSupportedFormats();
  return std::find(formats.begin(), formats.end(), ext) != formats.end();
}

} // namespace gpu_image

#else // !GPU_IMAGE_HAS_STB

// 无 stb 时的空实现，所有函数抛出异常
namespace gpu_image {

HostImage ImageIO::loadFromFile(const std::string&) {
  throw std::runtime_error(
      "ImageIO requires stb_image. Build with -DGPU_IMAGE_ENABLE_IO=ON");
}

bool ImageIO::saveToFile(const HostImage&, const std::string&) {
  throw std::runtime_error(
      "ImageIO requires stb_image_write. Build with -DGPU_IMAGE_ENABLE_IO=ON");
}

HostImage ImageIO::loadFromMemory(const unsigned char*, size_t) {
  throw std::runtime_error(
      "ImageIO requires stb_image. Build with -DGPU_IMAGE_ENABLE_IO=ON");
}

std::vector<unsigned char> ImageIO::encodeToMemory(const HostImage&,
                                                   const std::string&) {
  throw std::runtime_error(
      "ImageIO requires stb_image_write. Build with -DGPU_IMAGE_ENABLE_IO=ON");
}

std::vector<std::string> ImageIO::getSupportedFormats() { return {}; }
std::vector<std::string> ImageIO::getWritableFormats() { return {}; }
bool ImageIO::isFormatSupported(const std::string&) { return false; }

} // namespace gpu_image

#endif // GPU_IMAGE_HAS_STB
