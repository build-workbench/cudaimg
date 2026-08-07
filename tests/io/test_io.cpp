/**
 * ImageIO Tests（纯 host / CPU-only，无需 GPU）
 *
 * 这些测试验证图像编解码的正确性，可在没有 CUDA 设备的 CI 环境中运行，
 * 为构建提供真实的数值信号（encode→decode 往返、错误输入、格式查询）。
 */

#include "cudaimg/cudaimg.hpp"
#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace cudaimg;

namespace {

// 构造一张确定性内容的测试图像（无需 GPU）
HostImage makeTestImage(int width, int height, int channels,
                        unsigned char base) {
  HostImage image = ImageUtils::createHostImage(width, height, channels);
  for (size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = static_cast<unsigned char>((base + i) % 256);
  }
  return image;
}

} // namespace

class ImageIOTest : public ::testing::Test {
protected:
  void SetUp() override {
    // ImageIO 为纯 host 实现，不依赖 GPU；仅在未启用 stb 的构建中跳过。
    HostImage tiny = makeTestImage(2, 2, 3, 0);
    try {
      (void)ImageIO::encodeToMemory(tiny, "png");
    } catch (const std::exception&) {
      GTEST_SKIP() << "ImageIO/stb not enabled in this build";
    }
  }
};

// PNG 无损往返：像素逐字节一致
TEST_F(ImageIOTest, PngRoundtripPreservesPixels) {
  HostImage image = makeTestImage(16, 12, 3, 0);
  auto encoded = ImageIO::encodeToMemory(image, "png");
  ASSERT_FALSE(encoded.empty());

  HostImage decoded = ImageIO::loadFromMemory(encoded.data(), encoded.size());
  EXPECT_EQ(decoded.width, image.width);
  EXPECT_EQ(decoded.height, image.height);
  EXPECT_EQ(decoded.channels, image.channels);
  ASSERT_EQ(decoded.data.size(), image.data.size());
  EXPECT_EQ(decoded.data, image.data);
}

// BMP 无损往返
TEST_F(ImageIOTest, BmpRoundtripPreservesPixels) {
  HostImage image = makeTestImage(8, 8, 3, 42);
  auto encoded = ImageIO::encodeToMemory(image, "bmp");
  HostImage decoded = ImageIO::loadFromMemory(encoded.data(), encoded.size());
  ASSERT_EQ(decoded.data.size(), image.data.size());
  EXPECT_EQ(decoded.data, image.data);
}

// 灰度图往返（单通道）
TEST_F(ImageIOTest, GrayscaleRoundtrip) {
  HostImage image = makeTestImage(10, 10, 1, 7);
  auto encoded = ImageIO::encodeToMemory(image, "png");
  HostImage decoded = ImageIO::loadFromMemory(encoded.data(), encoded.size());
  EXPECT_EQ(decoded.channels, 1);
  EXPECT_EQ(decoded.data, image.data);
}

// RGBA 往返（四通道）
TEST_F(ImageIOTest, RgbaRoundtrip) {
  HostImage image = makeTestImage(6, 6, 4, 99);
  auto encoded = ImageIO::encodeToMemory(image, "png");
  HostImage decoded = ImageIO::loadFromMemory(encoded.data(), encoded.size());
  EXPECT_EQ(decoded.channels, 4);
  EXPECT_EQ(decoded.data, image.data);
}

// 空/空指针缓冲应抛 invalid_argument
TEST_F(ImageIOTest, LoadFromMemoryRejectsNullAndEmpty) {
  EXPECT_THROW(ImageIO::loadFromMemory(nullptr, 10), std::invalid_argument);
  unsigned char byte = 0;
  EXPECT_THROW(ImageIO::loadFromMemory(&byte, 0), std::invalid_argument);
}

// 无法解码的数据应抛 runtime_error
TEST_F(ImageIOTest, LoadFromMemoryRejectsGarbage) {
  std::vector<unsigned char> garbage(64, 0xAB);
  EXPECT_THROW(ImageIO::loadFromMemory(garbage.data(), garbage.size()),
               std::runtime_error);
}

// 不支持的编码格式应抛 invalid_argument
TEST_F(ImageIOTest, EncodeUnsupportedFormatThrows) {
  HostImage image = makeTestImage(4, 4, 3, 0);
  EXPECT_THROW(ImageIO::encodeToMemory(image, "gif"), std::invalid_argument);
}

// 无效图像编码应抛 invalid_argument
TEST_F(ImageIOTest, EncodeInvalidImageThrows) {
  HostImage invalid; // 默认构造，无效
  EXPECT_THROW(ImageIO::encodeToMemory(invalid, "png"), std::invalid_argument);
}

// 格式查询：写格式是读格式的子集；扩展名判断正确
TEST_F(ImageIOTest, FormatQueries) {
  auto readable = ImageIO::getSupportedFormats();
  auto writable = ImageIO::getWritableFormats();
  EXPECT_FALSE(readable.empty());
  EXPECT_FALSE(writable.empty());

  for (const auto& fmt : writable) {
    EXPECT_NE(std::find(readable.begin(), readable.end(), fmt), readable.end())
        << "writable format not in readable set: " << fmt;
  }

  EXPECT_TRUE(ImageIO::isFormatSupported("photo.png"));
  EXPECT_TRUE(ImageIO::isFormatSupported("PHOTO.JPG"));
  EXPECT_FALSE(ImageIO::isFormatSupported("noextension"));
  EXPECT_FALSE(ImageIO::isFormatSupported(""));
}

// 不支持的扩展名保存返回 false（含无扩展名边界）
TEST_F(ImageIOTest, SaveToFileUnsupportedExtensionReturnsFalse) {
  HostImage image = makeTestImage(4, 4, 3, 0);
  EXPECT_FALSE(ImageIO::saveToFile(image, "foo.xyz"));
  EXPECT_FALSE(ImageIO::saveToFile(image, "noextension"));
}

// 文件保存 + 读取往返
TEST_F(ImageIOTest, SaveAndLoadFileRoundtrip) {
  HostImage image = makeTestImage(12, 9, 3, 17);
  const std::string path = "image_io_test_roundtrip.png";

  ASSERT_TRUE(ImageIO::saveToFile(image, path));
  HostImage loaded = ImageIO::loadFromFile(path);
  EXPECT_EQ(loaded.data, image.data);

  std::remove(path.c_str());
}

// 读取不存在的文件应抛 runtime_error
TEST_F(ImageIOTest, LoadFromFileMissingThrows) {
  EXPECT_THROW(ImageIO::loadFromFile("definitely_does_not_exist_12345.png"),
               std::runtime_error);
}
