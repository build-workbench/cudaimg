/**
 * PixelOperator 单元测试
 *
 * Feature: gpu-image-processing
 * Property 2: 反色操作自逆性
 * Property 3: 灰度化公式正确性
 * Property 4: 亮度调整范围不变性
 */

#include "cudaimg/cudaimg.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace cudaimg;

class PixelOperatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  // 创建测试图像
  HostImage createTestImage(int width, int height, int channels) {
    HostImage image = ImageUtils::createHostImage(width, height, channels);
    unsigned int seed = 42;
    for (size_t i = 0; i < image.data.size(); ++i) {
      seed = seed * 1103515245 + 12345;
      image.data[i] = static_cast<unsigned char>((seed >> 16) & 0xFF);
    }
    return image;
  }
};

// Property 2: 反色操作自逆性
// For any 图像，invert(invert(image)) == image
TEST_F(PixelOperatorTest, InvertInvolution) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // 第一次反色
  CudaImage inverted;
  PixelOperator::invert(gpuOriginal, inverted);
  cudaDeviceSynchronize();

  // 第二次反色
  CudaImage doubleInverted;
  PixelOperator::invert(inverted, doubleInverted);
  cudaDeviceSynchronize();

  // 下载并比较
  HostImage result = ImageUtils::downloadFromGpu(doubleInverted);

  ASSERT_EQ(original.data.size(), result.data.size());
  for (size_t i = 0; i < original.data.size(); ++i) {
    EXPECT_EQ(original.data[i], result.data[i]) << "Mismatch at index " << i;
  }
}

// 测试反色的具体值
TEST_F(PixelOperatorTest, InvertValues) {
  const int width = 4;
  const int height = 4;
  const int channels = 1;

  HostImage original = ImageUtils::createHostImage(width, height, channels);
  // 设置已知值
  for (int i = 0; i < 16; ++i) {
    original.data[i] = static_cast<unsigned char>(i * 16); // 0, 16, 32, ...
  }

  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);
  CudaImage inverted;
  PixelOperator::invert(gpuOriginal, inverted);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(inverted);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(result.data[i], 255 - original.data[i]);
  }
}

// Property 3: 灰度化公式正确性
TEST_F(PixelOperatorTest, GrayscaleFormula) {
  const int width = 4;
  const int height = 4;
  const int channels = 3;

  HostImage original = ImageUtils::createHostImage(width, height, channels);

  // 设置已知 RGB 值
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      original.at(x, y, 0) = 100; // R
      original.at(x, y, 1) = 150; // G
      original.at(x, y, 2) = 200; // B
    }
  }

  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);
  CudaImage grayscale;
  PixelOperator::toGrayscale(gpuOriginal, grayscale);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(grayscale);

  // 预期灰度值: 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 =
  // 140.75 ≈ 141
  float expectedGray = 0.299f * 100 + 0.587f * 150 + 0.114f * 200;
  unsigned char expected = static_cast<unsigned char>(std::round(expectedGray));

  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_NEAR(result.data[i], expected, 1) << "Mismatch at index " << i;
  }
}

// Property 4: 亮度调整范围不变性
// For any 图像和偏移量，结果像素值应在 [0, 255] 范围内
TEST_F(PixelOperatorTest, BrightnessRangeInvariant) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // 测试各种偏移量
  std::vector<int> offsets = {-300, -100, -50, 0, 50, 100, 300};

  for (int offset : offsets) {
    CudaImage adjusted;
    PixelOperator::adjustBrightness(gpuOriginal, adjusted, offset);
    cudaDeviceSynchronize();

    HostImage result = ImageUtils::downloadFromGpu(adjusted);

    ASSERT_EQ(result.data.size(), original.data.size());
    for (size_t i = 0; i < result.data.size(); ++i) {
      // 期望值：clamp(original + offset, 0, 255)，与内核语义一致
      // （原先对 unsigned char 的 EXPECT_GE(v,0)/EXPECT_LE(v,255)
      // 是恒真空断言）
      const int expected =
          std::clamp(static_cast<int>(original.data[i]) + offset, 0, 255);
      EXPECT_EQ(result.data[i], static_cast<unsigned char>(expected))
          << "index " << i << " with offset " << offset;
    }
  }
}

// 测试亮度调整的具体值
TEST_F(PixelOperatorTest, BrightnessValues) {
  const int width = 4;
  const int height = 4;
  const int channels = 1;

  HostImage original = ImageUtils::createHostImage(width, height, channels);
  original.data[0] = 100;
  original.data[1] = 200;
  original.data[2] = 50;
  original.data[3] = 250;

  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // 测试正偏移
  CudaImage brighter;
  PixelOperator::adjustBrightness(gpuOriginal, brighter, 30);
  cudaDeviceSynchronize();
  HostImage brighterResult = ImageUtils::downloadFromGpu(brighter);

  EXPECT_EQ(brighterResult.data[0], 130); // 100 + 30
  EXPECT_EQ(brighterResult.data[1], 230); // 200 + 30
  EXPECT_EQ(brighterResult.data[2], 80);  // 50 + 30
  EXPECT_EQ(brighterResult.data[3], 255); // 250 + 30 = 280, clamped to 255

  // 测试负偏移
  CudaImage darker;
  PixelOperator::adjustBrightness(gpuOriginal, darker, -60);
  cudaDeviceSynchronize();
  HostImage darkerResult = ImageUtils::downloadFromGpu(darker);

  EXPECT_EQ(darkerResult.data[0], 40);  // 100 - 60
  EXPECT_EQ(darkerResult.data[1], 140); // 200 - 60
  EXPECT_EQ(darkerResult.data[2], 0);   // 50 - 60 = -10, clamped to 0
  EXPECT_EQ(darkerResult.data[3], 190); // 250 - 60
}

// 测试原地操作
TEST_F(PixelOperatorTest, InPlaceOperations) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuImage = ImageUtils::uploadToGpu(original);

  // 原地反色
  PixelOperator::invertInPlace(gpuImage);
  cudaDeviceSynchronize();

  HostImage inverted = ImageUtils::downloadFromGpu(gpuImage);

  for (size_t i = 0; i < original.data.size(); ++i) {
    EXPECT_EQ(inverted.data[i], 255 - original.data[i]);
  }
}

// 测试无效输入
TEST_F(PixelOperatorTest, InvalidInput) {
  CudaImage invalid; // 无效图像
  CudaImage output;

  EXPECT_THROW(PixelOperator::invert(invalid, output), std::invalid_argument);
  EXPECT_THROW(PixelOperator::toGrayscale(invalid, output),
               std::invalid_argument);
  EXPECT_THROW(PixelOperator::adjustBrightness(invalid, output, 50),
               std::invalid_argument);
}

// 测试灰度图像的灰度化（应该失败）
TEST_F(PixelOperatorTest, GrayscaleOnGrayscale) {
  HostImage grayscale = ImageUtils::createHostImage(32, 32, 1);
  CudaImage gpuGrayscale = ImageUtils::uploadToGpu(grayscale);
  CudaImage output;

  EXPECT_THROW(PixelOperator::toGrayscale(gpuGrayscale, output),
               std::invalid_argument);
}
