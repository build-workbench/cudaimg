/**
 * ColorSpace 单元测试
 */

#include "cudaimg/cudaimg.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace cudaimg;

class ColorSpaceTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

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

// 测试 RGB to HSV 往返
TEST_F(ColorSpaceTest, RgbHsvRoundTrip) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // RGB -> HSV
  CudaImage hsv;
  ColorSpace::rgbToHsv(gpuOriginal, hsv);
  cudaDeviceSynchronize();

  // HSV -> RGB
  CudaImage restored;
  ColorSpace::hsvToRgb(hsv, restored);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(restored);

  // 允许一定的舍入误差
  int maxDiff = 0;
  for (size_t i = 0; i < original.data.size(); ++i) {
    int diff = std::abs(static_cast<int>(original.data[i]) -
                        static_cast<int>(result.data[i]));
    maxDiff = std::max(maxDiff, diff);
  }

  // HSV 转换可能有较大误差，允许 ±5
  EXPECT_LE(maxDiff, 5) << "Max difference: " << maxDiff;
}

// 测试 RGB to YUV 往返
TEST_F(ColorSpaceTest, RgbYuvRoundTrip) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // RGB -> YUV
  CudaImage yuv;
  ColorSpace::rgbToYuv(gpuOriginal, yuv);
  cudaDeviceSynchronize();

  // YUV -> RGB
  CudaImage restored;
  ColorSpace::yuvToRgb(yuv, restored);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(restored);

  int maxDiff = 0;
  for (size_t i = 0; i < original.data.size(); ++i) {
    int diff = std::abs(static_cast<int>(original.data[i]) -
                        static_cast<int>(result.data[i]));
    maxDiff = std::max(maxDiff, diff);
  }

  EXPECT_LE(maxDiff, 2) << "Max difference: " << maxDiff;
}

// 测试通道分离和合并
TEST_F(ColorSpaceTest, SplitMergeChannels) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage original = createTestImage(width, height, channels);
  CudaImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // 分离通道
  CudaImage ch0, ch1, ch2;
  ColorSpace::splitChannels(gpuOriginal, ch0, ch1, ch2);
  cudaDeviceSynchronize();

  // 验证单通道尺寸
  EXPECT_EQ(ch0.channels, 1);
  EXPECT_EQ(ch1.channels, 1);
  EXPECT_EQ(ch2.channels, 1);

  // 合并通道
  CudaImage merged;
  ColorSpace::mergeChannels(ch0, ch1, ch2, merged);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(merged);

  // 应该与原图完全相同
  EXPECT_EQ(original.data, result.data);
}

// 测试已知颜色的 HSV 转换
TEST_F(ColorSpaceTest, KnownColorHsv) {
  const int width = 1;
  const int height = 1;
  const int channels = 3;

  // 纯红色
  HostImage red = ImageUtils::createHostImage(width, height, channels);
  red.data[0] = 255; // R
  red.data[1] = 0;   // G
  red.data[2] = 0;   // B

  CudaImage gpuRed = ImageUtils::uploadToGpu(red);
  CudaImage hsvRed;
  ColorSpace::rgbToHsv(gpuRed, hsvRed);
  cudaDeviceSynchronize();

  HostImage hsvResult = ImageUtils::downloadFromGpu(hsvRed);

  // 红色的 H 应该接近 0，S 和 V 应该是 255
  EXPECT_NEAR(hsvResult.data[0], 0, 5);   // H ≈ 0
  EXPECT_NEAR(hsvResult.data[1], 255, 5); // S = 255
  EXPECT_NEAR(hsvResult.data[2], 255, 5); // V = 255
}

// 测试灰度图像的 YUV 转换
TEST_F(ColorSpaceTest, GrayYuv) {
  const int width = 4;
  const int height = 4;
  const int channels = 3;

  // 创建灰度图像（R=G=B）
  HostImage gray = ImageUtils::createHostImage(width, height, channels);
  for (int i = 0; i < width * height; ++i) {
    unsigned char val = static_cast<unsigned char>(i * 16);
    gray.data[i * 3] = val;
    gray.data[i * 3 + 1] = val;
    gray.data[i * 3 + 2] = val;
  }

  CudaImage gpuGray = ImageUtils::uploadToGpu(gray);
  CudaImage yuv;
  ColorSpace::rgbToYuv(gpuGray, yuv);
  cudaDeviceSynchronize();

  HostImage yuvResult = ImageUtils::downloadFromGpu(yuv);

  // 对于灰度图像，U 和 V 应该接近 128
  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(yuvResult.data[i * 3 + 1], 128, 5); // U ≈ 128
    EXPECT_NEAR(yuvResult.data[i * 3 + 2], 128, 5); // V ≈ 128
  }
}

// 测试无效输入
TEST_F(ColorSpaceTest, InvalidInput) {
  CudaImage invalid;
  CudaImage output;

  EXPECT_THROW(ColorSpace::rgbToHsv(invalid, output), std::invalid_argument);
  EXPECT_THROW(ColorSpace::rgbToYuv(invalid, output), std::invalid_argument);

  // 单通道图像
  HostImage gray = ImageUtils::createHostImage(32, 32, 1);
  CudaImage gpuGray = ImageUtils::uploadToGpu(gray);

  EXPECT_THROW(ColorSpace::rgbToHsv(gpuGray, output), std::invalid_argument);
}
