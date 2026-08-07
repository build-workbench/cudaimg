/**
 * HistogramCalculator 单元测试
 *
 * Feature: gpu-image-processing
 * Property 7: 直方图总和不变性
 */

#include "cudaimg/cudaimg.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace cudaimg;

class HistogramTest : public ::testing::Test {
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

// Property 7: 直方图总和不变性
// For any 图像，直方图所有 bin 的总和应等于像素数
TEST_F(HistogramTest, HistogramSumInvariant) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  auto histogram = HistogramCalculator::calculate(gpuInput);

  int sum = std::accumulate(histogram.begin(), histogram.end(), 0);
  int expectedPixels = width * height;

  EXPECT_EQ(sum, expectedPixels);
}

// 测试已知分布
TEST_F(HistogramTest, KnownDistribution) {
  const int width = 16;
  const int height = 16;
  const int channels = 1;

  HostImage input = ImageUtils::createHostImage(width, height, channels);

  // 创建已知分布：一半是 100，一半是 200
  for (int i = 0; i < 128; ++i) {
    input.data[i] = 100;
  }
  for (int i = 128; i < 256; ++i) {
    input.data[i] = 200;
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  auto histogram = HistogramCalculator::calculate(gpuInput);

  EXPECT_EQ(histogram[100], 128);
  EXPECT_EQ(histogram[200], 128);

  // 其他 bin 应该为 0
  for (int i = 0; i < 256; ++i) {
    if (i != 100 && i != 200) {
      EXPECT_EQ(histogram[i], 0) << "Non-zero count at bin " << i;
    }
  }
}

// 测试全黑图像
TEST_F(HistogramTest, AllBlackImage) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = ImageUtils::createHostImage(width, height, channels);
  std::fill(input.data.begin(), input.data.end(), 0);

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  auto histogram = HistogramCalculator::calculate(gpuInput);

  EXPECT_EQ(histogram[0], width * height);

  for (int i = 1; i < 256; ++i) {
    EXPECT_EQ(histogram[i], 0);
  }
}

// 测试全白图像
TEST_F(HistogramTest, AllWhiteImage) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = ImageUtils::createHostImage(width, height, channels);
  std::fill(input.data.begin(), input.data.end(), 255);

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  auto histogram = HistogramCalculator::calculate(gpuInput);

  EXPECT_EQ(histogram[255], width * height);

  for (int i = 0; i < 255; ++i) {
    EXPECT_EQ(histogram[i], 0);
  }
}

// 测试 RGB 直方图
TEST_F(HistogramTest, RGBHistogram) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage input = ImageUtils::createHostImage(width, height, channels);

  // 设置已知颜色
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      input.at(x, y, 0) = 50;  // R
      input.at(x, y, 1) = 100; // G
      input.at(x, y, 2) = 150; // B
    }
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  auto histograms = HistogramCalculator::calculateRGB(gpuInput);

  int totalPixels = width * height;

  // R 通道
  EXPECT_EQ(histograms[0][50], totalPixels);
  // G 通道
  EXPECT_EQ(histograms[1][100], totalPixels);
  // B 通道
  EXPECT_EQ(histograms[2][150], totalPixels);
}

// 测试单通道直方图
TEST_F(HistogramTest, SingleChannelHistogram) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  for (int c = 0; c < channels; ++c) {
    auto histogram = HistogramCalculator::calculateChannel(gpuInput, c);

    int sum = std::accumulate(histogram.begin(), histogram.end(), 0);
    EXPECT_EQ(sum, width * height) << "Channel " << c;
  }
}

// 测试直方图均衡化
TEST_F(HistogramTest, HistogramEqualization) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  // 创建低对比度图像
  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (size_t i = 0; i < input.data.size(); ++i) {
    input.data[i] = static_cast<unsigned char>(100 + (i % 50)); // 100-149
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage equalized;
  HistogramCalculator::equalize(gpuInput, equalized);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(equalized);

  // 均衡化后应该有更大的动态范围
  unsigned char minVal = 255, maxVal = 0;
  for (unsigned char v : result.data) {
    minVal = std::min(minVal, v);
    maxVal = std::max(maxVal, v);
  }

  // 原始范围是 50，均衡化后应该更大
  int originalRange = 50;
  int equalizedRange = maxVal - minVal;

  EXPECT_GT(equalizedRange, originalRange);
}

// 测试无效输入
TEST_F(HistogramTest, InvalidInput) {
  CudaImage invalid;

  EXPECT_THROW(HistogramCalculator::calculate(invalid), std::invalid_argument);
  EXPECT_THROW(HistogramCalculator::calculateRGB(invalid),
               std::invalid_argument);
}

// 测试大图像的直方图总和
TEST_F(HistogramTest, LargeImageHistogramSum) {
  const int width = 512;
  const int height = 512;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  auto histogram = HistogramCalculator::calculate(gpuInput);

  int sum = std::accumulate(histogram.begin(), histogram.end(), 0);
  int expectedPixels = width * height;

  EXPECT_EQ(sum, expectedPixels);
}

// 测试 RGB 图像的灰度直方图
TEST_F(HistogramTest, RGBToGrayscaleHistogram) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  auto histogram = HistogramCalculator::calculate(gpuInput);

  int sum = std::accumulate(histogram.begin(), histogram.end(), 0);
  int expectedPixels = width * height; // 不是 width * height * channels

  EXPECT_EQ(sum, expectedPixels);
}
