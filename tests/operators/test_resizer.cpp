/**
 * ImageResizer 单元测试
 *
 * Feature: gpu-image-processing
 * Property 8: 缩放操作近似可逆性
 */

#include "gpu_image/gpu_image_processing.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace gpu_image;

class ResizerTest : public ::testing::Test {
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

  // 计算 PSNR
  double calculatePSNR(const HostImage& img1, const HostImage& img2) {
    if (img1.data.size() != img2.data.size()) {
      return 0.0;
    }

    double mse = 0.0;
    for (size_t i = 0; i < img1.data.size(); ++i) {
      double diff = static_cast<double>(img1.data[i]) - img2.data[i];
      mse += diff * diff;
    }
    mse /= img1.data.size();

    if (mse == 0)
      return 100.0; // 完全相同

    return 10.0 * std::log10(255.0 * 255.0 / mse);
  }
};

// 测试基本缩放
TEST_F(ResizerTest, BasicResize) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  // 放大
  GpuImage enlarged;
  ImageResizer::resize(gpuInput, enlarged, 128, 128);
  cudaDeviceSynchronize();

  HostImage enlargedHost = ImageUtils::downloadFromGpu(enlarged);
  EXPECT_EQ(enlargedHost.width, 128);
  EXPECT_EQ(enlargedHost.height, 128);
  EXPECT_EQ(enlargedHost.channels, channels);

  // 缩小
  GpuImage shrunk;
  ImageResizer::resize(gpuInput, shrunk, 32, 32);
  cudaDeviceSynchronize();

  HostImage shrunkHost = ImageUtils::downloadFromGpu(shrunk);
  EXPECT_EQ(shrunkHost.width, 32);
  EXPECT_EQ(shrunkHost.height, 32);
  EXPECT_EQ(shrunkHost.channels, channels);
}

// Property 8: 缩放操作近似可逆性
// 放大再缩小应该与原图相似
TEST_F(ResizerTest, ApproximateRoundTrip) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage original = ImageUtils::createHostImage(width, height, channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      original.at(x, y, 0) = static_cast<unsigned char>(x * 255 / (width - 1));
      original.at(x, y, 1) = static_cast<unsigned char>(y * 255 / (height - 1));
      original.at(x, y, 2) =
          static_cast<unsigned char>((x + y) * 255 / (width + height - 2));
    }
  }
  GpuImage gpuOriginal = ImageUtils::uploadToGpu(original);

  // 放大 2x
  GpuImage enlarged;
  ImageResizer::resize(gpuOriginal, enlarged, width * 2, height * 2);
  cudaDeviceSynchronize();

  // 缩小回原尺寸
  GpuImage restored;
  ImageResizer::resize(enlarged, restored, width, height);
  cudaDeviceSynchronize();

  HostImage restoredHost = ImageUtils::downloadFromGpu(restored);

  // 计算 PSNR
  double psnr = calculatePSNR(original, restoredHost);

  // PSNR > 25dB 表示相似度较高
  EXPECT_GT(psnr, 25.0) << "PSNR: " << psnr << " dB";
}

// 测试按比例缩放
TEST_F(ResizerTest, ResizeByScale) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage scaled;
  ImageResizer::resizeByScale(gpuInput, scaled, 1.5f, 2.0f);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(scaled);

  EXPECT_EQ(result.width, 96);   // 64 * 1.5
  EXPECT_EQ(result.height, 128); // 64 * 2.0
}

// 测试保持宽高比缩放
TEST_F(ResizerTest, ResizeFit) {
  const int width = 100;
  const int height = 50;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage fitted;
  ImageResizer::resizeFit(gpuInput, fitted, 200, 200);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(fitted);

  // 应该保持 2:1 的宽高比
  // 最大宽度 200，所以高度应该是 100
  EXPECT_EQ(result.width, 200);
  EXPECT_EQ(result.height, 100);
}

// 测试最近邻插值
TEST_F(ResizerTest, NearestNeighborInterpolation) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage resized;
  ImageResizer::resize(gpuInput, resized, 64, 64,
                       InterpolationMode::NearestNeighbor);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(resized);

  EXPECT_EQ(result.width, 64);
  EXPECT_EQ(result.height, 64);

  // 最近邻插值应该保留原始像素值
  // 检查一些像素是否与原图中的某个像素完全相同
  bool foundMatch = false;
  for (size_t i = 0; i < result.data.size(); ++i) {
    for (size_t j = 0; j < input.data.size(); ++j) {
      if (result.data[i] == input.data[j]) {
        foundMatch = true;
        break;
      }
    }
    if (foundMatch)
      break;
  }
  EXPECT_TRUE(foundMatch);
}

// 测试 1x1 图像缩放
TEST_F(ResizerTest, SinglePixelResize) {
  const int channels = 3;

  HostImage input = ImageUtils::createHostImage(1, 1, channels);
  input.data[0] = 100;
  input.data[1] = 150;
  input.data[2] = 200;

  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage resized;
  ImageResizer::resize(gpuInput, resized, 10, 10);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(resized);

  // 所有像素应该与原始像素相同
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 10; ++x) {
      EXPECT_EQ(result.at(x, y, 0), 100);
      EXPECT_EQ(result.at(x, y, 1), 150);
      EXPECT_EQ(result.at(x, y, 2), 200);
    }
  }
}

// 测试无效参数
TEST_F(ResizerTest, InvalidParameters) {
  HostImage input = createTestImage(32, 32, 3);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);
  GpuImage output;

  // 无效尺寸
  EXPECT_THROW(ImageResizer::resize(gpuInput, output, 0, 32),
               std::invalid_argument);
  EXPECT_THROW(ImageResizer::resize(gpuInput, output, 32, 0),
               std::invalid_argument);
  EXPECT_THROW(ImageResizer::resize(gpuInput, output, -1, 32),
               std::invalid_argument);

  // 无效缩放比例
  EXPECT_THROW(ImageResizer::resizeByScale(gpuInput, output, 0.0f, 1.0f),
               std::invalid_argument);
  EXPECT_THROW(ImageResizer::resizeByScale(gpuInput, output, -1.0f, 1.0f),
               std::invalid_argument);

  // 无效图像
  GpuImage invalid;
  EXPECT_THROW(ImageResizer::resize(invalid, output, 32, 32),
               std::invalid_argument);
}

// 测试大图像缩放
TEST_F(ResizerTest, LargeImageResize) {
  const int width = 512;
  const int height = 512;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage resized;
  ImageResizer::resize(gpuInput, resized, 1024, 1024);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(resized);

  EXPECT_EQ(result.width, 1024);
  EXPECT_EQ(result.height, 1024);

  // 验证内核确实产生了内容（输出并非全零）。
  // 原先对 unsigned char 的 EXPECT_GE(v,0)/EXPECT_LE(v,255) 是恒真空断言。
  EXPECT_TRUE(std::any_of(result.data.begin(), result.data.end(),
                          [](unsigned char v) { return v != 0; }));
}

// 测试非整数缩放比例
TEST_F(ResizerTest, NonIntegerScale) {
  const int width = 100;
  const int height = 100;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  GpuImage gpuInput = ImageUtils::uploadToGpu(input);

  GpuImage resized;
  ImageResizer::resize(gpuInput, resized, 73, 91); // 非整数比例
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(resized);

  EXPECT_EQ(result.width, 73);
  EXPECT_EQ(result.height, 91);
}
