/**
 * Morphology 单元测试
 */

#include "cudaimg/cudaimg.hpp"
#include <gtest/gtest.h>

using namespace cudaimg;

class MorphologyTest : public ::testing::Test {
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

// 测试腐蚀操作
TEST_F(MorphologyTest, Erode) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Morphology::erode(gpuInput, output, 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  EXPECT_EQ(result.width, width);
  EXPECT_EQ(result.height, height);

  // 腐蚀后的值应该 <= 原值
  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_LE(result.data[i], input.data[i]);
  }
}

// 测试膨胀操作
TEST_F(MorphologyTest, Dilate) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Morphology::dilate(gpuInput, output, 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 膨胀后的值应该 >= 原值
  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_GE(result.data[i], input.data[i]);
  }
}

// 测试开运算
TEST_F(MorphologyTest, Open) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Morphology::open(gpuInput, output, 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  EXPECT_EQ(result.width, width);
  EXPECT_EQ(result.height, height);
}

// 测试闭运算
TEST_F(MorphologyTest, Close) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Morphology::close(gpuInput, output, 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  EXPECT_EQ(result.width, width);
  EXPECT_EQ(result.height, height);
}

// 测试形态学梯度
TEST_F(MorphologyTest, Gradient) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Morphology::gradient(gpuInput, output, 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  EXPECT_EQ(result.width, width);
  EXPECT_EQ(result.height, height);
}

// 测试不同结构元素
TEST_F(MorphologyTest, DifferentStructuringElements) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage rectOutput, crossOutput, ellipseOutput;

  Morphology::erode(gpuInput, rectOutput, 3, StructuringElement::Rectangle);
  Morphology::erode(gpuInput, crossOutput, 3, StructuringElement::Cross);
  Morphology::erode(gpuInput, ellipseOutput, 3, StructuringElement::Ellipse);

  cudaDeviceSynchronize();

  HostImage rectResult = ImageUtils::downloadFromGpu(rectOutput);
  HostImage crossResult = ImageUtils::downloadFromGpu(crossOutput);
  HostImage ellipseResult = ImageUtils::downloadFromGpu(ellipseOutput);

  // 不同结构元素应该产生不同结果
  bool allSame = true;
  for (size_t i = 0; i < rectResult.data.size(); ++i) {
    if (rectResult.data[i] != crossResult.data[i]) {
      allSame = false;
      break;
    }
  }

  // 矩形和十字形结果应该不同
  EXPECT_FALSE(allSame);
}
