/**
 * Threshold 单元测试
 */

#include "cudaimg/cudaimg.hpp"
#include <gtest/gtest.h>

using namespace cudaimg;

class ThresholdTest : public ::testing::Test {
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

// 测试二值化
TEST_F(ThresholdTest, BinaryThreshold) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (int i = 0; i < 512; ++i) {
    input.data[i] = static_cast<unsigned char>(i / 2); // 0-255
  }
  for (int i = 512; i < 1024; ++i) {
    input.data[i] = static_cast<unsigned char>((i - 512) / 2);
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;

  Threshold::threshold(gpuInput, output, 128, 255, ThresholdType::Binary);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 验证二值化结果
  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_TRUE(result.data[i] == 0 || result.data[i] == 255)
        << "Non-binary value at index " << i << ": "
        << static_cast<int>(result.data[i]);
  }
}

// 测试反二值化
TEST_F(ThresholdTest, BinaryInvThreshold) {
  const int width = 16;
  const int height = 16;
  const int channels = 1;

  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (size_t i = 0; i < input.data.size(); ++i) {
    input.data[i] = static_cast<unsigned char>(i);
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage binaryOutput, binaryInvOutput;

  Threshold::threshold(gpuInput, binaryOutput, 128, 255, ThresholdType::Binary);
  Threshold::threshold(gpuInput, binaryInvOutput, 128, 255,
                       ThresholdType::BinaryInv);
  cudaDeviceSynchronize();

  HostImage binaryResult = ImageUtils::downloadFromGpu(binaryOutput);
  HostImage binaryInvResult = ImageUtils::downloadFromGpu(binaryInvOutput);

  // Binary 和 BinaryInv 应该互补
  for (size_t i = 0; i < binaryResult.data.size(); ++i) {
    EXPECT_EQ(binaryResult.data[i] + binaryInvResult.data[i], 255);
  }
}

// 测试截断阈值
TEST_F(ThresholdTest, TruncateThreshold) {
  const int width = 16;
  const int height = 16;
  const int channels = 1;
  const unsigned char thresh = 100;

  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (size_t i = 0; i < input.data.size(); ++i) {
    input.data[i] = static_cast<unsigned char>(i);
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;

  Threshold::threshold(gpuInput, output, thresh, 255, ThresholdType::Truncate);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  for (size_t i = 0; i < result.data.size(); ++i) {
    if (input.data[i] > thresh) {
      EXPECT_EQ(result.data[i], thresh);
    } else {
      EXPECT_EQ(result.data[i], input.data[i]);
    }
  }
}

// 测试 Otsu 阈值
TEST_F(ThresholdTest, OtsuThreshold) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  // 创建双峰分布图像
  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // 左半部分暗，右半部分亮
      input.at(x, y, 0) = (x < width / 2) ? 50 : 200;
    }
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  unsigned char otsuThresh = Threshold::otsuThreshold(gpuInput);

  // 对这种双峰分布，Otsu 可能返回平台区间内任一点，包含左峰端点
  EXPECT_GE(otsuThresh, 50);
  EXPECT_LT(otsuThresh, 200);
}

// 测试 Otsu 二值化
TEST_F(ThresholdTest, OtsuBinarize) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Threshold::otsuBinarize(gpuInput, output);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 验证结果是二值的
  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_TRUE(result.data[i] == 0 || result.data[i] == 255);
  }
}

// 测试自适应阈值
TEST_F(ThresholdTest, AdaptiveThreshold) {
  const int width = 64;
  const int height = 64;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage output;
  Threshold::adaptiveThreshold(gpuInput, output, 255, AdaptiveMethod::MeanC,
                               ThresholdType::Binary, 11, 2);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 验证结果是二值的
  for (size_t i = 0; i < result.data.size(); ++i) {
    EXPECT_TRUE(result.data[i] == 0 || result.data[i] == 255);
  }
}

// 测试无效参数
TEST_F(ThresholdTest, InvalidParameters) {
  CudaImage invalid;
  CudaImage output;

  EXPECT_THROW(Threshold::threshold(invalid, output, 128),
               std::invalid_argument);

  HostImage input = createTestImage(32, 32, 1);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  // 偶数 blockSize
  EXPECT_THROW(Threshold::adaptiveThreshold(gpuInput, output, 255,
                                            AdaptiveMethod::MeanC,
                                            ThresholdType::Binary, 10, 2),
               std::invalid_argument);
}
