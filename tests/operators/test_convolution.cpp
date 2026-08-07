/**
 * ConvolutionEngine 单元测试
 *
 * Feature: gpu-image-processing
 * Property 5: 卷积操作与参考实现一致性
 * Property 6: 边界处理正确性
 */

#include "cudaimg/cudaimg.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace cudaimg;

class ConvolutionTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  // CPU 参考实现
  void cpuConvolve(const HostImage& input, HostImage& output,
                   const std::vector<float>& kernel, int kernelSize) {
    int half = kernelSize / 2;

    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        for (int c = 0; c < input.channels; ++c) {
          float sum = 0.0f;

          for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
              int srcX = x + kx - half;
              int srcY = y + ky - half;

              float value = 0.0f;
              if (srcX >= 0 && srcX < input.width && srcY >= 0 &&
                  srcY < input.height) {
                value = input.at(srcX, srcY, c);
              }

              sum += value * kernel[ky * kernelSize + kx];
            }
          }

          sum = std::max(0.0f, std::min(255.0f, sum));
          output.at(x, y, c) = static_cast<unsigned char>(sum + 0.5f);
        }
      }
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

// 测试高斯核生成
TEST_F(ConvolutionTest, GaussianKernelGeneration) {
  auto kernel = ConvolutionEngine::generateGaussianKernel(3, 1.0f);

  EXPECT_EQ(kernel.size(), 9u);

  // 验证归一化（总和应为 1）
  float sum = 0.0f;
  for (float v : kernel) {
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0f, 0.001f);

  // 验证对称性
  EXPECT_NEAR(kernel[0], kernel[2], 0.001f); // 左上 = 右上
  EXPECT_NEAR(kernel[0], kernel[6], 0.001f); // 左上 = 左下
  EXPECT_NEAR(kernel[1], kernel[7], 0.001f); // 上中 = 下中

  // 中心应该是最大值
  EXPECT_GT(kernel[4], kernel[0]);
  EXPECT_GT(kernel[4], kernel[1]);
}

// Property 5: 卷积操作与参考实现一致性
TEST_F(ConvolutionTest, ConvolveMatchesCPU) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;
  const int kernelSize = 3;

  HostImage input = createTestImage(width, height, channels);

  // 简单的均值滤波核
  std::vector<float> kernel(kernelSize * kernelSize, 1.0f / 9.0f);

  // CPU 参考实现
  HostImage cpuOutput = ImageUtils::createHostImage(width, height, channels);
  cpuConvolve(input, cpuOutput, kernel, kernelSize);

  // GPU 实现
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage gpuOutput;
  ConvolutionEngine::convolve(gpuInput, gpuOutput, kernel.data(), kernelSize);
  cudaDeviceSynchronize();
  HostImage gpuResult = ImageUtils::downloadFromGpu(gpuOutput);

  // 比较结果（允许 ±1 的舍入误差）
  int maxDiff = 0;
  for (size_t i = 0; i < cpuOutput.data.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpuOutput.data[i]) -
                        static_cast<int>(gpuResult.data[i]));
    maxDiff = std::max(maxDiff, diff);
  }

  EXPECT_LE(maxDiff, 1) << "Max difference: " << maxDiff;
}

// 测试高斯模糊
TEST_F(ConvolutionTest, GaussianBlur) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage blurred;
  ConvolutionEngine::gaussianBlur(gpuInput, blurred, 5, 1.5f);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(blurred);

  // 验证输出尺寸
  EXPECT_EQ(result.width, width);
  EXPECT_EQ(result.height, height);
  EXPECT_EQ(result.channels, channels);

  // 验证内核确实产生了内容（输出并非全零）。
  // 原先对 unsigned char 的 EXPECT_GE(v,0)/EXPECT_LE(v,255) 是恒真空断言。
  EXPECT_TRUE(std::any_of(result.data.begin(), result.data.end(),
                          [](unsigned char v) { return v != 0; }));
}

// 测试 Sobel 边缘检测
TEST_F(ConvolutionTest, SobelEdgeDetection) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  // 创建有明显边缘的图像
  HostImage input = ImageUtils::createHostImage(width, height, channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      unsigned char value = (x < width / 2) ? 50 : 200;
      for (int c = 0; c < channels; ++c) {
        input.at(x, y, c) = value;
      }
    }
  }

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage edges;
  ConvolutionEngine::sobelEdgeDetection(gpuInput, edges);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(edges);

  // 输出应该是单通道
  EXPECT_EQ(result.channels, 1);

  // 边缘处（中间列）应该有较高的值
  int edgeX = width / 2;
  int nonEdgeX = width / 4;

  // 计算边缘区域和非边缘区域的平均值
  float edgeAvg = 0, nonEdgeAvg = 0;
  int edgeCount = 0, nonEdgeCount = 0;

  for (int y = 10; y < height - 10; ++y) {
    edgeAvg += result.at(edgeX, y, 0);
    edgeCount++;
    nonEdgeAvg += result.at(nonEdgeX, y, 0);
    nonEdgeCount++;
  }

  edgeAvg /= edgeCount;
  nonEdgeAvg /= nonEdgeCount;

  // 边缘区域应该比非边缘区域亮
  EXPECT_GT(edgeAvg, nonEdgeAvg);
}

// Property 6: 边界处理正确性（零填充）
TEST_F(ConvolutionTest, ZeroPaddingBoundary) {
  const int width = 8;
  const int height = 8;
  const int channels = 1;

  // 创建全白图像
  HostImage input = ImageUtils::createHostImage(width, height, channels);
  std::fill(input.data.begin(), input.data.end(), 255);

  // 使用 3x3 均值滤波
  std::vector<float> kernel(9, 1.0f / 9.0f);

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;
  ConvolutionEngine::convolve(gpuInput, output, kernel.data(), 3,
                              BorderMode::Zero);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 角落像素应该比中心像素暗（因为零填充）
  // 角落只有 4 个有效邻居，中心有 9 个
  unsigned char corner = result.at(0, 0, 0);
  unsigned char center = result.at(width / 2, height / 2, 0);

  EXPECT_LT(corner, center);
}

// 测试无效参数
TEST_F(ConvolutionTest, InvalidParameters) {
  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;

  std::vector<float> kernel(9, 1.0f / 9.0f);

  // 偶数核大小
  EXPECT_THROW(ConvolutionEngine::convolve(gpuInput, output, kernel.data(), 4),
               std::invalid_argument);

  // 核大小太大
  EXPECT_THROW(ConvolutionEngine::convolve(gpuInput, output, kernel.data(), 9),
               std::invalid_argument);

  // 空指针
  EXPECT_THROW(ConvolutionEngine::convolve(gpuInput, output, nullptr, 3),
               std::invalid_argument);

  // 无效图像
  CudaImage invalid;
  EXPECT_THROW(ConvolutionEngine::convolve(invalid, output, kernel.data(), 3),
               std::invalid_argument);
}

// 测试恒等卷积
TEST_F(ConvolutionTest, IdentityConvolution) {
  const int width = 32;
  const int height = 32;
  const int channels = 1;

  HostImage input = createTestImage(width, height, channels);

  // 恒等核
  std::vector<float> kernel = {0, 0, 0, 0, 1, 0, 0, 0, 0};

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;
  ConvolutionEngine::convolve(gpuInput, output, kernel.data(), 3);
  cudaDeviceSynchronize();

  HostImage result = ImageUtils::downloadFromGpu(output);

  // 内部像素应该不变（边界可能不同）
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      EXPECT_EQ(input.at(x, y, 0), result.at(x, y, 0))
          << "Mismatch at (" << x << ", " << y << ")";
    }
  }
}

TEST_F(ConvolutionTest, InvalidGaussianSigma) {
  HostImage input = createTestImage(32, 32, 1);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage output;

  EXPECT_THROW(ConvolutionEngine::gaussianBlur(gpuInput, output, 3, 0.0f),
               std::invalid_argument);
  EXPECT_THROW(ConvolutionEngine::gaussianBlur(gpuInput, output, 3, -1.0f),
               std::invalid_argument);
  EXPECT_THROW(ConvolutionEngine::generateGaussianKernel(3, 0.0f),
               std::invalid_argument);
  EXPECT_THROW(ConvolutionEngine::generateGaussianKernel1D(3, -1.0f),
               std::invalid_argument);
}

TEST_F(ConvolutionTest, ConcurrentDifferentKernelsDoNotInterfere) {
  HostImage input = ImageUtils::createHostImage(32, 32, 1);
  std::fill(input.data.begin(), input.data.end(), 0);
  input.at(16, 16, 0) = 255;

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage identityOutput;
  CudaImage blurOutput;

  std::vector<float> identityKernel = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<float> blurKernel(9, 1.0f / 9.0f);

  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  ConvolutionEngine::convolve(gpuInput, identityOutput, identityKernel.data(),
                              3, BorderMode::Zero, stream1);
  ConvolutionEngine::convolve(gpuInput, blurOutput, blurKernel.data(), 3,
                              BorderMode::Zero, stream2);

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));

  HostImage identityResult = ImageUtils::downloadFromGpu(identityOutput);
  HostImage blurResult = ImageUtils::downloadFromGpu(blurOutput);

  EXPECT_EQ(identityResult.at(16, 16, 0), 255);
  EXPECT_EQ(identityResult.at(16, 15, 0), 0);
  EXPECT_EQ(blurResult.at(16, 16, 0), 28);
  EXPECT_EQ(blurResult.at(16, 15, 0), 28);
}

TEST_F(ConvolutionTest, ConcurrentSeparableConvolutionsDoNotInterfere) {
  HostImage input = ImageUtils::createHostImage(32, 32, 1);
  std::fill(input.data.begin(), input.data.end(), 0);
  input.at(16, 16, 0) = 255;

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage horizontalOutput;
  CudaImage verticalOutput;

  std::vector<float> horizontal = {0.25f, 0.5f, 0.25f};
  std::vector<float> identity = {0.0f, 1.0f, 0.0f};

  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  ConvolutionEngine::separableConvolve(gpuInput, horizontalOutput,
                                       horizontal.data(), identity.data(), 3,
                                       stream1);
  ConvolutionEngine::separableConvolve(
      gpuInput, verticalOutput, identity.data(), horizontal.data(), 3, stream2);

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));

  HostImage horizontalResult = ImageUtils::downloadFromGpu(horizontalOutput);
  HostImage verticalResult = ImageUtils::downloadFromGpu(verticalOutput);

  EXPECT_EQ(horizontalResult.at(15, 16, 0), 64);
  EXPECT_EQ(horizontalResult.at(16, 16, 0), 128);
  EXPECT_EQ(horizontalResult.at(17, 16, 0), 64);
  EXPECT_EQ(horizontalResult.at(16, 15, 0), 0);

  EXPECT_EQ(verticalResult.at(16, 15, 0), 64);
  EXPECT_EQ(verticalResult.at(16, 16, 0), 128);
  EXPECT_EQ(verticalResult.at(16, 17, 0), 64);
  EXPECT_EQ(verticalResult.at(15, 16, 0), 0);
}

TEST_F(ConvolutionTest, ConvolutionSimpleKernelMatchesReference) {
  HostImage input = createTestImage(16, 16, 1);
  std::vector<float> kernel = {0, 0, 0, 0, 1, 0, 0, 0, 0};

  HostImage cpuOutput = ImageUtils::createHostImage(16, 16, 1);
  cpuConvolve(input, cpuOutput, kernel, 3);

  CudaImage gpuInput = ImageUtils::uploadToGpu(input);
  CudaImage gpuOutput;
  ConvolutionEngine::convolve(gpuInput, gpuOutput, kernel.data(), 3);
  cudaDeviceSynchronize();

  HostImage gpuResult = ImageUtils::downloadFromGpu(gpuOutput);
  EXPECT_EQ(cpuOutput.data, gpuResult.data);
}
