#include "cudaimg/core/image.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/filters.hpp"
#include <gtest/gtest.h>
#include <algorithm>

using namespace cudaimg;

class FiltersTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    testImage = ImageUtils::createCudaImage(16, 16, 1);
    std::vector<unsigned char> data(256);
    for (int i = 0; i < 256; ++i) {
      data[i] = static_cast<unsigned char>(i);
    }
    cudaMemcpy(testImage.buffer.data(), data.data(), 256,
               cudaMemcpyHostToDevice);
  }

  CudaImage testImage;
};

class ArithmeticTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    image1 = ImageUtils::createCudaImage(8, 8, 1);
    image2 = ImageUtils::createCudaImage(8, 8, 1);

    std::vector<unsigned char> data1(64, 100);
    std::vector<unsigned char> data2(64, 50);

    cudaMemcpy(image1.buffer.data(), data1.data(), 64, cudaMemcpyHostToDevice);
    cudaMemcpy(image2.buffer.data(), data2.data(), 64, cudaMemcpyHostToDevice);
  }

  CudaImage image1, image2;
};

TEST_F(FiltersTest, MedianFilter) {
  CudaImage output;
  Filters::medianFilter(testImage, output, 3);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, testImage.width);
  EXPECT_EQ(output.height, testImage.height);
  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, MedianFilter_5x5) {
  CudaImage output;
  Filters::medianFilter(testImage, output, 5);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, BilateralFilter) {
  CudaImage output;
  Filters::bilateralFilter(testImage, output, 5, 10.0f, 50.0f);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, testImage.width);
  EXPECT_EQ(output.height, testImage.height);
}

TEST_F(FiltersTest, BoxFilter) {
  CudaImage output;
  Filters::boxFilter(testImage, output, 3);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, Sharpen) {
  CudaImage output;
  Filters::sharpen(testImage, output, 1.0f);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, Laplacian) {
  CudaImage output;
  Filters::laplacian(testImage, output);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, InvalidKernelSize) {
  CudaImage output;
  EXPECT_THROW(Filters::medianFilter(testImage, output, 2),
               std::invalid_argument);
  EXPECT_THROW(Filters::medianFilter(testImage, output, 8),
               std::invalid_argument);
}

TEST_F(FiltersTest, BilateralFilterRejectsInvalidSigma) {
  CudaImage output;
  EXPECT_THROW(Filters::bilateralFilter(testImage, output, 5, 0.0f, 50.0f),
               std::invalid_argument);
  EXPECT_THROW(Filters::bilateralFilter(testImage, output, 5, -1.0f, 50.0f),
               std::invalid_argument);
  EXPECT_THROW(Filters::bilateralFilter(testImage, output, 5, 10.0f, 0.0f),
               std::invalid_argument);
  EXPECT_THROW(Filters::bilateralFilter(testImage, output, 5, 10.0f, -1.0f),
               std::invalid_argument);
}

TEST_F(ArithmeticTest, Add) {
  CudaImage output;
  ImageArithmetic::add(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 150); // 100 + 50
}


TEST_F(ArithmeticTest, Blend) {
  CudaImage output;
  ImageArithmetic::blend(image1, image2, output, 0.5f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 75); // 0.5*100 + 0.5*50
}


TEST_F(ArithmeticTest, AddScalar) {
  CudaImage output;
  ImageArithmetic::addScalar(image1, output, 20);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 120); // 100 + 20
}



TEST_F(ArithmeticTest, DimensionMismatch) {
  CudaImage small = ImageUtils::createCudaImage(4, 4, 1);
  CudaImage output;

  EXPECT_THROW(ImageArithmetic::add(image1, small, output),
               std::invalid_argument);
}

// ===== CPU 参考实现测试 =====

class FiltersCpuRefTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  HostImage createRandomImage(int width, int height, int channels) {
    HostImage image = ImageUtils::createHostImage(width, height, channels);
    unsigned int seed = 42;
    for (size_t i = 0; i < image.data.size(); ++i) {
      seed = seed * 1103515245 + 12345;
      image.data[i] = static_cast<unsigned char>((seed >> 16) & 0xFF);
    }
    return image;
  }

  // CPU 盒式滤波参考
  void cpuBoxFilter(const HostImage& input, HostImage& output, int kernelSize) {
    int radius = kernelSize / 2;
    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        for (int c = 0; c < input.channels; ++c) {
          float sum = 0.0f;
          for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
              int nx = std::max(0, std::min(x + kx, input.width - 1));
              int ny = std::max(0, std::min(y + ky, input.height - 1));
              sum += input.at(nx, ny, c);
            }
          }
          output.at(x, y, c) =
              static_cast<unsigned char>(sum / (kernelSize * kernelSize) + 0.5f);
        }
      }
    }
  }

  // CPU 中值滤波参考
  void cpuMedianFilter(const HostImage& input, HostImage& output, int kernelSize) {
    int radius = kernelSize / 2;
    std::vector<unsigned char> window(kernelSize * kernelSize);
    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        for (int c = 0; c < input.channels; ++c) {
          int count = 0;
          for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
              int nx = std::max(0, std::min(x + kx, input.width - 1));
              int ny = std::max(0, std::min(y + ky, input.height - 1));
              window[count++] = input.at(nx, ny, c);
            }
          }
          std::sort(window.begin(), window.begin() + count);
          output.at(x, y, c) = window[count / 2];
        }
      }
    }
  }

  // CPU 锐化参考
  void cpuSharpen(const HostImage& input, HostImage& output, float strength) {
    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        for (int c = 0; c < input.channels; ++c) {
          float center = input.at(x, y, c);
          float sum = center * (1.0f + 4.0f * strength);
          if (x > 0) sum -= strength * input.at(x - 1, y, c);
          if (x < input.width - 1) sum -= strength * input.at(x + 1, y, c);
          if (y > 0) sum -= strength * input.at(x, y - 1, c);
          if (y < input.height - 1) sum -= strength * input.at(x, y + 1, c);
          sum = std::max(0.0f, std::min(255.0f, sum));
          output.at(x, y, c) = static_cast<unsigned char>(sum);
        }
      }
    }
  }
};

TEST_F(FiltersCpuRefTest, BoxFilterMatchesCpu) {
  HostImage input = createRandomImage(32, 32, 1);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage gpuOutput;
  Filters::boxFilter(gpuInput, gpuOutput, 3);
  cudaDeviceSynchronize();

  HostImage gpuResult = ImageUtils::downloadFromGpu(gpuOutput);
  HostImage cpuResult = ImageUtils::createHostImage(32, 32, 1);
  cpuBoxFilter(input, cpuResult, 3);

  for (size_t i = 0; i < gpuResult.data.size(); ++i) {
    EXPECT_NEAR(gpuResult.data[i], cpuResult.data[i], 1)
        << "Mismatch at index " << i;
  }
}

TEST_F(FiltersCpuRefTest, MedianFilterMatchesCpu) {
  HostImage input = createRandomImage(32, 32, 1);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage gpuOutput;
  Filters::medianFilter(gpuInput, gpuOutput, 3);
  cudaDeviceSynchronize();

  HostImage gpuResult = ImageUtils::downloadFromGpu(gpuOutput);
  HostImage cpuResult = ImageUtils::createHostImage(32, 32, 1);
  cpuMedianFilter(input, cpuResult, 3);

  for (size_t i = 0; i < gpuResult.data.size(); ++i) {
    EXPECT_EQ(gpuResult.data[i], cpuResult.data[i])
        << "Mismatch at index " << i;
  }
}

TEST_F(FiltersCpuRefTest, SharpenMatchesCpu) {
  HostImage input = createRandomImage(32, 32, 1);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  CudaImage gpuOutput;
  Filters::sharpen(gpuInput, gpuOutput, 1.0f);
  cudaDeviceSynchronize();

  HostImage gpuResult = ImageUtils::downloadFromGpu(gpuOutput);
  HostImage cpuResult = ImageUtils::createHostImage(32, 32, 1);
  cpuSharpen(input, cpuResult, 1.0f);

  for (size_t i = 0; i < gpuResult.data.size(); ++i) {
    EXPECT_NEAR(gpuResult.data[i], cpuResult.data[i], 1)
        << "Mismatch at index " << i;
  }
}
