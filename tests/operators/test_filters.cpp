#include "gpu_image/core/gpu_image.hpp"
#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/operators/filters.hpp"
#include <gtest/gtest.h>

using namespace gpu_image;

class FiltersTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    testImage = ImageUtils::createGpuImage(16, 16, 1);
    std::vector<unsigned char> data(256);
    for (int i = 0; i < 256; ++i) {
      data[i] = static_cast<unsigned char>(i);
    }
    cudaMemcpy(testImage.buffer.data(), data.data(), 256,
               cudaMemcpyHostToDevice);
  }

  GpuImage testImage;
};

class ArithmeticTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    image1 = ImageUtils::createGpuImage(8, 8, 1);
    image2 = ImageUtils::createGpuImage(8, 8, 1);

    std::vector<unsigned char> data1(64, 100);
    std::vector<unsigned char> data2(64, 50);

    cudaMemcpy(image1.buffer.data(), data1.data(), 64, cudaMemcpyHostToDevice);
    cudaMemcpy(image2.buffer.data(), data2.data(), 64, cudaMemcpyHostToDevice);
  }

  GpuImage image1, image2;
};

TEST_F(FiltersTest, MedianFilter) {
  GpuImage output;
  Filters::medianFilter(testImage, output, 3);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, testImage.width);
  EXPECT_EQ(output.height, testImage.height);
  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, MedianFilter_5x5) {
  GpuImage output;
  Filters::medianFilter(testImage, output, 5);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, BilateralFilter) {
  GpuImage output;
  Filters::bilateralFilter(testImage, output, 5, 10.0f, 50.0f);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, testImage.width);
  EXPECT_EQ(output.height, testImage.height);
}

TEST_F(FiltersTest, BoxFilter) {
  GpuImage output;
  Filters::boxFilter(testImage, output, 3);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, Sharpen) {
  GpuImage output;
  Filters::sharpen(testImage, output, 1.0f);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, Laplacian) {
  GpuImage output;
  Filters::laplacian(testImage, output);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
}

TEST_F(FiltersTest, InvalidKernelSize) {
  GpuImage output;
  EXPECT_THROW(Filters::medianFilter(testImage, output, 2),
               std::invalid_argument);
  EXPECT_THROW(Filters::medianFilter(testImage, output, 8),
               std::invalid_argument);
}

TEST_F(FiltersTest, BilateralFilterRejectsInvalidSigma) {
  GpuImage output;
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
  GpuImage output;
  ImageArithmetic::add(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 150); // 100 + 50
}

TEST_F(ArithmeticTest, Subtract) {
  GpuImage output;
  ImageArithmetic::subtract(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 50); // 100 - 50
}

TEST_F(ArithmeticTest, Blend) {
  GpuImage output;
  ImageArithmetic::blend(image1, image2, output, 0.5f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 75); // 0.5*100 + 0.5*50
}

TEST_F(ArithmeticTest, AbsDiff) {
  GpuImage output;
  ImageArithmetic::absDiff(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 50); // |100 - 50|
}

TEST_F(ArithmeticTest, AddScalar) {
  GpuImage output;
  ImageArithmetic::addScalar(image1, output, 20);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 120); // 100 + 20
}

TEST_F(ArithmeticTest, MultiplyScalar) {
  GpuImage output;
  ImageArithmetic::multiplyScalar(image1, output, 2.0f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 200); // 100 * 2
}

TEST_F(ArithmeticTest, AddWeighted) {
  GpuImage output;
  ImageArithmetic::addWeighted(image1, 0.7f, image2, 0.3f, output, 10.0f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  // 0.7*100 + 0.3*50 + 10 = 70 + 15 + 10 = 95
  EXPECT_EQ(result[0], 95);
}

TEST_F(ArithmeticTest, DimensionMismatch) {
  GpuImage small = ImageUtils::createGpuImage(4, 4, 1);
  GpuImage output;

  EXPECT_THROW(ImageArithmetic::add(image1, small, output),
               std::invalid_argument);
}
