#include "cudaimg/core/image.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/filters.hpp"
#include <gtest/gtest.h>

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

TEST_F(ArithmeticTest, Subtract) {
  CudaImage output;
  ImageArithmetic::subtract(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 50); // 100 - 50
}

TEST_F(ArithmeticTest, Blend) {
  CudaImage output;
  ImageArithmetic::blend(image1, image2, output, 0.5f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 75); // 0.5*100 + 0.5*50
}

TEST_F(ArithmeticTest, AbsDiff) {
  CudaImage output;
  ImageArithmetic::absDiff(image1, image2, output);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 50); // |100 - 50|
}

TEST_F(ArithmeticTest, AddScalar) {
  CudaImage output;
  ImageArithmetic::addScalar(image1, output, 20);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 120); // 100 + 20
}

TEST_F(ArithmeticTest, MultiplyScalar) {
  CudaImage output;
  ImageArithmetic::multiplyScalar(image1, output, 2.0f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  EXPECT_EQ(result[0], 200); // 100 * 2
}

TEST_F(ArithmeticTest, AddWeighted) {
  CudaImage output;
  ImageArithmetic::addWeighted(image1, 0.7f, image2, 0.3f, output, 10.0f);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  // 0.7*100 + 0.3*50 + 10 = 70 + 15 + 10 = 95
  EXPECT_EQ(result[0], 95);
}

TEST_F(ArithmeticTest, DimensionMismatch) {
  CudaImage small = ImageUtils::createCudaImage(4, 4, 1);
  CudaImage output;

  EXPECT_THROW(ImageArithmetic::add(image1, small, output),
               std::invalid_argument);
}
