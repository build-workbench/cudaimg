#include "cudaimg/core/image.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/geometric.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace cudaimg;

namespace {
std::vector<unsigned char> downloadPixels(const CudaImage& image) {
  std::vector<unsigned char> pixels(image.totalBytes());
  cudaMemcpy(pixels.data(), image.buffer.data(), image.totalBytes(),
             cudaMemcpyDeviceToHost);
  return pixels;
}
} // namespace

class GeometricTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }

    // 创建 8x8 测试图像
    testImage = ImageUtils::createCudaImage(8, 8, 1);
    std::vector<unsigned char> data(64);
    for (int i = 0; i < 64; ++i) {
      data[i] = static_cast<unsigned char>(i * 4);
    }
    cudaMemcpy(testImage.buffer.data(), data.data(), 64,
               cudaMemcpyHostToDevice);
  }

  CudaImage testImage;
};

TEST_F(GeometricTest, Rotate90_Once) {
  CudaImage output;
  Geometric::rotate90(testImage, output, 1);
  cudaDeviceSynchronize();

  ASSERT_EQ(output.width, 8);
  ASSERT_EQ(output.height, 8);
  ASSERT_TRUE(output.isValid());

  // 顺时针 90°：dst(x, y) = src(7 - x, y) 的转置映射
  auto result = downloadPixels(output);
  EXPECT_EQ(result[0 * 8 + 0], 224); // src(0, 7) = 56 * 4
  EXPECT_EQ(result[0 * 8 + 7], 0);   // src(0, 0)
  EXPECT_EQ(result[7 * 8 + 0], 252); // src(7, 7) = 63 * 4
  EXPECT_EQ(result[7 * 8 + 7], 28);  // src(7, 0) = 7 * 4
}

TEST_F(GeometricTest, Rotate90_Twice) {
  CudaImage output;
  Geometric::rotate90(testImage, output, 2);
  cudaDeviceSynchronize();

  ASSERT_EQ(output.width, 8);
  ASSERT_EQ(output.height, 8);

  // 180°：dst(x, y) = src(7 - x, 7 - y)
  auto result = downloadPixels(output);
  EXPECT_EQ(result[0], 252); // src(7, 7)
  EXPECT_EQ(result[63], 0);  // src(0, 0)
  EXPECT_EQ(result[7], 224); // src(0, 7)
  EXPECT_EQ(result[56], 28); // src(7, 0)
}

TEST_F(GeometricTest, Rotate90_Thrice) {
  CudaImage output;
  Geometric::rotate90(testImage, output, 3);
  cudaDeviceSynchronize();

  ASSERT_EQ(output.width, 8);
  ASSERT_EQ(output.height, 8);

  // 顺时针 270°：dst(x, y) = src(y, 7 - x) 的逆向映射
  auto result = downloadPixels(output);
  EXPECT_EQ(result[0], 28);   // src(7, 0)
  EXPECT_EQ(result[7], 252);  // src(7, 7)
  EXPECT_EQ(result[56], 0);   // src(0, 0)
  EXPECT_EQ(result[63], 224); // src(0, 7)
}

TEST_F(GeometricTest, Rotate90_FourTimesIsIdentity) {
  CudaImage output;
  Geometric::rotate90(testImage, output, 4);
  cudaDeviceSynchronize();

  auto result = downloadPixels(output);
  auto original = downloadPixels(testImage);
  EXPECT_EQ(result, original);
}

// 通用 rotate(90°) 修复中心偏移后应与 rotate90 在角点/中心一致
TEST_F(GeometricTest, Rotate90_GenericMatchesRotate90) {
  CudaImage genericOut;
  Geometric::rotate(testImage, genericOut, 90.0f);
  cudaDeviceSynchronize();

  ASSERT_EQ(genericOut.width, 8);
  ASSERT_EQ(genericOut.height, 8);

  auto result = downloadPixels(genericOut);
  EXPECT_EQ(result[0 * 8 + 0], 224);
  EXPECT_EQ(result[0 * 8 + 7], 0);
  EXPECT_EQ(result[7 * 8 + 0], 252);
  EXPECT_EQ(result[7 * 8 + 7], 28);
  EXPECT_EQ(result[4 * 8 + 4], 112); // src(4, 3) = 28 * 4
}

TEST_F(GeometricTest, FlipHorizontal) {
  CudaImage output;
  Geometric::flip(testImage, output, FlipDirection::Horizontal);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, testImage.width);
  EXPECT_EQ(output.height, testImage.height);

  // 验证翻转
  std::vector<unsigned char> result(64);
  cudaMemcpy(result.data(), output.buffer.data(), 64, cudaMemcpyDeviceToHost);

  std::vector<unsigned char> original(64);
  cudaMemcpy(original.data(), testImage.buffer.data(), 64,
             cudaMemcpyDeviceToHost);

  // 全像素校验：dst(x, y) = src(7 - x, y)
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT_EQ(result[y * 8 + x], original[y * 8 + (7 - x)])
          << "mismatch at (" << x << ", " << y << ")";
    }
  }
}

TEST_F(GeometricTest, FlipVertical) {
  CudaImage output;
  Geometric::flip(testImage, output, FlipDirection::Vertical);
  cudaDeviceSynchronize();

  ASSERT_EQ(output.width, testImage.width);
  ASSERT_EQ(output.height, testImage.height);

  auto result = downloadPixels(output);
  auto original = downloadPixels(testImage);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT_EQ(result[y * 8 + x], original[(7 - y) * 8 + x])
          << "mismatch at (" << x << ", " << y << ")";
    }
  }
}

TEST_F(GeometricTest, FlipBoth) {
  CudaImage output;
  Geometric::flip(testImage, output, FlipDirection::Both);
  cudaDeviceSynchronize();

  auto result = downloadPixels(output);
  auto original = downloadPixels(testImage);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT_EQ(result[y * 8 + x], original[(7 - y) * 8 + (7 - x)]);
    }
  }
}

TEST_F(GeometricTest, Crop) {
  CudaImage output;
  Geometric::crop(testImage, output, 2, 2, 4, 4);
  cudaDeviceSynchronize();

  ASSERT_EQ(output.width, 4);
  ASSERT_EQ(output.height, 4);

  // crop 区域 (2,2)-(5,5)：out(x, y) = in(x + 2, y + 2)
  auto result = downloadPixels(output);
  auto original = downloadPixels(testImage);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      EXPECT_EQ(result[y * 4 + x], original[(y + 2) * 8 + (x + 2)]);
    }
  }
}

TEST_F(GeometricTest, Pad) {
  CudaImage output;
  Geometric::pad(testImage, output, 2, 2, 2, 2, 128);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, 12);
  EXPECT_EQ(output.height, 12);

  // 验证填充值
  std::vector<unsigned char> result(144);
  cudaMemcpy(result.data(), output.buffer.data(), 144, cudaMemcpyDeviceToHost);
  EXPECT_EQ(result[0], 128);            // 左上角应该是填充值
  EXPECT_EQ(result[11 * 12 + 11], 128); // 右下角应该是填充值

  // 内部 8x8 区域应为原图：out(x + 2, y + 2) = in(x, y)
  std::vector<unsigned char> original(64);
  cudaMemcpy(original.data(), testImage.buffer.data(), 64,
             cudaMemcpyDeviceToHost);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT_EQ(result[(y + 2) * 12 + (x + 2)], original[y * 8 + x]);
    }
  }
}

TEST_F(GeometricTest, Rotate) {
  CudaImage output;
  Geometric::rotate(testImage, output, 45.0f);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.isValid());
  // 45度旋转后尺寸会变大
  EXPECT_GT(output.width, testImage.width);
}

TEST_F(GeometricTest, AffineTransform_Identity) {
  CudaImage output;
  float matrix[6] = {1, 0, 0, 0, 1, 0}; // 单位矩阵
  Geometric::affineTransform(testImage, output, matrix, 8, 8);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, 8);
  EXPECT_EQ(output.height, 8);

  std::vector<unsigned char> original = downloadPixels(testImage);
  std::vector<unsigned char> result = downloadPixels(output);
  EXPECT_EQ(result, original);
}

TEST_F(GeometricTest, PerspectiveTransform_Identity) {
  CudaImage output;
  float matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  Geometric::perspectiveTransform(testImage, output, matrix, 8, 8);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, 8);
  EXPECT_EQ(output.height, 8);

  std::vector<unsigned char> original = downloadPixels(testImage);
  std::vector<unsigned char> result = downloadPixels(output);
  EXPECT_EQ(result, original);
}

TEST_F(GeometricTest, PerspectiveTransform_BoundaryPreserved) {
  CudaImage output;
  float matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  Geometric::perspectiveTransform(testImage, output, matrix, 8, 8);
  cudaDeviceSynchronize();

  std::vector<unsigned char> result = downloadPixels(output);
  EXPECT_EQ(result[7], 28);
  EXPECT_EQ(result[56], 224);
  EXPECT_EQ(result[63], 252);
}

TEST_F(GeometricTest, AffineTransform_Scale) {
  CudaImage output;
  float matrix[6] = {2, 0, 0, 0, 2, 0}; // 2x 缩放
  Geometric::affineTransform(testImage, output, matrix, 16, 16);
  cudaDeviceSynchronize();

  EXPECT_EQ(output.width, 16);
  EXPECT_EQ(output.height, 16);
}

TEST_F(GeometricTest, InvalidInput) {
  CudaImage invalid;
  CudaImage output;

  EXPECT_THROW(Geometric::rotate(invalid, output, 45.0f),
               std::invalid_argument);
  EXPECT_THROW(Geometric::flip(invalid, output, FlipDirection::Horizontal),
               std::invalid_argument);
  EXPECT_THROW(Geometric::crop(invalid, output, 0, 0, 4, 4),
               std::invalid_argument);
}

TEST_F(GeometricTest, InvalidCropDimensions) {
  CudaImage output;
  EXPECT_THROW(Geometric::crop(testImage, output, 0, 0, 0, 0),
               std::invalid_argument);
  EXPECT_THROW(Geometric::crop(testImage, output, 0, 0, -1, 4),
               std::invalid_argument);
}

TEST_F(GeometricTest, InvalidPadding) {
  CudaImage output;
  EXPECT_THROW(Geometric::pad(testImage, output, -1, 0, 0, 0),
               std::invalid_argument);
}

TEST_F(GeometricTest, NullMatrix) {
  CudaImage output;
  EXPECT_THROW(Geometric::affineTransform(testImage, output, nullptr, 8, 8),
               std::invalid_argument);
}
