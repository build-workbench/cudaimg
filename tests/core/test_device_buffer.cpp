/**
 * DeviceBuffer 单元测试
 *
 * Feature: gpu-image-processing, Property 1: 数据传输往返一致性
 * Validates: Requirements 1.1, 1.2
 */

#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/device_buffer.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace cudaimg;

class DeviceBufferTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 检查 CUDA 是否可用
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }
};

// 测试基本分配和释放
TEST_F(DeviceBufferTest, BasicAllocation) {
  const size_t size = 1024;
  DeviceBuffer buffer(size);

  EXPECT_TRUE(buffer.isValid());
  EXPECT_EQ(buffer.size(), size);
  EXPECT_NE(buffer.data(), nullptr);
}

// 测试空缓冲区
TEST_F(DeviceBufferTest, EmptyBuffer) {
  DeviceBuffer buffer;

  EXPECT_FALSE(buffer.isValid());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.data(), nullptr);
}

// 测试零大小分配
TEST_F(DeviceBufferTest, ZeroSizeAllocation) {
  DeviceBuffer buffer(0);

  EXPECT_FALSE(buffer.isValid());
  EXPECT_EQ(buffer.size(), 0);
}

// 测试移动语义
TEST_F(DeviceBufferTest, MoveSemantics) {
  const size_t size = 1024;
  DeviceBuffer buffer1(size);
  void* originalPtr = buffer1.data();

  // 移动构造
  DeviceBuffer buffer2(std::move(buffer1));

  EXPECT_FALSE(buffer1.isValid());
  EXPECT_TRUE(buffer2.isValid());
  EXPECT_EQ(buffer2.data(), originalPtr);
  EXPECT_EQ(buffer2.size(), size);

  // 移动赋值
  DeviceBuffer buffer3;
  buffer3 = std::move(buffer2);

  EXPECT_FALSE(buffer2.isValid());
  EXPECT_TRUE(buffer3.isValid());
  EXPECT_EQ(buffer3.data(), originalPtr);
}

// Property 1: 数据传输往返一致性
// For any 有效的数据，上传到 GPU 再下载回来应该完全相同
TEST_F(DeviceBufferTest, RoundTripConsistency) {
  const size_t size = 4096;

  // 创建测试数据
  std::vector<unsigned char> original(size);
  std::iota(original.begin(), original.end(), 0); // 0, 1, 2, ...

  // 上传到 GPU
  DeviceBuffer buffer(size);
  buffer.copyFromHost(original.data(), size);

  // 下载回来
  std::vector<unsigned char> downloaded(size);
  buffer.copyToHost(downloaded.data(), size);

  // 验证一致性
  EXPECT_EQ(original, downloaded);
}

// 测试部分复制
TEST_F(DeviceBufferTest, PartialCopy) {
  const size_t bufferSize = 1024;
  const size_t copySize = 512;

  std::vector<unsigned char> original(copySize, 42);

  DeviceBuffer buffer(bufferSize);
  buffer.copyFromHost(original.data(), copySize);

  std::vector<unsigned char> downloaded(copySize);
  buffer.copyToHost(downloaded.data(), copySize);

  EXPECT_EQ(original, downloaded);
}

// 测试无效参数
TEST_F(DeviceBufferTest, InvalidParameters) {
  DeviceBuffer buffer(1024);

  // 空指针
  EXPECT_THROW(buffer.copyFromHost(nullptr, 100), std::invalid_argument);
  EXPECT_THROW(buffer.copyToHost(nullptr, 100), std::invalid_argument);

  // 超出大小
  std::vector<unsigned char> data(2048);
  EXPECT_THROW(buffer.copyFromHost(data.data(), 2048), std::invalid_argument);
}

// 测试释放
TEST_F(DeviceBufferTest, Release) {
  DeviceBuffer buffer(1024);
  EXPECT_TRUE(buffer.isValid());

  buffer.release();

  EXPECT_FALSE(buffer.isValid());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.data(), nullptr);
}

// 随机数据往返测试
TEST_F(DeviceBufferTest, RandomDataRoundTrip) {
  const size_t size = 10000;

  // 生成随机数据
  std::vector<unsigned char> original(size);
  unsigned int seed = 12345;
  for (size_t i = 0; i < size; ++i) {
    seed = seed * 1103515245 + 12345;
    original[i] = static_cast<unsigned char>((seed >> 16) & 0xFF);
  }

  DeviceBuffer buffer(size);
  buffer.copyFromHost(original.data(), size);

  std::vector<unsigned char> downloaded(size);
  buffer.copyToHost(downloaded.data(), size);

  EXPECT_EQ(original, downloaded);
}
