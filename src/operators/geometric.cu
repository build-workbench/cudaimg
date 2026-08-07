#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/geometric.hpp"
#include <cmath>
#include <stdexcept>

namespace cudaimg {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
__device__ float sampleBilinearOrZero(const unsigned char* input, int srcWidth,
                                      int srcHeight, int channels, float srcX,
                                      float srcY, int channel) {
  // 反向映射在整数角度旋转时坐标可能因浮点误差越出边界约 1e-7，
  // 直接判零会让角点采到黑像素；对微小越界做钳位而非丢弃。
  const float kEps = 1e-4f;
  if (srcX < -kEps || srcX > static_cast<float>(srcWidth - 1) + kEps ||
      srcY < -kEps || srcY > static_cast<float>(srcHeight - 1) + kEps) {
    return 0.0f;
  }
  srcX = fminf(fmaxf(srcX, 0.0f), static_cast<float>(srcWidth - 1));
  srcY = fminf(fmaxf(srcY, 0.0f), static_cast<float>(srcHeight - 1));

  int x0 = static_cast<int>(floorf(srcX));
  int y0 = static_cast<int>(floorf(srcY));
  int x1 = min(x0 + 1, srcWidth - 1);
  int y1 = min(y0 + 1, srcHeight - 1);

  float fx = srcX - x0;
  float fy = srcY - y0;

  float v00 = input[(y0 * srcWidth + x0) * channels + channel];
  float v10 = input[(y0 * srcWidth + x1) * channels + channel];
  float v01 = input[(y1 * srcWidth + x0) * channels + channel];
  float v11 = input[(y1 * srcWidth + x1) * channels + channel];

  return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy +
         v11 * fx * fy;
}
} // namespace

// 旋转 Kernel
__global__ void rotateKernel(const unsigned char* input, unsigned char* output,
                             int srcWidth, int srcHeight, int dstWidth,
                             int dstHeight, int channels, float cosAngle,
                             float sinAngle, float centerX, float centerY) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  // 计算目标中心（使用像素坐标中心 (n-1)/2，与 host 侧 centerX/Y 一致，
  // 保证 90° 整数倍旋转与 rotate90 的结果对齐，无半像素偏移）
  float dstCenterX = (dstWidth - 1) / 2.0f;
  float dstCenterY = (dstHeight - 1) / 2.0f;

  // 反向映射到源图像
  float dx = x - dstCenterX;
  float dy = y - dstCenterY;

  float srcX = cosAngle * dx + sinAngle * dy + centerX;
  float srcY = -sinAngle * dx + cosAngle * dy + centerY;

  for (int c = 0; c < channels; ++c) {
    float value = sampleBilinearOrZero(input, srcWidth, srcHeight, channels,
                                       srcX, srcY, c);

    output[(y * dstWidth + x) * channels + c] =
        static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 255.0f));
  }
}

// 90度旋转 Kernel
__global__ void rotate90Kernel(const unsigned char* input,
                               unsigned char* output, int srcWidth,
                               int srcHeight, int dstWidth, int dstHeight,
                               int channels, int times) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  int srcX, srcY;

  switch (times % 4) {
  case 1: // 90°
    srcX = y;
    srcY = dstWidth - 1 - x;
    break;
  case 2: // 180°
    srcX = dstWidth - 1 - x;
    srcY = dstHeight - 1 - y;
    break;
  case 3: // 270°
    srcX = dstHeight - 1 - y;
    srcY = x;
    break;
  default: // 0°
    srcX = x;
    srcY = y;
    break;
  }

  for (int c = 0; c < channels; ++c) {
    output[(y * dstWidth + x) * channels + c] =
        input[(srcY * srcWidth + srcX) * channels + c];
  }
}

// 翻转 Kernel
__global__ void flipKernel(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels, int direction) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int srcX = x, srcY = y;

  if (direction == 0 || direction == 2) { // Horizontal or Both
    srcX = width - 1 - x;
  }
  if (direction == 1 || direction == 2) { // Vertical or Both
    srcY = height - 1 - y;
  }

  for (int c = 0; c < channels; ++c) {
    output[(y * width + x) * channels + c] =
        input[(srcY * width + srcX) * channels + c];
  }
}

// 仿射变换 Kernel
__global__ void affineKernel(const unsigned char* input, unsigned char* output,
                             int srcWidth, int srcHeight, int dstWidth,
                             int dstHeight, int channels, float a, float b,
                             float tx, float c_val, float d, float ty) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  // 计算逆变换矩阵
  float det = a * d - b * c_val;
  if (fabsf(det) < 1e-6f) {
    for (int ch = 0; ch < channels; ++ch) {
      output[(y * dstWidth + x) * channels + ch] = 0;
    }
    return;
  }

  float invA = d / det;
  float invB = -b / det;
  float invC = -c_val / det;
  float invD = a / det;
  float invTx = (b * ty - d * tx) / det;
  float invTy = (c_val * tx - a * ty) / det;

  float srcX = invA * x + invB * y + invTx;
  float srcY = invC * x + invD * y + invTy;

  for (int ch = 0; ch < channels; ++ch) {
    float value = sampleBilinearOrZero(input, srcWidth, srcHeight, channels,
                                       srcX, srcY, ch);

    output[(y * dstWidth + x) * channels + ch] =
        static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 255.0f));
  }
}


// 裁剪 Kernel
__global__ void cropKernel(const unsigned char* input, unsigned char* output,
                           int srcWidth, int srcHeight, int dstWidth,
                           int dstHeight, int channels, int offsetX,
                           int offsetY) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  int srcX = x + offsetX;
  int srcY = y + offsetY;

  for (int c = 0; c < channels; ++c) {
    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
      output[(y * dstWidth + x) * channels + c] =
          input[(srcY * srcWidth + srcX) * channels + c];
    } else {
      output[(y * dstWidth + x) * channels + c] = 0;
    }
  }
}

// 填充 Kernel
__global__ void padKernel(const unsigned char* input, unsigned char* output,
                          int srcWidth, int srcHeight, int dstWidth,
                          int dstHeight, int channels, int top, int left,
                          unsigned char padValue) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  int srcX = x - left;
  int srcY = y - top;

  for (int c = 0; c < channels; ++c) {
    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
      output[(y * dstWidth + x) * channels + c] =
          input[(srcY * srcWidth + srcX) * channels + c];
    } else {
      output[(y * dstWidth + x) * channels + c] = padValue;
    }
  }
}

// Geometric 实现
void Geometric::rotate(const CudaImage& input, CudaImage& output,
                       float angleDegrees, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  float angleRad = angleDegrees * static_cast<float>(M_PI) / 180.0f;
  float cosAngle = cosf(angleRad);
  float sinAngle = sinf(angleRad);

  // 计算旋转后的尺寸
  float abscos = fabsf(cosAngle);
  float abssin = fabsf(sinAngle);
  int newWidth =
      static_cast<int>(input.width * abscos + input.height * abssin + 0.5f);
  int newHeight =
      static_cast<int>(input.width * abssin + input.height * abscos + 0.5f);

  ImageUtils::ensureOutputSize(output, newWidth, newHeight, input.channels);

  dim3 block(16, 16);
  dim3 grid((newWidth + block.x - 1) / block.x,
            (newHeight + block.y - 1) / block.y);

  float centerX = (input.width - 1) / 2.0f;
  float centerY = (input.height - 1) / 2.0f;

  rotateKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      newWidth, newHeight, input.channels, cosAngle, sinAngle, centerX,
      centerY);

  CUDA_CHECK(cudaGetLastError());
}

void Geometric::rotate90(const CudaImage& input, CudaImage& output, int times,
                         cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  times = ((times % 4) + 4) % 4; // 归一化到 0-3

  int newWidth = (times % 2 == 0) ? input.width : input.height;
  int newHeight = (times % 2 == 0) ? input.height : input.width;

  ImageUtils::ensureOutputSize(output, newWidth, newHeight, input.channels);

  dim3 block(16, 16);
  dim3 grid((newWidth + block.x - 1) / block.x,
            (newHeight + block.y - 1) / block.y);

  rotate90Kernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      newWidth, newHeight, input.channels, times);

  CUDA_CHECK(cudaGetLastError());
}

void Geometric::flip(const CudaImage& input, CudaImage& output,
                     FlipDirection direction, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  flipKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, static_cast<int>(direction));

  CUDA_CHECK(cudaGetLastError());
}

void Geometric::affineTransform(const CudaImage& input, CudaImage& output,
                                const float* matrix, int outputWidth,
                                int outputHeight, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (matrix == nullptr) {
    throw std::invalid_argument("Matrix is null");
  }
  if (outputWidth <= 0 || outputHeight <= 0) {
    throw std::invalid_argument("Invalid output dimensions");
  }

  ImageUtils::ensureOutputSize(output, outputWidth, outputHeight,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((outputWidth + block.x - 1) / block.x,
            (outputHeight + block.y - 1) / block.y);

  affineKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      outputWidth, outputHeight, input.channels, matrix[0], matrix[1],
      matrix[2], matrix[3], matrix[4], matrix[5]);

  CUDA_CHECK(cudaGetLastError());
}


void Geometric::crop(const CudaImage& input, CudaImage& output, int x, int y,
                     int width, int height, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  ImageUtils::ensureOutputSize(output, width, height, input.channels);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  cropKernel<<<grid, block, 0, stream>>>(input.buffer.dataAs<unsigned char>(),
                                         output.buffer.dataAs<unsigned char>(),
                                         input.width, input.height, width,
                                         height, input.channels, x, y);

  CUDA_CHECK(cudaGetLastError());
}

void Geometric::pad(const CudaImage& input, CudaImage& output, int top,
                    int bottom, int left, int right, unsigned char padValue,
                    cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (top < 0 || bottom < 0 || left < 0 || right < 0) {
    throw std::invalid_argument("Padding values must be non-negative");
  }

  int newWidth = input.width + left + right;
  int newHeight = input.height + top + bottom;

  ImageUtils::ensureOutputSize(output, newWidth, newHeight, input.channels);

  dim3 block(16, 16);
  dim3 grid((newWidth + block.x - 1) / block.x,
            (newHeight + block.y - 1) / block.y);

  padKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      newWidth, newHeight, input.channels, top, left, padValue);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
