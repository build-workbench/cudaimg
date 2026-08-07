#pragma once

// 设备端共享工具：供各 .cu 文件的 kernel 复用，消除重复的内联样板。
//
// 本头文件为纯增量：现有 .cu 文件可逐步采用其中的工具，未采用的部分
// 不受影响。所有工具均为 __device__ inline，零运行时开销。
//
// 用法：在 .cu 文件中 #include "cudaimg/core/device_kernels.cuh"，
// 然后在 kernel 内调用 cudaimg::dev::xxx(...)。

#include <cuda_runtime.h>

namespace cudaimg {
namespace dev {

// ── 值 clamp ─────────────────────────────────────────────────────
// 将 float/int 值裁剪到 [0, 255] 并转为 unsigned char。
// 与现有 .cu 中反复出现的 min(max(v, 0), 255) / fminf(fmaxf(v,0),255) 等价。

__device__ inline unsigned char clampToUchar(float value) {
  return static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 255.0f));
}

__device__ inline unsigned char clampToUchar(int value) {
  return static_cast<unsigned char>(min(max(value, 0), 255));
}

// ── 坐标 clamp（Replicate 边界）──────────────────────────────────
// 将坐标限制在 [0, extent-1]，等价于多文件中重复的 max(0, min(x, extent-1))。
// 使用 CUDA 内建的 min/max（device 端无需 <algorithm>）。

__device__ inline int clampCoord(int x, int extent) {
  return max(0, min(x, extent - 1));
}

// ── 边界检查 ─────────────────────────────────────────────────────
// 线程坐标是否落在图像范围内。两种写法对应现有 .cu 中的两种风格：
//   isInBounds(x, y, w, h)      → if (x < w && y < h)
//   isOutside(x, y, w, h)       → if (x >= w || y >= h)

__device__ inline bool isInBounds(int x, int y, int width, int height) {
  return x >= 0 && x < width && y >= 0 && y < height;
}

__device__ inline bool isOutside(int x, int y, int width, int height) {
  return x < 0 || x >= width || y < 0 || y >= height;
}

// ── 像素索引 ─────────────────────────────────────────────────────
// 行优先、channels 交错的像素线性索引。
// 等价于 .cu 中重复的 (y * width + x) * channels + c。

__device__ inline int pixelIndex(int x, int y, int width, int channels,
                                 int c = 0) {
  return (y * width + x) * channels + c;
}

// ── 灰度转换 ─────────────────────────────────────────────────────
// ITU-R BT.601 加权。与 pixel_operator.cu / histogram_calculator.cu 中
// 重复的 0.299f*r + 0.587f*g + 0.114f*b 一致。

__device__ inline float rgbToGray(float r, float g, float b) {
  return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ inline unsigned char rgbToGrayUchar(unsigned char r, unsigned char g,
                                               unsigned char b) {
  return clampToUchar(rgbToGray(r, g, b) + 0.5f);
}

} // namespace dev
} // namespace cudaimg
