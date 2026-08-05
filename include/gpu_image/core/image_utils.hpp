#pragma once

#include "gpu_image/core/device_buffer.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace gpu_image {

// Forward declaration
struct GpuImage;
struct HostImage;

// ImageUtils: Image creation and transfer utilities
// 模块化图像工具：图像创建与传输工具
//
// This module provides the primary interface for creating and transferring
// image data between host and device memory. All GpuImage allocations should
// go through this module to enable future memory pooling optimizations.
//
// Interface contract:
// - All returned GpuImages have valid, properly-sized buffers
// - Parameters are validated before allocation
// - Memory can optionally be pooled via MemoryManager integration
namespace ImageUtils {

// ===== Image Creation =====

/// Create a GpuImage with specified dimensions
/// 创建指定尺寸的 GpuImage
///
/// @param width   Image width in pixels (must be > 0)
/// @param height  Image height in pixels (must be > 0)
/// @param channels Number of channels (1=gray, 3=RGB, 4=RGBA)
/// @return GpuImage with allocated device buffer
/// @throws std::invalid_argument if parameters are invalid
GpuImage createGpuImage(int width, int height, int channels);

/// Create an empty HostImage with specified dimensions
/// 创建指定尺寸的空 HostImage
HostImage createHostImage(int width, int height, int channels);

// ===== Host-Device Transfer =====

/// Upload HostImage to GPU, returning a new GpuImage
/// 上传 HostImage 到 GPU，返回新的 GpuImage
GpuImage uploadToGpu(const HostImage& hostImage);

/// Download GpuImage to host, returning a new HostImage
/// 下载 GpuImage 到主机，返回新的 HostImage
HostImage downloadFromGpu(const GpuImage& gpuImage);

/// Deep-copy a GpuImage (device-to-device), returning an independent image
/// 深拷贝 GpuImage（设备到设备），返回相互独立的新图像
/// An invalid input yields an empty (invalid) GpuImage.
GpuImage clone(const GpuImage& gpuImage);

/// Async upload: resizes gpuImage if needed
/// 异步上传：需要时调整 gpuImage 大小
void uploadToGpuAsync(const HostImage& hostImage, GpuImage& gpuImage,
                      cudaStream_t stream);

/// Async download: resizes hostImage if needed
/// 异步下载：需要时调整 hostImage 大小
void downloadFromGpuAsync(const GpuImage& gpuImage, HostImage& hostImage,
                          cudaStream_t stream);

// ===== Output Buffer Management =====

/// Ensure output GpuImage matches input dimensions (reallocate if needed)
/// 确保 GpuImage 输出与输入尺寸匹配（需要时重新分配）
void ensureOutputSize(const GpuImage& input, GpuImage& output);

/// Ensure output GpuImage has specified dimensions (reallocate if needed)
/// 确保 GpuImage 输出具有指定尺寸（需要时重新分配）
void ensureOutputSize(GpuImage& output, int width, int height, int channels);

// ===== Validation =====

/// Validate image parameters
/// 验证图像参数
/// @return true if valid (width>0, height>0, channels in {1,3,4})
bool validateImageParams(int width, int height, int channels);

// ===== Memory Pool Configuration =====

/// Enable or disable memory pooling for subsequent allocations
/// 为后续分配启用或禁用内存池
/// When enabled, GpuImage allocations use MemoryManager pool
void setMemoryPoolingEnabled(bool enabled);

/// Check if memory pooling is currently enabled
/// 检查内存池当前是否启用
bool isMemoryPoolingEnabled();

} // namespace ImageUtils

} // namespace gpu_image
