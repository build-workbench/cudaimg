#pragma once

#include "cudaimg/core/device_buffer.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace cudaimg {

// Forward declaration
struct CudaImage;
struct HostImage;

// ImageUtils: Image creation and transfer utilities
// 模块化图像工具：图像创建与传输工具
//
// This module provides the primary interface for creating and transferring
// image data between host and device memory. All CudaImage allocations should
// go through this module to enable future memory pooling optimizations.
//
// Interface contract:
// - All returned CudaImages have valid, properly-sized buffers
// - Parameters are validated before allocation
namespace ImageUtils {

// ===== Image Creation =====

/// Create a CudaImage with specified dimensions
/// 创建指定尺寸的 CudaImage
///
/// @param width   Image width in pixels (must be > 0)
/// @param height  Image height in pixels (must be > 0)
/// @param channels Number of channels (1=gray, 3=RGB, 4=RGBA)
/// @return CudaImage with allocated device buffer
/// @throws std::invalid_argument if parameters are invalid
CudaImage createCudaImage(int width, int height, int channels);

/// Create an empty HostImage with specified dimensions
/// 创建指定尺寸的空 HostImage
HostImage createHostImage(int width, int height, int channels);

// ===== Host-Device Transfer =====

/// Upload HostImage to GPU, returning a new CudaImage
/// 上传 HostImage 到 GPU，返回新的 CudaImage
CudaImage uploadToGpu(const HostImage& hostImage);

/// Download CudaImage to host, returning a new HostImage
/// 下载 CudaImage 到主机，返回新的 HostImage
HostImage downloadFromGpu(const CudaImage& gpuImage);

/// Deep-copy a CudaImage (device-to-device), returning an independent image
/// 深拷贝 CudaImage（设备到设备），返回相互独立的新图像
/// An invalid input yields an empty (invalid) CudaImage.
CudaImage clone(const CudaImage& gpuImage);

/// Async upload: resizes gpuImage if needed
/// 异步上传：需要时调整 gpuImage 大小
void uploadToGpuAsync(const HostImage& hostImage, CudaImage& gpuImage,
                      cudaStream_t stream);

/// Async download: resizes hostImage if needed
/// 异步下载：需要时调整 hostImage 大小
void downloadFromGpuAsync(const CudaImage& gpuImage, HostImage& hostImage,
                          cudaStream_t stream);

// ===== Output Buffer Management =====

/// Ensure output CudaImage matches input dimensions (reallocate if needed)
/// 确保 CudaImage 输出与输入尺寸匹配（需要时重新分配）
void ensureOutputSize(const CudaImage& input, CudaImage& output);

/// Ensure output CudaImage has specified dimensions (reallocate if needed)
/// 确保 CudaImage 输出具有指定尺寸（需要时重新分配）
void ensureOutputSize(CudaImage& output, int width, int height, int channels);

// ===== Validation =====

/// Validate image parameters
/// 验证图像参数
/// @return true if valid (width>0, height>0, channels in {1,3,4})
bool validateImageParams(int width, int height, int channels);

// ===== Memory Pool Configuration =====

/// Enable or disable memory pooling for subsequent allocations
/// 为后续分配启用或禁用内存池

/// Check if memory pooling is currently enabled
/// 检查内存池当前是否启用

} // namespace ImageUtils

} // namespace cudaimg
