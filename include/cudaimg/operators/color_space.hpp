#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>

namespace cudaimg {

// 颜色空间转换类
class ColorSpace {
public:
  // RGB <-> HSV
  static void rgbToHsv(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);
  static void hsvToRgb(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);

  // RGB <-> YUV
  static void rgbToYuv(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);
  static void yuvToRgb(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);

  // RGB <-> Lab (CIE L*a*b*)
  static void rgbToLab(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);
  static void labToRgb(const CudaImage& input, CudaImage& output,
                       cudaStream_t stream = nullptr);

  // 通道分离
  static void splitChannels(const CudaImage& input, CudaImage& channel0,
                            CudaImage& channel1, CudaImage& channel2,
                            cudaStream_t stream = nullptr);

  // 通道合并
  static void mergeChannels(const CudaImage& channel0, const CudaImage& channel1,
                            const CudaImage& channel2, CudaImage& output,
                            cudaStream_t stream = nullptr);
};

} // namespace cudaimg
