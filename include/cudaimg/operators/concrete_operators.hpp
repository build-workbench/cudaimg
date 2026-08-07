#pragma once

#include "cudaimg/operators/convolution_engine.hpp"
#include "cudaimg/operators/histogram_calculator.hpp"
#include "cudaimg/operators/image_operator.hpp"
#include "cudaimg/operators/image_resizer.hpp"
#include "cudaimg/operators/pixel_operator.hpp"

namespace cudaimg {

class InvertOperator : public UnaryOperator<InvertOperator> {
public:
  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    PixelOperator::invert(input, output, ctx.stream());
  }

  void applyInPlace(CudaImage& image, ExecutionContext& ctx) override {
    PixelOperator::invertInPlace(image, ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"invert", true, false, false};
  }
};

class GrayscaleOperator : public UnaryOperator<GrayscaleOperator> {
public:
  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    PixelOperator::toGrayscale(input, output, ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"grayscale", false, false, true};
  }
};

class BrightnessOperator : public UnaryOperator<BrightnessOperator> {
public:
  explicit BrightnessOperator(int offset = 0) : offset_(offset) {}

  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    PixelOperator::adjustBrightness(input, output, offset_, ctx.stream());
  }

  void applyInPlace(CudaImage& image, ExecutionContext& ctx) override {
    PixelOperator::adjustBrightnessInPlace(image, offset_, ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"brightness", true, false, false};
  }

  void setOffset(int offset) { offset_ = offset; }
  int offset() const { return offset_; }

private:
  int offset_;
};

class GaussianBlurOperator : public UnaryOperator<GaussianBlurOperator> {
public:
  explicit GaussianBlurOperator(int kernelSize = 5, float sigma = 1.0f)
      : kernelSize_(kernelSize), sigma_(sigma) {}

  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    ConvolutionEngine::gaussianBlur(input, output, kernelSize_, sigma_,
                                    ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"gaussian_blur", false, false, false};
  }

  void setKernelSize(int size) { kernelSize_ = size; }
  void setSigma(float sigma) { sigma_ = sigma; }
  int kernelSize() const { return kernelSize_; }
  float sigma() const { return sigma_; }

private:
  int kernelSize_;
  float sigma_;
};

class SobelOperator : public UnaryOperator<SobelOperator> {
public:
  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    ConvolutionEngine::sobelEdgeDetection(input, output, ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"sobel", false, false, true};
  }
};

class ResizeOperator : public UnaryOperator<ResizeOperator> {
public:
  ResizeOperator()
      : width_(0), height_(0), scaleX_(0), scaleY_(0), useScale_(false) {}

  explicit ResizeOperator(int width, int height)
      : width_(width), height_(height), scaleX_(0), scaleY_(0),
        useScale_(false) {}

  static ResizeOperator byScale(float scaleX, float scaleY) {
    return ResizeOperator(scaleX, scaleY, true);
  }

  static ResizeOperator byDimensions(int width, int height) {
    return ResizeOperator(width, height);
  }

  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    if (useScale_) {
      ImageResizer::resizeByScale(input, output, scaleX_, scaleY_,
                                  InterpolationMode::Bilinear, ctx.stream());
    } else {
      ImageResizer::resize(input, output, width_, height_,
                           InterpolationMode::Bilinear, ctx.stream());
    }
  }

  OperatorTraits traits() const override {
    return {"resize", false, true, false};
  }

  void setDimensions(int width, int height) {
    width_ = width;
    height_ = height;
    useScale_ = false;
  }

  void setScale(float scaleX, float scaleY) {
    scaleX_ = scaleX;
    scaleY_ = scaleY;
    useScale_ = true;
  }

  bool isScaleMode() const { return useScale_; }
  int width() const { return width_; }
  int height() const { return height_; }
  float scaleX() const { return scaleX_; }
  float scaleY() const { return scaleY_; }

private:
  ResizeOperator(float scaleX, float scaleY, bool)
      : width_(0), height_(0), scaleX_(scaleX), scaleY_(scaleY),
        useScale_(true) {}

  int width_, height_;
  float scaleX_, scaleY_;
  bool useScale_;
};

class HistogramEqualizeOperator
    : public UnaryOperator<HistogramEqualizeOperator> {
public:
  void execute(const CudaImage& input, CudaImage& output, ExecutionContext& ctx) {
    HistogramCalculator::equalize(input, output, ctx.stream());
  }

  OperatorTraits traits() const override {
    return {"histogram_equalize", false, false, false};
  }
};

} // namespace cudaimg
