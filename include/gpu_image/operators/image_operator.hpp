#pragma once

#include "gpu_image/core/execution_context.hpp"
#include "gpu_image/core/gpu_image.hpp"
#include <memory>
#include <stdexcept>
#include <string>

namespace gpu_image {

struct OperatorTraits {
  std::string name;
  bool inPlaceCapable;
  bool changesDimensions;
  bool changesChannels;
};

class ImageOperator {
public:
  virtual ~ImageOperator() = default;

  [[nodiscard]] virtual GpuImage apply(const GpuImage& input,
                                        ExecutionContext& ctx) = 0;

  virtual void applyInPlace(GpuImage& image, ExecutionContext& ctx) {
    (void)image;
    (void)ctx;
    throw std::runtime_error("In-place operation not supported");
  }

  [[nodiscard]] virtual OperatorTraits traits() const = 0;
  [[nodiscard]] virtual std::unique_ptr<ImageOperator> clone() const = 0;
  [[nodiscard]] bool canApplyInPlace() const { return traits().inPlaceCapable; }
};

template <typename Derived>
class UnaryOperator : public ImageOperator {
public:
  GpuImage apply(const GpuImage& input, ExecutionContext& ctx) override {
    GpuImage output;
    static_cast<Derived*>(this)->execute(input, output, ctx);
    return output;
  }

  std::unique_ptr<ImageOperator> clone() const override {
    return std::make_unique<Derived>(*static_cast<const Derived*>(this));
  }
};

} // namespace gpu_image
