#pragma once

#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/operators/image_operator.hpp"
#include <vector>

namespace gpu_image {

class OperatorPipeline : public ImageOperator {
public:
  OperatorPipeline() = default;

  template <typename Op, typename... Args>
  OperatorPipeline& then(Args&&... args) {
    operators_.push_back(std::make_unique<Op>(std::forward<Args>(args)...));
    return *this;
  }

  OperatorPipeline& then(std::unique_ptr<ImageOperator> op) {
    operators_.push_back(std::move(op));
    return *this;
  }

  GpuImage apply(const GpuImage& input, ExecutionContext& ctx) override {
    if (!input.isValid()) {
      return GpuImage{};
    }
    if (operators_.empty()) {
      // 恒等：返回一个独立副本（GpuImage 为 move-only，不能直接返回 input）
      return ImageUtils::clone(input);
    }

    // 每个算子都以 const& 读取输入并产出全新输出，因此无需拷贝 input：
    // 直接把它喂给首个算子，之后在线程间传递工作缓冲区并回收中间结果。
    GpuImage current = operators_.front()->apply(input, ctx);
    for (size_t i = 1; i < operators_.size(); ++i) {
      GpuImage next = operators_[i]->apply(current, ctx);
      if (current.isValid()) {
        ctx.recycleToPool(std::move(current));
      }
      current = std::move(next);
    }
    return current;
  }

  OperatorTraits traits() const override {
    OperatorTraits result{"pipeline", false, false, false};
    for (const auto& op : operators_) {
      auto t = op->traits();
      result.name += (result.name == "pipeline" ? ": " : " -> ") + t.name;
      result.changesDimensions = result.changesDimensions || t.changesDimensions;
      result.changesChannels = result.changesChannels || t.changesChannels;
    }
    return result;
  }

  std::unique_ptr<ImageOperator> clone() const override {
    auto pipeline = std::make_unique<OperatorPipeline>();
    for (const auto& op : operators_) {
      pipeline->operators_.push_back(op->clone());
    }
    return pipeline;
  }

  [[nodiscard]] size_t size() const { return operators_.size(); }
  [[nodiscard]] bool empty() const { return operators_.empty(); }
  void clear() { operators_.clear(); }

private:
  std::vector<std::unique_ptr<ImageOperator>> operators_;
};

} // namespace gpu_image
