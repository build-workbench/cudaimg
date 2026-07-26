#pragma once

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
    if (operators_.empty()) {
      return input;
    }
    if (!input.isValid()) {
      return GpuImage{};
    }

    GpuImage current = input;
    for (auto& op : operators_) {
      GpuImage next = op->apply(current, ctx);
      if (current.isValid() && &current != &input) {
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
