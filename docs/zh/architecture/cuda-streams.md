# CUDA 流

使用 CUDA 流实现异步执行。

## 概述

CUDA 流可以重叠计算和数据传输，提高吞吐量。

## 流基础

CUDA 流是按顺序执行的操作序列：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步操作
cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);

cudaStreamDestroy(stream);
```

## ExecutionPolicy 与 ExecutionContext

Mini-OpenCV 将原始的流操作隐藏在 `ExecutionPolicy` 之后，提供三种模式：

- **Sync** —— 无流；`synchronize()` 调用 `cudaDeviceSynchronize`
- **Async** —— 创建并持有一个内部 `cudaStream_t`（或通过 `ExecutionPolicy::async(stream)` 包装调用方提供的流）
- **Batch** —— 创建并持有一个内部流，用于流水线操作

`ExecutionPolicy` 仅可移动（流不能被共享），并在析构时销毁其流。`ExecutionContext` 将策略与输出分配组合在一起：

```cpp
ExecutionContext ctx(ExecutionPolicy::async());  // 创建内部流
GpuImage out = ctx.allocateOutput(input);
PixelOperator::invert(input, out, ctx.stream());
ctx.synchronize();  // 等待内部流
```

便捷工厂函数：`syncContext()`、`asyncContext()`、`batchContext()`。

## PipelineProcessor

`PipelineProcessor` 持有一个固定大小的流池（构造参数，默认 3 条），并对批量图像执行用户自定义的处理步骤。每个步骤接收图像及分配给它的流：

```cpp
PipelineProcessor pipeline(4);  // 4 条流

pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
});

// 上传 -> 步骤 -> 下载，图像轮流分配到各条流，
// 返回前在内部完成同步
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

完整 API：`addStep`、`clearSteps`、`stepCount`、`process`、`processHost`、`processBatch`、`processBatchHost`、`synchronize`。`processBatch*` 将每张图像提交到 `streams[i % numStreams]`，并在返回前同步所有流。完整程序参见 `examples/pipeline_example.cpp`。

## 性能收益

多流批处理使不同图像的 H2D 传输、内核执行和 D2H 传输相互重叠（时间线见[架构概览](./overview)）。实际加速比取决于图像大小、内核组合和 GPU；`examples/pipeline_example.cpp` 会打印顺序处理与流水线处理的实测耗时及流数量对比，[性能基准](../benchmarks/)一节也提供了实测数据。

## 最佳实践

1. **从默认的 3 条流开始**并实测；更多流收益递减
2. **批量相似操作**: 用同一组步骤处理一批图像
3. **边界同步**: `processBatch*` 已经内部同步；仅在你自行提交任务时才需要显式调用 `synchronize()`

## 下一步

- [设计决策](./design-decisions) - 为什么选择这种方式
- [性能基准](../benchmarks/) - 性能数据
