# CUDA Streams

Asynchronous execution using CUDA streams.

## Overview

CUDA streams enable overlapping computation and data transfer for higher throughput.

## Stream Basics

A CUDA stream is a sequence of operations that execute in order:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async operations
cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);

cudaStreamDestroy(stream);
```

## ExecutionPolicy and ExecutionContext

Mini-OpenCV hides raw stream handling behind `ExecutionPolicy`, with three modes:

- **Sync** — no stream; `synchronize()` calls `cudaDeviceSynchronize`
- **Async** — creates and owns an internal `cudaStream_t` (or wraps a caller-supplied stream via `ExecutionPolicy::async(stream)`)
- **Batch** — creates and owns an internal stream for pipelined operations

`ExecutionPolicy` is move-only (streams cannot be shared) and destroys its stream on destruction. `ExecutionContext` pairs a policy with output allocation:

```cpp
ExecutionContext ctx(ExecutionPolicy::async());  // internal stream created
GpuImage out = ctx.allocateOutput(input);
PixelOperator::invert(input, out, ctx.stream());
ctx.synchronize();  // wait for the internal stream
```

Convenience factories: `syncContext()`, `asyncContext()`, `batchContext()`.

## PipelineProcessor

`PipelineProcessor` owns a fixed pool of streams (constructor argument, default 3) and runs user-defined steps over batches of images. Each step receives the image and the stream assigned to it:

```cpp
PipelineProcessor pipeline(4);  // 4 streams

pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
});

// Upload -> steps -> download, images round-robined across streams,
// synchronizes internally before returning
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

Full API: `addStep`, `clearSteps`, `stepCount`, `process`, `processHost`, `processBatch`, `processBatchHost`, `synchronize`. `processBatch*` enqueue each image on `streams[i % numStreams]` and synchronize all streams before returning. See `examples/pipeline_example.cpp` for a complete program.

## Performance

Multi-stream batching overlaps H2D transfer, kernel execution, and D2H transfer of different images (see the timeline in [Architecture Overview](./overview)). The actual speedup depends on image size, kernel mix, and GPU; `examples/pipeline_example.cpp` prints measured sequential-vs-pipeline timings and a stream-count sweep, and the [Benchmarks](../benchmarks/) section contains measured numbers.

## Best Practices

1. **Start from the default of 3 streams** and measure; more streams have diminishing returns
2. **Batch similar operations**: process a vector of images with the same step list
3. **Synchronize at boundaries**: `processBatch*` already synchronizes; an explicit `synchronize()` is only needed if you enqueue work yourself

## Next Steps

- [Design Decisions](./design-decisions) - Why we chose this approach
- [Benchmarks](../benchmarks/) - Performance data
