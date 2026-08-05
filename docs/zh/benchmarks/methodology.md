# 测试方法

Mini-OpenCV 如何测量和报告性能。

## 范围

基准测试套件只测量 **GPU 绝对延迟**。它不测量 CPU 实现，也不计算相对 OpenCV 或其他任何库的加速比。这里刻意不记录固定的硬件环境：数字取决于你的 GPU、驱动和系统状态，我们唯一背书的结果是你自己复现出来的结果。基准测试可执行文件会在每次运行开始时打印它检测到的 CUDA 设备。

## 测量方法

### 预热

每次测量前（`benchmarks/benchmark_main.cpp`）：
1. 运行操作 10 次（不计时）预热 GPU
2. 调用 `cudaDeviceSynchronize()`

### 计时

手写的 `std::chrono` 计时器包裹计时循环（构建虽然链接了 Google Benchmark 库，但当前框架并未使用它）：

```cpp
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < iterations; i++) {   // iterations = 100
    operation();
}
cudaDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
auto avg_time = (end - start) / iterations;
```

### 指标

- **延迟**：每次操作的平均毫秒数——框架报告的唯一指标

## 图像尺寸

框架扫描五种正方形尺寸：

| 分辨率 | 像素数 |
|--------|--------|
| 256×256 | 66K |
| 512×512 | 262K |
| 1024×1024 | 1.0M |
| 2048×2048 | 4.2M |
| 4096×4096 | 16.8M |

## 可复现性

基准测试目标为 `gpu_image_benchmark`（默认关闭）：

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/bin/gpu_image_benchmark
```

可执行文件会打印检测到的 CUDA 设备、按图像尺寸各输出一张延迟表，以及一个流水线部分（对比顺序处理一批 10 张图像与使用 1 / 2 / 4 / 8 条 CUDA 流）。

## 注意事项

- 本仓库没有任何 CPU 基准测试；这些文档中的任何地方都不暗示 CPU/GPU 对比
- 各操作的计时测量的是数据已驻留在设备上时的内核执行；主机↔设备传输开销单独体现为"Upload (H2D)"/"Download (D2H)"条目
- 结果随 GPU 型号、温度、时钟频率和驱动版本而变化
