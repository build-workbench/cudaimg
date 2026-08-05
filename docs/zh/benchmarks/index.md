# 性能基准测试

基准测试套件测量 Mini-OpenCV 自身算子的 **GPU 绝对延迟**。它不与 CPU OpenCV 或其他任何库做对比，本页也刻意不给出加速比数字：结果取决于你的硬件、驱动和系统状态，唯一值得引用的数据是你自己复现出来的数据。

## 测量内容

唯一的可执行文件 `gpu_image_benchmark`（源码：`benchmarks/benchmark_main.cpp`）在五种正方形图像尺寸（256×256、512×512、1024×1024、2048×2048、4096×4096）下测量每个算子，并以毫秒为单位打印单次调用的平均耗时。

覆盖的操作：

| 类别 | 操作 |
|------|------|
| 像素操作 | 反色、灰度化、亮度调节 |
| 卷积 | 高斯模糊 3×3 / 5×5、Sobel 边缘检测 |
| 直方图 | 计算、均衡化 |
| 几何 | 双线性缩放 2× / 0.5× |
| 形态学 | 腐蚀 3×3、膨胀 3×3 |
| 阈值 | 固定阈值、Otsu 二值化 |
| 色彩空间 | RGB→HSV、RGB→YUV |
| 数据传输 | 主机→设备上传、设备→主机下载 |

流水线部分随后通过 `PipelineProcessor` 以顺序方式以及 1 / 2 / 4 / 8 条 CUDA 流处理一批 10 张图像，并报告每种配置的整批耗时。

## 计时方式

测试框架使用手写的 `std::chrono` 计时器（构建虽然链接了 Google Benchmark 库，但当前框架并未使用它）：

- 10 次不计时的预热调用
- 100 次计时迭代
- 计时循环前后各调用一次 `cudaDeviceSynchronize()`
- 报告值 = 总耗时 / 迭代次数

各操作的计时只测量数据已驻留在设备上时的内核执行；传输开销单独体现为"上传/下载"两个条目。

## 涉及的优化技术

以下技术在被测量的内核中真实实现：

### 1. 共享内存分块

卷积内核将输入分块连同边缘（halo）区域缓存到共享内存（`src/operators/convolution_engine.cu`），每个像素只从全局内存加载一次，即可在所有重叠的卷积核位置复用。

### 2. 原子直方图

直方图计算使用 `atomicAdd` 累加bin（`src/operators/histogram_calculator.cu`），所有线程可并发更新直方图，无需单独的归约步骤。

### 3. uchar4 向量化

逐像素算子通过 `uchar4` 每线程读写 4 字节（`src/operators/pixel_operator.cu`），将四次字节访问合并为一次向量化访问。

纹理内存、warp 级原语（`__shfl`/`__reduce`）、pinned/zero-copy 内存在当前代码中**均未使用**。

## 复现基准测试

基准测试默认关闭（`BUILD_BENCHMARKS=OFF`）：

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/bin/gpu_image_benchmark
```

可执行文件会先打印检测到的 CUDA 设备，然后按图像尺寸各输出一张延迟表，最后是流水线结果。

## 测试方法

详见 [测试方法](./methodology) 了解计时方式与结果解读。
