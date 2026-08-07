# CUDA 概念速查

按概念查找源码位置。每个概念标注了演示该概念的文件和函数。

## 线程模型

| 概念 | 位置 | 说明 |
|------|------|------|
| grid/block/thread 层次 | `pixel_operator.cu` `invertKernelScalar` | 最简单的 2D kernel |
| 2D 线程索引映射 | `pixel_operator.cu` 所有 scalar kernel | `(x, y) -> 线性索引` |
| 1D 线程索引映射 | `pixel_operator.cu` `invertKernelVec4` | 向量化版本的 1D grid |
| 边界检查 | 所有 kernel 的 `if (x < width && y < height)` | grid 向上取整产生的多余线程 |
| block 尺寸选择 | `kernel_helpers.hpp` `kBlockSize2D = 16` | 16×16 = 256 线程/block |

## 内存模型

| 概念 | 位置 | 说明 |
|------|------|------|
| Global memory 读写 | 所有 kernel | 最基本的 GPU 内存访问 |
| Shared memory tiling | `convolution_engine.cu` `convolveKernelShared` | tile + halo 协作加载 |
| Shared memory 静态数组 | `histogram_calculator.cu` `__shared__ int localHist[256]` | 编译时已知大小 |
| Shared memory 动态分配 | `convolution_engine.cu` `extern __shared__ float sharedMem[]` | 运行时指定大小 |
| `uchar4` 向量化 | `pixel_operator.cu` `invertKernelVec4` | 4 字节/线程 |
| `__constant__` 内存 | （未使用） | 教学扩展方向 |
| 纹理内存 | （未使用） | 教学扩展方向，见 `image_resizer.cu` 注释 |

## 同步

| 概念 | 位置 | 说明 |
|------|------|------|
| `__syncthreads()` | `convolution_engine.cu`, `histogram_calculator.cu` | block 内同步屏障 |
| `cudaDeviceSynchronize()` | `execution_context.hpp` Sync 模式 | 全设备同步 |
| `cudaStreamSynchronize()` | `execution_context.hpp` Async 模式 | 单 stream 同步 |
| stream 并发 | `pipeline_processor.cu` | 多 stream 异步流水线 |

## 原子操作

| 概念 | 位置 | 说明 |
|------|------|------|
| `atomicAdd` (block 级) | `histogram_calculator.cu` `atomicAdd(&localHist[value], 1)` | shared memory 上原子加 |
| `atomicAdd` (全局级) | `histogram_calculator.cu` `atomicAdd(&histogram[i], localHist[i])` | global memory 上原子加 |
| 两级规约 | `histogram_calculator.cu` | block 级 shared -> 全局 atomicAdd |

## 内存管理

| 概念 | 位置 | 说明 |
|------|------|------|
| RAII 显存管理 | `device_buffer.hpp` | 构造分配、析构释放 |
| `cudaMalloc` / `cudaFree` | `device_buffer.cu` | 底层 CUDA 内存分配 |
| H2D / D2H 拷贝 | `device_buffer.hpp` `copyFromHost` / `copyToHost` | 主机-设备数据传输 |
| D2D 拷贝 | `device_buffer.hpp` `copyFromDevice` | 设备间拷贝 |
| 异步拷贝 | `device_buffer.hpp` `copyFromHostAsync` | stream 异步传输 |
| 内存池 | `memory_manager.hpp` | 可选的缓冲区复用（默认关闭） |
| 移动语义 | `device_buffer.hpp` move constructor/assignment | 独占所有权转移 |

## Kernel 启动

| 概念 | 位置 | 说明 |
|------|------|------|
| `<<<grid, block>>>` | 所有 `.cu` 文件 | kernel 启动语法 |
| `<<<grid, block, sharedBytes, stream>>>` | `convolution_engine.cu` | 带 shared memory 和 stream 的完整启动 |
| `cudaGetLastError()` | 所有 kernel 启动后 | 捕获 kernel 启动错误 |
| `calcGridBlock2D()` | `kernel_helpers.hpp` | 自动计算 grid 维度 |

## 设备端工具

| 概念 | 位置 | 说明 |
|------|------|------|
| `__device__` 函数 | `image_resizer.cu` `bilinearInterpolate` | kernel 可调用的设备端函数 |
| `__device__ inline` 工具 | `device_kernels.cuh` | 跨文件复用的设备端工具 |
| `min` / `max` (CUDA 内建) | 多个 kernel | 设备端无需 `<algorithm>` |
| `fminf` / `fmaxf` | 多个 kernel | 浮点版本的 min/max |
| `make_uchar4` | `pixel_operator.cu` | 构造 `uchar4` 向量 |

## 架构模式

| 概念 | 位置 | 说明 |
|------|------|------|
| 门面模式 | `image_processor.hpp` `ImageProcessor` | 统一入口，隐藏 stream/buffer 细节 |
| 策略模式 | `execution_context.hpp` `ExecutionPolicy` | sync/async/batch 可切换 |
| 模板方法 | `image_operator.hpp` `UnaryOperator<T>` | CRTP 消除虚函数开销 |
| 管道模式 | `operator_pipeline.hpp` `OperatorPipeline` | 链式组合多个算子 |
| dispatch fallback | `pixel_operator.cu` `launchInvert` | 快路径 + 通用 fallback |
