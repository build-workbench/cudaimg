# GpuImage

GPU 内存图像容器。

## 概述

`GpuImage` 是一个轻量级 struct，将 GPU 内存缓冲区与图像元数据组合在一起。
内存生命周期由内嵌的 `DeviceBuffer`（RAII）管理。

## 定义

```cpp
struct GpuImage {
    DeviceBuffer buffer;   // GPU 内存
    int width = 0;         // 图像宽度（像素）
    int height = 0;        // 图像高度（像素）
    int channels = 0;      // 颜色通道数（1、3 或 4）

    size_t pitch() const;       // 每行字节数（width * channels）
    size_t totalBytes() const;  // 总字节数（width * height * channels）
    bool isValid() const;       // 缓冲区有效且尺寸已设置时为 true
    size_t pixelCount() const;  // 像素总数（width * height）
};
```

所有字段均为 public——直接读取 `gpu.width`、`gpu.height`、`gpu.channels`
（它们是字段，不是访问器方法）。

## 创建

GpuImage 对象通常由 `ImageProcessor` 创建：

```cpp
ImageProcessor processor;
GpuImage gpu = processor.loadFromHost(hostImage);
```

## 内存管理

### RAII 模式

当 `GpuImage` 超出作用域时自动释放内存
（其 `DeviceBuffer` 成员被释放）：

```cpp
{
    GpuImage gpu = processor.loadFromHost(host);
    // 使用 gpu...
}  // 内存在此释放
```

### 移动语义

`GpuImage` 可移动；移动会转移缓冲区所有权：

```cpp
GpuImage gpu1 = processor.loadFromHost(host);
GpuImage gpu2 = std::move(gpu1);  // gpu1 不再拥有缓冲区
```

## 配合操作使用

```cpp
GpuImage gpu = processor.loadFromHost(host);

// 操作返回新的 GpuImage
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);

// 下载到主机
HostImage result = processor.download(blurred);
```

## HostImage

主机端对应类型 `HostImage` 使用 `std::vector` 存储像素数据：

```cpp
struct HostImage {
    std::vector<unsigned char> data;
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char& at(int x, int y, int c);  // 带边界检查的像素访问
    size_t totalBytes() const;
    bool isValid() const;
};
```

## 另见

- [DeviceBuffer](./device-buffer) - GPU 内存管理
- [ImageProcessor](./image-processor) - 主 API
