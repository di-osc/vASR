# vASR

基于Rust实现的高性能语音理解引擎。


## 特点

- **Rust**：使用Rust实现，无Pytorch等Python库依赖，单个二进制文件即可运行。
- **高性能**：使用GPU处理大批量音频时，能达到900+实时率的吞吐量，SOTA级别的处理速度。
- **多后端**：支持CUDA、Metal加速，可部署于 NVIDIA GPU、Apple Silicon 和纯 CPU 环境。

## 安装

Apple Silicon（Metal）：

```bash
cargo install --path vasr-transcribe --force --features metal
```

NVIDIA GPU（CUDA）：

```bash
cargo install --path vasr-transcribe --force --features cuda
```

## 性能

| 应用 | 模型 | 设备 | 吞吐（音频秒 / 墙钟秒） | 实时率 RTF | 加权 CER | 运行命令 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `vasr-transcribe` | Qwen3-ASR-0.6B | Apple M4 Pro（48 GB，Metal） | **123.30×**（中位） | 0.0081 | 16.5876% | `vasr-transcribe serve --model Qwen3-ASR-0.6B`<br>`vasr-transcribe perf -i datasets/call-100.db` |
| `vasr-transcribe` | Qwen3-ASR-0.6B | Apple M4 Pro（48 GB，Metal） | **118.65×**（最高实测） | 0.008428 | 4.3336% | `vasr-transcribe serve --model Qwen3-ASR-0.6B`<br>`vasr-transcribe perf -i datasets/aishell1-test-200.db --concurrency 14 --batch-size 16` |
| `vasr-transcribe` | Qwen3-ASR-0.6B | NVIDIA RTX 4090（24 GB，CUDA） | **831.20×**（中位） | 0.001203 | 16.8500% | `VASR_CUDA_TUNED=1 vasr-transcribe serve`<br>`vasr-transcribe perf -i datasets/call-100.db --concurrency 8 --batch-size 16` |
