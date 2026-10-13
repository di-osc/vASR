# vASR

基于 Rust 与 Candle 的通用语音识别推理框架，支持离线音频转写与实时语音识别。

## 简介

框架提供模型推理、语音活动检测（VAD）、音频分段、请求调度和评测等模块，可按需组合，并支持不同的 ASR 模型。

支持 CUDA、Apple Metal 和 CPU，可部署在 NVIDIA GPU、Apple Silicon 或纯 CPU 环境。

## 特点

- **Rust 原生实现**：音频特征提取、分词和模型推理等环节均由 Rust 完成，部署时无需依赖 Python 等外部运行时。
- **高效推理**：面向长音频和并发请求优化处理流程，结合连续批处理与 PagedAttention 提升资源利用率，并按音频时长估算 KV cache 需求。性能数据见[性能](#性能)。
- **多硬件支持**：基于 Candle 统一运行时，支持 CUDA、Apple Metal 与 CPU，可部署于 NVIDIA GPU、Apple Silicon 和纯 CPU 环境。

## 安装

从仓库根目录执行。Apple Silicon（Metal）：

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
