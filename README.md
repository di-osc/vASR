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
| `vasr-transcribe` | Qwen3-ASR-0.6B | Apple M4 Pro（48 GB，Metal） | **118.65×**（最高实测） | 0.008428 | 4.3336% | `vasr-transcribe serve --model Qwen3-ASR-0.6B`<br>`vasr-transcribe perf -i datasets/aishell1-test-200.db --concurrency 14 --batch-size 16` |
| `vasr-transcribe` | Qwen3-ASR-0.6B | NVIDIA RTX 4090（24 GB，CUDA） | **833.76×**（中位） | 0.001199 | 16.7558% | `vasr-transcribe serve --vad-device cpu --num-vad-workers 8 --num-load-workers 4 --encoder-batch-max-audio-seconds 240`<br>`vasr-transcribe perf -i datasets/call-100.db --concurrency 8 --batch-size 16` |

CUDA 一行是 RTX 4090 上 `call-100`（99 条、5,025 秒音频）的实测中位数，同时设置 `VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS=32`、`VASR_CUDA_CONV_CROSS_SAMPLE=1`、`VASR_CUDA_CONV_CHUNKS=64`、`VASR_CUDA_SERIAL_FORWARD=1`、`VASR_VAD_NUM_THREADS=1`、`VASR_CUDA_GRAPH_BATCH_STEP=4`、`VASR_ASR_PREFILL_COALESCE_WAIT_MS=50`，并以 `cargo install --path vasr-transcribe --force --features cuda,cudnn` 构建。

编码器默认按长度分桶：连续批处理下，长度相近的分段会进入同一编码批次，减少批内补零。`call-100` 上 `encoder_frames` 的 executed/valid 帧由 2.91× 降到 1.66×，端到端吞吐由约 790× 提升到约 830×。可用 `VASR_ASR_BUCKET_BY_LENGTH=0` 退回按到达顺序分组。
