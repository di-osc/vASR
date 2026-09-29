# vASR

vASR 是用 Rust 和 Candle 实现的 Qwen3-ASR 推理框架。离线转写和实时识别共用同一套模型运行时：音频按时间线组织，请求进入全局流水线后连续批处理，而不是每个请求单独占住 GPU。

## 特点

- **连续批处理。** 所有请求共享一个常驻 decode 循环。序列生成完立刻让出槽位，新序列马上补进去，不等一个 HTTP 请求里的音频全部结束。
- **理论最佳吞吐。** decode 不再按请求边界停顿，GPU 按配置的最大并发序列数打满。吞吐接近该批大小下的计算上限，而不是请求级组批留下的空档。
- **PagedAttention。** KV cache 按块分配。每条序列只占用实际长度，不用按最大上下文预留整段显存。CUDA 与 Metal 都有实现。
- **FlashAttention。** CUDA 上使用 bf16 的 varlen FlashAttention，覆盖定长和变长、因果与非因果注意力。Metal 走对应的加速 SDPA。
- **CUDA decode graph。** 预先捕获 decode 步，重放时省掉重复的 kernel 启动。
- **分段流水线。** 加载、VAD、音频编码、prefill、decode 分成独立阶段。编码阶段会把已经排队的语音段合成一批，prefill 期间还可以并入后续批次。
- **离线与实时。** `vasr-transcribe` 提供 `GET /health` 和 `POST /transcribe`。`vasr-realtime` 提供 WebSocket 实时会话。
- **VAD。** 离线默认使用 FunASR FSMN，把长音频切成语音段；无语音输入直接返回空结果。实时使用 FireRed Stream-VAD。
- **量化与精度。** CPU 使用 f32，Metal 和 CUDA 使用 bf16。支持就地量化，例如 `q8_0`。
- **时间线数据。** 音频、标注和识别结果落在 `Timeline` 上，数据集用 SQLite AudioDb 保存，评测直接对标注算字错误率。
- **重复输入合并。** 同一请求里相同的音频只推理一次，结果按原始顺序展开。

## 离线 HTTP 服务

```bash
vasr-transcribe serve
```

`serve` 默认按所选设备、模型和可用内存估算 KV 容量，再解析出实际执行批量，启动日志会打印最终生效的策略：

```
serve 执行策略 | device=metal capacity=512 active_slots=32 decode_rows=32 mixed_step=true prefill_wave=8
```

日常启动不需要手写并发参数。需要覆盖时可用 CLI 参数（`--max-num-seqs`、`--gpu-memory-utilization`、`--encoder-batch-max-audio-seconds`）或环境变量（`VASR_METAL_MAX_ACTIVE_SEQS`、`VASR_METAL_PAGED_DECODE_BATCH_SIZE`、`VASR_PREFILL_MAX_WAVE`、`VASR_DISABLE_MIXED_STEP_DECODE`）。显式容量超过估算的安全范围，或低于 VAD 最大分段加 padding 的要求时，服务会在启动阶段报错。

服务也会在启动时读取当前工作目录下的 `.env`。仓库提供了 [`.env.example`](.env.example)；复制为 `.env` 后可在其中设置部署参数。优先级为 CLI 参数、已有进程环境变量、`.env`，最后是自动值。所有子命令都会加载 `.env`，文件不存在时会忽略。

## 推理速度

测量环境：Apple M4 Pro（48 GB 统一内存，Metal，bf16）。模型 `Qwen3-ASR-0.6B`，数据集 `datasets/call-100.vasr`（99 条，合计 5025 音频秒，单条 3.6–705 秒）。并发 14 / batch 1。

| 指标 | 默认配置 | 基线（1 活跃槽 / 1 decode 行） |
| --- | ---: | ---: |
| 吞吐（音频秒 / 墙钟秒） | **43.8×** | 21.8× |
| 实时率 RTF | **0.0228** | 0.0459 |
| 1 小时音频耗时 | **约 82 秒** | 约 165 秒 |
| 延迟 p50 | 14.7 s | 30.3 s |
| 延迟 p99 | 38.3 s | 62.4 s |

即每秒可转写约 44 秒音频，比基线快 **2.01×**。`eval` 口径（含评测开销）为 45.4×。单条短音频（`example1.vasr`，4.2 秒）耗时 0.153 秒，约 27.6×——短音频受固定开销影响，倍速低于长音频批量。

吞吐随并发上升并很快饱和，服务内部流水线在单个客户端下也能跑满：

| 并发 | 1 | 4 | 8 | 14 | 28 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 吞吐 | 32.9× | 42.5× | 45.0× | 44.7× | 45.2× |

并发 ≥ 8 后稳定在约 45×；即使只有 1 路串行客户端也有 32.9×（已超过基线并发 14 时的 21.8×），因为收益来自服务内的 32 槽连续批处理，而不是靠客户端堆并发。

30 分钟持续来流（打乱输入，并发 14）实测 44.9×，1598 个请求零失败、零 OOM，延迟 p50 14.1 s / 最大 42.8 s，进程 RSS 稳定在 2.2–2.9 GiB 无泄漏。

### 复现

服务起来后（见上一节），另开终端：

```bash
vasr-transcribe perf -i datasets/call-100.vasr --concurrency 14 --batch-size 1
vasr-transcribe eval -i datasets/call-100.vasr --timeline mono --normalizer bench
```

### 与基线的质量对照

| 指标 | 基线 | 默认配置 |
| --- | ---: | ---: |
| 加权 CER | 0.167497 | 0.167858 |
| `example1.vasr` CER | 0.000000 | 0.000000 |

默认配置的加权 CER 比基线高 0.000361（+0.22% 相对）。经隔离实验确认，该偏移只由 decode 行批处理触发（把 decode 行降到 1 时逐项错误数与基线完全相同），encoder 多段与 prefill wave 均已排除，且它是吞吐收益的唯一来源。完整数据、容量扫描与根因分析见 [docs/performance/m4-pro-qwen3-asr-2026-09-30.md](docs/performance/m4-pro-qwen3-asr-2026-09-30.md)。
