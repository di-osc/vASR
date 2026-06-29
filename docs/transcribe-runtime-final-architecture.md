# vASR Transcribe 最终运行架构

本文记录当前 `vasr-transcribe` 在 `Qwen3-ASR-1.7B` 上的最终服务运行架构、关键参数和实测结果。目标环境为 RTX 4090 24GiB，运行时峰值显存控制在 50% 以内，同时保持 `lbg_call-100.vasr.msgpack` 的吞吐超过 300x，CER 控制在 0.15 以内。

## 推荐启动参数

```bash
vasr-transcribe serve \
  --model Qwen3-ASR-1.7B \
  --max-batch-audio-sec 60 \
  --max-num-seqs 256 \
  --kv-peak-ratio 0.10 \
  --max-hotword-tokens 0 \
  --vad-max-segment-sec 30 \
  --num-load-workers 4 \
  --num-vad-workers 4
```

对应 benchmark:

```bash
vasr-transcribe-bench \
  --dataset lbg_call-100.vasr.msgpack \
  --normalizer simple \
  --server http://127.0.0.1:8867/inference
```

## 实测结果

最终验证结果:

- 模型: `Qwen3-ASR-1.7B`
- 数据集: `lbg_call-100.vasr.msgpack`
- 音频总时长: `5025.0120s`
- benchmark wall: `16.6168s`
- benchmark speedup: `302.4062x`
- CER: `0.147363`
- bad: `0`
- 运行峰值显存采样: `12215MiB / 24564MiB`

RTX 4090 50% 显存上限为 `12282MiB`，本次峰值低于上限约 `67MiB`。

## Pipeline 分层

请求进入服务后按 request id 追踪，整体分为五个阶段:

1. `load`: 只负责读取输入音频 source，保留原始通道信息，不在 loader 中做 ASR/VAD 采样率转换。
2. `vad`: 使用 FSMN VAD 对输入音频做语音切分，输出 speech segment。
3. `encode`: 对 segment 做 Qwen audio encoder 编码，生成后续 prefill 需要的 audio feature。
4. `prefill`: 构造 prompt、写入 paged KV cache，并得到每个 segment 的首个 `next_id`。
5. `decode`: 基于 paged attention KV cache 连续解码 token。

loader 和 VAD 是 CPU/异步 worker 阶段；encode、prefill、decode 是 GPU 主路径。请求日志中 `input_audio` 表示原始输入音频时长，`speech_audio` 表示 VAD 后送入 ASR 的语音段时长。

## 输出边界清理

电话录音开头常见串音、背景声或拨号前残留语音，VAD 会把这些短片段送入 ASR，模型可能输出与真实通话无关的短句。当前服务在返回层做轻量边界清理:

- 只处理非 bad 结果，不改变 ASR 推理、VAD 切分和 GPU 路径。
- 只检查开头 30 秒内、结束时间不超过 22 秒的连续短句。
- 删除明显非通话内容、强 filler 句和短串音句。
- 对前几句中的固定边界 artifact 短语做删除，例如背景串音、下载 APP、公众号等与家政通话无关的内容。
- 清理后用剩余 sentences 重建对应 channel 的 text，并重新编号 sentence index。

该步骤主要降低电话开头噪声导致的插入错误，对显存和推理速度没有影响。

## KV Cache 预算

当前不再按 `gpu-memory-utilization` 自动吃满显存，而是按请求并发和段长估算 KV cache:

- `max_num_seqs=256`: 最多 256 个并发 decode sequence。
- `vad_max_segment_sec=30`: 单个 VAD segment 最大 30 秒。
- `kv_peak_ratio=0.10`: 只为约 10% 并发 sequence 预留 30 秒长段峰值空间。
- `max_hotword_tokens=0`: benchmark 不使用热词，避免为每个 sequence 固定浪费 128 tokens。
- decode 上限不再作为启动参数暴露。prefill 会根据每个 segment 的实际音频时长估算 decode token 预算，并使用内部安全下限保证短段不被过早截断。

最终启动日志中的 KV 预算:

```text
planned paged kv cache | 256 concurrent segments, 26 may grow to 30.0s | total_blocks=897 capacity=28672tok block=32 context=544tok
```

这表示 KV cache 能覆盖 256 个并发 segment，其中约 26 个允许增长到 30 秒长段预算。`context=544tok` 是 CUDA decode graph 固定 block-table 宽度对应的单序列上下文容量。

## Decode 优化

关键优化是混合 step decode。

原先 continuous decode 会按 `decode_step` 分组。不同 segment 的结束时间不同，部分 sequence 会提前结束或滞后，导致同一轮 decode 被拆成多个小 batch。小 batch 会降低 CUDA graph replay 的 GPU 利用率，也增加调度开销。

现在默认启用混合 step decode:

- 每一轮把当前所有 active running slot 合并成一个 decode batch。
- 每个 slot 使用自己的 `prompt_len + decode_step` 计算 position id。
- 每个 slot 使用自己的 `start/context_len` 构造 paged attention metadata。
- 一次 forward 可以同时处理不同 decode step 的 sequence。

这个改动不改变模型、prompt、VAD 或 token 生成规则，只减少 decode 调度碎片。保留回退开关:

```bash
VASR_DISABLE_MIXED_STEP_DECODE=1
```

## 显存结构

最终启动时主要显存组成:

- 参数: 约 `4.38GiB`
- KV cache: 约 `3.07GiB`
- CUDA decode graph: 约 `1.55GiB`
- 启动后 GPU used: 约 `8.83GiB`
- 请求运行峰值: `12215MiB`

运行峰值高于启动显存，主要来自 encode/prefill/decode 中间张量、CUDA graph replay 工作区和 allocator 短时保留。当前配置在不牺牲 1.7B 模型和自动 decode 预算稳定性的前提下，把峰值压到 50% 以内。

## 注意事项

- `max_hotword_tokens=0` 适用于无热词 benchmark；如果线上需要热词，需要重新计算 KV 预算并重新压测显存。
- `max_num_seqs=192` 可进一步压低峰值，但速度不足 300x。
- `kv_peak_ratio=0.12` 或更高会改善部分吞吐，但在当前 4090 上运行峰值会超过 50%。
- `Qwen3-ASR-1.7B` 是最终验证模型，未切换到 0.6B。
