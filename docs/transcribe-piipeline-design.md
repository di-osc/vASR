# vASR Transcribe Pipeline 设计

本文档记录 `vasr-transcribe` 当前服务模式下的异步流水线设计、阶段职责、队列关系、音频处理边界和请求级耗时日志口径。

## 目标

Transcribe 服务需要在一个 HTTP 请求中处理多个音频输入，并让加载、VAD、ASR 编码和解码阶段并行流水执行。核心目标是：

- loader 只负责获取和解码音频，不负责模型输入采样率转换。
- VAD 和 ASR 各自负责自身需要的音频预处理。
- 同一个请求内的多个音频、多个 VAD segment 可以跨阶段并行推进。
- 服务端最终 `request_done` 日志只保留请求总览。
- 每个请求的每个阶段结束时单独打印阶段日志，用于定位瓶颈来自计算还是排队。

## 输入模型

HTTP transcribe input 的核心字段是：

- `audio source`: URL 或 base64 音频源。
- `mono`: 是否将原始音频作为单通道处理。

服务端不再依赖输入显式声明 channel。loader 会根据 `mono` 和原始音频的实际通道数决定输出：

- `mono=true`: 输出一个 `mono` channel，必要时对原始多通道做 downmix。
- `mono=false` 且原始音频是双通道或多通道: 输出 `left` 和 `right` 两路。
- 原始单通道: 输出 `mono`。

请求内会基于 `source + mono` 做去重，同一个请求中重复输入只跑一份 canonical job，结果完成后再展开回原始 index。

## 音频加载与重采样边界

当前 loader 使用 `AudioLoader::load_raw()`：

- Path: 解码本地音频。
- URL: 支持本地 file URL 或远端 URL 解码。
- Base64: 解码 base64 音频。
- Bytes: 按现有 PCM16/16k 约定构造 waveform。
- Waveform: clone 传入 waveform。

`load_raw()` 不做 normalize，也不做 resample。原有 `AudioLoader::load()` 仍保留旧语义，会按 `AudioLoadOptions` 做 normalize/resample，供其它调用继续使用。

VAD 在自己的阶段内做 VAD 需要的预处理：

- 先转 mono。
- 再 normalize/resample 到 `SAMPLE_RATE_HZ = 16000`。
- VAD detect 在 16k mono waveform 上执行。
- VAD 产出的时间段是毫秒时间范围，后续切片仍然切 loader 产出的原始采样率 waveform。

ASR 接收原始采样率的 segment waveform。Qwen ASR 路径根据 waveform 的 `sample_rate` 在 ASR 内部完成模型需要的 normalize/resample。

## 流水线结构

服务模式使用全局异步流水线：

```text
HTTP request
  -> submit worker
  -> loader workers
  -> loader_to_vad queue
  -> VAD workers
  -> vad_to_asr queue
  -> ASR worker
       -> segment encoder worker
       -> encoder_to_decoder queue
       -> segment decoder worker
  -> result router
  -> HTTP response
```

### Submit Worker

Submit worker 为每个 HTTP 请求分配内部 `PipelineRequestId`，保存 pending request 状态，并为每个 unique input 生成 `GlobalLoadJob`。

Pending request 中记录：

- 剩余结果数。
- duplicate index 映射。
- internal job index 列表。
- request-level stage 进度。
- response channel。

服务端的短请求 id 会传入 pipeline，用于阶段日志，保证阶段日志和 `request_start/request_done` 共享同一个 id。

### Loader

Loader worker 的职责：

- 获取并解码原始音频。
- 根据 `mono` 产出一个或两个 channel waveform。
- 给每个 `LoadedJob` 附加 loader run span。
- 将结果送入 `loader_to_vad` queue。

Loader 阶段不做 resample，不做 normalize。

### VAD

VAD worker 从 `loader_to_vad` queue 取出 `LoadedJob` 后启动 VAD task。

VAD task 的职责：

- 为 VAD 构造 16k mono waveform。
- 执行 VAD detect。
- 如果有 speech segment，使用 VAD 的毫秒时间范围切原始 waveform。
- 每个 segment 转成 `AsrSegmentJob` 送入 `vad_to_asr` queue。
- 如果 VAD disabled 或 no speech，按对应逻辑直接送到 ASR/result 路径。

### ASR Segment Encode

ASR worker 收到 `AsrSegmentJob` 后，如果 ASR 是 Qwen3ASR，会发送到 segment encoder worker。

Segment encoder worker 会：

- 按 `max_batch_audio_sec` 收集 segment batch。
- 执行音频 encoder，产出 CPU audio features。
- 给 segment 附加 `audio_encoder` run span。
- 将 encoded batch 送入 `encoder_to_decoder` queue。

### ASR Prefill / Decode

Segment decoder worker 从 `encoder_to_decoder` queue 取 encoded batch。

Decoder worker 会：

- 执行 prompt prefill。
- 执行 token decode。
- 将 Qwen 输出转换为 timeline。
- 将 segment 结果聚合回 job 结果。

在开启 timing feature 时，prefill span 由 decoder span 和 Qwen generation timing 推导出来：

- prefill run: decoder worker 开始到 prefill 结束。
- decode run: prefill 结束到 decoder worker 完成。

如果无法得到 prefill timing，则 decode fallback 为整个 decoder worker span。

### Result Router

ASR result 通过 result router 回到 pending request：

- 根据 `request_item` 或 `job_routes` 找到所属 request。
- 收集 outcome。
- request remaining 归零后，展开 duplicate outcomes。
- 移除 job routes。
- 通过 oneshot channel 返回 HTTP handler。

## 队列和阶段日志口径

阶段日志为 request-level 统计，不是每个 job/batch 的日志。

当前阶段日志模板：

```text
{id} {stage:<7} DONE  | audio={audio}s | elapsed={elapsed}s | active={active}s | wait={wait}s | speed={speed}x
```

阶段名固定为：

- `load`
- `vad`
- `encode`
- `prefill`
- `decode`

日志级别是 `INFO`。

### active

`active` 表示本阶段实际执行的 request-level 覆盖窗口。

如果一个请求在该阶段拆成多个并行 task 或多个 batch，`active` 不是所有 task 时间求和，而是：

```text
active = min(active_start) 到 max(active_end)
```

这样可以表达“这个请求在该阶段实际有工作运行的覆盖时间”。

### wait

`wait` 表示上游产物已经可用，但当前阶段尚未开始处理的 request-level 等待覆盖窗口。

同样，它不是所有 item/segment 等待时间求和，也不是平均等待，而是：

```text
wait = min(wait_start) 到 max(wait_end)
```

各阶段 wait 来源：

- `load`: 无上游队列，`wait=0`。
- `vad`: loader 完成到 VAD task 开始。
- `encode`: VAD 完成到 encoder batch 开始。
- `prefill`: encode 完成到 decoder/prefill 开始。
- `decode`: 有 prefill timing 时为 `0`，因为 decode 紧跟 prefill，在同一个 decoder worker 内；没有 prefill timing 时 fallback 为 encode 完成到 decoder worker 开始。

### elapsed

`elapsed` 是 wait span 和 active span 的并集覆盖时间：

```text
elapsed = union(wait_span, active_span)
```

不能简单使用：

```text
elapsed = wait + active
```

因为 request 内多个 item/segment 会并行推进，wait 和 active 覆盖窗口可能重叠。直接相加会双算重叠部分，甚至导致阶段 elapsed 大于整个 request elapsed。

### speed

`speed` 使用：

```text
speed = audio / elapsed
```

它表达该 request 在当前阶段的端到端阶段吞吐，包含排队影响。

## 日志示例

```text
7b622a8f request START | items=99
7b622a8f load    DONE  | audio=5025.0s | elapsed= 1.703s | active= 1.703s | wait= 0.000s | speed=2950.7x
7b622a8f vad     DONE  | audio=5025.0s | elapsed= 3.858s | active= 3.749s | wait= 0.109s | speed=1302.6x
7b622a8f encode  DONE  | audio=4169.4s | elapsed=11.959s | active=11.959s | wait= 0.480s | speed= 348.6x
7b622a8f prefill DONE  | audio=4169.4s | elapsed=12.276s | active=12.276s | wait= 0.090s | speed= 339.6x
7b622a8f decode  DONE  | audio=4169.4s | elapsed=12.916s | active=12.916s | wait= 0.000s | speed= 322.8x
7b622a8f request DONE  | items=99 bad=0 | audio=5025.0s ch=mono:99 stereo:0 total:99 | elapsed=13.081s | speed= 384.1x
```

注意：示例中的 `elapsed` 是覆盖窗口，不保证等于 `active + wait`。

## 如何解读 wait/active

`active` 大通常说明当前阶段自身计算耗时明显。

`wait` 大通常说明当前阶段前存在排队、调度等待或 backpressure。例如：

- VAD wait 大：loader 产出速度高于 VAD 消费速度，或者 VAD worker 不足。
- encode wait 大：VAD segment 产出速度高于 encoder 消费速度，或者 encoder 受 GPU/批处理调度影响。
- prefill wait 大：encoded features 已准备好，但 decoder/prefill 侧排队，可能受连续 decode、KV cache 或 GPU 资源占用影响。

当前 wait 是 request-level 覆盖窗口，适合判断整体是否有明显排队。它不适合回答“每个 segment 平均等多久”或“p90 等待多久”。如果需要分布诊断，应增加：

```text
q_avg q_p50 q_p90 q_max
```

## request_done

最终 `request_done` 只打印请求总览：

```text
{id} request DONE  | items={} bad={} | audio={:.1}s ch=mono:{} stereo:{} total:{} | elapsed={:.3}s | speed={:.1}x
```

它不包含阶段耗时。阶段耗时由每个 `{id} {stage} DONE` 日志单独给出。

## 当前限制

- request-level wait/active 是覆盖窗口，不是累计耗时。
- 阶段日志主要服务于请求级瓶颈定位，不提供 item/segment 等待分布。
- prefill span 依赖 Qwen timing 推导；无法取得 timing 时，decode 会 fallback 到 decoder worker span。
- raw ASR fallback 路径没有完整的 encode/prefill/decode 分段语义。
