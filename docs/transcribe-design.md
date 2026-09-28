# Transcribe 设计

## Pipeline

当前服务使用 `GlobalAsyncTranscribePipeline`。`serve` 模式下，所有 HTTP 请求共享同一套 loader、VAD、ASR、encoder、prefill、decoder worker。请求进入服务后会先拆成去重后的音频/通道 job，结果由 `Global Result Router` 按原始输入顺序合并。

```text
                    HTTP request
                       |
                       v
+--------------------------------------------------+
| Submit Worker                                     |
| 拆分输入、去重、记录 request_id/job route           |
+--------------------------------------------------+
                       |
                       | load_job_tx, unbounded
                       v
+--------------------------------------------------+
| Global Loader Worker                              |
| 默认并发 1；--num-load-workers=0 使用 CPU 核心数     |
| 加载 URL/path/base64/bytes，解码为 Waveform          |
+--------------------------------------------------+
                       |
                       | loader_to_vad
                       | 消息不按数量限流，按 waveform 内存 permit 限流
                       v
+--------------------------------------------------+
| VAD Worker                                        |
| 默认并发 1；--num-vad-workers=0 使用 CPU 核心数      |
| 无语音 item 直接返回空结果                          |
| 有语音 item 切成 AsrSegmentJob                      |
+--------------------------------------------------+
                       |
                       | vad_to_asr
                       | 消息不按数量限流，按 segment waveform 内存 permit 限流
                       v
+--------------------------------------------------+
| ASR Worker                                        |
| 单 async worker                                   |
| Job fallback: collect_job_microbatch()             |
| NoSpeech/Disabled job 走 job microbatch            |
| Speech segment 直接送 encoder                       |
+--------------------------------------------------+
                       |
                       | asr_to_encoder, unbounded
                       v
+--------------------------------------------------+
| Segment Encoder Worker                            |
| 单 async worker + spawn_blocking                   |
| collect_encoder_segment_batch()                    |
| 收集已排队 segment，并额外等待 10ms 聚合             |
| 执行 audio encoder，输出 CPU audio features          |
+--------------------------------------------------+
                       |
                       | encoder_to_prefill, unbounded
                       v
+--------------------------------------------------+
| Segment Prefill Worker                            |
| paged-attn feature 开启时存在                       |
| 单 async worker + prefill producer thread           |
| 对 encoded batch 执行 dynamic prefill                |
| prefill 期间可继续合并后续 encoded batch              |
| 输出 prefill stage / metadata 给全局 decoder          |
+--------------------------------------------------+
                       |
                       | prefill_to_decoder, unbounded
                       v
+--------------------------------------------------+
| Segment Decoder Worker                            |
| 单 async worker                                   |
| 启动后保持一个全局 continuous decode loop              |
| 持续消费所有请求的 PrefillStageOutput                  |
| chunk 完成时回调 token ids 并释放 paged KV slot       |
| StreamingMetadataDecoder 聚合 chunk -> segment       |
| segment 完成后立即发送 RoutedTranscribeItemOutcome   |
+--------------------------------------------------+
                       |
                       | asr_to_result, bounded
                       | serve capacity = 1024
                       v
+--------------------------------------------------+
| Global Result Router                              |
| 收集 RoutedTranscribeItemOutcome                   |
| 一个 request 的所有 item 完成后组装 HTTP JSON 响应     |
| duplicate input 展开回原始输入顺序                   |
+--------------------------------------------------+
```

## 队列与并发规则

```text
Loader:
  serve 默认 --num-load-workers=2；run 默认 --num-load-workers=1
  设置 N 时最多 N 个 load job in-flight
  设置 0 时使用 std::thread::available_parallelism()

VAD:
  serve 默认 --num-vad-workers=2；run 默认 --num-vad-workers=1
  设置 N 时最多 N 个 VAD job in-flight
  设置 0 时使用 std::thread::available_parallelism()
  VASR_VAD_NUM_THREADS 控制单个 VAD 模型内部 CPU 线程数；vad-device=cuda 时不用于 CUDA kernel 并行度

`serve` 的 ASR 设备与 VAD 设备默认都是 `auto`。ASR 按 CUDA、Metal、CPU 顺序选取可初始化的后端；VAD 按 CUDA、CPU 顺序选择。设备确定后若模型加载或资源分配失败，启动会直接报错，不会换设备重试。

未指定 `--max-num-seqs`、`--gpu-memory-utilization` 和 `--encoder-batch-max-audio-seconds` 时，`serve` 根据设备内存、模型大小、VAD 最大分段时长和 hotword 预留估算容量。CUDA KV cache 会为 encoder batch、CUDA graph 和 10% 安全余量预留内存。估算依赖的内存信息不可用时使用基线：512 并发、0.8 显存利用率、30 秒 / 32 段；基线分配仍然失败时直接退出。

CLI 数值优先于进程环境变量，进程环境变量优先于当前工作目录的 `.env`，缺省项再使用自动值。`.env.example` 列出了可配置环境变量。所有子命令解析参数前都会加载 `.env`；文件不存在时忽略。

ASR Worker:
  单 worker，负责把 VAD 后的 job/segment 路由到 fallback 或动态分阶段路径

Segment Encoder:
  单 worker，每次处理 collect_encoder_segment_batch() 聚合出的 batch

Segment Prefill:
  paged-attn feature 开启时单 worker
  负责 dynamic prefill，可短暂等待更多 encoded batch 以提升合批

Segment Decoder:
  单 worker
  当前是跨请求常驻的全局 decode loop
  所有 prefill worker 产生的 PrefillStageOutput 都进入同一个 decode queue
  有空 slot 就 attach prefilled seq，slot 满了就继续 decode running slots
  channel 未关闭时，即使当前没有 running/prefilled seq，也会等待新消息
  chunk/segment 完成事件即时处理，不等整个 HTTP request 的所有 item 完成

Result Router:
  serve 模式 bounded channel capacity = 1024
  local pipeline 模式 capacity = unique_inputs.len()
```

`loader_to_vad`、`vad_to_asr` 不按消息数量限流，主要靠 waveform/segment 内存 permit 控制排队内存。`asr_to_encoder`、`encoder_to_prefill`、`prefill_to_decoder` 当前使用 unbounded channel。

## 动态 Decode 返回规则

Qwen3 动态分阶段路径中，decode 的最小完成单位是 ASR chunk。一个 VAD segment 可能被模型侧切成多个 chunk，所以“chunk 完成”不等于“segment 可返回”。

```text
PrefillStageOutput
  -> Segment Decoder Worker
  -> continuous decode slot
  -> chunk tokens completed
  -> completion callback
  -> StreamingMetadataDecoder
  -> segment completed
  -> RoutedTranscribeItemOutcome
  -> Global Result Router
```

当前行为：

```text
chunk 完成:
  把 token ids 写入 outputs
  调用 completion callback
  释放该 chunk 的 paged KV slot

metadata 尚未到达:
  completed chunk 暂存在 pending map
  后续 prefill metadata 到达后再继续聚合

segment 完成:
  当该 segment 的所有 chunk 都完成后
  立即 decode 文本并生成 Timeline
  立即发送 RoutedTranscribeItemOutcome 到 result router

request 完成:
  Global Result Router 等该 request 下所有 item 完成
  再按原始输入顺序返回 HTTP JSON response
```

这意味着内部 pipeline 已经是全局流式 decode：所有请求共享同一个 decode scheduler，chunk/segment 完成即聚合、完成即路由。HTTP `/transcribe` 仍是一次请求一次 JSON 响应，不是 SSE/WebSocket 流式 API。后续如果要做对外流式服务，可以复用 `RoutedTranscribeItemOutcome` 这一层事件。

## Segment Batch 形成规则

动态 Qwen3 segment 路径中，ASR Worker 不再先聚合 segment，而是把 segment 发送给 Segment Encoder Worker。真正的 encoder batch 在 `collect_encoder_segment_batch()` 中形成：

```text
Segment Encoder Worker 收到第一个 SegmentEncodeRequest
  |
  v
collect_encoder_segment_batch()
  |
  +-- 从 asr_to_encoder 非阻塞 try_recv()
  +-- 收集所有已经排队的 SegmentEncodeRequest
  +-- 额外等待 ASR_ENCODER_BATCH_COLLECT_WAIT_MS，当前为 10ms
  +-- 按 (job_index, offset) 排序
  v
Vec<AsrSegmentJob> -> Encoder Worker
```

该动态 encoder batch 不按音频秒数截断，也不由 decoder 空闲 slot 决定。segment ASR 路径依赖 paged-attn；旧的同步 segment fallback 已删除。

## 简化流程

```text
input
  -> loader 并发加载
  -> audio waveform 内存限流队列
  -> VAD 并发切分
  -> segment waveform 内存限流队列
  -> ASR worker 路由 job/segment
  -> encoder 单 worker 生成 CPU audio features
  -> prefill 单 worker dynamic prefill
  -> decoder 单 worker 常驻执行全局 continuous decode
  -> chunk 完成即回调并释放 KV slot
  -> segment 完成即发送结果
  -> request 聚合完成后 HTTP 返回
```

## HTTP API

HTTP 服务由 `vasr-transcribe serve` 启动，目前暴露两个接口：

```text
GET  /health
POST /transcribe
```

JSON body 上限是 512 MiB。

### GET /health

用于服务存活检查，响应：

```json
{
  "status": "ok"
}
```

### POST /transcribe

请求示例：

```json
{
  "inputs": [
    {
      "url": "https://example.com/audio.wav",
      "b64_str": null,
      "mono": false,
      "hotword": "可选热词文本"
    }
  ]
}
```

请求字段说明：

```text
inputs      音频输入数组。响应 data 会保持与 inputs 相同的 item 顺序。
url         可选，音频 URL。
b64_str     可选，base64 编码后的音频 bytes。
mono        可选 bool。缺省为 false。
hotword     可选文本。当前 API 接收该字段，但服务 pipeline 尚未将它接入 ASR options。
```

每个 input 必须提供 `url` 或 `b64_str` 之一。如果两个字段同时存在，优先使用 `url`。如果两个字段都缺失，该 item 会作为 loader 错误写入响应，不会让整个 HTTP 请求失败。

`mono` 控制通道处理：

```text
mono=true   下混为一个 mono 通道。
mono=false  尽量保留 stereo；当源音频超过一个通道时，分别处理 left/right。
```

响应示例：

```json
{
  "data": [
    {
      "serviceId": "transcribe-<uuid>",
      "spentSeconds": 1.234,
      "spentDetails": {
        "loader": 0.1,
        "detector": 0.2,
        "recognizer": 0.9,
        "sentencizer": 0.0
      },
      "timings": {
        "total": 1.234,
        "loader": 0.1,
        "pipeline": 1.134,
        "audio": 12.3
      },
      "text": {
        "mono": "",
        "left": "左声道文本",
        "right": "右声道文本"
      },
      "sentences": [
        {
          "text": "左声道文本",
          "startTime": 0,
          "endTime": 1230,
          "duration": 1230,
          "channel": "left",
          "sentenceIndex": 0
        }
      ],
      "is_bad": false,
      "bad_reason": null,
      "bad_component": null
    }
  ],
  "inference_performance": {
    "batch_wall_seconds": 1.234,
    "num_items": 1,
    "throughput_items_per_second": 0.81,
    "total_audio_duration_seconds": 12.3,
    "speedup": 9.97,
    "rtf": 0.1003
  }
}
```

响应字段说明：

```text
data                         每个请求 input 对应一个结果。
serviceId                    当前服务进程生成的服务 id。
spentSeconds                 单个 item 的处理耗时，单位秒。
spentDetails.loader          加载/解码耗时，单位秒。
spentDetails.detector        VAD 耗时，单位秒。
spentDetails.recognizer      ASR 耗时，单位秒。
spentDetails.sentencizer     预留字段，当前通常为 0。
timings.total                单个 item 的总耗时，单位秒。
timings.loader               loader 耗时，单位秒。
timings.pipeline             非 loader pipeline 耗时，单位秒。
timings.audio                处理的音频时长，单位秒。
text.mono                    mono 输出文本。
text.left/text.right         stereo left/right 通道输出文本。
sentences[].startTime        句子开始时间，单位毫秒。
sentences[].endTime          句子结束时间，单位毫秒。
sentences[].duration         句子时长，单位毫秒。
sentences[].channel          mono、left 或 right。
sentences[].sentenceIndex    通道合并并排序后的句子序号。
is_bad                       该 item 是否失败。
bad_reason                   is_bad=true 时的错误详情。
bad_component                is_bad=true 时的失败组件：loader、vad 或 recognizer。
```

`inference_performance` 描述整个 HTTP 请求：

```text
batch_wall_seconds           从请求进入服务到响应组装完成的 wall-clock 秒数。
num_items                    响应 item 数。
throughput_items_per_second  num_items / batch_wall_seconds。
total_audio_duration_seconds 成功处理的输出通道音频总时长。
speedup                      total_audio_duration_seconds / batch_wall_seconds。
rtf                          batch_wall_seconds / total_audio_duration_seconds。
```

handler 通常返回 HTTP 200，并通过 `data[].is_bad` 表达 item 级错误。JSON 解析、body 超限等错误会在进入 transcribe handler 之前由 Axum 处理。

## AudioSource 映射

`AudioSource` 是 runtime 内部输入抽象，不是 HTTP API 或 msgpack 的持久化 schema。

```text
HTTP url                  -> AudioSource::Url
HTTP b64_str              -> AudioSource::Base64
本地音频文件              -> AudioSource::Path
msgpack AudioAsset::Uri   -> AudioSource::Url 或 AudioSource::Path
msgpack AudioAsset::Embedded.bytes -> AudioSource::EncodedBytes
msgpack waveform_cache    -> AudioSource::Waveform
```

`AudioSource::EncodedBytes` 表示完整音频文件 bytes，例如 mp3、wav、flac、ogg 等，进入 loader 后由 Symphonia 解码。它不是裸 PCM。内部已经解码好的音频应使用 `AudioSource::Waveform`。

## 服务日志

`RUST_LOG=info` 下，服务日志保留请求开始、请求结束，以及开启 `--profile` 后的周期 PERF 表：

```text
<id> request START | items=<n> src=url:<n> base64:<n> missing:<n> mono=true:<n> false:<n>
<id> request DONE  | items=<n> bad=<n> | audio=<s> ch=mono:<n> stereo:<n> total:<n> | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>

+---------------------------------------------------------------------------------------------+
|                         Service transcribe-<uuid> Perf | window=5.0s                         |
+---------+-------+-------+---------+-------+--------------------------------------------------+
|  stage  | count | audio | elapsed | speed |                      detail                      |
+---------+-------+-------+---------+-------+--------------------------------------------------+
|  load   |  ...  |  ...  |   ...   |  ...  |                       ...                        |
+---------+-------+-------+---------+-------+--------------------------------------------------+
|   vad   |  ...  |  ...  |   ...   |  ...  |                       ...                        |
+---------+-------+-------+---------+-------+--------------------------------------------------+
| encode  |  ...  |  ...  |   ...   |  ...  |                       ...                        |
+---------+-------+-------+---------+-------+--------------------------------------------------+
| prefill |  ...  |  ...  |   ...   |  ...  |                       ...                        |
+---------+-------+-------+---------+-------+--------------------------------------------------+
| decode  |  ...  |  ...  |   ...   |  ...  |                       ...                        |
+---------+-------+-------+---------+-------+--------------------------------------------------+
```

周期 PERF 表由 serve 参数控制：

```text
--profile                 开启周期 profile 表；默认关闭。
--profile-window <secs>   profile 统计窗口，单位秒；默认 5.0。
```

PERF 表标题是跨整张表的 panel，包含当前服务 id 和实际统计窗口：

```text
Service <service_id> Perf | window=<seconds>s
```

单请求阶段日志是 `DEBUG` 日志：

```text
<id> load    DONE | input_audio=<s>  | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>
<id> vad     DONE | input_audio=<s>  | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>
<id> encode  DONE | speech_audio=<s> | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>
<id> prefill DONE | speech_audio=<s> | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>
<id> decode  DONE | speech_audio=<s> | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>
```

`request BAD` 是 `ERROR` 日志，按失败的 item/channel 打印：

```text
<id> request BAD | index=<n> channel=<mono|left|right> component=<loader|vad|recognizer> audio=<s> reason=<error>
```

### request DONE 字段

`items` 是响应组装后的输入 item 数。`bad` 是标记为失败的响应 item 数。

`audio` 是成功处理的音频总时长，按返回通道累加。`mono=false` 的 stereo 输入会同时贡献 left 和 right 两个通道时长。`ch=mono/stereo/total` 统计的是返回通道数，不只是原始请求 item 数。

`elapsed` 和 `wall` 都是请求从进入服务到响应组装完成的 wall-clock 耗时。`active` 是该请求所有活跃工作 span 合并后的时长，并裁剪到请求 wall 时间以内，避免并行 worker 的时间被重复相加。`queue = wall - active`，表示该请求在共享 pipeline 中的空闲、排队或被阻塞时间。

`speed = audio / active`，表示去掉请求 idle 时间后的计算吞吐估计。`e2e = audio / wall`，表示用户可感知的端到端速度。

### 阶段 DEBUG 字段

阶段日志描述单个请求在某个 pipeline 阶段的统计，适合排查单个请求；常规吞吐对比应优先看周期 PERF 表。

`load` 和 `vad` 使用 `input_audio`。`encode`、`prefill`、`decode` 使用 `speech_audio`，即 VAD 去掉非语音后的时长。

`active` 是该请求在该阶段的活跃 span 合并时长。对于多 worker 阶段，重叠的 item 工作不会被重复相加。`queue` 是该请求在该阶段的排队 span 合并时长。`elapsed = active + queue`。

`wall` 是该阶段从最早 queue/run 开始到最晚 run 结束的跨度。如果 item/channel span 之间存在间隔，`wall` 可能大于 `elapsed`。

`speed = stage_audio / active`。`e2e = stage_audio / elapsed`。

### 周期 PERF 表

周期 PERF 表由全局服务 pipeline 在存在阶段活动时打印。默认窗口是 5 秒。它是服务级吞吐分析的优先日志，因为它聚合了所有共享 pipeline 的请求。

列说明：

```text
stage    pipeline 阶段。
count    当前窗口内完成的 job 或 batch 数。
audio    当前窗口内该阶段完成的音频秒数。
elapsed  当前窗口内该阶段用于吞吐计算的阶段耗时。
speed    audio / elapsed。
detail   阶段特定计数器。
```

`elapsed` 的口径按阶段不同：

```text
load/vad:
  多 worker 阶段使用窗口内最忙 worker 的计算耗时。
  这样 speed 表示该组件在当前 worker 配置下的有效服务吞吐，而不是把多个 worker 的耗时相加后得到单 worker 均速。

encode/prefill/decode:
  使用当前窗口内完成 batch 的模型计算耗时累加。
```

阶段行说明：

```text
load:
  count   job=<加载完成的 item/channel 数>
  audio   加载的输入音频时长
  elapsed 最忙 loader worker 的计算耗时
  detail  bad=<加载失败数>

vad:
  count   job=<VAD item/channel 数>
  audio   VAD 处理的输入音频时长
  elapsed 最忙 VAD worker 的计算耗时
  detail  speech=<VAD 后语音秒数> ratio=<语音/输入比例>
          seg=<输出 segment 数> bad=<VAD 失败数> no_speech=<无语音 item 数>

encode:
  count   bat=<encoder microbatch 数>
  audio   编码的语音音频时长
  elapsed audio encoder batch 耗时累加
  detail  seg=<编码 segment 数>

prefill:
  count   bat=<decoder batch 数>
  audio   进入 decoder batch 的语音音频时长
  elapsed model prefill 计算耗时
  detail  tok=<prefill tokens> tok/s=<prefill tokens per second> seg=<segments>

decode:
  count   bat=<decoder batch 数>
  audio   decode 的语音音频时长
  elapsed model decode 计算耗时
  detail  tok=<生成 tokens> tok/s=<生成 tokens per second> step=<decode scheduler steps> seg=<segments>
```

PERF 表回答的是“当前服务组件整体吞吐是多少”。单个请求的端到端耗时和排队情况看 `request DONE` 或阶段 `DEBUG` 日志。
