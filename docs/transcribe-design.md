# Transcribe 设计

## Pipeline

当前服务使用 `GlobalAsyncTranscribePipeline`。`serve` 模式下，所有 HTTP 请求共享一套 loader、VAD、ASR、encoder、prefill、decoder worker。请求进入服务后会先拆成去重后的音频/通道 job，最后再按原始输入顺序合并响应。

```text
                    HTTP request
                       |
                       v
+--------------------------------------------------+
| Submit Worker                                     |
| 将请求拆成去重后的 TranscribeInput job             |
| 记录 request_id / job route / duplicate mapping    |
+--------------------------------------------------+
                       |
                       | load_job_tx
                       | unbounded channel
                       v
+--------------------------------------------------+
| Global Loader Worker                              |
| 并发数 = --num-load-workers 或 CPU 并行度           |
| 每个任务 spawn loader task                         |
| 加载/解码音频，得到 waveform                        |
+--------------------------------------------------+
                       |
                       | loader_to_vad
                       | 无消息数上限
                       | 由 waveform 内存 permit 限制
                       v
+--------------------------------------------------+
| VAD Worker                                        |
| 并发数 = --num-vad-workers 或 CPU 并行度           |
| 从 loader_to_vad 消费 LoadedJob                    |
| 执行 VAD                                           |
|                                                  |
| 如果无语音:                                       |
|   直接返回结果，不进入 ASR                         |
|                                                  |
| 如果有语音:                                       |
|   切分为语音 segment                              |
|   每个 segment 携带 waveform 数据                  |
+--------------------------------------------------+
                       |
                       | vad_to_asr
                       | 无消息数上限
                       | 由 segment waveform 内存 permit 限制
                       v
+--------------------------------------------------+
| ASR Worker                                        |
| 单 async worker                                   |
| 从 vad_to_asr 消费消息                             |
|                                                  |
| 如果消息是 Job:                                   |
|   collect_job_microbatch()                        |
|   走整条音频 ASR fallback 路径                      |
|                                                  |
| 如果消息是 Segment 且启用 Qwen3 动态分阶段路径:     |
|   立即发送到 Segment Encoder Worker                |
|                                                  |
| 如果消息是 Segment 但未启用动态分阶段路径:          |
|   collect_segment_microbatch()                    |
|   按 max_batch_audio_sec 聚合 segment              |
|   走 run_segment_asr_inference fallback 路径        |
+--------------------------------------------------+
                       |
                       | asr_to_encoder
                       | unbounded channel
                       | 只用于 Qwen3 动态分阶段路径
                       v
+--------------------------------------------------+
| Segment Encoder Worker                            |
| 单 async worker + spawn_blocking                   |
|                                                  |
| 输入: Vec<AsrSegmentJob>                           |
| batch 来自 collect_encoder_segment_batch()         |
| 收集已排队 segment，并额外等待 10ms 聚合            |
|                                                  |
| Qwen3 ASR 路径:                                   |
|   run_segment_asr_encode()                        |
|   执行 audio encoder                               |
|   将 audio features 存到 CPU                       |
+--------------------------------------------------+
                       |
                       | encoder_to_prefill
                       | unbounded channel
                       v
+--------------------------------------------------+
| Segment Prefill Worker                            |
| 单 async worker                                   |
|                                                  |
| paged-attn feature 开启时:                         |
|   执行 dynamic prefill                             |
|   可从 encoder_to_prefill 继续合并 encoded batch    |
|   输出 prefilling session 给 decoder               |
|                                                  |
| paged-attn feature 关闭时:                         |
|   不单独启动该 worker                              |
+--------------------------------------------------+
                       |
                       | prefill_to_decoder
                       | unbounded channel
                       v
+--------------------------------------------------+
| Segment Decoder Worker                            |
| 单 async worker                                   |
|                                                  |
| paged-attn feature 开启时:                         |
|   消费 prefilled session                           |
|   执行 continuous decode                           |
|                                                  |
| paged-attn feature 关闭时:                         |
|   直接消费 encoded feature batch                   |
|   在 decoder worker 内完成 prefill/decode           |
+--------------------------------------------------+
                       |
                       | asr_to_result
                       | bounded channel
                       | serve 模式 capacity = 1024
                       v
+--------------------------------------------------+
| Global Result Router                              |
| 收集 RoutedTranscribeItemOutcome                   |
| 按 request_id 聚合请求结果                          |
| 将重复输入展开回原始请求顺序                         |
+--------------------------------------------------+
```

## 队列与并发规则

```text
Loader 并发:
  设置了 --num-load-workers 时使用该值，否则使用 CPU available_parallelism()

VAD 并发:
  设置了 --num-vad-workers 时使用该值，否则使用 CPU available_parallelism()

loader_to_vad 队列:
  不按消息数量限制
  按 waveform 内存 permit 限制

vad_to_asr 队列:
  不按消息数量限制
  按 segment waveform 内存 permit 限制

asr_to_encoder 队列:
  unbounded channel
  只传递动态 Qwen3 segment encode 请求

encoder_to_prefill 队列:
  unbounded channel
  传递 encoded audio features

prefill_to_decoder 队列:
  unbounded channel
  paged-attn feature 开启时存在，传递 prefilled session 或失败结果

asr_to_result 队列:
  serve 模式 bounded channel，capacity = 1024
  local pipeline 模式 capacity = unique_inputs.len()

Segment Encoder:
  单 worker
  每次处理一个 collect_encoder_segment_batch() 聚合出的 batch

Segment Prefill:
  paged-attn feature 开启时单 worker
  负责 dynamic prefill，可短暂等待更多 encoded batch 以提升合批

Segment Decoder:
  单 worker
  paged-attn feature 开启时消费 prefilled session
  paged-attn feature 关闭时直接消费 encoded feature batch

整条音频 Job fallback:
  ASR Worker 内 spawn_blocking 执行 run_asr_jobs_batch()

非动态 Segment fallback:
  ASR Worker 内 collect_segment_microbatch()
  然后 spawn_blocking 执行 run_segment_asr_inference()
```

`ASR_ENCODED_FEATURE_QUEUE = 1` 仍存在于代码中，但当前全局动态分阶段路径的 `asr_to_encoder`、`encoder_to_prefill`、`prefill_to_decoder` 都使用 unbounded channel。

## Segment Batch 形成规则

动态 Qwen3 segment 路径中，ASR Worker 不再按 `max_batch_audio_sec` 先聚合 segment，而是把 segment 发送给 Segment Encoder Worker。真正的 encoder batch 在 `collect_encoder_segment_batch()` 中形成：

```text
Segment Encoder Worker 收到第一个 SegmentEncodeRequest
  |
  v
collect_encoder_segment_batch()
  |
  +-- 从 asr_to_encoder 非阻塞 try_recv()
  |
  +-- 收集所有已经排队的 SegmentEncodeRequest
  |
  +-- 额外等待 ASR_ENCODER_BATCH_COLLECT_WAIT_MS，当前为 10ms
  |
  +-- 按 (job_index, offset) 排序
  v
Vec<AsrSegmentJob> -> Encoder Worker
```

该动态 encoder batch 当前不按 `max_batch_audio_sec` 截断，也不由 decoder 空闲 slot 决定。

fallback segment 路径仍使用 `collect_segment_microbatch()`：

```text
fallback 当前行为:
  vad_to_asr 中已经可取的 segment
  + max_batch_audio_sec
  -> encoder batch size

动态 Qwen3 当前行为:
  asr_to_encoder 中已经可取的 segment
  + 10ms collect wait
  -> encoder batch size

不是当前行为:
  decoder free slots
  -> encoder 拉取的 segment 数量
```

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
  -> decoder 单 worker continuous decode
  -> result
```

## HTTP API

HTTP 服务由 `vasr-transcribe serve` 启动，目前暴露三个接口：

```text
GET  /health
POST /transcribe
POST /inference
```

`/transcribe` 和 `/inference` 是别名，使用完全相同的请求和响应结构。JSON body 上限是 512 MiB。

### GET /health

用于服务存活检查，响应：

```json
{
  "status": "ok"
}
```

### POST /transcribe 和 /inference

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

## 服务日志

`RUST_LOG=info` 下，服务日志只保留核心信息：

```text
<id> request START | items=<n> src=url:<n> base64:<n> missing:<n> mono=true:<n> false:<n>
<id> request DONE  | items=<n> bad=<n> | audio=<s> ch=mono:<n> stereo:<n> total:<n> | elapsed=<s> active=<s> queue=<s> wall=<s> | speed=<x> e2e=<x>

+---------------------------------------------------------------------------------------------+
|                         Service transcribe-<uuid> Perf | window=5.0s                         |
+---------+-------+-------+--------+-------+---------------------------------------------------+
|  stage  | count | audio | worker | speed |                      detail                       |
+---------+-------+-------+--------+-------+---------------------------------------------------+
|  load   |  ...  |  ...  |  ...   |  ...  |                        ...                        |
+---------+-------+-------+--------+-------+---------------------------------------------------+
|   vad   |  ...  |  ...  |  ...   |  ...  |                        ...                        |
+---------+-------+-------+--------+-------+---------------------------------------------------+
| encode  |  ...  |  ...  |  ...   |  ...  |                        ...                        |
+---------+-------+-------+--------+-------+---------------------------------------------------+
| prefill |  ...  |  ...  |  ...   |  ...  |                        ...                        |
+---------+-------+-------+--------+-------+---------------------------------------------------+
| decode  |  ...  |  ...  |  ...   |  ...  |                        ...                        |
+---------+-------+-------+--------+-------+---------------------------------------------------+
```

周期 PERF 表由 serve 参数控制：

```text
--no-profile              关闭周期 profile 表；默认开启。
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

`wall` 是该阶段从最早 queue/run 开始到最晚 run 结束的跨度。如果 item/channel span 之间存在间隔，`wall` 可能大于 `elapsed`；它也不同于 worker 时间求和。

`speed = stage_audio / active`。`e2e = stage_audio / elapsed`。

### 周期 PERF 表

周期 PERF 表由全局服务 pipeline 在存在阶段活动时打印。默认窗口是 5 秒。它是服务级吞吐分析的优先日志，因为它聚合了所有共享 pipeline 的请求。

列说明：

```text
stage   pipeline 阶段。
count   当前窗口内完成的 job 或 batch 数。
audio   当前窗口内该阶段完成的音频秒数。
worker  当前窗口内累加的 worker/model 计算秒数。
speed   audio / worker。
detail  阶段特定计数器。
```

阶段行说明：

```text
load:
  count  job=<加载完成的 item/channel 数>
  audio  加载的输入音频时长
  worker loader 解码/读取计算耗时
  detail bad=<加载失败数>

vad:
  count  job=<VAD item/channel 数>
  audio  VAD 处理的输入音频时长
  worker 所有 VAD worker 计算耗时求和
  detail speech=<VAD 后语音秒数> speech_ratio=<语音/输入比例>
         seg=<输出 segment 数> bad=<VAD 失败数>
         no_speech=<无语音 item 数> send_wait=<等待送入 ASR 队列的时间>

encode:
  count  bat=<encoder microbatch 数>
  audio  编码的语音音频时长
  worker 这些 batch 的 audio encoder wall/model 时间
  detail seg=<编码 segment 数>

prefill:
  count  bat=<decoder batch 数>
  audio  进入 decoder batch 的语音音频时长
  worker model prefill 计算耗时
  speed  语音音频时长 / prefill 计算耗时
  detail tok=<prefill tokens> tok/s=<prefill tokens per second> seg=<segments>

decode:
  count  bat=<decoder batch 数>
  audio  decode 的语音音频时长
  worker model decode 计算耗时
  speed  语音音频时长 / decode 计算耗时
  detail tok=<生成 tokens> tok/s=<生成 tokens per second>
         step=<decode scheduler steps> seg=<segments>
```

PERF 表使用 worker/model 时间，而不是请求 wall 时间。对于 loader 和 VAD 这类多 worker 阶段，`worker` 会按 worker 求和。它回答的是“组件真正计算时有多快”，不是“某个请求端到端完成得有多快”。
