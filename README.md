# vASR

基于 Rust 与 Candle 的通用语音识别推理框架，支持离线音频转写与实时语音识别。

## 简介

vASR 面向本地部署和语音识别服务，提供通用的推理运行时与服务基础设施，将模型推理、语音活动检测（VAD）、音频分段、请求调度与结果评测整合在一个 Rust 工作空间中。当前已实现 Qwen3-ASR 模型支持。框架支持 Apple Silicon（Metal）、NVIDIA GPU（CUDA）和 CPU，并提供离线 HTTP 转写服务与 WebSocket 实时识别服务，可用于录音转写、批量音频处理和实时语音应用。

离线转写与实时识别共用同一套模型运行时。请求进入全局流水线后，通过连续批处理、分段编码和 PagedAttention 提高计算资源利用率；音频、标注与识别结果统一按时间线组织，配套工具支持数据集管理、吞吐测试和识别质量评测。

## 特点

- **连续批处理。** 所有请求共享一个常驻 decode 循环。序列生成完立刻让出槽位，新序列马上补进去，不等一个 HTTP 请求里的音频全部结束。
- **自动执行策略。** KV 内存安全容量、活跃槽数、decode 行数、prefill wave 和 encoder 批量分别管理，按设备、模型与可用内存选择，启动日志打印最终配置。
- **PagedAttention。** KV cache 按块分配。每条序列只占用实际长度，不用按最大上下文预留整段显存。CUDA 与 Metal 都有实现。
- **Attention 加速。** CUDA 支持 bf16 varlen FlashAttention；Metal 提供 PagedAttention 与 packed prefill，encoder 保持原有窗口和音频边界。
- **CUDA decode graph。** 预先捕获 decode 步，重放时省掉重复的 kernel 启动。
- **分段流水线。** 加载、VAD、音频编码、prefill、decode 分成独立阶段。编码阶段会把已经排队的语音段合成一批，prefill 期间还可以并入后续批次。
- **离线与实时。** `vasr-transcribe` 提供 `GET /health` 和 `POST /transcribe`。`vasr-realtime` 提供 WebSocket 实时会话。
- **VAD。** 离线默认使用 FunASR FSMN，把长音频切成语音段；本轮 M4 Pro 验证配置使用 CPU VAD。无语音输入直接返回空结果，实时使用 FireRed Stream-VAD。
- **量化与精度。** CPU 使用 f32，Metal 和 CUDA 使用 bf16。支持就地量化，例如 `q8_0`。
- **时间线数据。** 音频、标注和识别结果落在 `Timeline` 上，数据集用 SQLite AudioDb 保存，评测直接对标注算字错误率。
- **ModelScope 数据源。** SQLite AudioDb 保存音频引用、元信息和标注，音频由 modelhub 下载或命中缓存。数据与评测接口使用 `asr-data`。
- **重复输入合并。** 同一请求里相同的音频只推理一次，结果按原始顺序展开。

## 离线 HTTP 服务

从仓库根目录执行。Apple Silicon 使用 `metal` feature 覆盖安装当前版本：

```bash
cargo install --path vasr-transcribe --force --features metal
vasr-transcribe serve --model Qwen3-ASR-0.6B
```

NVIDIA 机器安装时改用 `--features cuda`。`serve` 未指定模型时默认也是 `Qwen3-ASR-0.6B`，监听 `0.0.0.0:8867`。

`serve` 默认按所选设备、模型和可用内存估算 KV 容量，再解析出实际执行批量，启动日志会打印最终生效的策略：

```
serve 执行策略 | device=metal capacity=159 active_slots=64 decode_rows=64 mixed_step=true prefill_wave=16
Metal CNN policy | nhwc=true im2col32=true bias_gelu=true native_chunks=false explicit_overrides=preserved
Metal encoder policy | conv_tile=Some(Narrow) attention=Some(GroupedNative) fc1_bias_gelu=true overrides=preserved
Metal GEMM policy | decode=Some(Adaptive) max_decode_rows=128 adaptive_split_rows=32 prefill=Some(Adaptive)
Metal temporary ownership policy | explicit_override=preserved autorelease_scope=true
```

上面是本机默认启动日志示例；`capacity` 和 encoder 音频预算随启动时可用内存变化。经过验证的 M4 Pro / 48 GB、`Qwen3-ASR-0.6B`、BF16、未量化组合自动选择64槽/64行decode、prefill16和encoder16段；安全容量独立估算。自动启用 Metal packed prefill、Adaptive GEMM、同长窗口 attention 合批、NHWC/3×3 im2col32、CNN与encoder FC1的 bias + GELU 融合，以及局部临时所有权回收。融合保留 dtype 舍入与原 GELU 公式。其他硬件、模型或 dtype 保留保守回退，显式配置优先。

日常启动不需要手写并发参数。需要覆盖时可用 CLI 参数（`--max-num-seqs`、`--gpu-memory-utilization`、`--encoder-batch-max-audio-seconds`）或环境变量（`VASR_METAL_MAX_ACTIVE_SEQS`、`VASR_METAL_PAGED_DECODE_BATCH_SIZE`、`VASR_PREFILL_MAX_WAVE`、`VASR_DISABLE_MIXED_STEP_DECODE`）。显式容量超过估算的安全范围，或低于 VAD 最大分段加 padding 的要求时，服务会在启动阶段报错。

服务也会在启动时读取当前工作目录下的 `.env`。仓库提供了 [`.env.example`](.env.example)；复制为 `.env` 后可在其中设置部署参数。优先级为 CLI 参数、已有进程环境变量、`.env`，最后是自动值。所有子命令都会加载 `.env`，文件不存在时会忽略。

## 数据集

数据集现在是单个 SQLite 文件 `datasets/<name>.db`。音频不再拆到 `assets/`，而是以 ModelScope 数据集引用（`repo_id` + 仓库内相对路径 + `revision`）存进库里，读取时由 modelhub 下载或命中缓存：

```text
datasets/
├── call-100.db
├── call-200.db
└── example1.db
```

`perf`、`eval`、`stress`、`run` 的 `-i` 接受 `.db` / `.vasr` / `.vasr.msgpack` 数据文件。旧的“拆成数据集目录”的 `pack` 子命令已经移除。

## 推理速度

100× 实验于 **2026-10-08** 验收真实自动默认：三轮 **122.54 / 122.61 / 122.65×**，中位 **122.61×**，同期关闭 RoPE 融合的参考中位 100.34×，提升 **22.19%**。使用迁移后的 `call-100.db`，99 条、5025.0118 秒，客户端并发 28/batch 4；完整预热后测量，模型加载另计。六轮正式及六轮预热的原始、Bench 规范化文本与正确基线逐条一致，加权 CER **16.5876%**，example1 零 CER，零失败与质量异常。

主要新改动是 packed prefill 的 RoPE 逐元素计算与 Q/K 布局融合：每次 forward 只准备一次位置数据，并直接写入后续所需布局；保留两个乘积的 dtype 舍入，实际 F32/F16/BF16 与参考逐元素一致。模型权重、BF16、VAD 与解码语义保持原样。已验证的 M4 Pro/48 GB、0.6B、未量化组合自动启用，服务仍自动选择安全容量及 64 活跃槽、encoder16、prefill16；显式禁用优先，其他组合保留回退。量化、wave32、仅 gate/up 融合和仅位置准备复用未显示稳定收益，未纳入默认。

真实自动默认的 **30 分钟持续吞吐为 120.80×**（含短探针及队列排空），1123 次请求零失败，4372 条主转写与基线逐条一致，30 次 example1 探针零 CER。按相同输入重复次数计算的加权 CER 两者均为 16.583913%；KV 3461/3461 全回收，直接计数的请求元数据排空。RSS 五分钟稳态中位先升后降，swap 未增长；短探针最大延迟 46.91 秒，吞吐验收不代表低延迟 SLA。完整优化、精度、负面结果与复现命令见[100× 实验记录](docs/performance/m4-pro-100x-2026-10-07.md)。

### CNN 优化的历史对照

以下为 **2026-10-01 CNN 优化的历史验收**；当前默认结果见下方 100× 实验。环境为 Apple M4 Pro、48 GB 统一内存、Metal、AC 供电，模型为 `Qwen3-ASR-0.6B`，使用 BF16、未量化。输入为 call-100 的固定音频与参考文本快照：99 条、合计约 5025 秒；固定输入顺序、mono 声道策略和 Bench 规范化。HTTP 并发 14、batch size 1，服务端容量与执行批量自动解析。

服务完整预热一轮，再运行三轮正式测试。以下吞吐、RTF 和延迟为各轮统计值的中位数；模型加载和预热时间单独记录，正式吞吐关闭额外 GPU 诊断同步。

| 指标 | 当时默认配置 | 同期对照：关闭 CNN bias + GELU 融合 |
| --- | ---: | ---: |
| 吞吐（音频秒 / 墙钟秒） | **87.81×** | 82.22× |
| 三轮吞吐范围 | **87.76–88.55×** | 82.17–82.33× |
| 实时率 RTF | **0.0114** | 0.0122 |
| 1 小时音频估算耗时（按吞吐折算） | **约 41 秒** | 约 44 秒 |
| 请求延迟 p50 | 7.92 s | 8.74 s |
| 请求延迟 p99 | 23.56 s | 25.29 s |
| 加权 CER | **16.5876%** | 16.5876% |
| 平均 CER | 21.7032% | 21.7032% |

两列均使用 32 活跃槽、最多 32 decode 行及 8 行 prefill wave，其余已验证的 Metal 优化保持启用。三轮完整数据集零失败，99 条原始转写文本逐条一致；example1 保持“我要修改商品帖子的价格”的零 CER。CNN 改动经过 F32 / F16 / BF16 算子数值对照及 encoder 回归验证。

30 分钟持续来流测试达到 **88.28×**（含短请求探针和队列排空），3142 次主请求加 30 次 example1 探针，共 **3172 次请求全部成功**。99 条音频各重复 31–32 次，原始与规范化转写文本全部与同期对照一致；按相同输入重复次数计算的加权 CER 两者均为 **16.594415%**，30 次 example1 均零 CER，最长延迟 15.85 秒。

服务 RSS 起始 3285 MiB、峰值 4308 MiB，排空后降至 3004 MiB，交换内存使用量未增加。运行期间 RSS 有增长，此结果只覆盖本次 30 分钟观察范围。独立诊断实例排空后 KV 可用 block 恢复为 3977/3977。

本轮测量与数据迁移同时进行，因此采用固定快照与冻结客户端维持输入和评测口径，推理服务使用当前 Metal 代码；以上数字来自该验收路径。当前仓库的常规数据入口是 `datasets/call-100.db` 的 ModelScope 引用，首次下载与缓存状态会影响端到端时间。完整原始记录、各项独立优化收益、早期性能波动和精确复现方式见[性能与精度报告](docs/performance/m4-pro-cnn-layout-2026-09-30.md)，逐条精度对照见[CSV](docs/performance/m4-pro-cnn-fusion-per-item-2026-09-30.csv)。

### 当前 checkout 的测试入口

从仓库根目录安装并启动服务后，另开终端。先运行一次 `perf` 完整预热，再运行三次相同命令做正式对照：

```bash
vasr-transcribe perf -i datasets/call-100.db \
  --mono true --concurrency 28 --batch-size 4 --retries 1 --cer
vasr-transcribe eval -i datasets/call-100.db --timeline mono --normalizer bench
vasr-transcribe eval -i datasets/example1.db --timeline mono --normalizer bench
```

`perf --cer` 直接评估并发请求的实际输出。精确重现上述固定快照验收的方法在性能报告及本地归档 `results/m4-pro-cnn-layout-2026-09-30/REPRODUCE.md` 中；归档包含完整响应、冻结二进制、源码校验清单和输入快照，按 `.gitignore` 保留在本地。


v17的30分钟持续测试为**80.64×**（含短探针及排空），757次请求全成功、2912条主转写与基线一致、29次短探针零CER，KV和直接计数的请求元数据排空；RSS没有持续上升趋势、swap未增长。这是旧路径的持续成绩，短探针最大延迟69.17秒，本次只证明能够完成。独立FSMN CPU VAD完整三轮计算吞吐1584.90×，音频解码和模型加载另计。配置、每项精度与收益、负面结果及复现命令见[100× 实验记录](docs/performance/m4-pro-100x-2026-10-07.md)。
