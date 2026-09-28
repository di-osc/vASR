# `vasr-transcribe serve` 参数精简与硬件自适应

## 目标

`serve` 缺省时根据设备和可用内存配置 ASR 并发及 encoder batch；容量项仍可手动覆盖。启动不运行 benchmark。自动值是基于模型和资源成本的高吞吐估算，不承诺对任意音频分布都达到数学意义上的全局峰值。用户显式选择的设备若初始化失败，`auto` 才尝试下一个后端；设备已选定后，模型加载或显存分配失败直接退出，不静默切换设备。

## CLI 调整

### `serve` 保留

`--model`、`--cache-dir`、`--isq`、`--language`、`--host`、`--port`、`--verbose` 保持行为。ASR `--device` 默认 `auto`，支持 `cpu`、`metal`、`cuda`；VAD `--vad-device` 默认从 `cpu` 改为 `auto`，支持 `cpu`、`cuda`。

| 参数 | 缺省行为 | 手动覆盖 / 环境变量 |
| --- | --- | --- |
| `--max-num-seqs` | 自动按内存和 KV 成本估算 | CLI、`VASR_MAX_NUM_SEQS` |
| `--gpu-memory-utilization` | 自动按预算设置 | CLI、`VASR_GPU_MEMORY_UTILIZATION` |
| `--encoder-batch-max-audio-seconds` | 自动按可用工作区估算，至少覆盖最大 VAD 分段及 padding | CLI、`VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS` |
| `--max-hotword-tokens` | 128 | CLI、`VASR_MAX_HOTWORD_TOKENS` |
| `--num-load-workers` | 2 | CLI、`VASR_NUM_LOAD_WORKERS`；0 表示全部可用 CPU 并行度 |
| `--num-vad-workers` | 2 | CLI、`VASR_NUM_VAD_WORKERS`；0 表示全部可用 CPU 并行度 |

`--profile`、VAD 模型、阈值、语音和静音长度、padding、最大分段时长继续显示在帮助中，默认行为不变。

### 从 `serve --help` 移除

移除以下选项的 CLI 接口；隐藏内部调优仍读取列出的环境变量：

| 移除项 | 内部行为 |
| --- | --- |
| `--kv-peak-ratio` | 默认 0.25；保留 `VASR_KV_PEAK_RATIO` |
| `--cuda-graph-chunked-argmax` | Q8 时按原规则自动开启 |
| `--cuda-graph-chunked-argmax-size` | 固定 4096 |
| `--q8-prefill-lm-head-mmq` | 默认开启 |
| `--q8-prefill-lm-head-mmq-max-logits-mib` | 固定 256 MiB |
| `--profile-window` | 默认 5 秒；保留 `VASR_PROFILE_WINDOW` |
| `--vad-num-threads` | 默认 0；保留 `VASR_VAD_NUM_THREADS` |
| `--encoder-batch-max-segments` | 自动设置，基线32；保留 `VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS` |

### `serve` 与 `run` 都移除

- 删除 `--decode-tokens-per-sec`，也不再接受环境变量覆盖。4.25 固定为 Qwen3-ASR 模型常量。
- 删除 `--decode-min-tokens` 和相应公共 Rust 配置字段。KV cache 估算继续保留32-token block 下限。
- `run` 的其他 CLI 项和默认值保持现状；不使用 `serve` 的自动容量和 worker 默认值。

`serve` 使用独立的 Clap 参数结构，再转换成内部 pipeline 参数；这样不会因复用 `run` 参数结构而泄露隐藏选项或改变 `run` 的默认值。

## 自动容量计算

启动阶段先初始化 VAD，再探测所选 ASR 设备和 CPU 并行度。模型名称、量化和运行设备用于估算 ASR 权重占用及 KV 每 token 成本；VAD 最大分段、padding 和 hotword 上限参与 KV 与 encoder 预算。CUDA KV 分配发生在模型权重加载后，paged-cache allocator 使用届时的实际空闲显存、利用率和预留空间决定可用 block 数；自动模式下调度器按已分配 cache 的实际容量工作。

- encoder batch 段数上限按可用内存估算，最多32段；合批音频上限按 encoder 工作区预算推算，自动值不会小于最大 VAD 分段加 padding。
- CUDA 预算预留 encoder 工作区、CUDA graph 工作区和剩余可用内存的10%作为余量。KV 每序列成本由音频 token、hotword、固定 prompt、模型 decode 常量和 KV peak ratio 估算。
- CPU 使用主机可用内存及并行度；Linux 同时考虑 cgroup 内存限制。Metal 使用统一内存与主机并行度。
- 不运行启动基准测试。内存或模型 profile 探测不可用时记录原因，回退到512并发、0.8 GPU 利用率、30秒 / 32段；后备分配仍 OOM 时直接报错。
- 手动容量项优先于自动值。非法值、超出估算安全并发、或 encoder batch 小于最大 VAD 段及 padding 时启动时报错。

自动容量解析器以独立硬件 profile 为输入，CPU、CUDA、Metal 合成容量测试检查余量、边界、覆盖、探测失败后备和资源不足错误。日志记录最终容量及 CLI、环境或自动来源。

## `.env`

- 所有子命令在 Clap 参数解析及 Tokio runtime 创建前，从**当前工作目录**加载 `.env`；文件不存在时忽略。
- 优先级：CLI 参数 > 已有进程环境变量 > `.env` > 自动值。文件值不会覆盖进程环境变量。
- 支持空行、整行 `#` 注释、`KEY=VALUE` 和首尾单/双引号。不支持变量展开、多行值或 `export`。
- 不添加 `--env-file`。根目录 `.env.example` 展示手动调优环境变量。

## 验收

- `serve --help` 移除项消失，保留项和默认值正确；`run --help` 除 decode 两项外保留当前参数行为。
- `.env` 测试覆盖当前目录、文件缺失、进程环境变量优先、CLI 优先，以及环境值进入 Clap 与内部调优。
- 自动调优测试覆盖 CPU、CUDA、Metal、余量、容量上下限、覆盖、后备和不足错误。
- 已选设备后的加载错误直接传播；无重试 CPU 或其他后端。
- 更新帮助文本、服务部署文档和 `.env.example`，并执行 `cargo install --path vasr-transcribe --force --features cuda`。
