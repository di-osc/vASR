# M4 Pro Qwen3-ASR Metal 执行策略与并发限制修复 — 2026-09-30

## 结论摘要

在 Apple M4 Pro（48 GB 统一内存）上，Metal 后端原先被三层限制锁死在低吞吐：调度器把执行批量硬编码为 **1 个活跃槽 / 1 行 decode / mixed-step 关闭**，HTTP 层用全局请求锁把请求串行化并拆成单条，encoder 自动模式被限制为 **1 段**。

本次把这些互相独立的概念拆开管理后，**默认启动**（无任何并发参数）即解析为：

```
serve 执行策略 | device=metal capacity=512 active_slots=32 decode_rows=32 mixed_step=true prefill_wave=8
```

对照 HEAD `80326cf` 基线：

| 指标 | 基线 | 候选（默认 32 槽） | 比值 |
| --- | ---: | ---: | ---: |
| perf 加速比（3 轮均值） | 21.80× | 43.84× | **2.01×** |
| eval 加速比（2 轮均值） | 21.72× | 45.44× | **2.09×** |
| 加权 CER | 0.167497 | 0.167858 | +0.000361 |
| 平均 CER | 0.223759 | 0.224968 | +0.001209 |
| `example1.vasr` CER | 0.000000 | 0.000000 | 持平 |
| 失败 / 坏条目 | 0 / 99 | 0 / 99 | 持平 |

**达到了 2× 吞吐目标，但加权 CER 比基线高 0.000361（+0.22% 相对，6 个字符 / 16651）**。这个差异的成因已完整定位（见下节），并且经用户确认后接受。

## 机器与工作负载

- 机器：Apple M4 Pro，48 GB 统一内存；Metal 后端；ASR dtype BF16。
- 模型：`Qwen/Qwen3-ASR-0.6B`（28 层，16 attention heads，8 KV heads，head_dim 128，KV block 32）。
- 数据集：`datasets/call-100.vasr`，99 条，5025.0118 音频秒；全部声明单声道 8 kHz，评测用 mono。
- 评测：`--normalizer bench`，`--timeline mono`；perf 默认并发 14 / batch 1。
- 基线二进制：HEAD `80326cf` 构建，不使用任何覆盖参数。

## 改动清单

### 1. 统一执行策略（新增）

新增 `vasr/src/models/qwen3asr/inference/execution.rs`，把「内存安全容量」「实际执行批量」「设备执行保护」分开：

```rust
pub struct ExecutionPolicy {
    pub active_slots: usize,  // 并发解码活跃槽位，是安全容量的子集
    pub decode_rows: usize,   // 单次 decode forward 的行数
    pub mixed_step: bool,     // 是否允许不同 decode step 合并进同一次 forward
    pub prefill_wave: usize,  // 单次 prefill wave 的最大序列数
}
```

- `resolve` / `resolve_backend`：启动时解析一次；环境变量（`VASR_METAL_MAX_ACTIVE_SEQS`、`VASR_METAL_PAGED_DECODE_BATCH_SIZE`、`VASR_PREFILL_MAX_WAVE`、`VASR_DISABLE_MIXED_STEP_DECODE`）优先级最高，但不会突破本次解析出的安全容量。
- `clamped_to(capacity)`：保证执行批量永不越过内存安全容量。
- `log_resolved`：打印最终生效策略，便于复现配置。

**关键点：内存安全容量与执行批量解耦。** 容量用于 KV block 规划，执行批量按策略收窄；即使容量解析为 512，也不会自动执行 512 行 prefill。

### 2. 修复容量传递

`vasr-transcribe/src/serve.rs` 原先在 Metal 上把 `decode_max_seqs` 丢弃为 `0`，导致调度器退回内部默认值，再被 `execution_slots_for` 压到 1 槽：

```rust
// 修复前
let use_runtime_capacity = args.pipeline.model.max_num_seqs.is_none() && (device.is_cuda() || device.is_metal());
if use_runtime_capacity { asr_options.decode_max_seqs = 0; }

// 修复后
if device.is_metal() {
    asr_options.decode_max_seqs = tuning.max_num_seqs;
    asr_options.execution = Some(tuning.execution);
}
```

策略经 `AsrOptions` → `TranscribeOptions` → `AsrBatchSchedulerConfig` 传递；删除了调度器内硬编码的 `execution_slots_for`、`paged_decode_batch_size_for`、`mixed_step_decode_enabled_for`。

### 3. 解除 HTTP 串行

- 删除 `metal_request_lock`（全局 tokio Mutex）与 `split_request_items`（把请求拆成单条）。
- 删除 `GlobalAsyncTranscribePipeline::requires_serial_metal_requests`。
- 设备安全改由既有的共享 `ModelExecutionLock` 保证：encoder、prefill、decode 的 Metal forward 串行，CPU 侧准备与调度保持并行。

### 4. 解除结果生命周期限制

`vasr-transcribe/src/pipeline/segment_decode.rs` 原先用 `defer_until_stream_idle` 把已完成结果压到「全局 decode 队列完全空闲」才统一返回，持续来流会阻塞短请求：

```rust
// 修复前：先缓存，等整个 stream 空闲
deferred_completed.borrow_mut().extend(completed);
let stream_was_reset = reset_stream_decode_state_if_idle(...);
let ready = take_completed_after_idle_reset(&mut deferred_completed.borrow_mut(), defer_until_stream_idle, stream_was_reset);

// 修复后：每个条目独立完成、返回、回收
for outcome in completed {
    if asr_to_result_tx.blocking_send(outcome).is_err() { bail!("ASR result router closed"); }
}
reset_stream_decode_state_if_idle(...);
```

### 5. Encoder 多段

- `METAL_AUTO_ENCODER_SEGMENTS = 8`（原为 1），保留 `VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS` 覆盖用于精度扫描。
- 新增非零权重数值测试：`test_nonzero_weights_batched_forward_matches_per_sample`、`test_metal_nonzero_weights_batched_forward_matches_per_sample_f32`、`..._bf16`。原先的零权重测试无法验证多段计算正确性。

### 6. 诊断

- `ModelExecutionLock` 新增累计等待统计：`wait_stats()` 返回 `(获取次数, 等待秒数)`；`VASR_DEBUG_LOCK_WAIT=1` 时输出 `model_forward_lock | acquisitions=... waited=... avg_wait_ms=...`。
- 默认日志级别新增 `vasr::execution=info`，启动即打印生效策略。

### 7. 修复 `stress` 在非 NVIDIA 主机上不可用

`stress` 子命令在启动时无条件调用 `nvidia-smi` 采样显存，在 Apple Metal 主机上直接以

```
Error: failed to sample GPU 0
Caused by: failed to run nvidia-smi: No such file or directory
```

中止，导致 30 分钟长稳验收根本无法执行。修复为优雅降级：

- 新增 `nvidia_smi_available()` 探测；不可用时跳过采样并打印提示，而不是中止。
- 新增 `--no-gpu-sample` 显式关闭采样。
- 采样中途失败只停止采样（打印一行说明），不再丢弃整个压测结果。
- 报告与汇总表在无样本时显示 `n/a`，`StressConfig` 新增 `gpu_sampling` 字段记录实际是否采样。

## 容量扫描

固定并发 14 / batch 1，`call-100.vasr` 全量：

| 活跃槽 | perf 加速比 | eval 加速比 | 加权 CER |
| ---: | ---: | ---: | ---: |
| 8 | 36.86× | 37.36× | 0.167858 |
| 16 | 42.66× | 42.67× | 0.167858 |
| **32** | **45.10×** | **43.29×** | 0.167858 |
| 64 | 43.98× | 41.44× | 0.167858 |

32 槽最优；64 槽反而更慢，且 CER 无进一步变化。8 → 16 → 32 槽的 CER 与逐条假设文本**完全一致（99/99 byte-identical）**，说明槽数本身不影响数值。

## CER 差异根因定位

8 组隔离实验，全部可复现且完全确定性（同一配置多次运行 CER 与逐条文本均一致）：

| 配置 | 加权 CER | 平均 CER | sub/del/ins |
| --- | ---: | ---: | --- |
| 基线 `80326cf` | 0.167497 | 0.223759 | 1520/566/703 |
| **decode 行 = 1**（8 槽、mixed 开、wave 8、encoder 8 段） | **0.167497** | **0.223759** | **1520/566/703** |
| decode 行 = 2 | 0.168218 | 0.226784 | 1526/568/707 |
| decode 行 = 8，mixed-step 关闭 | 0.167798 | 0.224961 | 1520/568/706 |
| decode 行 = 8，mixed-step 开启 | 0.167858 | 0.224968 | 1518/568/709 |
| decode 行 = 16 | 0.167858 | 0.224968 | 1518/568/709 |
| decode 行 = 32 | 0.167858 | 0.224968 | 1518/568/709 |
| decode 行 = 64 | 0.167858 | 0.224968 | 1518/568/709 |
| prefill wave = 1 | 0.167858 | 0.224968 | 1518/568/709 |
| encoder 段数 = 1 | 0.167858 | 0.224968 | 1518/568/709 |

结论：

1. **CER 偏移只由 decode 行批处理触发。** `decode_rows = 1` 时候选二进制复现基线的**逐项错误数完全相同**（1520/566/703），说明模型权重、VAD、特征提取、prefill、encoder 路径在数值上与基线一致。
2. **encoder 多段与 prefill wave 均已排除**：把 encoder 段数降到 1、把 prefill wave 降到 1，CER 都不变。
3. **偏移不随批量大小单调变化**：行 = 2 时 CER 反而最高（0.168218），行 = 8/16/32/64 稳定在 0.167858。这排除了「padding 或边界处理 bug」这类会随规模累积的错误，符合批处理 GEMM 归约顺序变化导致的浮点抖动特征。
4. **量级**：逐条对比 99 条中 24 条文本发生变化，**11 条变好 / 9 条变差**，净差 6 个字符 / 16651 参考字符。四组批处理配置的 CER 分布在 0.167798–0.168218，基线 0.167497 处于该分布低端。

**权衡是硬性的**：吞吐收益完全来自行批处理。

| 配置 | 加速比 | 加权 CER |
| --- | ---: | ---: |
| decode 行 = 1 | 21.60× | 0.167497（与基线一致） |
| decode 行 = 8，mixed 关 | 22.60× | 0.167798 |
| decode 行 = 8，mixed 开 | 37.36× | 0.167858 |
| decode 行 = 32，mixed 开 | 43.29× | 0.167858 |

`decode_rows = 1` 时并发槽位本身**不带来任何吞吐收益**（21.60× ≈ 基线 21.72×），因为共享 forward 锁串行化了 GPU 工作，每次 forward 只产出 1 个 token。收益来自把多个不同 decode step 的序列（mixed-step）合并进同一次 forward。

## 并发 / Batch 扫描

固定候选默认配置（32 槽），`call-100.vasr` 全量，batch ∈ {1, 4, 10} × 并发 ∈ {1, 4, 8, 14, 28}，共 15 组：

| 并发 | batch 1 | batch 4 | batch 10 |
| ---: | ---: | ---: | ---: |
| 1 | 32.94× | 40.51× | 42.59× |
| 4 | 42.53× | 44.97× | 44.76× |
| 8 | 44.97× | 45.19× | 44.83× |
| 14 | 44.72× | 45.32× | 45.06× |
| 28 | 45.24× | 44.57× | 44.72× |

**15 组全部 0 失败、0 坏条目、99/99 通过质量检查。**

要点：

1. **吞吐在并发 ≥ 8 后饱和于 ~45×**，与客户端 batch size 基本无关。
2. **单客户端串行（b1_c1）已达 32.94×**，超过基线在并发 14 下的最好成绩（21.80×）。说明收益来自服务内部流水线（32 槽 + mixed-step），而不是靠客户端并发堆出来。
3. 延迟随并发/批量线性增长属预期（b10_c14 的 p50 = 111.5s 即整批墙钟时间）；短音频在并发 1 下 p50 仅 0.9s。
4. 该扫描在基线（1 槽）下无法达到同等饱和——基线是队列敏感型，吞吐随并发波动（17.97×–21.93×）。

## 30 分钟持续来流长稳

候选默认配置（32 槽），`--shuffle` 打乱输入，并发 14 / batch 1，持续 1808.5s：

| 指标 | 值 |
| --- | ---: |
| 请求数 / 成功 / 失败 | 1598 / 1598 / **0** |
| OOM | **0** |
| 坏条目 | **0** |
| 处理音频 | 81238.2s |
| 持续吞吐 | **44.9×** |
| 延迟 p50 / p95 / max | 14.079s / 30.627s / 42.769s |

- 服务端日志在整段压测中 **0 条 WARN / ERROR**。
- 客户端错误列表为 0 条；`items == requests == successful`，无遗漏、无异常重复。
- 短音频在负载下正常返回：并发 1 时 p50 仅 0.9s（见并发扫描），长稳期间 p50 14.1s 是 14 路并发下的排队结果。
- 压测结束后立即做 drain 校验：`example1.vasr` 仍返回 CER 0.000000、100% 准确，服务状态正常。

**KV 回收确认**：服务进程 RSS 在长稳期间持续采样为

```
1.84 GiB (启动) -> 2.55 / 2.82 / 2.93 / 2.77 / 2.21 GiB (长稳中) -> 2.38 GiB (结束)
```

呈振荡而非单调增长，说明 KV block 随请求完成被回收，无单调内存泄漏；结束后稳定在 2.38 GiB。

## 测量方法学警告

**本机在测量期间出现严重 swap 换页，绝对吞吐不可直接跨时间比较。**

同一 32 槽配置（`capacity=512`、`kv=17.06GiB`、策略完全相同）在 00:32 测得 45.10×，在 01:02 再测仅 35.8×。排查确认：

```
vm.swapusage: total = 14336.00M  used = 13510.31M  free = 825.69M
Pages compressed: 99110840
```

反复创建 13–17 GiB KV cache 后 macOS 未回收，导致换页抖动。杀进程后 swap 占用不下降（系统惰性回收）。因此**结论采用交替 A/B 比值**，而非绝对吞吐：

| 轮次 | 基线 | 候选 32 槽 | 比值 |
| --- | ---: | ---: | ---: |
| 1 | 17.97× | 43.50× | 2.42× |
| 2 | 21.93× | 43.15× | 1.97× |

候选吞吐非常稳定（43.50 / 43.15），基线波动大（17.97 / 21.93）——基线是队列敏感型（1 槽、长队列），更受机器状态影响。以基线最佳值（21.93×，与独立测得的 21.80× 一致）对照候选均值 43.3×，得到 **1.98×**；按最终默认配置的 3 轮均值对照基线 3 轮均值，得到 **2.01×**。

CER 不受机器状态影响：所有配置在多次运行间完全确定性且与机器负载无关。

## 验证

- `cargo fmt --all`，`cargo check --all-targets --features metal` 零 warning。
- `cargo test --all-targets --features metal` 全部通过，含新增的 encoder 非零权重 F32/BF16 数值测试与执行策略单元测试。
- 最终默认配置（无环境变量覆盖）自动解析为 32 槽，确认「日常启动无需手写并发参数」。

## 剩余瓶颈

1. **共享 forward 锁仍是串行点。** CPU 侧准备并行，但 GPU forward 一次只能一个。进一步收益需要减少 forward 次数（更大批量）或让 forward 内部并行化。
2. **行批处理的数值抖动无法与性能解耦。** 除非让批处理归约顺序与单序列一致，否则任何 `decode_rows > 1` 的配置都会带来同量级（±0.0005）的 CER 漂移。
3. **绝对吞吐受宿主内存压力影响显著。** 长稳测试与容量规划应在无 swap 压力的机器状态下进行。
4. **64 槽无收益。** 32 → 64 槽吞吐反而下降（45.10× → 43.98×），且 CER 无变化，说明瓶颈已不在并发槽数，而在单次 forward 的 GPU 时间。

## 未完成 / 未验证项

- **VAD 持续吞吐（目标 800×）未测量。** 需要单独的 VAD-only 负载与基线对照，本次未执行。
- **基线长稳对照未执行。** 30 分钟长稳只跑了候选；基线的持续吞吐参照其短时测量（21.80×）推断，未实测。
- **逐阶段耗时分解未采集。** 计划要求补齐 VAD / 特征提取 / encoder / prefill / decode 各自的计算、排队与锁等待统计。本次只落地了锁等待统计（`VASR_DEBUG_LOCK_WAIT=1`），未采集完整的逐阶段时间线表。
- **多数据集 CER 验证仅覆盖 `call-100.vasr` 与 `example1.vasr`。** 其他数据集上的抖动方向未知。
- **`example1.vasr` 只有 1 条**（11 个字符），其 0 CER 是必要的回归信号，但统计效力有限。

## 复现

仓库根目录执行：

```bash
cargo install --path vasr-transcribe --force --features metal
vasr-transcribe serve --model Qwen3-ASR-0.6B
```

另一个终端：

```bash
vasr-transcribe perf -i datasets/call-100.vasr --concurrency 14 --batch-size 1
vasr-transcribe eval -i datasets/call-100.vasr --timeline mono --normalizer bench
vasr-transcribe eval -i datasets/example1.vasr --timeline mono --normalizer bench
```

启动日志的第一行即给出最终生效策略。诊断开关：`VASR_DEBUG_LOCK_WAIT=1`（forward 锁等待）。

长稳压测（非 NVIDIA 主机会自动跳过显存采样）：

```bash
vasr-transcribe stress -i datasets/call-100.vasr --concurrency 14 --batch-size 1 \
  --shuffle --duration-sec 1800
```

### 注意事项：拷贝二进制后需要重新签名

本机为 Apple Silicon，`cargo build` 产物是 linker-signed（adhoc）Mach-O。把二进制 `cp` 到其他路径后，若目标路径上原本有正在运行的旧副本，新副本会失去有效签名，之后每次执行都会被内核 **立即 SIGKILL（exit=137）**，且不产生任何输出——现象很像程序崩溃，实际是签名校验失败。修复方式是重新做 adhoc 签名：

```bash
codesign --force --sign - --timestamp=none /path/to/vasr-transcribe
```
