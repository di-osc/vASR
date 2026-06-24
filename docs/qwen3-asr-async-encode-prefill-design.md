# Qwen3-ASR 异步 Encode + Prefill 设计

日期：2026-06-24

本文记录 Qwen3-ASR segment 推理调度的目标设计。核心目标是把
`audio encode + prompt prefill` 从 decode slot 的空闲条件里拆出来，让
prefill 可以提前完成 KV cache 写入；decode 侧只负责持续解码，并在 slot
空闲时直接挂载已经 prefill 好的序列。

## 当前问题

当前 segment ASR 的主要流程是：

```text
segment queue
  -> audio encoder worker
  -> audio features queue
  -> decoder worker
       只有 decode slot 可用时才 prefill
       然后进入 continuous decode
```

这个流程里，prefill 和 decode slot admission 绑在一起。结果是：

- decode slot 空出来之后，可能还要先等待 audio feature 和 prefill；
- decode 不能稳定保持满 slot；
- audio features 可能堆积在 waiting 队列里，占用显存；
- GPU 利用率在前半段容易偏低，后半段 waiting 消化时才升高。

目标流程是：

```text
segment queue
  -> audio encoder + prefill worker
       audio encode
       构造 prompt
       根据 segment 时长估算总 KV block 预算
       分配完整 KV blocks
       执行 prefill
       enqueue prefilled sequence
  -> decoder worker
       持续 decode running slots
       slot 空闲时挂载已经 prefill 完成的 sequence
```

## 核心原则

prefill 不等待 decode slot。

KV blocks 和 decode slots 是两个不同资源：

- KV blocks 在 prefill 前分配；
- decode slots 只表示当前正在执行 decode 的并发序列；
- prefill 是否能执行，只看是否有足够 KV blocks；
- decode 是否能挂载新序列，只看是否有空闲 decode slot 和 prefilled queue。

因此调度策略应改为：

```text
prefill admission: 由 free KV blocks 决定
decode admission: 由 free decode slots 决定
```

## 状态流转

```text
VAD Segment
  -> Encoding
  -> PrefillPending
  -> Prefilling
  -> PrefilledWaitingForSlot
  -> RunningDecode
  -> Finished
```

更具体的执行路径：

```text
segment waveform
  |
  v
audio encoder
  |
  v
audio tokens / audio features
  |
  v
构造 text/audio prompt
  |
  v
计算 prompt tokens
  |
  v
根据 segment_audio_sec 估算 generated tokens budget
  |
  v
计算 total token capacity 和 required KV blocks
  |
  v
等待 free KV blocks 足够
  |
  v
一次性分配完整 KV blocks
  |
  v
prefill 写入 prompt KV
  |
  v
进入 prefilled queue
  |
  v
decode slot 空闲时挂载
  |
  v
decode 直到 EOS 或超过预分配 token capacity
  |
  v
释放 KV blocks
```

## KV Block 预算

每个 segment 的生成 token 预算由音频时长估算：

```text
generated_tokens_budget =
  max(
    ceil(segment_audio_sec * generated_tokens_per_sec),
    min_generated_tokens
  )
```

建议默认值：

```text
generated_tokens_per_sec = 8.0
min_generated_tokens = 32
```

这个值来自 `raw_audios` 的真实 segment 统计，使用 Qwen3 tokenizer 粗略统计
sentence/segment 的生成 token 密度：

```text
segments = 357

tokens/sec:
  p50 = 2.60
  p90 = 4.08
  p95 = 4.71
  p99 = 6.19
  max = 6.67
```

prefill 侧需要分配的是完整容量：

```text
total_token_capacity = prompt_tokens + generated_tokens_budget
required_blocks = ceil(total_token_capacity / block_size)
```

例如 `block_size = 32`：

```text
segment_audio_sec = 3.0
generated_tokens_budget = max(ceil(3.0 * 8.0), 32) = 32
prompt_tokens = 80
total_token_capacity = 112
required_blocks = ceil(112 / 32) = 4
```

decode 阶段不再动态扩 block，只能写入 prefill 前已经分配好的 blocks。
如果 decode 超过该 sequence 的 `total_token_capacity`，直接报错。

## Batch Prefill 放行规则

prefill 必须是 batch 输入，不是单 segment 串行提交。

对于一个待 prefill 的 segment batch：

```text
1. 先计算每个 segment 的 total_token_capacity 和 required_blocks。
2. 查询当前 free KV blocks。
3. 从 batch 中选择当前 free blocks 能容纳的前 N 个 segment。
4. 如果 N > 0，则对这 N 个 segment 分配 KV blocks 并执行 batch prefill。
5. 如果 N = 0，则等待 decode 释放 blocks。
6. 只要 free blocks 足够容纳至少一个 segment，就可以继续 prefill。
```

这里不检查 decode slot 数量。即使当前 running slots 已满，只要 KV blocks
足够，prefill 仍然可以提前执行，结果进入 `prefilled queue` 等待 slot。

## Worker 职责

### Encoder + Prefill Worker

职责：

```text
1. 从 segment queue 收集 segment batch。
2. 执行 audio encoder。
3. 构造 prompt 和模型输入。
4. 计算 prompt token 数。
5. 根据 segment duration 计算 generated token budget。
6. 计算 total token capacity 和 required KV blocks。
7. 等待 KV blocks 足够。
8. 一次性分配完整 KV blocks。
9. 执行 batch prefill。
10. 将 PrefilledSequence 发送给 decoder worker。
```

这个 worker 不关心 decode slots。

### Decoder Worker

职责：

```text
1. 持续 decode 当前 running slots。
2. 发现 EOS 时结束对应 sequence。
3. 如果 generated tokens 超过预分配容量，直接报错。
4. 释放 finished sequence 的 KV blocks。
5. slot 空闲时，从 prefilled queue 取 sequence 挂载。
6. 继续下一轮 decode。
```

decoder 不再为新 token 动态分配 KV blocks。

## PrefilledSequence

`PrefilledSequence` 是 prefill worker 交给 decoder worker 的对象。它应包含
decode 挂载时需要的全部信息：

```rust
struct PrefilledSequence {
    client_index: usize,
    request_id: usize,
    prompt_tokens: usize,
    token_capacity: usize,
    generated_tokens_budget: usize,
    reserved_blocks: usize,
    next_id: u32,
    // 以及当前 paged runtime decode 所需的 block table / position / metadata 状态
}
```

如果当前 `PagedDecodeSlot` 已经可以表示 request/KV 状态，它可以作为
`PrefilledSequence` 的一部分复用。否则需要拆分：

```text
prefilled request state: 已分配 KV、已完成 prompt prefill
decode slot state: 当前挂载在哪个 decode slot 上运行
```

## Paged Runtime 需要满足的语义

当前设计要求 paged runtime 支持：

```text
1. prefill 前按 total_token_capacity 分配完整 blocks；
2. paged_batch_prefill 使用给定 blocks 写入 prompt KV；
3. decode step 只校验 capacity，不再 try_allocate / allocate 新 blocks；
4. decode 超过 token_capacity 时直接返回错误；
5. sequence finished 或请求取消时释放全部 blocks；
6. prefilled 但尚未挂载 decode slot 的 sequence 也必须支持取消释放。
```

也就是说，`paged_batch_prefill` 需要从“内部按 prompt 动态分配”改成支持：

```text
外部或调用方已经确定每个 row 的 token capacity；
prefill 根据这些 capacity 完成 block 分配或使用已分配 block；
decode 后续只写入这些已分配 block。
```

第一版可以先让 `paged_batch_prefill` 接收 `token_capacities`，并在 prefill
内部按这些容量分配完整 blocks。后续如果要把 block admission 完全放到
scheduler 层，可以再拆出显式的 allocate API。

## 队列变化

当前的 waiting 语义比较混杂，后续应拆成：

```text
segment queue
prefill pending batch
prefilled queue
running decode slots
```

目标不是通知 segment queue，也不是让 decoder 直接从 segment queue 取数据。
decoder 只从 `prefilled queue` 补 slot。

## 参数变化

新增或保留的核心参数：

```text
max_num_seqs
  同时 decode 的最大 slot 数，同时也是 CUDA graph 最大 seqs 的控制参数。

generated_tokens_per_sec
  根据 segment 秒数估算生成 token 预算，默认 8.0。

min_generated_tokens
  短音频最小生成 token 预算，默认 32。
```

应该删除或不再作为正常预算依据的参数：

```text
max_new_tokens
generated_token_margin
```

decode 的限制来自每个 segment 预分配的 `token_capacity`，而不是全局
`max_new_tokens`。如果 decode 超过预分配容量，说明预算不足，应直接报错并
通过日志暴露。

## 日志和验证

prefill 前建议增加预算日志：

```text
prefill budget |
  seqs=...
  audio_sec min/p50/p90/max=...
  prompt_tokens min/p50/p90/max=...
  gen_budget min/p50/p90/max=...
  token_capacity min/p50/p90/max=...
  required_blocks min/p50/p90/max=...
  free_blocks=...
```

decode 完成时建议增加预算使用日志：

```text
decode budget |
  audio=...
  prompt_tokens=...
  gen_budget=...
  generated_tokens=...
  token_capacity=...
  hit_capacity=true/false
  reserved_blocks=...
```

关键验证指标：

```text
prefill_queue_len
prefilled_queue_len
decode_slot_occupancy
free_kv_blocks
reserved_blocks
used_blocks
wasted_blocks
over_capacity_count
audio_encoder_speed
prefill_tok_s
decode_tok_s
gpu_memory_peak
end_to_end_wall
```

## 实现计划

建议按以下顺序做：

```text
1. 在 paged_batch_engine 中支持 per-row token_capacity。
2. 让 prefill 按 token_capacity 分配完整 KV blocks。
3. 移除 decode 中的动态 block allocate，只保留 capacity 校验。
4. 在 scheduler 中根据 segment duration 计算 generated_tokens_budget。
5. admission 从 prompt-only blocks 改为 total_token_capacity blocks。
6. 引入 PrefilledSequence。
7. 拆分 waiting 队列和 prefilled queue。
8. 把 prefill 移入 encoder worker。
9. decoder worker 只从 prefilled queue 补 slot。
10. 删除 max_new_tokens / generated_token_margin 的正常路径依赖。
11. 对 benchmark / run / serve 统一验证吞吐、显存峰值和日志。
```

## 当前实现状态

截至 2026-06-24，当前代码已经完成以下部分：

```text
1. paged_batch_engine 支持 per-row token_capacity。
2. paged_batch_prefill 会按 token_capacity 一次性分配完整 KV blocks。
3. decode 阶段不再动态 allocate/extend KV blocks。
4. decode 每步会校验 needed_tokens <= token_capacity。
5. scheduler 已经引入 PrefilledSequence。
6. waiting 和 prefilled queue 已拆开。
7. prefill admission 已改为只看 free KV blocks，不看 decode slot。
8. decode admission 只负责把 prefilled sequence 挂载到空闲 slot。
9. dynamic continuous 路径已抽出 PrefillInputBuffer 和 drive_dynamic_prefill_stage。
10. dynamic prefill 已使用 paged_batch_prefill_shared，runtime Mutex 只覆盖 allocate/block table/cache clone/trim/free。
11. dynamic decode 已使用 paged_batch_decode_slots_at_step_shared，普通 decode forward 不再持有 runtime Mutex。
12. run_continuous_prepared_dynamic_async 已使用 scoped thread，将 prefill producer 和 decode consumer 分成两个物理执行线程。
13. dynamic input polling 的 metas/batch_lens/timing 计数已收进 DynamicPreparedSource，poll 闭包现在满足 Send 约束。
14. async producer/consumer 共享 AtomicUsize request id allocator，不再通过预留大区间规避 request_id 冲突。
15. shared decode 已恢复 CUDA graph replay：能 replay graph 时短锁进入 graph 路径，失败后禁用对应 key 并 fallback 到普通 forward。
16. async producer 到 decode consumer 的消息通道已改为无界 channel，不再用固定 channel 容量限制已经 prefilled 的 sequence。
17. decode capacity 校验语义已对齐 `prompt_tokens + generated_tokens_budget`：第 N 个生成 token 写入时需要 `prompt_tokens + N` 个 KV token。
18. dynamic continuous ASR 路径不再使用 `max_new_tokens` 判断正常完成或打印 hit 日志；正常限制来自 token_capacity。
19. prefill producer 已抽成 `DynamicPrefillProducer` / `PrefillStageOutput` / `PrefilledSequence` 公共 API，现有 dynamic decode 路径也改为通过该 producer 执行 prefill stage。
20. decode consumer 已抽成 `run_continuous_prefilled_dynamic`，输入是 `PrefillStageOutput` 流；现有 async dynamic 路径已改为 producer thread + 该 consumer 方法。
```

当前实际执行边界：

```text
pipeline segment_encoder_worker:
  segment waveform -> PreparedInputs + CPU audio_features
  通过 unbounded encoder_to_decoder channel 发送 EncodedAsrBatch

Qwen3ASR dynamic decode API:
  启动 prefill producer thread
  producer 从 EncodedAsrBatch poll source 拉取 CPU audio_features
  producer 执行 prefill，分配 KV blocks，生成 PrefilledSequence
  consumer 执行 decode，只从 prefilled queue 挂载空闲 slot
```

也就是说，当前已经做到“audio encode/prefill 与 decode 异步执行”。prefill
producer 的实现已经从闭包拆成公共 API，但它的物理 worker 仍由 dynamic
decode API 内部创建，不是 pipeline 的 `segment_encoder_worker` 直接调用
prefill。decode consumer 也已经拆成可消费 `PrefillStageOutput` 流的方法。
要进一步变成“encoder worker 直接执行 encode + prefill”，还需要把 metadata
routing 和结果聚合从 `decode_cpu_audio_features_dynamic*` 上移到
`vasr-transcribe` pipeline 层，让 pipeline 层直接连接
`DynamicPrefillProducer -> run_continuous_prefilled_dynamic`。

当前仍未完成“真正 CUDA stream 级并发最优形态”的原因：

```text
1. 同一个 ThinkerForConditionalGeneration 的 CUDA forward 是否能跨线程/跨 stream 并发执行还没有明确隔离。
2. CUDA graph map 仍在 PagedCacheRuntime 内；shared decode 的 graph replay 仍需要短锁进入 runtime。
3. Paged KV cache 写入与 decode 读取之间还需要明确 stream/event 同步语义。
4. prefilled queue 策略还没有做公平性/优先级调度。
5. prefill worker 目前在 dynamic decode API 内部，不在 pipeline segment_encoder_worker 内部；如果严格按“encoder worker = encode + prefill worker”落地，还需要继续重构 pipeline 的 metadata routing 和结果聚合。
```

因此下一步需要继续优化执行层：

```text
1. 进一步把 CUDA graph replay state 从 PagedCacheRuntime 中拆出，减少 graph replay 的 runtime 锁范围。
2. 为 prefill 写 KV 和 decode 读 KV 增加明确的 CUDA stream/event 同步。
3. 用真实 benchmark 比较 async producer/consumer 与旧 single-loop 路径的吞吐、显存峰值和 decode slot occupancy。
4. 根据 benchmark 结果调整 prefilled queue 策略。
```

## 当前明确决策

```text
1. prefill 不等待 decode slot。
2. prefill 是否执行只看 free KV blocks。
3. prefill 是 batch 输入。
4. decode 不动态扩 block。
5. decode 超过 token_capacity 直接报错。
6. 不保留旧路径作为长期 fallback。
7. 使用一个 encoder + prefill worker。
8. 预算基于 VAD 之后的 segment 时长。
9. max_new_tokens 后续删除。
10. generated_token_margin 不需要。
```

## 后续仍需细化

```text
1. prefilled sequence 取消或请求超时时如何释放 KV blocks。
2. prefilled queue 是否严格 FIFO，还是支持短任务优先或 block-aware 排序。
3. prefill 和 decode 是否最终使用不同 CUDA stream 并发执行。
4. 如果预算不足频繁发生，是否只调大 generated_tokens_per_sec，还是按语言/场景自适应。
5. serve 模式下跨请求的 prefilled queue 公平性策略。
```
