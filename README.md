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
