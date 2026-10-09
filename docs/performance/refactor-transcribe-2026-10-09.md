# Transcribe 全面重构记录

基线为 `82b4951`，隔离分支 `codex/refactor-transcribe`。模型、BF16、VAD、解码与浮点设置保持。

## 验证进度

- 冻结旧版二进制，SHA256 `b934ed0905f9a3513c9bd91f052bb331b3a3a217b759adc5202ced03b707a889`；数据及既有交付清单归档到 `results/refactor-2026-10-09/baseline-manifest.json`。
- 旧版 CLI Metal 配置：97 项 Rust 测试通过。
- 新增三个真实失败测试：加载通道关闭、结果通道关闭均使请求不结束；重复结果使请求提前结束。旧代码运行三项均失败，日志 `lifecycle-red.log`。
- 统一请求完成与失败通知，校验条目及声道，忽略同条目同声道的重复完成；失败保留已完成条目，为未完成声道补齐结果；关闭通道清理路由。修复后 CLI 100 项测试通过，日志 `registry-green.log`。

## 剩余验收

- [ ] worker 监督、关闭状态与取消回收
- [ ] 请求／条目／编码音频预算和入口背压
- [ ] 配置与分段状态整理
- [ ] Metal 编译缓存辅助代码及模块拆分
- [ ] CPU、Rust、Metal 数值、脚本回归
- [ ] 同期各三轮完整数据集，文本逐条一致、CER 不升高、吞吐回退不超过 3% 且超过 100x
- [ ] 30 分钟持续来流、短请求、元数据与 KV 排空

历史 122.6x 仅为参考，尚未构成本次重构的性能验收。

## Metal 公共编译与缓存

pointwise、residual norm、rotary 共用编译选项和缓存；设备、源码、函数、常量和精度均进入 key。保持 Candle 5404348 的现代及旧版 API 选项与环境变量解析规则，未修改 `.metal` 算术。

审查发现 Candle 常量对象的 F32 Eq/Hash 对 NaN 和正负零不一致。公共 key 改用具名类型和浮点位模式，原常量值、类型及重复 setter 顺序保持不变。新增四项回归测试先失败再通过；独立 spec 和质量审查通过。

父任务重新执行默认和关闭 fast-math 的相关 Metal 测试，两组各 19 项通过，含实际非零 F32/F16/BF16 数值、stride/offset/尾部和原生 pipeline 重用。日志 `metal-default-green.log`、`metal-precise-green.log`。旧版系统分支在本机验证配置选择与 setter，不声称已在旧版 macOS 实机验收。

重新计算七个模型文件及 99 个主数据集缓存音频的 SHA256，均与原正确基线一致；归档 `model-manifest.json` 和 `audio-manifest.json`。

## 具名分段状态与结果组装

将 StreamingMetadataDecoder 从转写编排中拆出，保持原公开导出和构造／方法签名。AudioDecodeProgress 收集文本、语言及完成进度，早期条目返回与最终 finish 共用按全局 chunk 索引排序的组装函数。

四项特征测试在状态替换和模块提取前后均通过：乱序／重复完成、跨批次 push 与结果索引、波形释放、零 chunk 及未完成错误。独立规格、质量审查均通过。日志 metadata-characterization.log、metadata-module-green.log。此阶段没有改动模型计算。
