# Transcribe 全面重构记录

基线为 `82b4951`，隔离分支 `codex/refactor-transcribe`。模型、BF16、VAD、解码与浮点设置保持。

## 验证进度

- 冻结旧版二进制，SHA256 `b934ed0905f9a3513c9bd91f052bb331b3a3a217b759adc5202ced03b707a889`；数据及既有交付清单归档到 `results/refactor-2026-10-09/baseline-manifest.json`。
- 旧版 CLI Metal 配置：97 项 Rust 测试通过。
- 新增三个真实失败测试：加载通道关闭、结果通道关闭均使请求不结束；重复结果使请求提前结束。旧代码运行三项均失败，日志 `lifecycle-red.log`。
- 统一请求完成与失败通知，校验条目及声道，忽略同条目同声道的重复完成；失败保留已完成条目，为未完成声道补齐结果；关闭通道清理路由。修复后 CLI 100 项测试通过，日志 `registry-green.log`。

## 验收清单

- [x] worker 监督、关闭状态与取消回收
- [x] 请求／条目／编码音频预算和入口背压
- [x] 配置与分段状态整理
- [x] Metal 编译缓存辅助代码及模块拆分
- [x] CPU、Rust、Metal 数值、脚本回归
- [x] 同期各三轮完整数据集，文本逐条一致、CER 不升高、吞吐回退不超过 3% 且超过 100x
- [x] 30 分钟持续来流、短请求、元数据与 KV 排空

历史 122.6x 仅为参考，本轮以以下同期测试作为性能验收。

## Metal 公共编译与缓存

pointwise、residual norm、rotary 共用编译选项和缓存；设备、源码、函数、常量和精度均进入 key。保持 Candle 5404348 的现代及旧版 API 选项与环境变量解析规则，未修改 `.metal` 算术。

审查发现 Candle 常量对象的 F32 Eq/Hash 对 NaN 和正负零不一致。公共 key 改用具名类型和浮点位模式，原常量值、类型及重复 setter 顺序保持不变。新增四项回归测试先失败再通过；独立 spec 和质量审查通过。

父任务重新执行默认和关闭 fast-math 的相关 Metal 测试，两组各 19 项通过，含实际非零 F32/F16/BF16 数值、stride/offset/尾部和原生 pipeline 重用。日志 `metal-default-green.log`、`metal-precise-green.log`。旧版系统分支在本机验证配置选择与 setter，不声称已在旧版 macOS 实机验收。

重新计算七个模型文件及 99 个主数据集缓存音频的 SHA256，均与原正确基线一致；归档 `model-manifest.json` 和 `audio-manifest.json`。

## 具名分段状态与结果组装

将 StreamingMetadataDecoder 从转写编排中拆出，保持原公开导出和构造／方法签名。AudioDecodeProgress 收集文本、语言及完成进度，早期条目返回与最终 finish 共用按全局 chunk 索引排序的组装函数。

四项特征测试在状态替换和模块提取前后均通过：乱序／重复完成、跨批次 push 与结果索引、波形释放、零 chunk 及未完成错误。独立规格、质量审查均通过。日志 metadata-characterization.log、metadata-module-green.log。此阶段没有改动模型计算。

## CNN 编译缓存迁移与精度模式对照

将既有 CNN bias/GELU 与 im2col32 的编译、缓存迁入公共辅助代码。保持原生默认编译选项 `None` 与显式 Fast 的区别；缓存键纳入这一区别。所有 dispatch、dtype、布局、offset、尾部及 Metal 源码不变。规格和独立质量审查通过。

默认浮点设置下，相关 28 项测试通过。关闭 fast-math 后，完整扫描发现 `nhwc_bias_gelu_matches_materialized_reference` 的 F32 精确相等断言失败；在原版 `82b4951` 相同配置下也失败。两版实际输出和参考输出分别具有完全相同的 SHA256，见 `precise-conv-comparison.json`。既有 NHWC kernel 使用原生默认浮点设置，而参考路径服从 Candle 精确设置；本轮不改变计算设置以消除这个原有差异。排除这一项后，扩展 Metal 测试 36 项通过；模型侧精确模式 90 项通过。不得将本轮结果表述为所有精确模式测试通过。

## 正式对照与持续来流脚本

新增 refactor 模式，对冻结的两个二进制交替运行三轮，均采用自动默认策略、并发 28、batch 4、mono=true，预热完整数据集。正式模式清除额外浮点覆盖和内部实验覆盖。审计同时检查原始文本、Bench 文本、条目映射、CER、预热、示例标签以及中位吞吐门槛；精确 3% 边界使用十进制比较。

持续来流验收新增基线示例原始文本对照、每个短请求文本对照，以及登记、路由、请求／条目／来源预算的容量与最终回收检查。脚本回归 21 项通过（正式 10、持续来流 11），bash 语法检查通过。脚本阶段先交付可复现的验收入口；后续真实六轮和 30 分钟结果见下方正式验收。

## 不可变模型策略与模块拆分

算子内部配置在模型构造时形成实例私有的不可变快照；调度器捕获执行覆盖、prefill 长度和诊断配置，消除 forward 热路径的重复环境读取和进程级策略缓存。保留严格布尔、规范化布尔、存在性开关以及无效／非 Unicode 值各自原有解析规则；公开构造和 ExecutionPolicy 字面量保持兼容。

模型和服务共用实际模型结构的资格判断；仅规范名称不足以启用已验证策略。新增 config-only 入口读取 checkpoint 配置，不额外加载模型权重或 GPU。公开导出保留。拆分 text 策略、投影／MLP、attention、测试及 audio 策略、attention、测试，数学函数保持。

相关规格与质量审查通过。模型普通 CPU 测试 117 项通过；模型侧默认浮点专项 45 项、关闭 fast-math 后扩展 90 项通过。实际 GPU 测试涵盖非零权重 encoder F32/BF16 混批、尾窗、窗口隔离、投影融合、stride 与 offset；保留参考路径和实验开关。本机不声称完成 CUDA 运行验证。

## 请求生命周期、入口预算与取消

请求登记器按稳定请求／条目／声道管理结果，重复完成仅计一次；统一失败出口保留完成项并补齐失败项、撤销路由、唤醒等待者。监督器保存 stage 的 abort handle 并监视 join，包括嵌套 encoder、prefill、decode；异常退出、panic、意外通道关闭转为 fatal，正常 stop 使用明确停止状态。

HTTP middleware 在 JSON 请求体读取前申请请求许可及编码来源预算，超载等待；活跃请求用 guard 自动释放。进入加载前按唯一条目逐条申请许可，大批次不会一次占满许可后等待自身释放。每条目的许可沿加载、VAD、encoder 和已提交 GPU 任务传递；取消跳过尚未开始的工作，已进入 GPU 的工作完成后回收，迟到结果不会进入其他请求。管理锁不跨通道等待或 GPU forward。

请求上限为最终 active_slots，唯一待处理条目上限为安全容量两倍；编码来源预算按可用内存 1/16 推导并限制在 512 MiB–2 GiB，纳入内存安全估算。HTTP 无 Content-Length 时预留 512 MiB 请求体上限。现有 PCM／segment 预算保留。内部来源超出总预算时按原条目索引报告失败，避免不可满足的 semaphore 等待；HTTP 超载不返回 429。

审查额外捕获三项实际缺陷并先 RED 后 GREEN：最后一个 VAD 子任务失败而上游保持打开时必须立即失败；stop 与 submit 注册竞态必须在排空前关闭 gate；立体声加载完成后、发表结果前取消不能产生无归属的右声道 ID。额外 ID 分配与完整声道登记在同一 pending 锁内完成，16 次重复取消的退役范围由 16 降为 1。完整 CLI Metal 回归 129 lib + 3 bin 通过；真实模型加载集成测试按既有标记保留待正式验收。日志 `lifecycle-review-red.log`、`lifecycle-stereo-red.log`、`lifecycle-final.log`。

拆分流水线统计／日志及测试模块；32 个统计函数体保持。ProducerChunkRange 用具名字段替换跨 producer 范围元组，集中索引换算和剩余计数检查，补充越界、乱序及重复回收测试。

## 混批取消、结果顺序与公开入口兼容

排队的编码批次进入初始及追加 prefill 前按存活请求筛选。用同一 mask 保序筛选 segments、waveforms、inference targets、encoder CPU 特征行、chunk 和语言元数据；全部存活直接保持原对象，全部取消则不提交 prefill。特征 token 行数按原 mask 非零长度计算并校验总行数。prefill 的 debug 存在性与 coalesce wait 在 worker 构造时捕获，不再使用进程级 OnceLock 或逐轮读取。

独立 RED 测试复现非原始分组顺序时 prepared 行与 metadata 不一致，修复仅将现有 CPU 行按原 chunk 顺序重排，默认顺序保持原路径。非零特征／混长短／交错分组测试 6 项通过；真实 prefill worker 的旧 any-live 语义会保留取消条目，先 RED 后 GREEN。

质量审查发现公开 start() 把 decode_max_seqs=0 的自动语义缩为单槽，以及不同来源使用相同 caller index 时请求无法结束。两项新回归先失败再通过。公开包装在有重复 caller index 时使用输入位置作为内部身份，并在成功／失败统一出口还原调用方索引；HTTP 已使用唯一输入位置，路径不变。零容量恢复原调度默认 8，显式执行槽数受容量约束。5 项回归涵盖不同来源、重复来源、立体声、失败及取消许可回收。

最终完整 CLI 测试 139 lib + 3 bin 通过，包含本地监听和 Metal 设备探测，模型加载集成测试仍按标记待正式验收。普通 CPU 模型 121 项通过，CPU CLI 构建检查通过（保留条件编译下的 unused/dead-code 警告）；日志 `cli-final-complete.log`、`cpu-model-complete.log`、`cpu-check-complete.log`、`public-compat-red.log`、`public-compat-complete.log`。本轮协作审查受使用额度中断，最终新增兼容包装由父任务复核并执行测试，不声称全范围最后版本均完成独立双重审查。

## 2026-10-10 同期正式验收

冻结原版与候选二进制，分别来自 `82b4951` 和 `278e609`；候选 SHA256 为 `71b93b3b459d5117f5ca483d4ab7f1dd53adb903bc9cc273ca7bb917cfb66709`。实际输入及二进制 110 个文件重新核验，另补充四个 FSMN VAD 权重／配置哈希。两版使用 M4 Pro/48 GB、Metal、BF16、相同默认 VAD、AC 供电；额外阶段同步、profiling 和实验覆盖均关闭。客户端并发 28、batch 4、mono=true。

| 版本 | 第一轮 | 第二轮 | 第三轮 | 中位吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 原版 | 122.6599× | 122.8886× | 123.1561× | 122.8886× |
| 重构版 | 123.5180× | 118.1813× | 123.2974× | 123.2974× |

候选中位相对同期原版 +0.3327%，超过 100×，满足回退不超过 3% 的要求。候选有一轮波动，不删除该轮、不将最快结果作为中位。六轮正式与六轮完整预热全部 99 条成功，原始及 Bench 文本逐条一致，编辑数 2762／参考字符 16651，加权 CER 均为 16.5875923%。六轮 example1 均零 CER，原始文本均保持“我要修改商品帖子的价格。”。自动执行策略保持 active64/decode64/mixed=true/prefill16/encoder16/audio120s；新增编码来源预留仅降低内存安全容量。

正式观察到候选 RSS 峰值最高 3.64 GiB，原版最高值和各轮明细保留在审计 JSON；swap 在全部正式样本中保持 5890.12 MiB，该用量在测试开始前已存在。正式测试的采样峰值不能替代持续内存增长验证。审计输出 `results/refactor-2026-10-09/formal-audit.log`，完整结果与元数据位于同目录 `formal/`。30 分钟持续来流已完成，结果见下方。

复现（从 worktree 仓库根目录运行）：

```bash
cargo install --path vasr-transcribe --force --features metal
bash docs/performance/scripts/m4_formal.sh \
  results/refactor-2026-10-09/candidate-binary \
  results/refactor-2026-10-09/formal-new refactor \
  results/refactor-2026-10-09/baseline-binary
python3 docs/performance/scripts/audit_formal.py results/refactor-2026-10-09/formal-new
bash docs/performance/scripts/m4_soak_run.sh \
  results/refactor-2026-10-09/candidate-binary \
  results/refactor-2026-10-09/soak-new \
  results/refactor-2026-10-09/formal/control-r1-perf.json \
  1 require automatic refactor \
  results/refactor-2026-10-09/formal/control-r1-example1.json
```

## 30 分钟持续来流最终验收

28 个客户端持续提交 batch 4，另每分钟提交 example1 短探针，所有请求都实际调用服务，无结果缓存和重试。完整来流 1800 秒，队列在 1802.6234 秒排空，并继续采样 59.90 秒。

- 1123 次请求全部成功，零失败、遗漏、串结果或异常重复。4372 条主转写的原始及 Bench 文本与同期原版逐条一致，所有 99 个样本各重复 44–45 次。
- 30 次短探针全部零 CER，覆盖整个持续负载，最大延迟 48.0223 秒；验证能完成，不代表低延迟 SLA。
- 含短探针和排空的持续吞吐 123.1751×；同输入重复次数加权 CER 两版均为 16.5839132%，平均 CER 均为 21.7240526%。
- KV 3100/3100 可用块全部恢复；登记与路由均归零，请求许可 64/64、条目许可 636/636、编码来源字节许可 1742505984/1742505984 全部归还。来源容量本轮约 1.62 GiB，由启动时可用内存推导，上限 2 GiB；这次启动固定后不随运行变动。
- decoder、producer ranges、prefill buffers 和结果日志的直接活跃条目计数均排空。producer 保留唯一当前 ID 标量；不将它表述为未回收的逐项元数据。登记峰值 29、路由峰值 113，未越过入口预算。
- RSS 采样峰值 3.9489 GiB。第 5–10／10–15／15–20／20–25／25–30 分钟中位分别 3.86956／3.79920／3.88702／3.86108／3.87926 GiB，波动约 0.09 GiB，无持续增长趋势。系统 swap 起点、终点及峰值均为 5890.12 MiB，未增长；初始 swap 用量属于测试前既有状态。

审计输出 `soak/audit.log`，逐条质量、计数和内存证据为 `soak/soak/quality.json`、`summary.json`、`responses.jsonl`、`memory.jsonl`。验收后模型、VAD、两个输入库、实际音频缓存、两个二进制及候选源文件哈希再次一致；归档 `post-acceptance-hashes.json` 和 `runtime-metadata.json`。本机为 macOS 26.5.2、Apple M4 Pro/48 GiB、Rust 1.99.0、Candle 5404348。测试服务已退出。

原版 ASR 加载时间中位 2.126 秒、范围 1.548–2.253 秒；候选中位 2.038 秒、范围 1.412–2.275 秒。完整预热墙钟中位原版 42.2589 秒、候选 41.0070 秒；各轮预热分别保留，正式吞吐不计加载和预热。

本轮完成七项职责重构，没有新增量化、改模型／VAD／解码或改变算子数学。监督、预算和取消所有权的后续修改集中在同一生命周期提交，避免暴露不能唤醒预算等待者的半成品；其余登记基础、状态、Metal 公共代码、策略和验收脚本分别提交。保留隔离分支与 worktree，主分支不合并。已知原有精确模式 CNN 断言例外和审查额度中断如上记录，不声称 CUDA 实机运行验证或所有旧版 macOS API 实机验证。
