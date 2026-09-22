# vasr-data

vASR 内部的音频、时间轴、标注和 SQLite 数据集模型。

代码从 [`asr-data` 0.1.2](https://github.com/di-osc/asr-data) 迁入，许可证为 MIT（见 `LICENSE`，Copyright (c) 2026 di-osc）。中文文本归一化用的 FST 资源来自 WeText，说明见 `assets/wetext/README.md`。

相对上游，这里去掉了两块本仓库不用的能力：

- Python 绑定
- 依赖 `modelhub` 的 ModelScope 数据集下载（`dataset` feature）

默认开启 `db` 与 `metrics`：SQLite 存储，以及 CER / 中文文本归一化。
