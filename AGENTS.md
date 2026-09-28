## benchmark

在仓库根目录执行。当前工作目录决定 `.env` 和 `datasets/` 是否能被找到。

本机是 Apple M4 Pro（Metal 4，48 GB 统一内存）。`vasr-transcribe` 用 `metal` feature。换到 NVIDIA 机器时改成 `--features cuda`。

### Transcribe

- 覆盖安装适配本机硬件的最新命令行。已安装的二进制可能还是旧子命令（`benchmark`、`extract-audio`），必须 `--force`。

```bash
cargo install --path vasr-transcribe --force --features metal
```

- 启动服务。未指定模型时默认是 `Qwen3-ASR-0.6B`，对应仓库 id `Qwen/Qwen3-ASR-0.6B`。`serve` 按设备和可用内存自动估算并发与 encoder batch，不要手写 `--max-num-seqs` 等容量参数，除非用户明确要求。默认监听 `0.0.0.0:8867`。

```bash
vasr-transcribe serve --model Qwen3-ASR-0.6B
```

- 服务起来后再测吞吐。默认请求 `http://127.0.0.1:8867/transcribe`，结果写入 `results/perf_<timestamp>.json`。

```bash
vasr-transcribe perf -i datasets/call-100.vasr
```
