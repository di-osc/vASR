# vASR

vASR is a Rust speech inference framework built around a timeline data model.

The current implementation keeps the existing Qwen3-ASR Candle runtime and wraps it as an
`AsrModel`. New framework crates define audio loading, model capabilities, fasr-compatible
protocol DTOs, and HTTP/WebSocket service entry points.

## Crates

- [`asr-data`](https://github.com/di-osc/asr-data): `Audio`, `Timeline`, `Annotation`, `Waveform`, and SQLite dataset storage.
- `vasr-audio`: audio source loading into normalized 16 kHz mono waveforms.
- `vasr-runtime`: model capability traits, Qwen3-ASR ASR model wrapper, FSMN/FireRed VAD, offline/realtime pipelines.
- `vasr-protocol`: fasr-compatible transcribe and realtime DTOs.
- `vasr-server`: axum routes for split transcribe and realtime services.
- `vasr`: Candle Qwen3-ASR model implementation.

## Development

```bash
cargo test --workspace
```

Build with Apple Metal support (includes PagedAttention for batch transcribe):

```bash
cargo test --workspace --features vasr-runtime/metal
```

Build with CUDA support:

```bash
cargo test --workspace --features vasr-runtime/cuda
```

## Serve

Start the CPU transcribe HTTP service:

```bash
cargo run -p vasr-cli --bin vasr-transcribe -- serve \
  --model /path/to/Qwen3-ASR-0.6B \
  --vad-model /path/to/funasr-fsmn-vad \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu
```

Start the Metal transcribe service:

```bash
cargo run -p vasr-cli --bin vasr-transcribe --features metal -- serve \
  --model /path/to/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --device metal \
  --isq q8_0
```

Start the standalone realtime WebSocket service:

```bash
cargo run -p vasr-realtime -- serve \
  --model /path/to/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu
```

Transcribe local audio files to JSON (`{stem}.transcribe.json`):

```bash
cargo run -p vasr-cli --bin vasr-transcribe --features metal -- run \
  --model /path/to/Qwen3-ASR-0.6B \
  --device metal \
  --input ./raw_audios/audio.wav \
  --output ./outputs/

cargo run -p vasr-cli --bin vasr-transcribe --features metal -- run \
  --model /path/to/Qwen3-ASR-0.6B \
  --device metal \
  --input ./raw_audios/ \
  --output ./outputs/ \
  --recursive
```

Benchmark ASR character error rate (CER) against a `VasrRecordList` MessagePack file:

```bash
cargo run -p vasr-cli --bin vasr-transcribe --features metal -- benchmark \
  --model /path/to/Qwen3-ASR-0.6B \
  --device metal \
  --input ./datasets/eval.records.msgpack \
  --output ./outputs/benchmark.json
```

Convert a FASR `FASRAL01` AudioList binary file to `VasrRecordList` MessagePack:

```bash
cargo run -p vasr-cli --bin vasr-transcribe -- convert-fasr \
  --input ./lbg_400call-200.bin \
  --output ./lbg_400call-200.vasr.msgpack
```

Initialize the realtime 2-pass TOML configuration:

```bash
cargo run -p vasr-cli -- init --cfg vasr.cfg
```

Edit `vasr.cfg` to select all three realtime models:

```toml
[models.vad]
kind = "firered"
device = "cpu"

[models.online]
kind = "sensevoice-online"
checkpoint = "iic/SenseVoiceSmall"
device = "cuda"

[models.offline]
kind = "qwen3-asr"
checkpoint = "Qwen/Qwen3-ASR-0.6B"
device = "cuda"
```

Start the service:

```bash
cargo run -p vasr-cli --features cuda -- serve --cfg vasr.cfg
```

The online model emits low-latency partial transcripts. FireRed Stream-VAD detects the utterance
boundary, then the complete buffered utterance is sent to the offline model for the final
transcript. Model selection, checkpoints, devices, quantization, and model-specific parameters
are configured only in the TOML file.

Build the unified release binary:

```bash
cargo build --release -p vasr-cli --features cuda
./target/release/vasr serve --cfg vasr.cfg
```

CUDA FlashAttention builds fetch CUTLASS from
`https://gitee.com/good-as-water/cutlass.git` by default. Override the mirror
with `VASR_CUTLASS_REPO` and, when necessary, the revision with
`VASR_CUTLASS_COMMIT`.

FunASR FSMN VAD is enabled by default for offline `/transcribe` segmentation. Pass
`--vad-model /path/to/funasr-fsmn-vad` to use a local directory containing `model.pt` and
`am.mvn`, or `--no-vad` on `vasr-transcribe` to disable offline VAD segmentation. The default
FSMN VAD `speech_noise_thres` is `0.5`; use `--vad-threshold <0..1>` or `VASR_VAD_THRESHOLD`
to tune segmentation. Additional offline VAD tuning flags:

- `--vad-min-speech-ms` (default `250`)
- `--vad-min-silence-ms` (default `500`)
- `--vad-max-segment-sec` (default `30`)

When `--vad-model` is omitted, vASR downloads/uses `funasr/fsmn-vad` from the Hugging Face cache.

The realtime service uses FireRed `Stream-VAD`. Its startup settings live under `[models.vad]`
instead of command-line flags. Per-session VAD fields can still be adjusted through
`session.update`.

Weight dtype is selected automatically from the device (`f32` on CPU, `bf16` on Metal/CUDA).

By default, component runtime logs (loader/VAD/ASR timings) are hidden. Pass `--verbose`
or set `VASR_LOG=warn,vasr_cli=info,vasr_runtime=info,vasr_server=info` to enable them.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Offline transcribe:

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -H 'content-type: application/json' \
  -d '{
    "inputs": [
      {
        "url": "file:///absolute/path/audio.wav",
        "mono": true,
        "hotword": "OpenAI fasr"
      }
    ]
  }'
```

Realtime WebSocket endpoints:

```text
ws://127.0.0.1:8001/v1/realtime
```

The realtime API accepts fasr-style JSON events such as `session.update`,
`input_audio_buffer.append`, `input_audio_buffer.commit`, and `session.finish`.

## Architecture

Offline flow:

```text
TranscribeRequest
  -> vasr-audio AudioLoader
  -> Waveform
  -> vasr-runtime OfflinePipeline
  -> Timeline + Annotation
  -> vasr-protocol TranscribeResponse
```

Realtime flow:

```text
WebSocket PCM16 base64 events
  -> AudioBytesStream
  -> AudioChunk
  -> RealtimePipeline
  -> ServerRealtimeEvent
```
