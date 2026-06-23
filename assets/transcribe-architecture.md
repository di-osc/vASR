# Current Transcribe Pipeline Architecture

```text
                    input jobs
                        |
                        v
+--------------------------------------------------+
| Loader Worker                                     |
| concurrency = CPU available_parallelism()         |
| loads/decodes each job into audio waveform         |
+--------------------------------------------------+
                        |
                        | loader_to_vad
                        | unbounded channel
                        | bounded by waveform memory permits
                        v
+--------------------------------------------------+
| VAD Worker                                        |
| concurrency = CPU available_parallelism()         |
| consumes LoadedJob from loader_to_vad              |
| runs VAD                                           |
|                                                   |
| if no speech:                                     |
|   returns result directly, no ASR                  |
|                                                   |
| if speech:                                        |
|   splits audio into segments                       |
|   each segment carries waveform data               |
+--------------------------------------------------+
                        |
                        | vad_to_asr
                        | unbounded channel
                        | bounded by segment waveform memory permits
                        v
+--------------------------------------------------+
| ASR Worker                                        |
| single async worker                                |
| consumes messages from vad_to_asr                  |
|                                                   |
| if message is Job:                                |
|   collect_job_microbatch()                         |
|   runs whole-job ASR path                          |
|                                                   |
| if message is Segment:                            |
|   collect_segment_microbatch()                     |
|   groups segments by max_batch_audio_sec           |
|   sends grouped segments to Encoder Worker         |
+--------------------------------------------------+
                        |
                        | asr_to_encoder
                        | bounded channel
                        | capacity = ASR_ENCODED_FEATURE_QUEUE = 1
                        | unit = encoded request batch count
                        v
+--------------------------------------------------+
| Segment Encoder Worker                            |
| single async worker + spawn_blocking               |
|                                                   |
| input: Vec<AsrSegmentJob>                          |
| batch comes from collect_segment_microbatch()      |
|                                                   |
| Qwen3 ASR path:                                   |
|   run_segment_asr_encode()                         |
|   runs audio encoder                               |
|   stores audio features on CPU                     |
+--------------------------------------------------+
                        |
                        | encoder_to_decoder
                        | bounded channel
                        | capacity = ASR_ENCODED_FEATURE_QUEUE = 1
                        | unit = encoded feature batch count
                        v
+--------------------------------------------------+
| Segment Decoder Worker                            |
| single async worker                                |
|                                                   |
| input: encoded audio features batch                |
|                                                   |
| Qwen3 ASR path:                                   |
|   decode_cpu_audio_features...                     |
|   prefill                                          |
|   continuous decode                                |
|   timeline/text result                             |
+--------------------------------------------------+
                        |
                        | asr_to_result
                        | bounded channel
                        | capacity = unique_inputs.len()
                        v
+--------------------------------------------------+
| Result Collector                                  |
| collects each job result                           |
| expands duplicate inputs back to original order    |
+--------------------------------------------------+
```

## Queue And Concurrency Rules

```text
Loader concurrency:
  CPU available_parallelism()

VAD concurrency:
  CPU available_parallelism()

loader_to_vad queue:
  not bounded by message count
  bounded by waveform memory permits

vad_to_asr queue:
  not bounded by message count
  bounded by segment waveform memory permits

Encoder:
  single worker
  processes one segment microbatch at a time

Decoder:
  single worker
  processes one encoded feature batch at a time

audio_features queues:
  still bounded by fixed message count
  capacity = ASR_ENCODED_FEATURE_QUEUE = 1
```

## Segment Microbatch Formation

```text
ASR Worker receives the first Segment
  |
  v
collect_segment_microbatch()
  |
  +-- non-blocking try_recv() from vad_to_asr
  |
  +-- keeps collecting already queued Segment messages
  |
  +-- stops before exceeding max_batch_audio_sec
  |
  +-- stops if the queue is temporarily empty
  v
Vec<AsrSegmentJob> -> Encoder Worker
```

Current segment batching is not driven by decoder free slots.

```text
Current behavior:
  available segments in vad_to_asr
  + max_batch_audio_sec
  -> encoder batch size

Not current behavior:
  decoder free slots
  -> number of segments pulled for encoder
```

## Compact Flow

```text
input
  -> loader concurrent load
  -> audio waveform memory-bounded queue
  -> VAD concurrent segmentation
  -> segment waveform memory-bounded queue
  -> ASR worker segment microbatch
  -> encoder single worker creates CPU audio features
  -> audio features queue, capacity=1
  -> decoder single worker prefill/decode
  -> result
```
