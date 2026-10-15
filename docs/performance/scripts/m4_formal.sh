#!/bin/bash
# Alternating same-binary controls or two-binary refactor controls, with a complete warmup after each restart.
# All outputs remain available for per-item CER and source-identity auditing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
binary=${1:?frozen binary path required}
output=${2:?new output directory required}
experiment=${3:-throughput}
[[ "$experiment" == throughput || "$experiment" == autorelease || "$experiment" == automatic || "$experiment" == prefill-gate-up || "$experiment" == prefill-wave || "$experiment" == prefill-rope || "$experiment" == prefill-rope-kernel || "$experiment" == rope-automatic || "$experiment" == refactor ]] || { echo "unknown experiment" >&2; exit 1; }
baseline_binary=
if [[ "$experiment" == refactor ]]; then
    baseline_binary=${4:?baseline binary required for refactor}
    [[ -x "$baseline_binary" && -x "$binary" ]] || { echo "both frozen binaries must be executable" >&2; exit 1; }
    baseline_binary=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$baseline_binary")
fi
[[ ! -e "$output" ]] || { echo "output directory already exists" >&2; exit 1; }
mkdir -p "$output"
binary=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$binary")
output=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$output")
printf '%s\n' "$experiment" > "$output/experiment.txt"
awake_pid=
server_pid=
monitor_pid=
cleanup() {
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    monitor_pid=
    server_pid=
}
trap 'cleanup; if [[ -n "$awake_pid" ]]; then kill "$awake_pid" 2>/dev/null || true; wait "$awake_pid" 2>/dev/null || true; fi' EXIT
[[ ! -e .env ]] || { echo ".env present; refuse an uncontrolled formal environment" >&2; exit 1; }
caffeinate -d -i -u -t 3600 &
awake_pid=$!
pmset -g batt > "$output/power.txt"
if [[ -n "$baseline_binary" ]]; then shasum -a 256 "$baseline_binary" > "$output/baseline-sha256.txt"; fi
shasum -a 256 "$binary" datasets/call-100.db datasets/example1.db > "$output/sha256.txt"
# Avoid inherited experimental options; explicit controls also work if these
# optimizations later become defaults. Existing validated CNN/packed paths stay on.
while IFS= read -r inherited_option; do
    if [[ -n "$inherited_option" ]]; then unset "$inherited_option"; fi
done < <(python3 -c 'import os;print("\n".join(sorted(k for k in os.environ if k.startswith("VASR_") or k == "CANDLE_METAL_COMPUTE_PER_BUFFER")))')
unset VASR_METAL_STAGE_PROFILE VASR_METAL_STAGE_SYNC VASR_LOG VASR_ISQ VASR_MAX_NUM_SEQS
unset VASR_DEVICE VASR_METAL_ASR_DTYPE VASR_GPU_MEMORY_UTILIZATION
unset VASR_METAL_CONV_NHWC VASR_METAL_CONV_IM2COL32 VASR_METAL_CONV_BIAS_GELU
unset VASR_METAL_PACKED_PREFILL VASR_METAL_PACKED_PAGED_ATTN
unset VASR_DEBUG_REQUEST_MEM VASR_DEBUG_CUDA_MEM VASR_DEBUG_LOCK_WAIT
if [[ "$experiment" == refactor ]]; then
    unset CANDLE_METAL_ENABLE_FAST_MATH
    printf '%s\n' 'CANDLE_METAL_ENABLE_FAST_MATH=unset (Candle default)' > "$output/floating-point.txt"
fi
export VASR_PROFILE=false
# The per-component execution policy lines are DEBUG and no longer part of the
# default log filter. Keep them in the archived server log explicitly; this sets
# log verbosity only, not any capacity/batching override the run must exercise.
export VASR_LOG=vasr_transcribe=info,vasr::execution=debug
unset VASR_DISABLE_MIXED_STEP_DECODE VASR_METAL_GEMM_TRACE
unset VASR_METAL_PREFILL_GATE_UP
unset VASR_METAL_PREPARED_MROPE
unset VASR_METAL_PREFILL_ROPE_KERNEL
unset VASR_METAL_DENSE_PROJECTION_FUSION VASR_METAL_CONV_CROSS_SAMPLE
unset VASR_METAL_CONV_CHUNKS CANDLE_METAL_COMPUTE_PER_BUFFER
export VASR_METAL_SILU_MUL=0 VASR_METAL_RESIDUAL_RMSNORM=0 VASR_METAL_CONV_NATIVE_CHUNKS=0
export VASR_METAL_AUTORELEASEPOOL=0
for round in 1 2 3; do
    # Reverse the order of the middle pair to reduce ordering/thermal bias.
    modes=(control candidate)
    if [[ "$round" == 2 ]]; then modes=(candidate control); fi
    for mode in "${modes[@]}"; do
        prefix="$output/$mode-r$round"
        run_binary=$binary
        if [[ "$experiment" == refactor && "$mode" == control ]]; then run_binary=$baseline_binary; fi
        if [[ "$experiment" == autorelease ]]; then
            # Isolate ownership scopes: all kernels, batches and safety estimates
            # are otherwise identical in the two groups.
            export VASR_QWEN3_ASR_AUDIO_ATTN=grouped-native VASR_METAL_CONV_GEMM_TILE=64x32
            export VASR_METAL_DECODE_GEMM_TILE=adaptive VASR_METAL_PREFILL_GEMM_TILE=adaptive VASR_METAL_ENCODER_BIAS_GELU=1
            export VASR_METAL_MAX_ACTIVE_SEQS=64 VASR_METAL_PAGED_DECODE_BATCH_SIZE=64
            export VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS=16 VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS=120 VASR_PREFILL_MAX_WAVE=16
            if [[ "$mode" == control ]]; then export VASR_METAL_AUTORELEASEPOOL=0; else export VASR_METAL_AUTORELEASEPOOL=1; fi
        elif [[ "$experiment" == prefill-gate-up || "$experiment" == prefill-wave || "$experiment" == prefill-rope || "$experiment" == prefill-rope-kernel || "$experiment" == rope-automatic || "$experiment" == refactor ]] || [[ "$experiment" == automatic && "$mode" == candidate ]]; then
            # Exercise the actual startup defaults. No capacity, batching or
            # operator knob is supplied to this group.
            unset VASR_QWEN3_ASR_AUDIO_ATTN VASR_METAL_CONV_GEMM_TILE
            unset VASR_METAL_DECODE_GEMM_TILE VASR_METAL_PREFILL_GEMM_TILE VASR_METAL_ENCODER_BIAS_GELU
            unset VASR_METAL_MAX_ACTIVE_SEQS VASR_METAL_PAGED_DECODE_BATCH_SIZE VASR_PREFILL_MAX_WAVE
            unset VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS
            unset VASR_METAL_CONV_NHWC VASR_METAL_CONV_IM2COL32 VASR_METAL_CONV_BIAS_GELU
            unset VASR_METAL_PACKED_PREFILL VASR_METAL_PACKED_PAGED_ATTN VASR_METAL_AUTORELEASEPOOL
            unset VASR_METAL_SILU_MUL VASR_METAL_RESIDUAL_RMSNORM VASR_METAL_CONV_NATIVE_CHUNKS
            if [[ "$experiment" == prefill-gate-up ]]; then
                if [[ "$mode" == control ]]; then export VASR_METAL_PREFILL_GATE_UP=0; else export VASR_METAL_PREFILL_GATE_UP=1; fi
            elif [[ "$experiment" == prefill-wave ]]; then
                if [[ "$mode" == control ]]; then export VASR_PREFILL_MAX_WAVE=16; else export VASR_PREFILL_MAX_WAVE=32; fi
            elif [[ "$experiment" == prefill-rope ]]; then
                if [[ "$mode" == control ]]; then export VASR_METAL_PREPARED_MROPE=0; else export VASR_METAL_PREPARED_MROPE=1; fi
            elif [[ "$experiment" == prefill-rope-kernel ]]; then
                if [[ "$mode" == control ]]; then
                    export VASR_METAL_PREFILL_ROPE_KERNEL=0 VASR_METAL_PREPARED_MROPE=0
                else
                    export VASR_METAL_PREFILL_ROPE_KERNEL=1
                    unset VASR_METAL_PREPARED_MROPE
                fi
            elif [[ "$experiment" == rope-automatic ]]; then
                if [[ "$mode" == control ]]; then
                    export VASR_METAL_PREFILL_ROPE_KERNEL=0 VASR_METAL_PREPARED_MROPE=0
                else
                    unset VASR_METAL_PREFILL_ROPE_KERNEL VASR_METAL_PREPARED_MROPE
                fi
            fi
        elif [[ "$mode" == control ]]; then
            export VASR_METAL_AUTORELEASEPOOL=0
            export VASR_QWEN3_ASR_AUDIO_ATTN=reference VASR_METAL_CONV_GEMM_TILE=0
            export VASR_METAL_DECODE_GEMM_TILE=0 VASR_METAL_PREFILL_GEMM_TILE=0 VASR_METAL_ENCODER_BIAS_GELU=0
            export VASR_METAL_MAX_ACTIVE_SEQS=32 VASR_METAL_PAGED_DECODE_BATCH_SIZE=32
            export VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS=8 VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS=120 VASR_PREFILL_MAX_WAVE=8
        else
            export VASR_QWEN3_ASR_AUDIO_ATTN=grouped-native VASR_METAL_CONV_GEMM_TILE=64x32
            export VASR_METAL_DECODE_GEMM_TILE=adaptive VASR_METAL_PREFILL_GEMM_TILE=adaptive VASR_METAL_ENCODER_BIAS_GELU=1
            export VASR_METAL_MAX_ACTIVE_SEQS=64 VASR_METAL_PAGED_DECODE_BATCH_SIZE=64
            export VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS=32 VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS=120 VASR_PREFILL_MAX_WAVE=32
        fi
        python3 - <<'PYPORT'
import socket
with socket.socket() as server:
    server.bind(('0.0.0.0', 8868))
PYPORT
        "$run_binary" serve --model Qwen3-ASR-0.6B --port 8868 > "$prefix-server.log" 2>&1 &
        server_pid=$!
        ready=0
        for attempt in {1..90}; do
            if kill -0 "$server_pid" 2>/dev/null && rg -q "ready \| http://.*:8868" "$prefix-server.log" && curl -fsS http://127.0.0.1:8868/health >/dev/null 2>&1; then ready=1; break; fi
            if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 30 "$prefix-server.log"; exit 1; fi
            sleep 1
        done
        if [[ "$ready" != 1 ]]; then echo "server did not become ready" >&2; exit 1; fi
        (
            while kill -0 "$server_pid" 2>/dev/null; do
                python3 - "$server_pid" <<'PY'
import json,subprocess,sys,time
def read(command):
    result=subprocess.run(command,capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else result.stderr.strip()
print(json.dumps({'unix_sec':time.time(),'rss_kib':read(['ps','-o','rss=','-p',sys.argv[1]]),'swap':read(['sysctl','vm.swapusage'])}),flush=True)
PY
                sleep 5
            done
        ) > "$prefix-memory.jsonl" &
        monitor_pid=$!
        "$run_binary" perf -i datasets/call-100.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 28 --batch-size 4 --retries 1 --cer -o "$prefix-warmup.json" > "$prefix-warmup.log" 2>&1
        python3 - "$prefix-warmup.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));s=d['summary']
assert s['items']==s['quality_passed_items']==99 and s['failed_requests']==s['bad_items']==0, 'invalid warmup'
assert d['cer']['summary']['quality_passed_items']==99
PY
        "$run_binary" perf -i datasets/call-100.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 28 --batch-size 4 --retries 1 --cer -o "$prefix-perf.json" > "$prefix-perf.log" 2>&1
        python3 - "$prefix-perf.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));s=d['summary']
assert s['items']==s['quality_passed_items']==99 and s['failed_requests']==s['bad_items']==0
c=d['cer']['summary']
assert c['quality_passed_items']==99
print(sys.argv[1],s['quality_audio_seconds_per_second'],c['cer_weighted'],flush=True)
PY
        "$run_binary" eval -i datasets/example1.db --server http://127.0.0.1:8868/transcribe --retries 1 -o "$prefix-example1.json" > "$prefix-example1.log" 2>&1
        python3 - "$prefix-example1.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['summary']['cer_weighted']==0
PY
        cleanup
    done
done
