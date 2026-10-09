#!/bin/bash
# Continuous validation of explicit or actual automatic BF16 execution.
# Run from any directory. Owns only its server and temporary power assertion.
set -euo pipefail
cd "$(dirname "$0")/../../.."
binary=${1:?frozen binary path required}
output=${2:?new output directory required}
baseline=${3:?correct perf --cer baseline required}
autorelease=${4:-0}
metadata_policy=${5:-optional}
execution_mode=${6:-explicit}
if [[ "$autorelease" != 0 && "$autorelease" != 1 ]]; then echo "autorelease must be 0 or 1" >&2; exit 1; fi
if [[ "$metadata_policy" != optional && "$metadata_policy" != require ]]; then echo "unknown metadata policy" >&2; exit 1; fi
if [[ "$execution_mode" != explicit && "$execution_mode" != automatic ]]; then echo "unknown execution mode" >&2; exit 1; fi
[[ ! -e "$output" ]] || { echo "output directory already exists" >&2; exit 1; }
[[ ! -e .env ]] || { echo ".env present; refuse an uncontrolled environment" >&2; exit 1; }
mkdir -p "$output"
binary=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$binary")
output=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$output")
printf 'autorelease=%s\nmetadata_policy=%s\nexecution_mode=%s\n' "$autorelease" "$metadata_policy" "$execution_mode" > "$output/experiment.txt"
server_pid=
awake_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    if [[ -n "$awake_pid" ]]; then kill "$awake_pid" 2>/dev/null || true; wait "$awake_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
python3 - <<'PYPORT'
import socket
with socket.socket() as server:
    server.bind(('0.0.0.0', 8868))
PYPORT
caffeinate -d -i -u -t 3600 &
awake_pid=$!
pmset -g batt > "$output/power.txt"
shasum -a 256 "$binary" datasets/call-100.db datasets/example1.db > "$output/sha256.txt"
while IFS= read -r inherited_option; do
    if [[ -n "$inherited_option" ]]; then unset "$inherited_option"; fi
done < <(python3 -c 'import os;print("\n".join(sorted(k for k in os.environ if k.startswith("VASR_") or k == "CANDLE_METAL_COMPUTE_PER_BUFFER")))')
unset VASR_METAL_STAGE_PROFILE VASR_METAL_STAGE_SYNC VASR_ISQ VASR_MAX_NUM_SEQS
unset VASR_DISABLE_MIXED_STEP_DECODE VASR_METAL_GEMM_TRACE CANDLE_METAL_COMPUTE_PER_BUFFER
unset VASR_METAL_DENSE_PROJECTION_FUSION VASR_METAL_CONV_CROSS_SAMPLE VASR_METAL_CONV_CHUNKS
export VASR_PROFILE=false VASR_LOG=vasr_transcribe=info,vasr::execution=info
export VASR_METAL_SILU_MUL=0 VASR_METAL_RESIDUAL_RMSNORM=0 VASR_METAL_CONV_NATIVE_CHUNKS=0
export VASR_METAL_AUTORELEASEPOOL="$autorelease"
export VASR_QWEN3_ASR_AUDIO_ATTN=grouped-native VASR_METAL_CONV_GEMM_TILE=64x32
export VASR_METAL_DECODE_GEMM_TILE=adaptive VASR_METAL_PREFILL_GEMM_TILE=adaptive VASR_METAL_ENCODER_BIAS_GELU=1
export VASR_METAL_MAX_ACTIVE_SEQS=64 VASR_METAL_PAGED_DECODE_BATCH_SIZE=64
export VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS=16 VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS=120 VASR_PREFILL_MAX_WAVE=16
if [[ "$execution_mode" == automatic ]]; then
    unset VASR_METAL_MAX_ACTIVE_SEQS VASR_METAL_PAGED_DECODE_BATCH_SIZE VASR_PREFILL_MAX_WAVE
    unset VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS
    unset VASR_METAL_AUTORELEASEPOOL VASR_QWEN3_ASR_AUDIO_ATTN VASR_METAL_CONV_GEMM_TILE
    unset VASR_METAL_DECODE_GEMM_TILE VASR_METAL_PREFILL_GEMM_TILE VASR_METAL_ENCODER_BIAS_GELU
    unset VASR_METAL_SILU_MUL VASR_METAL_RESIDUAL_RMSNORM VASR_METAL_CONV_NATIVE_CHUNKS
fi
# KV snapshots are for lifetime validation; this is not a formal throughput run.
export VASR_DEBUG_REQUEST_MEM=1
"$binary" serve --model Qwen3-ASR-0.6B --port 8868 > "$output/server.log" 2>&1 &
server_pid=$!
ready=0
for attempt in {1..90}; do
    if kill -0 "$server_pid" 2>/dev/null && rg -q "server ready .*:8868" "$output/server.log" && curl -fsS http://127.0.0.1:8868/health >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 30 "$output/server.log"; exit 1; fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then echo "server did not become ready" >&2; exit 1; fi
"$binary" perf -i datasets/call-100.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 28 --batch-size 4 --retries 1 --cer -o "$output/warmup.json" > "$output/warmup.log" 2>&1
"$binary" perf -i datasets/example1.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 1 --batch-size 1 --retries 1 --cer -o "$output/short.json" > "$output/short.log" 2>&1
python3 - "$output/warmup.json" "$output/short.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));s=d['summary']
assert s['items']==s['quality_passed_items']==99 and s['failed_requests']==s['bad_items']==0, 'invalid warmup'
assert d['cer']['summary']['quality_passed_items']==99
d=json.load(open(sys.argv[2]));s=d['summary']
assert s['items']==s['quality_passed_items']==1 and s['failed_requests']==s['bad_items']==0
assert d['cer']['summary']['cer_weighted']==0, 'invalid short probe reference'
PY
python3 docs/performance/scripts/m4_soak.py --reference-perf "$output/warmup.json" --short-perf "$output/short.json" --output-dir "$output/soak" --server-pid "$server_pid" --duration-sec 1800 --concurrency 28 --batch-size 4 > "$output/progress.log" 2>&1
python3 docs/performance/scripts/audit_soak.py --baseline "$baseline" --validation-dir "$output" --soak-dir "$output/soak" --source-perf "$output/warmup.json" --short-perf "$output/short.json" --server-log "$output/server.log" --metadata-policy "$metadata_policy" > "$output/audit.log" 2>&1
cat "$output/audit.log"
