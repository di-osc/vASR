#!/bin/bash
# Full-dataset screening followed by completion timing at whole-forward boundaries.
# Nested probe durations remain host elapsed time; never sum them into the roots.
set -euo pipefail
cd "$(dirname "$0")/../../.."
binary=${1:?frozen binary path required}
output=${2:?new output directory required}
[[ ! -e "$output" ]] || { echo "output directory already exists" >&2; exit 1; }
[[ ! -e .env ]] || { echo ".env present; refuse uncontrolled measurements" >&2; exit 1; }
mkdir -p "$output"
binary=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$binary")
output=$(python3 -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve())' "$output")
server_pid=
monitor_pid=
awake_pid=
cleanup() {
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    monitor_pid=
    server_pid=
}
trap 'cleanup; if [[ -n "$awake_pid" ]]; then kill "$awake_pid" 2>/dev/null || true; wait "$awake_pid" 2>/dev/null || true; fi' EXIT
while IFS= read -r option_name; do
    if [[ -n "$option_name" ]]; then unset "$option_name"; fi
done < <(compgen -e | rg '^VASR_' || true)
unset CANDLE_METAL_COMPUTE_PER_BUFFER
caffeinate -d -i -u -t 1800 &
awake_pid=$!
pmset -g batt > "$output/power.txt"
shasum -a 256 "$binary" datasets/call-100.db datasets/example1.db > "$output/sha256.txt"
for mode in normal forward; do
    prefix="$output/$mode"
    if [[ "$mode" == normal ]]; then
        export VASR_PROFILE=false
        unset VASR_METAL_STAGE_PROFILE VASR_METAL_STAGE_SYNC VASR_DEBUG_LOCK_WAIT VASR_LOG
    else
        export VASR_PROFILE=true VASR_METAL_STAGE_PROFILE=1 VASR_METAL_STAGE_SYNC=forward VASR_DEBUG_LOCK_WAIT=1
        export VASR_LOG='info,vasr::stage_profile=info,vasr_transcribe::pipeline=info'
    fi
    python3 - <<'PY'
import socket
with socket.socket() as server:
    server.bind(('0.0.0.0', 8868))
PY
    "$binary" serve --model Qwen3-ASR-0.6B --port 8868 > "$prefix-server.log" 2>&1 &
    server_pid=$!
    ready=0
    for attempt in {1..90}; do
        if kill -0 "$server_pid" 2>/dev/null && rg -q 'ready \| http://.*:8868' "$prefix-server.log" && curl -fsS http://127.0.0.1:8868/health >/dev/null 2>&1; then ready=1; break; fi
        if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 30 "$prefix-server.log"; exit 1; fi
        sleep 1
    done
    [[ "$ready" == 1 ]] || { echo "server did not become ready" >&2; exit 1; }
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
    "$binary" perf -i datasets/call-100.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 28 --batch-size 4 --retries 1 --cer -o "$prefix-warmup.json" > "$prefix-warmup.log" 2>&1
    # The reporter publishes cumulative snapshots every five seconds. Capture
    # quiescent boundaries so the analysis subtracts the complete warmup.
    if [[ "$mode" == forward ]]; then sleep 6; fi
    wc -l < "$prefix-server.log" > "$prefix-warmup-line.txt"
    "$binary" perf -i datasets/call-100.db --server http://127.0.0.1:8868/transcribe --mono true --concurrency 28 --batch-size 4 --retries 1 --cer -o "$prefix-perf.json" > "$prefix-perf.log" 2>&1
    if [[ "$mode" == forward ]]; then sleep 6; fi
    wc -l < "$prefix-server.log" > "$prefix-end-line.txt"
    "$binary" eval -i datasets/example1.db --server http://127.0.0.1:8868/transcribe --retries 1 -o "$prefix-example1.json" > "$prefix-example1.log" 2>&1
    python3 - "$output" "$mode" <<'PY'
import json,pathlib,sys
sys.path.insert(0,'docs/performance/scripts')
from audit_formal import checked_report
folder=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
baseline, original=checked_report(pathlib.Path('results/m4-pro-100x-2026-10-07/baseline-c14-b1-perf1.json'))
for suffix in ['warmup','perf']:
    report, items=checked_report(folder/f'{mode}-{suffix}.json')
    for index,item in items.items():
        control=original[index]
        for field in ['audio_id','url','reference','reference_normalized','hypothesis','hypothesis_normalized']:
            assert item[field]==control[field], (mode,suffix,index,field)
example=json.loads((folder/f'{mode}-example1.json').read_text())
assert example['summary']['cer_weighted']==0
print(mode,report['summary']['quality_audio_seconds_per_second'],report['cer']['summary']['cer_weighted'],flush=True)
PY
    cleanup
done
