#!/usr/bin/env python3
"""Continuous HTTP soak; saves every response for an offline quality audit.

Run from repository root with a warmed service. Input is a validated perf --cer
JSON, used only for its ordered URLs and references, never to answer requests.
All requests are executed against the service, with no retries or result cache.
"""
import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-perf', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8868/transcribe')
    parser.add_argument('--duration-sec', type=float, default=1800)
    parser.add_argument('--concurrency', type=int, default=14)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--short-url', default='datasets/example1/assets/example1.mp3')
    args = parser.parse_args()
    assert args.duration_sec > 0 and args.concurrency > 0
    reference = json.loads(pathlib.Path(args.reference_perf).read_text())
    items = sorted(reference['cer']['items'], key=lambda item: item['index'])
    assert len(items) == reference['items'] and all(not item['is_bad'] for item in items)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stopped = threading.Event()
    counter = 0
    started = time.monotonic()
    deadline = started + args.duration_sec
    successes = 0
    errors = []
    audio_seconds = 0.0
    short_latencies = []
    responses = (output / 'responses.jsonl').open('w')
    memories = (output / 'memory.jsonl').open('w')

    def request(index, url, short=False):
        nonlocal successes, audio_seconds
        begin = time.monotonic()
        row = {'index': index, 'url': url, 'short': short, 'started_sec': begin - started}
        try:
            body = json.dumps({'inputs': [{'url': url, 'mono': True}]}).encode()
            req = urllib.request.Request(args.server, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.load(response)
            row['response'] = data
            assert len(data['data']) == 1, 'response count mismatch'
            item = data['data'][0]
            assert not item['is_bad'], str(item.get('bad_reason'))
            assert item['text']['mono'].strip(), 'unexplained empty transcription'
            if short:
                assert ''.join(c for c in item['text']['mono'] if c not in '，。！？,.!? \n') == '我要修改商品帖子的价格', 'short transcript mismatch'
            row['ok'] = True
        except Exception as error:
            row['ok'] = False
            row['error'] = str(error)
            stopped.set()
        row['latency_sec'] = time.monotonic() - begin
        row['completed_sec'] = time.monotonic() - started
        with lock:
            responses.write(json.dumps(row, ensure_ascii=False) + '\n')
            responses.flush()
            if row['ok']:
                successes += 1
                audio_seconds += data['inferencePerformance']['totalAudioDurationSeconds'] if 'inferencePerformance' in data else data['inference_performance']['total_audio_duration_seconds']
                if short:
                    short_latencies.append(row['latency_sec'])
            else:
                errors.append(row)
        return row['ok']

    def worker():
        nonlocal counter
        while time.monotonic() < deadline and not stopped.is_set():
            with lock:
                item = items[counter % len(items)]
                counter += 1
            if not request(item['index'], item['url']):
                break

    def short_worker():
        next_probe = started + 30
        while next_probe < deadline and not stopped.is_set():
            if stopped.wait(max(0, next_probe - time.monotonic())):
                break
            if not request(-1, args.short_url, True):
                break
            next_probe = max(next_probe + 60, time.monotonic() + 1)

    def monitor():
        while not stopped.is_set():
            row = {'elapsed_sec': time.monotonic() - started}
            for name, command in [('rss_kib', ['ps', '-o', 'rss=', '-p', str(args.server_pid)]), ('swap', ['sysctl', 'vm.swapusage']), ('vm_stat', ['vm_stat'])]:
                result = subprocess.run(command, capture_output=True, text=True)
                row[name] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            with lock:
                row['completed'] = successes
                row['failed'] = len(errors)
            memories.write(json.dumps(row) + '\n')
            memories.flush()
            print(json.dumps({key: row[key] for key in ['elapsed_sec', 'completed', 'failed', 'rss_kib', 'swap']}), flush=True)
            stopped.wait(10)

    monitor_thread = threading.Thread(target=monitor)
    short_thread = threading.Thread(target=short_worker)
    monitor_thread.start()
    short_thread.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda _: worker(), range(args.concurrency)))
    short_thread.join()
    drained = time.monotonic() - started
    # Capture the drained service for one minute, with no new requests.
    time.sleep(60)
    stopped.set()
    monitor_thread.join()
    responses.close()
    memories.close()
    summary = {'duration_sec': args.duration_sec, 'drained_wall_sec': drained, 'scheduled': counter, 'successful': successes, 'failed': len(errors), 'audio_seconds': audio_seconds, 'throughput_including_probes_and_drain': audio_seconds / drained, 'short_probes': len(short_latencies), 'short_max_latency_sec': max(short_latencies, default=None), 'errors': errors}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
