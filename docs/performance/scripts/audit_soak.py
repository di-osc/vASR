#!/usr/bin/env python3
"""Audit every soak response; validate Bench normalization against Rust perf artifacts."""
import argparse
import collections
import functools
import json
import pathlib
import math
import re
import statistics
import unicodedata

DIGITS = {c: n for n, chars in enumerate(['零〇', '一壹', '二两贰', '三叁', '四肆', '五伍', '六陆', '七柒', '八捌', '九玖']) for c in chars}
SMALL = {'十': 10, '百': 100, '千': 1000}
LARGE = {'万': 10000, '亿': 100000000}


def normalize(text):
    chars = ''.join(' ' if c == '\u3000' else chr(ord(c) - 0xfee0) if '\uff01' <= c <= '\uff5e' else c for c in text).lower()
    result = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isspace() or unicodedata.category(c).startswith('P'):
            i += 1
            continue
        end = i
        while end < len(chars) and chars[end] in DIGITS | SMALL | LARGE:
            end += 1
        if end > i:
            group = chars[i:end]
            if all(c in DIGITS for c in group):
                result.append(''.join(str(DIGITS[c]) for c in group))
            else:
                total = section = number = 0
                for c in group:
                    if c in DIGITS:
                        number = DIGITS[c]
                    elif c in SMALL:
                        section += max(number, 1) * SMALL[c]
                        number = 0
                    else:
                        section += number
                        total += max(section, 1) * LARGE[c]
                        section = number = 0
                result.append(str(total + section + number))
            i = end
            continue
        def cjk(ch):
            return '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf'
        if c == '儿' and i > 0 and cjk(chars[i-1]) and (i+1 == len(chars) or not cjk(chars[i+1]) or chars[i+1] in '啊呀呢吧吗嘛啦了的'):
            i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)


@functools.lru_cache(maxsize=None)
def distance(a, b):
    # Myers bit-vector Levenshtein. Python integers cover arbitrary string lengths.
    if not a:
        return len(b)
    matches = {}
    for i, c in enumerate(a):
        matches[c] = matches.get(c, 0) | (1 << i)
    positive, negative, score, top = ~0, 0, len(a), 1 << (len(a)-1)
    for c in b:
        equal = matches.get(c, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        score += bool(ph & top) - bool(mh & top)
        ph = (ph << 1) | 1
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    return score


def repeated_pattern(text):
    """Same boolean guard as vasr-transcribe/src/quality.rs, before normalization."""
    chars = ''.join(c for c in text if not c.isspace())
    for width in range(2, min(12, len(chars) // 5) + 1):
        for start in range(len(chars) - width * 5 + 1):
            pattern = chars[start:start + width]
            if len(set(pattern)) == 1:
                continue
            repetitions = 1
            next_start = start + width
            while chars[next_start:next_start + width] == pattern:
                repetitions += 1
                next_start += width
            covered = repetitions * width
            if repetitions >= 5 and covered >= 24 and covered * 4 >= len(chars):
                return True
    return False


def quality_anomaly(raw_text, edits, reference_chars):
    return not normalize(raw_text) or edits >= reference_chars or repeated_pattern(raw_text)


def check_metadata_evidence(server_log):
    """Require direct observations before claiming metadata was reclaimed."""
    fields = {
        'producer': ['producers', 'ranges', 'waiting_chunks'],
        'prefill_buffer': ['prepared', 'waiting', 'audio_tokens'],
        'prefill_log': ['items'],
        'decoder': ['batches', 'jobs', 'pending_chunks'],
        'router': ['requests', 'routes'],
    }
    observations = collections.defaultdict(list)
    for match in re.finditer(r'metadata_lifetime \| stage=(\w+) ([^\r\n]*)', server_log or ''):
        stage = match.group(1)
        if stage not in fields:
            continue
        values = {key: int(value) for key, value in re.findall(r'\b(\w+)=(\d+)\b', match.group(2))}
        assert all(key in values for key in fields[stage]), (stage, 'incomplete metadata snapshot')
        observations[stage].append({key: values[key] for key in fields[stage]})
    assert set(observations) == set(fields), 'missing direct metadata evidence'
    last = {stage: rows[-1] for stage, rows in observations.items()}
    peaks = {stage: {key: max(row[key] for row in rows) for key in fields[stage]}
             for stage, rows in observations.items()}
    # The serial producer retains one current ID/counter while idle. Completed
    # ranges and all per-item metadata must be empty; IDs themselves are scalars.
    assert peaks['producer']['producers'] <= 1, 'old producer IDs accumulated'
    for stage, snapshot in last.items():
        assert all(value == 0 for key, value in snapshot.items()
                   if not (stage == 'producer' and key == 'producers')), (stage, 'metadata not drained')
    return {'drained': True, 'last': last, 'peaks': peaks,
            'samples': {stage: len(rows) for stage, rows in observations.items()},
            'scope': 'direct live-entry counts at stage transitions; not allocator byte accounting'}


def check_run_evidence(rows, summary, memories, server_log=None):
    """Quality alone cannot certify duration, continuous probes or drained KV."""
    duration = summary['duration_sec']
    assert duration >= 1800, 'soak must request at least 30 minutes'
    assert summary['drained_wall_sec'] >= duration, 'run stopped before the deadline'
    assert summary['failed_requests'] == 0 and not summary.get('errors'), 'request failure'
    assert summary['successful_requests'] == len(rows) and all(row['ok'] for row in rows)
    regular = [row for row in rows if not row['short']]
    short = sorted((row for row in rows if row['short']), key=lambda row: row['started_sec'])
    assert regular and max(row['started_sec'] for row in regular) >= duration - 180, 'load stopped early'
    assert all(0 <= row['started_sec'] < duration and row['started_sec'] <= row['completed_sec']
               <= summary['drained_wall_sec'] + .1 for row in rows), 'invalid request interval'
    assert len(short) == summary['short_probes'] and len(short) >= math.ceil(duration / 120) - 1, 'missing sustained short probes'
    assert short[0]['started_sec'] <= 90 and short[-1]['started_sec'] >= duration - 150, 'short probes do not cover the run'
    assert all(b['started_sec'] - a['started_sec'] <= 150 for a, b in zip(short, short[1:])), 'short probes stalled'
    assert all(any(row['started_sec'] < probe['completed_sec'] and row['completed_sec'] > probe['started_sec']
                   for row in regular) for probe in short), 'short probe had no overlapping load'
    assert memories and memories[0]['elapsed_sec'] <= 20
    assert memories[-1]['elapsed_sec'] >= summary['drained_wall_sec'] + 50, 'missing drained observation'
    assert all(0 < b['elapsed_sec'] - a['elapsed_sec'] <= 30 for a, b in zip(memories, memories[1:])), 'memory sampling gap'
    rss = [int(row['rss_kib']) for row in memories]
    windows = [[int(row['rss_kib']) for row in memories if start <= row['elapsed_sec'] < start + 300]
               for start in range(300, int(duration), 300)]
    medians = [statistics.median(window) for window in windows if window]
    swaps = [float(re.search(r'used\s*=\s*([\d.]+)M', row['swap']).group(1)) for row in memories]
    kv_recovered = None
    if server_log is not None:
        snapshots = re.findall(r'kv_free=(\d+)/(\d+) usable', server_log)
        assert snapshots, 'missing KV evidence'
        kv_recovered = snapshots[-1][0] == snapshots[-1][1]
        assert kv_recovered, 'KV blocks not fully recovered after drain'
    return {'observed_duration_sec': duration, 'observed_last_regular_start_sec': max(row['started_sec'] for row in regular),
            'short_probes_overlap_load': True, 'drained_observation_sec': memories[-1]['elapsed_sec'] - summary['drained_wall_sec'],
            'sampled_peak_rss_kib': max(rss), 'steady_five_minute_rss_medians_kib': medians,
            'swap_start_mb': swaps[0], 'swap_end_mb': swaps[-1], 'swap_peak_mb': max(swaps),
            'kv_fully_recovered': kv_recovered,
            'producer_metadata_growth': 'not instrumented; no claim of a complete metadata lifetime audit'}


def check_sources(rows, source_report, short_report, refs):
    from m4_soak import replay_inputs
    inputs = replay_inputs(source_report)
    source_items = {item['index']: item for item in source_report['cer']['items']}
    assert len(inputs) == len(refs) == len(source_items) == 99
    for index, item in source_items.items():
        assert (item['url'], item['reference_normalized']) == (refs[index]['url'], refs[index]['reference_normalized'])
    short_inputs = replay_inputs(short_report)
    assert len(short_inputs) == 1
    service_ids = {item['serviceId'] for report in [source_report, short_report]
                   for request in report['requests'] for item in request['data']}
    assert len(service_ids) == 1 and next(iter(service_ids)), 'warmup service identity mismatch'
    for row in rows:
        expected = short_inputs if row['short'] else [inputs[index] for index in row['indices']]
        assert row['inputs'] == expected, 'audio DTO, mono/hotword or revision mismatch'
        assert len(row['response']['data']) == len(expected)
        assert {item['serviceId'] for item in row['response']['data']} == service_ids, 'service identity changed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--validation-dir', required=True)
    parser.add_argument('--soak-dir', required=True)
    parser.add_argument('--cer-tolerance', type=float, default=0.0)
    parser.add_argument('--source-perf', help='the replayed 99-item perf --cer artifact')
    parser.add_argument('--short-perf', help='the replayed example1 perf --cer artifact')
    parser.add_argument('--server-log', help='request KV snapshots including drained completion')
    parser.add_argument('--metadata-policy', choices=['optional', 'require'], default='optional',
                        help='require direct producer, decoder and router lifetime counters')
    args = parser.parse_args()
    validated = 0
    for path in pathlib.Path(args.validation_dir).glob('*.json'):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        quality = data.get('cer', data)
        if not isinstance(quality, dict) or not isinstance(quality.get('items'), list):
            continue
        for item in quality['items']:
            if not isinstance(item, dict) or 'reference_normalized' not in item:
                continue
            assert normalize(item['reference']) == item['reference_normalized'], (str(path), item['index'], 'reference')
            assert normalize(item['hypothesis']) == item['hypothesis_normalized'], (str(path), item['index'], 'hypothesis')
            assert distance(item['reference_normalized'], item['hypothesis_normalized']) == item['substitutions'] + item['insertions'] + item['deletions'], (str(path), item['index'], 'distance')
            if 'quality_anomaly' in item and not item.get('is_bad'):
                assert bool(item['quality_anomaly']) == quality_anomaly(item['hypothesis'], distance(item['reference_normalized'], item['hypothesis_normalized']), len(item['reference_normalized'])), (str(path), item['index'], 'quality guard')
            validated += 1
    assert validated >= 99
    base = json.loads(pathlib.Path(args.baseline).read_text())
    refs = {item['index']: item for item in base['cer']['items']}
    folder = pathlib.Path(args.soak_dir)
    counts = collections.Counter()
    errors = []
    total_chars = total_edits = baseline_edits = 0
    differences = 0
    raw_differences = 0
    cer_sum = baseline_cer_sum = 0.0
    short_count = 0
    short_latencies = []
    per_item = collections.defaultdict(lambda: {'runs': 0, 'edits': 0, 'reference_chars': 0, 'min_cer': 10, 'max_cer': 0})
    rows = [json.loads(line) for line in (folder / 'responses.jsonl').read_text().splitlines()]
    for row in rows:
        if not row['ok']:
            errors.append(row.get('error', 'request failed'))
            continue
        data = row['response']['data']
        indices = row.get('indices', [row.get('index')])
        urls = row.get('urls', [row.get('url')])
        if len(data) != len(indices) or len(urls) != len(indices) or any(item['is_bad'] for item in data):
            errors.append('bad or missing response')
            continue
        hypothesis = normalize(data[0]['text']['mono'])
        if row['short']:
            short_count += 1
            short_latencies.append(row['latency_sec'])
            if hypothesis != normalize('我要修改商品帖子的价格'):
                errors.append('short reference mismatch')
            continue
        for index, url, response in zip(indices, urls, data):
            hypothesis = normalize(response['text']['mono'])
            item = refs[index]
            assert url == item['url']
            counts[index] += 1
            edits = distance(item['reference_normalized'], hypothesis)
            n = len(item['reference_normalized'])
            total_chars += n
            total_edits += edits
            baseline_edits += item['substitutions'] + item['insertions'] + item['deletions']
            differences += hypothesis != item['hypothesis_normalized']
            raw_differences += response['text']['mono'] != item['hypothesis']
            cer_sum += edits / n
            baseline_cer_sum += (item['substitutions'] + item['insertions'] + item['deletions']) / n
            if quality_anomaly(response['text']['mono'], edits, n):
                errors.append(f"empty, repeated pattern or CER at least 100%, index={index}")
            entry = per_item[index]
            entry['runs'] += 1
            entry['edits'] += edits
            entry['reference_chars'] += n
            entry['min_cer'] = min(entry['min_cer'], edits/n)
            entry['max_cer'] = max(entry['max_cer'], edits/n)
    summary = json.loads((folder / 'summary.json').read_text())
    assert sum(counts.values()) == summary['scheduled'], 'missing completion'
    assert len(counts) == 99, 'not all dataset items covered'
    memories = [json.loads(line) for line in (folder / 'memory.jsonl').read_text().splitlines()]
    server_log = pathlib.Path(args.server_log).read_text() if args.server_log else None
    evidence = check_run_evidence(rows, summary, memories, server_log)
    if args.metadata_policy == 'require' or (server_log and 'metadata_lifetime |' in server_log):
        evidence['metadata_lifetime'] = check_metadata_evidence(server_log)
        evidence['producer_metadata_growth'] = 'direct live-entry counts bounded and drained in this run; allocator bytes not measured'
    assert bool(args.source_perf) == bool(args.short_perf), 'provide both replay source artifacts'
    if args.source_perf:
        check_sources(rows, json.loads(pathlib.Path(args.source_perf).read_text()),
                      json.loads(pathlib.Path(args.short_perf).read_text()), refs)
    result = {'validated_rust_items': validated, 'audited_responses': sum(counts.values()), 'short_probes': short_count, 'short_max_latency_sec': max(short_latencies, default=None), 'sampled_weighted_cer': total_edits/total_chars, 'same_input_multiplicity_baseline_cer': baseline_edits/total_chars, 'normalized_transcripts_different_from_baseline': differences, 'min_repeats_per_item': min(counts.values()), 'max_repeats_per_item': max(counts.values()), 'errors': errors, 'per_item': dict(per_item)}
    result['run_evidence'] = evidence
    result['audio_dto_identity_verified'] = bool(args.source_perf)
    result['average_cer'] = cer_sum / sum(counts.values())
    result['same_input_multiplicity_baseline_average_cer'] = baseline_cer_sum / sum(counts.values())
    result['raw_transcripts_different_from_baseline'] = raw_differences
    (folder / 'quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'per_item'}, ensure_ascii=False, indent=2))
    assert not errors
    assert total_edits/total_chars <= baseline_edits/total_chars + args.cer_tolerance, 'CER exceeds the same-input baseline tolerance'


if __name__ == '__main__':
    main()
