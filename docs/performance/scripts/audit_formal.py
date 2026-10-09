#!/usr/bin/env python3
"""Audit complete controls and candidates, without excluding samples."""
import argparse
import json
import pathlib
import statistics
from decimal import Decimal


def ordered_inputs(report):
    requests = sorted(report['requests'], key=lambda request: request['request_index'])
    expected_requests = (99 + report['summary']['batch_size'] - 1) // report['summary']['batch_size']
    assert [request['request_index'] for request in requests] == list(range(expected_requests))
    for request in requests:
        assert request['ok'] and request['error'] is None
        assert request['batch_size'] == len(request['inputs']) == len(request['data'])
        assert request['quality_passed_items'] == request['batch_size'] and request['bad_items'] == 0
    return [request['inputs'] for request in requests]


def checked_report(path):
    report = json.loads(path.read_text())
    summary = report['summary']
    items = report['cer']['items']
    assert summary['items'] == summary['quality_passed_items'] == len(items) == 99, path
    assert summary['failed_requests'] == summary['bad_items'] == 0, path
    assert report['cer']['summary']['quality_passed_items'] == 99, path
    assert sorted(item['index'] for item in items) == list(range(99)), path
    assert sum(len(request['inputs']) for request in report['requests']) == 99, path
    ordered_inputs(report)
    assert not any(item['is_bad'] or item['quality_anomaly'] for item in items), path
    by_index = {item['index']: item for item in items}
    responses = [item for request in sorted(report['requests'], key=lambda request: request['request_index'])
                 for item in request['data']]
    assert all(item['text']['mono'] == by_index[index]['hypothesis']
               for index, item in enumerate(responses)), (path, 'response to CER mapping mismatch')
    # The service accumulates decoded f32 durations in completion order, then
    # perf rounds to four decimals. Allow only sub-millisecond rounding drift.
    assert abs(summary['audio_seconds'] - 5025.0118) < 0.001, path
    return report, {item['index']: item for item in items}


def edits(item):
    return item['substitutions'] + item['insertions'] + item['deletions']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=pathlib.Path)
    args = parser.parse_args()
    folder = args.directory
    refactor = (folder / 'experiment.txt').read_text().strip() == 'refactor' if (folder / 'experiment.txt').exists() else False
    first, baseline = checked_report(folder / 'control-r1-perf.json')
    reference_chars = sum(len(item['reference_normalized']) for item in baseline.values())
    reference_edits = sum(edits(item) for item in baseline.values())
    baseline_inputs = ordered_inputs(first)
    rows = []
    service_ids = set()
    baseline_example = None
    for mode in ['control', 'candidate']:
        for round_number in [1, 2, 3]:
            prefix = folder / f'{mode}-r{round_number}'
            report, items = checked_report(pathlib.Path(f'{prefix}-perf.json'))
            ids = {item['serviceId'] for request in report['requests'] for item in request['data']}
            assert len(ids) == 1 and not ids & service_ids, (prefix, 'service was not restarted')
            service_ids.update(ids)
            for field in ['concurrency', 'batch_size']:
                assert report['summary'][field] == first['summary'][field], (prefix, field)
            # Sort by request identity; completion order may legitimately differ.
            inputs = ordered_inputs(report)
            if refactor:
                assert report['summary']['concurrency'] == 28 and report['summary']['batch_size'] == 4, (prefix, 'client load must be 28/4')
                assert all(audio.get('mono') is True for request_inputs in inputs for audio in request_inputs), (prefix, 'mono must be true')
            assert inputs == baseline_inputs, (prefix, 'audio sources or order changed')
            raw_changed = normalized_changed = 0
            for index, item in items.items():
                control = baseline[index]
                for field in ['audio_id', 'url', 'reference', 'reference_normalized']:
                    assert item[field] == control[field], (prefix, index, field)
                raw_changed += item['hypothesis'] != control['hypothesis']
                normalized_changed += item['hypothesis_normalized'] != control['hypothesis_normalized']
            if refactor:
                assert raw_changed == normalized_changed == 0, (prefix, 'refactor changed transcript')
                warmup, warmup_items = checked_report(pathlib.Path(f'{prefix}-warmup.json'))
                assert ordered_inputs(warmup) == baseline_inputs, (prefix, 'warmup input mapping changed')
                assert warmup['summary']['concurrency'] == 28 and warmup['summary']['batch_size'] == 4, (prefix, 'warmup client load must be 28/4')
                for index, item in warmup_items.items():
                    for field in ['audio_id', 'url', 'reference', 'reference_normalized', 'hypothesis', 'hypothesis_normalized']:
                        assert item[field] == baseline[index][field], (prefix, 'warmup', index, field)
            candidate_edits = sum(edits(item) for item in items.values())
            assert candidate_edits <= reference_edits, (prefix, 'weighted CER increased')
            example = json.loads(pathlib.Path(f'{prefix}-example1.json').read_text())
            assert example['summary']['cer_weighted'] == 0, prefix
            if refactor:
                assert example['summary']['quality_passed_items'] == len(example['items']) == 1, prefix
                item = example['items'][0]
                assert item['index'] == 0 and item['audio_id'] == 'example1', prefix
                assert item['reference_normalized'] == item['hypothesis_normalized'] == '我要修改商品帖子的价格', prefix
                assert not item['is_bad'] and not item['quality_anomaly'], prefix
                if baseline_example is None:
                    baseline_example = item['hypothesis']
                assert item['hypothesis'] == baseline_example, (prefix, 'example1 raw text changed')
            samples = [json.loads(line) for line in pathlib.Path(f'{prefix}-memory.jsonl').read_text().splitlines()]
            rss = [int(sample['rss_kib']) for sample in samples if sample['rss_kib'].isdigit()]
            rows.append({'mode': mode, 'round': round_number,
                         'throughput_x': report['summary']['quality_audio_seconds_per_second'],
                         'wall_seconds': report['summary']['wall_seconds'],
                         'weighted_cer': candidate_edits / reference_chars,
                         'raw_text_differences': raw_changed,
                         'normalized_text_differences': normalized_changed,
                         'peak_rss_gib': max(rss) / 1024**2 if rss else None,
                         'swap_samples': sorted(set(sample['swap'] for sample in samples))})
    result = {'rounds': rows, 'baseline_weighted_cer': reference_edits / reference_chars}
    for mode in ['control', 'candidate']:
        speeds = [row['throughput_x'] for row in rows if row['mode'] == mode]
        result[mode] = {'median_x': statistics.median(speeds), 'min_x': min(speeds), 'max_x': max(speeds)}
    result['median_gain_percent'] = (result['candidate']['median_x'] / result['control']['median_x'] - 1) * 100
    result['target_above_100x'] = result['candidate']['median_x'] > 100
    if refactor:
        # Compare the reported decimal medians directly; binary percentage arithmetic
        # can reject exactly 3 percent (for example 194 / 200).
        within_regression_limit = Decimal(str(result['candidate']['median_x'])) >= Decimal(str(result['control']['median_x'])) * Decimal('0.97')
        result['refactor_acceptance_passed'] = result['target_above_100x'] and within_regression_limit
    (folder / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if refactor:
        assert result['refactor_acceptance_passed'], 'refactor throughput must stay above 100x and within 3 percent of contemporary baseline'


if __name__ == '__main__':
    main()
