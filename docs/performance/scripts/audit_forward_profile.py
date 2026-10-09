#!/usr/bin/env python3
"""Audit quality and subtract warmup from whole-forward diagnostic counters."""
import argparse
import json
import pathlib
import re

from audit_formal import checked_report, ordered_inputs


def last_snapshot(lines, field):
    pattern = re.compile(r'\b' + field + r'=("(?:\\.|[^"\\])*")')
    matches = [match for line in lines if (match := pattern.search(line))]
    assert matches, f'no {field} snapshot'
    return json.loads(json.loads(matches[-1].group(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('baseline', type=pathlib.Path)
    args = parser.parse_args()
    folder = args.directory
    baseline, original = checked_report(args.baseline)
    result = {'quality': [], 'diagnostic_only': True}
    service_ids = set()
    first_inputs = None
    for mode in ['normal', 'forward']:
        for suffix in ['warmup', 'perf']:
            report, items = checked_report(folder / f'{mode}-{suffix}.json')
            ids = {item['serviceId'] for request in report['requests'] for item in request['data']}
            assert len(ids) == 1
            if suffix == 'perf':
                assert not ids & service_ids
                service_ids.update(ids)
            inputs = ordered_inputs(report)
            if first_inputs is None:
                first_inputs = inputs
            assert inputs == first_inputs, (mode, suffix, 'input order')
            for index, item in items.items():
                for field in ['audio_id', 'url', 'reference', 'reference_normalized', 'hypothesis', 'hypothesis_normalized']:
                    assert item[field] == original[index][field], (mode, suffix, index, field)
            result['quality'].append({'mode': mode, 'phase': suffix,
                                      'throughput_x': report['summary']['quality_audio_seconds_per_second'],
                                      'wall_seconds': report['summary']['wall_seconds'],
                                      'weighted_cer': report['cer']['summary']['cer_weighted'],
                                      'raw_text_differences': 0,
                                      'normalized_text_differences': 0})
        example = json.loads((folder / f'{mode}-example1.json').read_text())
        assert example['summary']['cer_weighted'] == 0
    lines = (folder / 'forward-server.log').read_text().splitlines()
    start_line = int((folder / 'forward-warmup-line.txt').read_text())
    end_line = int((folder / 'forward-end-line.txt').read_text())
    before = lines[:start_line]
    after = lines[:end_line]
    start = last_snapshot(before, 'measurements')
    end = last_snapshot(after, 'measurements')
    stages = {}
    for name, measurement in end.items():
        old = start.get(name, {'calls': 0, 'seconds': 0})
        assert measurement['calls'] >= old['calls'] and measurement['seconds'] >= old['seconds']
        stages[name] = {'calls': measurement['calls'] - old['calls'],
                        'seconds': measurement['seconds'] - old['seconds'],
                        'completion_sync': measurement['completion_sync']}
    roots = ['encoder_forward', 'prefill_forward', 'decode_forward']
    assert all(stages[name]['completion_sync'] is True for name in roots)
    assert all(value['completion_sync'] is False for name, value in stages.items() if name not in roots)
    root_seconds = sum(stages[name]['seconds'] for name in roots)
    for name in roots:
        stages[name]['fraction_of_forward_seconds'] = stages[name]['seconds'] / root_seconds
    start_batches = last_snapshot(before, 'batches')
    end_batches = last_snapshot(after, 'batches')
    batches = {}
    for name, values in end_batches.items():
        old = start_batches.get(name, {})
        batch = {key: values[key] - old.get(key, 0) for key in ['batches', 'rows', 'valid_tokens', 'executed_tokens']}
        batch['max_rows_including_warmup'] = values['max_rows']
        batch['mean_rows'] = batch['rows'] / batch['batches'] if batch['batches'] else None
        batches[name] = batch
    result.update({'start_line': start_line, 'end_line': end_line,
                   'stages': stages, 'batches': batches, 'total_forward_seconds': root_seconds,
                   'timing_semantics': 'Host elapsed including submission and completion waits at roots only. Nested times are host elapsed and overlap roots. Prefill root excludes last-token lm_head; decode root excludes argmax/token readback. Diagnostic throughput is not an acceptance score.'})
    (folder / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
