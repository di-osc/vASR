import unittest
import json
import pathlib
import subprocess
import sys
import tempfile
import copy
from m4_soak import replay_inputs, next_batch
from audit_soak import check_run_evidence, check_sources, check_metadata_evidence, quality_anomaly, repeated_pattern, check_lifecycle_evidence, check_short_text


class ReplayInputsTest(unittest.TestCase):
    def test_batches_wrap_tail_and_keep_source_mapping(self):
        items = [{'index': i} for i in range(5)]
        inputs = [{'url': f'{i}.wav'} for i in range(5)]
        batch, sources = next_batch(items, inputs, 3, 4)
        self.assertEqual([item['index'] for item in batch], [3, 4, 0, 1])
        self.assertEqual(sources, [inputs[i] for i in [3, 4, 0, 1]])
    def test_preserves_modelscope_revision_and_input_order(self):
        first = {'modelscope': {'repo_id': 'owner/audio', 'file_path': 'a.mp3', 'revision': 'abc'}, 'mono': True}
        second = {'url': 'b.wav', 'mono': False, 'hotword': '商品'}
        report = {'items': 2, 'requests': [
            {'request_index': 1, 'inputs': [second]},
            {'request_index': 0, 'inputs': [first]},
        ]}
        self.assertEqual(replay_inputs(report), [first, second])

    def test_rejects_unreplayable_or_missing_inputs(self):
        for report in [
            {'items': 1, 'requests': [{'request_index': 0, 'inputs': [{'mono': True}]}]},
            {'items': 2, 'requests': [{'request_index': 0, 'inputs': [{'url': 'a.wav'}]}]},
            {'items': 1, 'requests': [{'request_index': 0, 'inputs': [{'url': 'data:audio/wav;base64,AQ=='}]}]},
        ]:
            with self.assertRaises(ValueError):
                replay_inputs(report)


class SoakAuditTest(unittest.TestCase):
    def test_metadata_requires_every_stage_and_drained_counts(self):
        lines = [
            'metadata_lifetime | stage=producer producers=1 ranges=0 waiting_chunks=0',
            'metadata_lifetime | stage=prefill_buffer prepared=0 waiting=0 audio_tokens=0',
            'metadata_lifetime | stage=prefill_log items=0',
            'metadata_lifetime | stage=decoder batches=0 jobs=0 pending_chunks=0',
            'metadata_lifetime | stage=router requests=0 routes=0',
        ]
        for invalid in [lines[:-1],
                        lines + ['metadata_lifetime | stage=producer producers=2 ranges=0 waiting_chunks=0'],
                        lines + ['metadata_lifetime | stage=prefill_log items=1'],
                        lines + ['metadata_lifetime | stage=decoder batches=1 jobs=0 pending_chunks=0'],
                        lines + ['metadata_lifetime | stage=router requests=0 routes=1'],
                        lines + ['metadata_lifetime | stage=prefill_buffer prepared=0 waiting=0 audio_tokens=12']]:
            with self.subTest(log=invalid), self.assertRaises(AssertionError):
                check_metadata_evidence('\n'.join(invalid))
        result = check_metadata_evidence('\n'.join([
            'metadata_lifetime | stage=decoder batches=7 jobs=24 pending_chunks=5',
            *lines,
        ]))
        self.assertTrue(result['drained'])
        self.assertEqual(result['peaks']['decoder']['jobs'], 24)

    def test_lifecycle_requires_registry_and_all_permits_drained(self):
        def line(requests=0, routes=0, requests_available=64, items_available=128, source_available=2048):
            return (f'metadata_lifetime | stage=registry requests={requests} routes={routes} '
                    f'request_budget_available={requests_available} request_budget_capacity=64 '
                    f'item_budget_available={items_available} item_budget_capacity=128 '
                    f'source_budget_available_bytes={source_available} source_budget_capacity_bytes=2048')
        for log in ['', line(requests=1), line(routes=1), line(requests_available=63),
                    line(items_available=127), line(source_available=2047),
                    line(items_available=129)+'\n'+line()]:
            with self.subTest(log=log), self.assertRaises(AssertionError):
                check_lifecycle_evidence(log)
        result=check_lifecycle_evidence(line(requests=3,routes=20,requests_available=61,items_available=108,source_available=1024)+'\n'+line())
        self.assertTrue(result['drained'])
        self.assertEqual(result['samples'],2)

    def test_refactor_short_probe_checks_raw_reference_as_well_as_cer(self):
        reference='我要修改商品帖子的价格。'
        check_short_text(reference,reference)
        for changed in ['我要修改商品帖子的价格', '我要修改商品帖子的价格！', '我要修改帖子价格。']:
            with self.subTest(text=changed), self.assertRaises(AssertionError):
                check_short_text(changed,reference)
        with self.assertRaises(AssertionError):
            check_short_text('错误标签','错误标签')

    def test_unready_service_never_receives_a_benchmark_workload(self):
        # macOS Bash 3.2 does not apply errexit to a bare [[ ... ]] command.
        # Replace only the wait delay and power assertion; the real scripts must
        # reject their own live but unready server before invoking perf or eval.
        import os
        scripts = pathlib.Path(__file__).resolve().parent
        for script in ['m4_formal.sh', 'm4_soak_run.sh']:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as folder:
                root = pathlib.Path(folder)
                tools = root / 'tools'
                tools.mkdir()
                for name, content in [('sleep', '#!/bin/bash\nexit 0\n'),
                                      ('caffeinate', '#!/bin/bash\nexec /bin/sleep 20\n')]:
                    path = tools / name
                    path.write_text(content)
                    path.chmod(0o755)
                binary = root / 'fake-transcribe'
                binary.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys,time
if sys.argv[1] == 'serve':
    print('loading model',flush=True)
    time.sleep(20)
else:
    pathlib.Path(os.environ['VASR_TEST_PERF_MARKER']).write_text(sys.argv[1])
    pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('{}')
''')
                binary.chmod(0o755)
                marker = root / 'workload-started'
                env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                           VASR_TEST_PERF_MARKER=str(marker))
                command = ['/bin/bash', str(scripts / script), str(binary), str(root / 'output')]
                if script == 'm4_soak_run.sh':
                    command.append(str(root / 'baseline.json'))
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
                self.assertFalse(marker.exists(), 'workload submitted before server readiness')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('server did not become ready', result.stderr)

    def test_quality_boundaries_match_rust_guards(self):
        self.assertTrue(quality_anomaly('再见', 2, 2))
        self.assertFalse(quality_anomaly('再见', 1, 2))
        self.assertTrue(quality_anomaly('', 0, 2))
        self.assertTrue(repeated_pattern('hello ibilibilibilibilibilibilibilibil end'))
        self.assertTrue(repeated_pattern('秧undryfaoknryfaoknryfaoknryfaoknryfaoknryfaoknry'))
        self.assertFalse(repeated_pattern('你好你好，今天怎么样？'))
        self.assertFalse(repeated_pattern('ha ha ha ha ha ha'))

    def test_requires_load_probes_full_duration_and_drained_kv(self):
        regular = [{'short': False, 'ok': True, 'started_sec': start,
                    'completed_sec': start + 65} for start in range(0, 1800, 60)]
        regular.append({'short': False, 'ok': True, 'started_sec': 1799, 'completed_sec': 1810})
        probes = [{'short': True, 'ok': True, 'started_sec': start,
                   'completed_sec': start + 1} for start in range(30, 1800, 120)]
        rows = regular + probes
        summary = {'duration_sec': 1800, 'drained_wall_sec': 1810,
                   'failed_requests': 0, 'successful_requests': len(rows),
                   'short_probes': len(probes)}
        memories = [{'elapsed_sec': elapsed, 'rss_kib': '1024', 'swap': 'used = 2.00M'}
                    for elapsed in range(0, 1871, 10)]
        valid = check_run_evidence(rows, summary, memories, 'kv_free=4/4 usable')
        self.assertTrue(valid['kv_fully_recovered'])
        for changed_rows, changed_summary, changed_memories, log in [
            (regular, dict(summary, successful_requests=len(regular), short_probes=0), memories, 'kv_free=4/4 usable'),
            (rows, dict(summary, duration_sec=60), memories, 'kv_free=4/4 usable'),
            (rows, summary, memories[:181], 'kv_free=4/4 usable'),
            (rows, summary, memories, 'kv_free=3/4 usable'),
        ]:
            with self.assertRaises(AssertionError):
                check_run_evidence(changed_rows, changed_summary, changed_memories, log)

    def test_actual_dtos_and_service_identity_cannot_be_swapped(self):
        refs = {i: {'url': f'{i}.wav', 'reference_normalized': '你好'} for i in range(99)}
        inputs = [{'modelscope': {'repo_id': 'owner/audio', 'file_path': f'{i}.wav', 'revision': 'abc'},
                   'mono': True} for i in range(99)]
        source = {'items': 99, 'requests': [{'request_index': 0, 'inputs': inputs,
                  'data': [{'serviceId': 'one'}] * 99}],
                  'cer': {'items': [dict(item, index=i) for i, item in refs.items()]}}
        short = {'items': 1, 'requests': [{'request_index': 0, 'inputs': [{'url': 'short.wav', 'mono': True}],
                 'data': [{'serviceId': 'one'}]}]}
        rows = [{'short': False, 'indices': [1, 0], 'inputs': copy.deepcopy([inputs[1], inputs[0]]),
                 'response': {'data': [{'serviceId': 'one'}, {'serviceId': 'one'}]}}]
        check_sources(rows, source, short, refs)
        wrong = copy.deepcopy(rows)
        wrong[0]['inputs'][0]['modelscope']['revision'] = 'other'
        with self.assertRaises(AssertionError):
            check_sources(wrong, source, short, refs)
        wrong = copy.deepcopy(rows)
        wrong[0]['response']['data'][0]['serviceId'] = 'other'
        with self.assertRaises(AssertionError):
            check_sources(wrong, source, short, refs)

    def test_quality_alone_cannot_certify_a_one_minute_run(self):
        # A complete, error-free 99-item transcription can finish too early to
        # provide any evidence about sustained load or short-request fairness.
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            items = [{'index': i, 'url': f'{i}.wav', 'reference': '你好啊',
                      'reference_normalized': '你好啊', 'hypothesis': '你好啊',
                      'hypothesis_normalized': '你好啊', 'substitutions': 0,
                      'insertions': 0, 'deletions': 0} for i in range(99)]
            baseline = {'cer': {'items': items}}
            (root / 'baseline.json').write_text(json.dumps(baseline))
            rows = [{'indices': [i], 'urls': [f'{i}.wav'], 'short': False,
                     'ok': True, 'started_sec': i / 10, 'completed_sec': 10,
                     'response': {'data': [{'is_bad': False, 'text': {'mono': '你好啊'}}]}}
                    for i in range(99)]
            (root / 'responses.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            (root / 'summary.json').write_text(json.dumps({
                'scheduled': 99, 'duration_sec': 60, 'drained_wall_sec': 61,
                'short_probes': 0, 'successful_requests': 99, 'failed_requests': 0}))
            (root / 'memory.jsonl').write_text('')
            result = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('audit_soak.py')),
                                     '--baseline', str(root / 'baseline.json'),
                                     '--validation-dir', str(root), '--soak-dir', str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'quality checks must not certify 30 minutes')
            self.assertIn('soak must request at least 30 minutes', result.stderr)


if __name__ == '__main__':
    unittest.main()
