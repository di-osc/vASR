import json
import os
import pathlib
import subprocess
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parent

class RefactorFormalTest(unittest.TestCase):
    def test_refactor_requires_baseline_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = pathlib.Path(folder) / 'output'
            result = subprocess.run(['/bin/bash', str(SCRIPTS / 'm4_formal.sh'), '/missing/candidate', str(output), 'refactor'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('baseline binary required', result.stderr)
            self.assertFalse(output.exists())

    def test_refactor_alternates_two_binaries_without_operator_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            utilities = root / 'tools'
            utilities.mkdir()
            for name, source in {
                'sleep': '#!/bin/bash\nexec /bin/sleep 0.02\n',
                'caffeinate': '#!/bin/bash\nexec /bin/sleep 30\n',
                'curl': '#!/bin/bash\nexit 0\n',
            }.items():
                path = utilities / name
                path.write_text(source)
                path.chmod(0o755)
            marker = root / 'invocations.jsonl'
            source = '''#!/usr/bin/env python3
import json,os,pathlib,sys,time
with open(os.environ['REFACTOR_TEST_INVOCATIONS'], 'a') as f:
    f.write(json.dumps({'binary':pathlib.Path(sys.argv[0]).name,'command':sys.argv[1],'fast_math':os.environ.get('CANDLE_METAL_ENABLE_FAST_MATH'),
        'options':{k:v for k,v in os.environ.items() if k.startswith('VASR_')}})+'\\n')
if sys.argv[1] == 'serve':
    print('ready | http://0.0.0.0:8868 endpoints=health,transcribe',flush=True)
    time.sleep(30)
else:
    if sys.argv[1] == 'perf':
        data={'summary':{'items':99,'quality_passed_items':99,'failed_requests':0,'bad_items':0,
            'quality_audio_seconds_per_second':120},'cer':{'summary':{'quality_passed_items':99,'cer_weighted':0}}}
    else:
        data={'summary':{'cer_weighted':0}}
    pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text(json.dumps(data))
'''
            binaries = [root / 'candidate', root / 'baseline']
            for binary in binaries:
                binary.write_text(source)
                binary.chmod(0o755)
            env = dict(os.environ, PATH=str(utilities)+os.pathsep+os.environ['PATH'], REFACTOR_TEST_INVOCATIONS=str(marker), VASR_METAL_PREFILL_ROPE_KERNEL='0', VASR_METAL_MAX_ACTIVE_SEQS='1', CANDLE_METAL_ENABLE_FAST_MATH='0')
            result = subprocess.run(['/bin/bash', str(SCRIPTS / 'm4_formal.sh'), str(binaries[0]), str(root / 'output'), 'refactor', str(binaries[1])], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows=[json.loads(line) for line in marker.read_text().splitlines()]
            serves=[row for row in rows if row['command']=='serve']
            self.assertEqual([row['binary'] for row in serves], ['baseline','candidate','candidate','baseline','baseline','candidate'])
            for row in serves:
                self.assertFalse(set(row['options']) - {'VASR_PROFILE'}, row)
                self.assertIsNone(row['fast_math'], 'refactor inherited changed floating point settings')
            self.assertEqual([row['binary'] for row in rows if row['command']=='perf'], [binary for binary in ['baseline','candidate','candidate','baseline','baseline','candidate'] for _ in range(2)])
            self.assertEqual(sum(row['command']=='eval' for row in rows),6)

class RefactorAuditTest(unittest.TestCase):
    def fixture(self, root, candidate_speed=119, raw_change=False, baseline_speed=120):
        (root / 'experiment.txt').write_text('refactor\n')
        for mode in ['control','candidate']:
            for number in [1,2,3]:
                items=[{'index':i,'audio_id':str(i),'url':f'{i}.wav','reference':'a','reference_normalized':'a',
                    'hypothesis':'a','hypothesis_normalized':'a','substitutions':0,'insertions':0,'deletions':0,
                    'is_bad':False,'quality_anomaly':False} for i in range(99)]
                if raw_change and mode=='candidate': items[0]['hypothesis']='a!'
                requests=[]
                for i in range(0,99,4):
                    chunk=items[i:i+4]
                    requests.append({'request_index':i//4,'ok':True,'error':None,'batch_size':len(chunk),
                        'quality_passed_items':len(chunk),'bad_items':0,
                        'inputs':[{'url':item['url'],'mono':True} for item in chunk],
                        'data':[{'serviceId':f'{mode}-{number}','text':{'mono':item['hypothesis']}} for item in chunk]})
                report={'summary':{'items':99,'quality_passed_items':99,'failed_requests':0,'bad_items':0,
                    'audio_seconds':5025.0118,'batch_size':4,'concurrency':28,
                    'quality_audio_seconds_per_second':baseline_speed if mode=='control' else candidate_speed,'wall_seconds':42},
                    'cer':{'summary':{'quality_passed_items':99},'items':items},'requests':requests}
                prefix=root / f'{mode}-r{number}'
                for name in ['perf','warmup']: pathlib.Path(f'{prefix}-{name}.json').write_text(json.dumps(report))
                pathlib.Path(f'{prefix}-example1.json').write_text(json.dumps({'summary':{'cer_weighted':0,'quality_passed_items':1},'items':[{'index':0,'audio_id':'example1','reference_normalized':'我要修改商品帖子的价格','hypothesis_normalized':'我要修改商品帖子的价格','hypothesis':'我要修改商品帖子的价格。','is_bad':False,'quality_anomaly':None}]}))
                pathlib.Path(f'{prefix}-memory.jsonl').write_text(json.dumps({'rss_kib':'1000','swap':'0'})+'\n')
    def audit(self, root):
        return subprocess.run([os.sys.executable,str(SCRIPTS/'audit_formal.py'),str(root)],capture_output=True,text=True)
    def test_refactor_rejects_performance_regression(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);self.fixture(root,candidate_speed=110)
            result=self.audit(root)
            self.assertNotEqual(result.returncode,0,'refactor audit accepted >3% regression')
    def test_refactor_requires_above_100_even_within_three_percent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);self.fixture(root,candidate_speed=99,baseline_speed=101)
            result=self.audit(root)
            self.assertNotEqual(result.returncode,0,'refactor audit accepted throughput below100x')
    def test_refactor_rejects_raw_changes_even_when_cer_is_equal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);self.fixture(root,raw_change=True)
            result=self.audit(root)
            self.assertNotEqual(result.returncode,0,'refactor audit accepted changed raw transcript')
    def test_refactor_audits_warmup_as_well_as_formal_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);self.fixture(root)
            (root/'candidate-r2-warmup.json').unlink()
            result=self.audit(root)
            self.assertNotEqual(result.returncode,0,'refactor audit ignored missing warmup')
    def test_refactor_rejects_missing_or_changed_example1_despite_zero_summary_cer(self):
        for items in [[], [{'index':0,'audio_id':'example1','reference_normalized':'我要修改商品帖子的价格',
                           'hypothesis_normalized':'我要修改商品帖子的价格','hypothesis':'changed',
                           'is_bad':False,'quality_anomaly':None}]]:
            with self.subTest(items=items), tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);self.fixture(root)
                (root/'candidate-r1-example1.json').write_text(json.dumps({'summary':{'cer_weighted':0,'quality_passed_items':1},'items':items}))
                result=self.audit(root)
                self.assertNotEqual(result.returncode,0,'refactor audit trusted an invalid example1 summary')
    def test_refactor_requires_exact_client_load_and_mono(self):
        for change in ['concurrency','mono']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);self.fixture(root)
                for path in list(root.glob('*-perf.json')) + list(root.glob('*-warmup.json')):
                    data=json.loads(path.read_text())
                    if change=='concurrency':
                        data['summary']['concurrency']=1
                    else:
                        for request in data['requests']:
                            for audio in request['inputs']: audio['mono']=False
                    path.write_text(json.dumps(data))
                result=self.audit(root)
                self.assertNotEqual(result.returncode,0,'refactor audit accepted a different client workload')
    def test_refactor_accepts_exactly_three_percent_and_rejects_more(self):
        for speed, allowed in [(194,True),(193.9999,False)]:
            with self.subTest(speed=speed), tempfile.TemporaryDirectory() as folder:
                root=pathlib.Path(folder);self.fixture(root,candidate_speed=speed,baseline_speed=200)
                result=self.audit(root)
                self.assertEqual(result.returncode==0,allowed,result.stdout+result.stderr)
    def test_refactor_accepts_identical_text_within_three_percent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);self.fixture(root,candidate_speed=117)
            result=self.audit(root)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__ == '__main__':
    unittest.main()
