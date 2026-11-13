import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark import measure,run
from compare import compare,save_comparison


class BenchmarkTests(unittest.TestCase):
    def test_workbench_requires_explicit_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'workbench-root'):
                run(Path(folder)/'out',backend='workbench',size=32,repeats=1)

    def test_comparison_rejects_different_recipe_or_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run(root/'a',size=32,warmup=0,repeats=1)
            source=root/'a/benchmark.json';other=root/'other.json'
            for key in ('recipe_sha256','workbench_engine_sha256','workbench_gpu_sha256'):
                report=json.loads(source.read_text());report[key]='different'
                other.write_text(json.dumps(report))
                with self.assertRaises(ValueError):compare([source,other],True)

    def test_environment_override_and_portable_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run(root/'a',size=32,warmup=0,repeats=2)
            paths=[root/'a/benchmark.json',root/'other.json']
            report=json.loads(paths[0].read_text());report['dependencies']={'different':'1'}
            paths[1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'environments differ'):compare(paths)
            rows=compare(paths,True)
            self.assertIn('dependencies',rows[1]['environment_differences'])
            for suffix in ('csv','html','json'):
                save_comparison(rows,root/('report.'+suffix))
                self.assertTrue((root/('report.'+suffix)).is_file())
            report['precision']='changed';paths[1].write_text(json.dumps(report))
            with self.assertRaises(ValueError):compare(paths,True)

    def test_failed_run_records_state(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            with patch('benchmark.measure',side_effect=RuntimeError('simulated failure')):
                with self.assertRaises(RuntimeError):run(root,size=32,repeats=2)
            state=json.loads((root/'run.json').read_text())
            self.assertEqual(state['status'],'failed');self.assertIn('simulated',state['error'])
    def test_warmup_and_sync_boundaries(self):
        calls=[]; sync=[]
        result,output=measure(lambda:(calls.append(1) or {'x':np.ones((2,2))}),warmup=2,repeats=3,
                              synchronize=lambda:sync.append(1))
        self.assertEqual(len(calls),5); self.assertEqual(len(result['latency_ms']),3)
        self.assertEqual(len(sync),6)
        self.assertGreaterEqual(result['rss_sampled_peak_bytes'],result['rss_baseline_bytes'])
        with self.assertRaises(ValueError): measure(lambda:None,repeats=0)

    def test_repeatable_input_and_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            first=run(root/'a',size=64,repeats=2,warmup=1)
            second=run(root/'b',size=64,repeats=2,warmup=1)
            self.assertEqual(first['input_sha256'],second['input_sha256'])
            self.assertTrue(first['synthetic']); self.assertGreater(first['sample_output_bytes'],0)
            paths=[root/'a/benchmark.json',root/'b/benchmark.json']
            self.assertEqual(len(compare(paths)),2)
            changed=json.loads(paths[1].read_text()); changed['input_sha256']='different'
            paths[1].write_text(json.dumps(changed))
            with self.assertRaises(ValueError): compare(paths)
            with self.assertRaises(ValueError): run(root/'a',repeats=1)

    def test_cpu_baseline_refuses_fake_gpu_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'CPU only'): run(Path(folder)/'out',device='cuda')
