import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark import measure,run
from compare import compare


class BenchmarkTests(unittest.TestCase):
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
