import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark import run
from compare import compare
from parity import check_outputs


class ParityTests(unittest.TestCase):
    def test_matching_benchmark_samples_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('a','b'):run(root/name,size=32,warmup=0,repeats=1)
            paths=[root/n/'benchmark.json' for n in ('a','b')]
            self.assertTrue(all(r['output_parity']['passed'] for r in compare(paths,check_outputs=True)))
            with (root/'b/sample-output.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):compare(paths,check_outputs=True)
    def test_tolerances_mismatch_categories_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            def check(left,right,**options):
                np.savez(a/'sample-output.npz',**left);np.savez(b/'sample-output.npz',**right)
                return check_outputs(a/'benchmark.json',b/'benchmark.json',{},{},**options)
            left={'image':np.array([0,255],dtype='uint8')};right={'image':np.array([1,254],dtype='uint8')}
            self.assertFalse(check(left,right)['passed']);self.assertTrue(check(left,right,atol=1)['passed'])
            self.assertFalse(check({'class_id':np.array([1])},{'class_id':np.array([2])},atol=5)['passed'])
            with self.assertRaises(ValueError):check(left,{'image':np.array([np.nan,0])})
            with self.assertRaises(ValueError):check(left,{'other':np.array([0])})
            with self.assertRaises(ValueError):check(left,{'image':np.array([0])})
            with self.assertRaises(ValueError):check(left,left,atol=float('nan'))
