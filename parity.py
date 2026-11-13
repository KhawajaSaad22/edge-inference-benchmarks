"""Compare saved numeric outputs; numerical agreement does not measure accuracy."""
import hashlib
import math
from pathlib import Path
import numpy as np


def check_outputs(reference_report,candidate_report,reference_metadata,candidate_metadata,atol=0.,rtol=0.):
    if any(not math.isfinite(v) or v<0 for v in (atol,rtol)):raise ValueError('Parity tolerances must be finite and nonnegative')
    archives=[]
    for report,metadata in ((reference_report,reference_metadata),(candidate_report,candidate_metadata)):
        path=Path(report).parent/'sample-output.npz'
        expected=metadata.get('sample_output_sha256')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Saved sample output changed: '+str(path))
        try:
            with np.load(path,allow_pickle=False) as data:archives.append({key:data[key] for key in data.files})
        except (ValueError,KeyError,EOFError) as exc:raise ValueError('Cannot read numeric sample output: '+str(path)) from exc
    left,right=archives
    if not left or set(left)!=set(right):raise ValueError('Sample output array names differ or are empty')
    arrays={}
    for key,a in left.items():
        b=right[key]
        if a.shape!=b.shape:raise ValueError('Sample output shapes differ: '+key)
        if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Sample outputs must contain finite numeric arrays: '+key)
        exact=np.array_equal(a,b)
        delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
        # Categories and boolean masks must agree exactly, regardless of tolerance.
        categorical=key in ('class_id','mask') or a.dtype.kind=='b' or b.dtype.kind=='b'
        passed=exact if categorical or atol==rtol==0 else bool(np.allclose(b,a,atol=atol,rtol=rtol))
        arrays[key]=dict(passed=bool(passed),exact=bool(exact),shape=list(a.shape),
                         reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),
                         max_abs_difference=float(delta.max()) if delta.size else 0.)
    return dict(passed=all(v['passed'] for v in arrays.values()),atol=atol,rtol=rtol,arrays=arrays,
                scope='One saved final output per run; numerical parity is not model accuracy.')
