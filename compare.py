"""Compare benchmark records without silently combining incompatible inputs."""
import argparse
import json
from pathlib import Path


def compare(paths):
    reports=[json.loads(Path(path).read_text(encoding='utf-8')) for path in paths]
    if len(reports)<2: raise ValueError('Provide at least two benchmark reports')
    keys=('backend','input_sha256','input_shape','weights_sha256','model_size','warmup','repeats')
    if any(any(report.get(key)!=reports[0].get(key) for key in keys) for report in reports[1:]):
        raise ValueError('Backend, input, model and measurement settings must match')
    return [dict(path=str(path),device=r['device'],platform=r['platform'],p50_ms=r['p50_ms'],
                 p95_ms=r['p95_ms'],rss_sampled_peak_bytes=r['rss_sampled_peak_bytes'],
                 sample_output_bytes=r['sample_output_bytes']) for path,r in zip(paths,reports)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('reports',nargs='+'); args=p.parse_args()
    print(json.dumps(compare(args.reports),indent=2))
