"""Compare benchmark records without silently combining incompatible inputs."""
import argparse
import json
import csv
import html
import math
from pathlib import Path


def compare(paths,allow_environment_differences=False):
    reports=[json.loads(Path(path).read_text(encoding='utf-8')) for path in paths]
    if len(reports)<2: raise ValueError('Provide at least two benchmark reports')
    keys=('backend','input_sha256','input_shape','weights_sha256','model_size','warmup','repeats','scope','precision','inference_threshold','recipe_sha256','workbench_engine_sha256','workbench_gpu_sha256')
    if any(any(report.get(key)!=reports[0].get(key) for key in keys) for report in reports[1:]):
        raise ValueError('Backend, input, model and measurement settings must match')
    environment=('dependencies','python','platform','device','cpu','cpu_logical_count','gpu','provider','opencv_threads','torch_threads','torch_cuda_runtime')
    differences=[[key for key in environment if r.get(key)!=reports[0].get(key)] for r in reports]
    if any(differences) and not allow_environment_differences:
        raise ValueError('Runtime environments differ; use --allow-environment-differences for an explicit comparison')
    for report in reports:
        for key in ('p50_ms','p95_ms','rss_sampled_peak_bytes','sample_output_bytes'):
            value=report.get(key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid benchmark metric: '+key)
    return [dict(path=str(path),device=r['device'],platform=r['platform'],p50_ms=r['p50_ms'],
                 p95_ms=r['p95_ms'],rss_sampled_peak_bytes=r['rss_sampled_peak_bytes'],
                 sample_output_bytes=r['sample_output_bytes'],environment_differences=diff)
            for path,r,diff in zip(paths,reports,differences)]


def save_comparison(rows,path):
    path=Path(path)
    if path.suffix.lower() not in ('.json','.csv','.html'):raise ValueError('Use .json, .csv or .html output')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as stream:
        if path.suffix.lower()=='.json':json.dump(rows,stream,indent=2)
        elif path.suffix.lower()=='.csv':
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        else:
            columns=list(rows[0]);esc=lambda x:html.escape(str(x))
            stream.write('<!doctype html><meta charset="utf-8"><title>Benchmark comparison</title><h1>Benchmark comparison</h1><p>Latency and resource use only; no accuracy claim.</p><table border="1"><tr>'+''.join('<th>'+esc(k)+'</th>' for k in columns)+'</tr>')
            for row in rows:stream.write('<tr>'+''.join('<td>'+esc(row[k])+'</td>' for k in columns)+'</tr>')
            stream.write('</table>')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('reports',nargs='+')
    p.add_argument('--allow-environment-differences',action='store_true');p.add_argument('--output')
    args=p.parse_args();rows=compare(args.reports,args.allow_environment_differences)
    if args.output:save_comparison(rows,args.output)
    print(json.dumps(rows,indent=2))
