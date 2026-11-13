"""Repeatable latency and memory measurements; accuracy is evaluated separately."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import threading
import time
import cv2
import numpy as np
import psutil


def measure(workload,*,warmup=3,repeats=20,synchronize=lambda:None):
    if type(warmup) is not int or warmup<0 or type(repeats) is not int or repeats<1:
        raise ValueError('Nonnegative warmup and positive repeat count required')
    for _ in range(warmup): workload(); synchronize()
    process=psutil.Process(); baseline=process.memory_info().rss; peak=[baseline]
    stop=threading.Event()
    def sample():
        while not stop.wait(.005): peak[0]=max(peak[0],process.memory_info().rss)
    worker=threading.Thread(target=sample,daemon=True); worker.start(); times=[]
    try:
        synchronize()
        for _ in range(repeats):
            start=time.perf_counter(); output=workload(); synchronize()
            times.append(time.perf_counter()-start)
        peak[0]=max(peak[0],process.memory_info().rss)
    finally:
        stop.set(); worker.join()
    return dict(warmup=warmup,repeats=repeats,latency_ms=[t*1000 for t in times],
                p50_ms=float(np.percentile(times,50)*1000),p95_ms=float(np.percentile(times,95)*1000),
                mean_ms=float(np.mean(times)*1000),iterations_per_s=repeats/sum(times),
                rss_baseline_bytes=baseline,rss_sampled_peak_bytes=peak[0],
                memory_note='Process RSS sampled every 5ms after warmup; includes model/process overhead and may miss short peaks.'),output


def run(output_dir,**options):
    output_dir=Path(output_dir)
    if output_dir.exists():raise ValueError('Choose a new output directory')
    if type(options.get('warmup',3)) is not int or options.get('warmup',3)<0 or type(options.get('repeats',20)) is not int or options.get('repeats',20)<1:
        raise ValueError('Nonnegative warmup and positive repeat count required')
    output_dir.mkdir(parents=True)
    state={'schema_version':1,'status':'running','configuration':options}
    def save_state():
        (output_dir/'run.json').write_text(json.dumps(state,indent=2,default=str),encoding='utf-8')
    save_state()
    try:
        result=_run(output_dir,**options)
        state['status']='completed';save_state();return result
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));save_state();raise


def _run(output_dir,*,backend='preprocess',image_path=None,size=512,seed=42,
        weights=None,model_size='nano',device='cpu',warmup=3,repeats=20,workbench_root=None,recipe=None):
    startup_started=time.perf_counter()
    output_dir=Path(output_dir)
    if size<16 or size>8192: raise ValueError('Image size must be 16..8192')
    metadata=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        backend=backend,device=device,seed=seed,platform=platform.platform(),python=platform.python_version(),
        cpu=platform.processor() or 'unknown',cpu_logical_count=psutil.cpu_count(),
        opencv_threads=cv2.getNumThreads(),precision='backend_default' if backend=='rfdetr' else 'uint8',
        provider='pytorch' if backend=='rfdetr' else 'opencv',inference_threshold=.5 if backend=='rfdetr' else None,
        scope='One image per iteration; excludes model startup, file decoding and result serialization.',
        quality_note='No accuracy, recall, defect validity or real-time drone capability is measured.')
    if image_path:
        data=Path(image_path).read_bytes(); bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError('Could not decode input image')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        metadata.update(input_sha256=hashlib.sha256(data).hexdigest(),synthetic=False)
    else:
        rgb=np.random.default_rng(seed).integers(0,256,(size,size,3),dtype=np.uint8)
        metadata.update(synthetic=True,input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    metadata['input_shape']=list(rgb.shape); sync=lambda:None; torch=None
    if backend=='preprocess':
        if device!='cpu': raise ValueError('The preprocessing baseline runs on CPU only')
        def workload():
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            return {'edges':cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),50,150)}
    elif backend=='rfdetr':
        if weights is None or not Path(weights).is_file(): raise ValueError('Supply an existing segmentation checkpoint')
        if model_size not in ('nano','small') or device not in ('cpu','cuda'): raise ValueError('Invalid model size or device')
        import torch
        from rfdetr import RFDETRSegNano,RFDETRSegSmall
        if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
        model={'nano':RFDETRSegNano,'small':RFDETRSegSmall}[model_size](pretrain_weights=str(weights),device=device)
        metadata.update(torch_threads=torch.get_num_threads(),torch_cuda_runtime=torch.version.cuda)
        metadata.update(model_size=model_size,weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest())
        if device=='cuda':
            sync=torch.cuda.synchronize; metadata['gpu']=torch.cuda.get_device_name(0)
            # Exclude allocation peaks during initialization; report inference peak including the model.
            torch.cuda.reset_peak_memory_stats()
        def workload():
            detections=model.predict(rgb,threshold=.5)
            arrays={}
            for name in ('xyxy','mask','confidence','class_id'):
                value=getattr(detections,name,None)
                if value is not None: arrays[name]=np.asarray(value)
            return arrays
    elif backend=='workbench':
        import sys
        if workbench_root is None or recipe is None:raise ValueError('Workbench benchmarks require --workbench-root and --recipe')
        root=Path(workbench_root).resolve()
        if not (root/'engine.py').is_file() or not (root/'gpu_backend.py').is_file():raise ValueError('Invalid workbench source directory')
        sys.path.insert(0,str(root))
        try:
            engine=importlib.import_module('engine');gpu_backend=importlib.import_module('gpu_backend')
            if Path(engine.__file__).resolve().parent!=root or Path(gpu_backend.__file__).resolve().parent!=root:
                raise ValueError('A different workbench module is already loaded in this process')
        finally:sys.path.pop(0)
        pipeline=engine.load_pipeline(recipe)
        metadata.update(provider='workbench',precision='uint8',inference_threshold=None,
                        recipe_sha256=hashlib.sha256(json.dumps(pipeline,sort_keys=True).encode()).hexdigest(),
                        workbench_engine_sha256=hashlib.sha256((root/'engine.py').read_bytes()).hexdigest(),
                        workbench_gpu_sha256=hashlib.sha256((root/'gpu_backend.py').read_bytes()).hexdigest())
        if device=='cuda':
            import torch
            if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
            torch.cuda.reset_peak_memory_stats();sync=torch.cuda.synchronize
            metadata['gpu']=torch.cuda.get_device_name(0)
        last_execution={}
        def workload():
            output,_,_,execution=gpu_backend.run(rgb,pipeline,device,retain_intermediates=False,strict=device=='cuda')
            last_execution.clear();last_execution.update(execution)
            return {'image':output}
    else: raise ValueError('Unknown backend')
    metadata['startup_and_input_load_s']=time.perf_counter()-startup_started
    result,arrays=measure(workload,warmup=warmup,repeats=repeats,synchronize=sync)
    if backend=='workbench':result['execution']=last_execution.copy()
    if torch is not None and device=='cuda':
        result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['gpu_memory_scope']='PyTorch allocator including model and warmup; excludes other processes and non-PyTorch allocations.'
    versions={}
    for package in ('numpy','opencv-python-headless','psutil','rfdetr','torch','torchvision'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    result.update(metadata,dependencies=versions)
    with (output_dir/'sample-output.npz').open('xb') as stream: np.savez_compressed(stream,**arrays)
    result['sample_output_bytes']=(output_dir/'sample-output.npz').stat().st_size
    result['sample_output_sha256']=hashlib.sha256((output_dir/'sample-output.npz').read_bytes()).hexdigest()
    result['sample_output_note']='One final output serialized as compressed NPZ, not full-video storage.'
    with (output_dir/'benchmark.json').open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--backend',choices=['preprocess','rfdetr','workbench'],default='preprocess')
    p.add_argument('--workbench-root');p.add_argument('--recipe')
    p.add_argument('--image'); p.add_argument('--size',type=int,default=512); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--weights'); p.add_argument('--model-size',choices=['nano','small'],default='nano')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=20)
    a=vars(p.parse_args()); a['output_dir']=a.pop('output'); a['image_path']=a.pop('image')
    print(json.dumps(run(**a),indent=2))


if __name__=='__main__': main()
