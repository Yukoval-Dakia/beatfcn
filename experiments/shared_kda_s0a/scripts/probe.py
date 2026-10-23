#!/usr/bin/env python3
"""Bounded S0a probe. Does not provision GPU, upload, train corpus, or buy compute."""
import argparse,hashlib,json,time,resource,subprocess,sys,platform,importlib.metadata,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from beatfcn_s0.model import SharedModel,Context
from beatfcn_s0.preprocess import waveform_mel

EXPECTED='6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');p.add_argument('--backend',choices=['reference','chunk','fla'],default='reference');p.add_argument('--audio');p.add_argument('--audio-offset-seconds',type=float,default=15.0);p.add_argument('--mel-frames',type=int,default=4);p.add_argument('--groups',type=int,default=2);p.add_argument('--steps',type=int,default=1);p.add_argument('--output',default='reports/probe.json');p.add_argument('--expected-sha256',default=EXPECTED);p.add_argument('--parity-report')
    a=p.parse_args()
    if not math.isfinite(a.audio_offset_seconds) or a.audio_offset_seconds<0:p.error('audio offset must be finite and nonnegative')
    # Native window needs a chunked scan: FLA on CUDA, or the exact PyTorch chunk form.
    max_frames=1024 if (a.device=='cuda' and a.backend=='fla') or a.backend=='chunk' else 64
    if not (4<=a.mel_frames<=max_frames and a.mel_frames%4==0 and 1<=a.groups<=8 and 1<=a.steps<=3):p.error(f'bounded S0a: frames4..{max_frames} multiple4;groups1..8;steps1..3')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback GPU claim')
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback GPU claim')
    if a.backend=='fla' and a.device!='cuda':p.error('FLA requires CUDA')
    if a.backend=='fla':
        if importlib.metadata.version('fla-core')!='0.5.2':raise RuntimeError('required pinned fla-core==0.5.2')
        if not a.parity_report:raise RuntimeError('run check_fla.py first and supply --parity-report')
        parity=json.loads(Path(a.parity_report).read_text())
        if parity.get('tested_lengths')!=[16,129] or not parity.get('passed') or parity.get('fla_core')!='0.5.2' or parity.get('torch')!=torch.__version__ or parity.get('gpu')!=torch.cuda.get_device_name() or parity.get('cuda')!=torch.version.cuda or parity.get('triton')!=importlib.metadata.version('triton') or parity.get('dtype')!='bfloat16' or parity.get('kda_source_sha256')!=sha(Path(__file__).resolve().parents[1]/'beatfcn_s0/kda.py'):raise RuntimeError('parity report mismatch; stop rather than fallback')
    torch.manual_seed(20261022);torch.set_num_threads(2)
    report={'scope':'S0a full-size shared numerical probe, synthetic event/condition features, NOT S0b training','torch':torch.__version__,'python':platform.python_version(),'device':a.device,'backend':a.backend,'mel_frames':a.mel_frames,'previous_groups':a.groups,'steps':a.steps,'dtype':'bfloat16','native_window_test':a.mel_frames==1024,'parity_report':a.parity_report,'gpu_measured':a.device in ('cuda','mps'),'cuda_peak_allocated':None,'cuda_peak_reserved':None,'mps_samples':[]}
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
        if a.device=='mps':torch.mps.synchronize()
    def sample(phase):
        # MPS has no peak counter: record allocator state at the likely peaks.
        if a.device=='mps':sync();report['mps_samples'].append({'phase':phase,'current_allocated':torch.mps.current_allocated_memory(),'driver_allocated':torch.mps.driver_allocated_memory()})
    start=time.perf_counter();digest=sha(a.checkpoint)
    if digest!=a.expected_sha256:raise ValueError('checkpoint SHA256 mismatch')
    report['checkpoint_sha256']=digest;report['checkpoint_sha256_verified']=True
    dtype=torch.bfloat16;model=SharedModel.from_checkpoint(a.checkpoint,dtype=dtype,device=a.device)
    report['inventory']=model.inventory();report['loaded_tensors']=len(model.backbone.state_dict());report['load_seconds']=time.perf_counter()-start
    if a.audio:
        # Decode one short existing local audio excerpt only; no download.
        raw=subprocess.run(['ffmpeg','-v','error','-ss',str(a.audio_offset_seconds),'-i',a.audio,'-t',str(a.mel_frames*.01),'-f','f32le','-ac','1','-ar','24000','pipe:1'],check=True,capture_output=True).stdout
        wave=np.frombuffer(raw,dtype='<f4').copy();needed=a.mel_frames*240
        if len(wave)<needed:raise ValueError('audio excerpt unexpectedly short')
        mel=waveform_mel(wave[:needed]);report['audio_sha256']=sha(a.audio);report['audio_offset_seconds']=a.audio_offset_seconds;report['audio_scope']='real local excerpt, synthetic all event/condition features; rights not established for training'
    else:
        mel=torch.randn(1,1,a.mel_frames,128);report['audio_scope']='synthetic normalized-shape input; no real audio claim'
    def rand(*shape):return torch.randn(*shape,device=a.device,dtype=dtype)
    c=Context(mel.to(a.device,dtype),rand(1,2,1024),rand(1,2,8),rand(1,2,8));y=rand(1,a.groups,8)
    frozen_before={n:p.detach().flatten()[::max(1,p.numel()//7)].clone() for n,p in model.backbone.named_parameters()}
    # No optimizer state allocated: one explicit SGD step suffices for path check.
    report['iterations']=[]
    if a.device=='cuda':
        report['cuda_load_peak_allocated']=torch.cuda.max_memory_allocated();report['cuda_load_peak_reserved']=torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();report['gpu_name']=torch.cuda.get_device_name()
    report['versions']={k:(importlib.metadata.version(k) if importlib.util.find_spec(k.replace('-','_')) is not None else None) for k in ['numpy','safetensors']}
    if a.device=='cuda':
        report['versions']['cuda']=torch.version.cuda
        report['versions']['triton']=importlib.metadata.version('triton')
    if a.backend=='fla':report['versions']['fla-core']=importlib.metadata.version('fla-core')
    sample('loaded');sync();t=time.perf_counter();memory=model.encode_audio(c.local_mel);sync()
    report['audio_encode_seconds']=time.perf_counter()-t;report['audio_memory_bytes']=memory.nbytes();sample('audio_memory')
    for step in range(a.steps):
        # Frozen audio memory is reused across updates; only trainable rows rerun.
        model.zero_grad(set_to_none=True);sync();t=time.perf_counter();out=model(c,y,audio=memory,backend=a.backend);sample(f'step{step}_forward')
        loss=sum(out[n].float().square().mean() for n in ['action','coarse','fine','selection'])
        if not torch.isfinite(loss) or any(not torch.isfinite(out[n]).all() for n in ['action','coarse','fine','selection']):raise RuntimeError('nonfinite outputs or loss; stop')
        loss.backward();sample(f'step{step}_backward')
        sync()
        elapsed=time.perf_counter()-t
        grad={};missing=[];bad=[];frozen_grad=[]
        for n,param in model.named_parameters():
            if not param.requires_grad:
                if param.grad is not None:frozen_grad.append(n)
                continue
            if param.grad is None:missing.append(n);continue
            if not torch.isfinite(param.grad).all():bad.append(n)
            key='.'.join(n.split('.')[:2]) if n.startswith('kda_blocks.') else n.split('.')[0]
            grad[key]=grad.get(key,0.)+param.grad.float().square().sum().item()
        if missing or bad or frozen_grad:raise RuntimeError(f'gradient failure missing={missing},nonfinite={bad},frozen={frozen_grad}')
        old=model.outer_action.weight.detach().clone()
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:param.add_(param.grad,alpha=-.01)
        model.mark_updated()
        changed=not torch.equal(old,model.outer_action.weight)
        if not changed:raise RuntimeError('new parameter did not update')
        report['iterations'].append({'loss':loss.item(),'forward_backward_seconds':elapsed,'gradient_l2':{k:v**.5 for k,v in grad.items()},'new_weight_changed':changed,'missing_gradients':missing,'nonfinite_gradients':bad,'frozen_gradients':frozen_grad})
        del out,loss
    model.eval();timings=[]
    with torch.no_grad():
        for label in ('edit_first_kda_prefix_fill','edit_memo_hit'):
            sync();t=time.perf_counter();model(c,y,audio=memory,backend=a.backend);sync();timings.append((label,time.perf_counter()-t))
        sync();t=time.perf_counter();model(c,y,backend=a.backend);sync();timings.append(('no_memory_full_reencode',time.perf_counter()-t))
    report['inference_seconds']=dict(timings);sample('inference')
    report['frozen_sample_values_unchanged']=all(torch.equal(frozen_before[n],p.detach().flatten()[::max(1,p.numel()//7)]) for n,p in model.backbone.named_parameters())
    # All frozen tensors excluded from updates by construction; sampled equality
    # is not mislabeled a full post-run tensor hash proof.
    if a.device=='cuda':
        report['cuda_step_peak_allocated']=torch.cuda.max_memory_allocated();report['cuda_step_peak_reserved']=torch.cuda.max_memory_reserved()
        report['cuda_peak_allocated']=max(report['cuda_load_peak_allocated'],report['cuda_step_peak_allocated']);report['cuda_peak_reserved']=max(report['cuda_load_peak_reserved'],report['cuda_step_peak_reserved'])
    report['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['status']='passed'
    report['total_seconds']=time.perf_counter()-start
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Fail closed, including OOM. Do not retry with smaller/different model.
        args=sys.argv;dest=Path(args[args.index('--output')+1]) if '--output' in args else Path('reports/probe.json')
        failure={'status':'failed','error_type':type(exc).__name__,'error':str(exc),'cuda_peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'cuda_peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,'automatic_retry':False}
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(failure,indent=2));raise

