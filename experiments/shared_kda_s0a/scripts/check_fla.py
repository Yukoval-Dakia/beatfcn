#!/usr/bin/env python3
"""Official-kernel preflight on an ALREADY authorized CUDA host; no installation.
Two lengths test within-chunk and cross-chunk recurrence, state and gradients.
"""
import sys,json,importlib.metadata,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from beatfcn_s0.kda import recurrence,official_fla

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/fla_parity.json');args=p.parse_args();dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps({'passed':False,'status':'pending'}))
    if importlib.metadata.version('fla-core')!='0.5.2':raise RuntimeError('required pinned fla-core==0.5.2')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check NOT run')
    torch.manual_seed(123);cases=[]
    for length in (16,129):
        leaves=[torch.randn(1,length,8,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        q,k,v=leaves;q=F.normalize(q.float(),dim=-1,eps=1e-6).to(q.dtype);k=F.normalize(k.float(),dim=-1,eps=1e-6).to(k.dtype)
        g=(-F.softplus(torch.randn_like(q).float())*.1).requires_grad_();beta=torch.rand(1,length,8,device='cuda').requires_grad_()
        a,sa=recurrence(q,k,v,g,beta);b,sb=official_fla(q,k,v,g,beta)
        torch.testing.assert_close(a,b,atol=.01,rtol=.03);torch.testing.assert_close(sa,sb,atol=.02,rtol=.03)
        ga=torch.autograd.grad(a.float().square().sum()+sa.square().mean(),leaves+[g,beta],retain_graph=True)
        gb=torch.autograd.grad(b.float().square().sum()+sb.square().mean(),leaves+[g,beta])
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=.02,rtol=.05)
        cases.append({'length':length,'output_max_error':(a-b).abs().max().item(),'state_max_error':(sa-sb).abs().max().item(),'gradient_max_errors':[(x-y).abs().max().item() for x,y in zip(ga,gb)]})
    result={'passed':True,'cuda':torch.version.cuda,'triton':importlib.metadata.version('triton'),'dtype':'bfloat16','kda_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'beatfcn_s0/kda.py').read_bytes()).hexdigest(),'torch':torch.__version__,'fla_core':importlib.metadata.version('fla-core'),'gpu':torch.cuda.get_device_name(),'tested_lengths':[16,129],'cases':cases,'tolerances':{'output':[.01,.03],'state':[.02,.03],'grad':[.02,.05]}}
    dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        argv=sys.argv;dest=Path(argv[argv.index('--output')+1]) if '--output' in argv else Path('reports/fla_parity.json')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps({'passed':False,'status':'failed','error_type':type(exc).__name__,'error':str(exc)},indent=2))
        raise

