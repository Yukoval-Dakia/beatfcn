"""Kimi Delta Attention reference + exact chunked form, isolated state routing.

Independent PyTorch implementation of published recurrence/parameterization.
No claim that a causal scan implements an arbitrary block attention mask.
Official sources/epsilon choices documented in ARCHITECTURE.md. State is [B,H,K,V].
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


def recurrence(q,k,v,log_alpha,beta,state=None):
    """Inputs [B,T,H,D], beta[B,T,H]; q/k ALREADY L2-normalized.

    Accumulate state FP32 for BF16/FP32 inputs, retain float64 for gradcheck.
    Gate decays KEY rows before delta prediction; beta scalar/head.
    """
    original=v.dtype
    work=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,k,v,g,beta=[z.to(work) for z in (q,k,v,log_alpha,beta)]
    b,t,h,d=q.shape
    s=q.new_zeros(b,h,d,v.shape[-1]) if state is None else state.to(work)
    outputs=[]
    for j in range(t):
        decayed=s*g[:,j].exp()[...,None]
        residual=v[:,j]-(k[:,j,...,None]*decayed).sum(-2)
        s=decayed+beta[:,j,:,None,None]*k[:,j,...,None]*residual[...,None,:]
        outputs.append(((q[:,j]*(d**-.5))[...,None]*s).sum(-2))
    out=torch.stack(outputs,1) if outputs else v.new_empty(b,0,h,v.shape[-1])
    return out.to(original),s


def chunk_recurrence(q,k,v,log_alpha,beta,state=None,chunk=16):
    """Exact chunked form of ``recurrence`` in plain PyTorch (CPU/MPS/CUDA).

    Per chunk, u=(I+A)^-1 diag(beta)(V-K+ S0) with A_tj=beta_t k_t.(G_t-G_j decay)k_j
    (j<t), solved by the nilpotent doubling product; only chunk-boundary states
    are kept for autograd. Decays enter only as exp(G_t-G_j<=0), never exp(-G),
    so strong gates cannot overflow. Pairwise decay tensor is [C,C,K] per chunk.
    """
    original=v.dtype
    work=torch.float64 if q.dtype==torch.float64 else torch.float32
    q,k,v,g,beta=[z.to(work) for z in (q,k,v,log_alpha,beta)]
    b,t,h,d=q.shape;dv=v.shape[-1];c=chunk;n=-(-t//c);pad=n*c-t
    if pad:
        # Zero key/value/beta and zero log-decay leave the state untouched.
        q,k,v,g=[F.pad(z,(0,0,0,0,0,pad)) for z in (q,k,v,g)];beta=F.pad(beta,(0,0,0,pad))
    q,k,v,g=[z.reshape(b,n,c,h,-1).transpose(2,3) for z in (q,k,v,g)]  # B,N,H,C,D
    beta=beta.reshape(b,n,c,h).transpose(2,3)
    G=g.cumsum(3)
    causal=torch.ones(c,c,dtype=torch.bool,device=q.device).tril()
    decay=torch.where(causal[...,None],G[...,:,None,:]-G[...,None,:,:],-torch.inf).exp()
    strict=torch.eye(c,dtype=torch.bool,device=q.device).logical_not()
    A=torch.einsum('bnhtd,bnhjd,bnhtjd->bnhtj',k,k,decay).masked_fill(~(causal&strict),0)*beta[...,None]
    eye=torch.eye(c,dtype=work,device=q.device);inv=eye-A;power=A;span=2
    while span<c:power=power@power;inv=inv@(eye+power);span*=2
    wv=inv@(beta[...,None]*v);wk=inv@(beta[...,None]*k*G.exp())
    qk=torch.einsum('bnhtd,bnhjd,bnhtjd->bnhtj',q*d**-.5,k,decay)
    qplus=q*G.exp()*d**-.5;tail=k*(G[...,-1:,:]-G).exp();last=G[...,-1,:].exp()
    s=q.new_zeros(b,h,d,dv) if state is None else state.to(work)
    outputs=[]
    for j in range(n):
        u=wv[:,j]-wk[:,j]@s
        outputs.append(qplus[:,j]@s+qk[:,j]@u)
        s=last[:,j,...,None]*s+tail[:,j].transpose(-1,-2)@u
    out=torch.stack(outputs,1).transpose(2,3).reshape(b,n*c,h,dv)[:,:t]
    return out.to(original),s


def official_fla(q,k,v,g,beta,state=None):
    """Optional actual official CUDA kernel, never silently substitute another op.

    Requires a pinned tested fla-core environment. Not installed or GPU-tested
    by this package. Already-transformed gates/norm/beta avoid fused ambiguity.
    """
    if not q.is_cuda:raise ValueError('FLA requires CUDA; use reference on CPU')
    from fla.ops.kda import chunk_kda
    return chunk_kda(q=q,k=k,v=v,g=g,beta=beta,initial_state=state,
                     output_final_state=True,use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False,use_beta_sigmoid_in_kernel=False,
                     state_v_first=False)


@dataclass
class StreamState:
    recurrent: torch.Tensor
    convolutions: tuple


class ShortConv(nn.Module):
    def __init__(self,dim,kernel=4):
        super().__init__();self.kernel=kernel
        self.weight=nn.Parameter(torch.empty(dim,kernel));self.bias=nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight,std=kernel**-.5)
    def forward(self,x,state=None):
        b,t,d=x.shape
        history=x.new_zeros(b,self.kernel-1,d) if state is None else state
        z=torch.cat([history,x],dim=1)
        # unfold shape B,T,D,K. Kernel convention matches torch.conv1d.
        out=(z.unfold(1,self.kernel,1)*self.weight[None,None]).sum(-1)+self.bias
        return F.silu(out),z[:,-(self.kernel-1):]


class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x):
        dtype=x.dtype; y=x.float() if dtype!=torch.float64 else x
        return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.eps)).to(dtype)*self.weight


class KDAMixer(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();assert dim%heads==0
        self.dim=dim;self.heads=heads;self.hd=dim//heads
        self.q_proj=nn.Linear(dim,dim,bias=False);self.k_proj=nn.Linear(dim,dim,bias=False);self.v_proj=nn.Linear(dim,dim,bias=False)
        self.q_conv=ShortConv(dim);self.k_conv=ShortConv(dim);self.v_conv=ShortConv(dim)
        self.alpha_down=nn.Linear(dim,self.hd,bias=False);self.alpha_up=nn.Linear(self.hd,dim,bias=False)
        self.gate_down=nn.Linear(dim,self.hd,bias=False);self.gate_up=nn.Linear(self.hd,dim,bias=False)
        self.beta_proj=nn.Linear(dim,heads,bias=False)
        self.A_log=nn.Parameter(torch.zeros(heads));self.dt_bias=nn.Parameter(torch.full((heads,self.hd),-3.))
        self.out_norm=RMSNorm(self.hd);self.out_proj=nn.Linear(dim,dim,bias=False)
    def forward(self,x,state=None,backend='reference'):
        b,t,d=x.shape
        if not t:raise ValueError('empty scan; skip explicitly at caller')
        old=(None,None,None) if state is None else state.convolutions
        q,cq=self.q_conv(self.q_proj(x),old[0]);k,ck=self.k_conv(self.k_proj(x),old[1]);v,cv=self.v_conv(self.v_proj(x),old[2])
        q=q.reshape(b,t,self.heads,self.hd);k=k.reshape_as(q);v=v.reshape_as(q)
        # Fixed epsilon, explicit pre-normalization shared by CPU and FLA.
        q=F.normalize(q.float(),dim=-1,eps=1e-6).to(x.dtype);k=F.normalize(k.float(),dim=-1,eps=1e-6).to(x.dtype)
        g=-self.A_log.float().exp()[None,None,:,None]*F.softplus(self.alpha_up(self.alpha_down(x)).float().reshape_as(q)+self.dt_bias.float())
        beta=self.beta_proj(x).float().sigmoid()
        fn={'reference':recurrence,'chunk':chunk_recurrence,'fla':official_fla}.get(backend)
        if fn is None:raise ValueError('unknown KDA backend')
        out,s=fn(q,k,v,g,beta,None if state is None else state.recurrent)
        gate=self.gate_up(self.gate_down(x)).reshape_as(out).sigmoid()
        out=(self.out_norm(out)*gate).reshape(b,t,d)
        return self.out_proj(out),StreamState(s,(cq,ck,cv))


class KDABlock(nn.Module):
    def __init__(self,dim,heads,hidden):
        super().__init__();self.norm1=RMSNorm(dim);self.mixer=KDAMixer(dim,heads);self.norm2=RMSNorm(dim)
        self.ff_gate=nn.Linear(dim,hidden,bias=False);self.ff_value=nn.Linear(dim,hidden,bias=False);self.ff_out=nn.Linear(hidden,dim,bias=False)
    def finish(self,x,mixed):
        x=x+mixed;z=self.norm2(x)
        return x+self.ff_out(F.silu(self.ff_gate(z))*self.ff_value(z))
    def stream(self,x,state=None,backend='reference'):
        mixed,state=self.mixer(self.norm1(x),state,backend)
        return self.finish(x,mixed),state
    def audio_prefix(self,audio,audio_valid,backend='reference'):
        """Per-example mixer state after scanning that example's valid audio.

        Audio rows themselves never receive this block's residual (frozen audio
        stream); the state only seeds the ordered prefix scan of later rows.
        Depends on frozen audio and THIS block's weights, never on labels.
        """
        states=[]
        for b in range(audio.shape[0]):
            z=audio[b:b+1].index_select(1,torch.nonzero(audio_valid[b]).squeeze(1))
            states.append(self.mixer(self.norm1(z),backend=backend)[1] if z.shape[1] else None)
        return states
    def routed(self,x,lengths,valid,prefix,backend='reference'):
        """Two shared-weight scans over trainable rows x=[G,R,I,W,E].

        Scan 1: R then I, emitting R and I rows. Scan 2: continues from the
        audio prefix state, then G,R,I,W,E, emitting G, W and E rows. A causal
        scan's output at a row depends only on earlier rows, so this equals the
        separate per-class routes R; R->I; A->G; A->G->R->I->W; A->...->E while
        sharing each prefix once. R/I never see audio, G, workspace or targets.
        Per-example compact valid positions: pad advances no conv/recurrence.
        """
        g,r,i,w,e=lengths;off=[0,g,g+r,g+r+i,g+r+i+w,g+r+i+w+e]
        def cls(j):return torch.zeros(off[-1],dtype=torch.bool,device=x.device).index_fill_(0,torch.arange(off[j],off[j+1],device=x.device),True)
        first=cls(1)|cls(2);emit2=cls(0)|cls(3)|cls(4)
        rows=[];event_states=[]
        # index_select/index_add only: advanced-index backward lacks BF16 on MPS.
        # The two scans emit disjoint rows, so accumulation onto zeros is assignment.
        for b in range(x.shape[0]):
            y=torch.zeros_like(x[b:b+1]);xb=x[b:b+1]
            for scan,emit,state in ((first,first,None),(torch.ones_like(first),emit2,prefix[b])):
                ids=torch.nonzero(scan&valid[b]).squeeze(1)
                if not ids.numel():continue
                mixed,out_state=self.mixer(self.norm1(xb.index_select(1,ids)),state,backend)
                take=torch.nonzero(emit[ids]).squeeze(1)
                if take.numel():
                    target=ids.index_select(0,take)
                    y=y.index_add(1,target,self.finish(xb.index_select(1,target),mixed.index_select(1,take)))
            rows.append(y);event_states.append(out_state)
        # Event suffix always has BOS, so scan 2 always ran and holds the E state.
        state=StreamState(torch.cat([s.recurrent for s in event_states],0),tuple(torch.cat([s.convolutions[j] for s in event_states],0) for j in range(3)))
        return torch.cat(rows,0),state
