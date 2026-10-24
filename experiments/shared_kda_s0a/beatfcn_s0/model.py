"""Bounded shared Pupu24/KDA4 scaffold, not the complete event decoder."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .pupu import FrozenLargeTeacher, load_teacher, make_rope, apply_rope
from .kda import KDABlock

@dataclass(frozen=True)
class Config:
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    kda_heads: int = 8
    insert_after: tuple = (6,12,18,24)
    ffn_hidden: int = 4096
    feature_dim: int = 8
    intent_tokens: int = 8
    local_max_frames: int = 1024  # S0 ONE native <=10.24s window; no implicit crop
    global_max_tokens: int = 600
    max_groups: int = 256
    max_prefix: int = 1536
    density_enabled: bool = True  # density control toggle: True=follow density range, False=freely generate

@dataclass
class Context:
    """Already legal, caller-supplied features. Hidden labels NEVER belong here.

    local_mel normalized Pupu mel [B,1,T,128]; global_music is bounded
    audio-only raw feature summary [B,N,dim], NOT chart label summaries.
    reference encodes paired reference evidence (incl reference audio upstream).
    known holds observed/controls/scene features only. Provenance/semantic
    validation and real audio summary/reference builders are NOT implemented.
    """
    local_mel: torch.Tensor
    global_music: torch.Tensor
    reference: torch.Tensor
    known: torch.Tensor
    audio_valid: torch.Tensor | None = None
    global_valid: torch.Tensor | None = None
    reference_valid: torch.Tensor | None = None
    known_valid: torch.Tensor | None = None
    # Control parameters (wired up from request controls)
    density_enabled: bool = True
    density_notes_per_second: tuple[float, float] | None = None
    hold_head_ratio: tuple[float, float] | None = None
    allowed_types: list[str] | None = None
    reference_strength: float = 0.8

    def fingerprint(self):
        h = hashlib.sha256()
        for name in self.__dataclass_fields__:
            x=getattr(self,name)
            h.update(name.encode())
            if x is not None:
                x=x.detach().cpu().contiguous()
                h.update(str((tuple(x.shape),x.dtype)).encode())
                h.update(x.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

@dataclass
class AudioMemory:
    """Frozen-backbone local audio stream, reusable across steps and requests.

    Local audio rows read only valid local audio and receive neither modality
    nor KDA residuals, so every layer equals the pretrained Pupu forward: a pure
    function of (mel, audio_valid, frozen backbone weights). It is built under
    no_grad once and read as keys by trainable rows in training AND inference;
    trainable-parameter updates never stale it. Any backbone change (load,
    LoRA, in-place edit) changes `key` and is rejected, never silently reused.
    kda_prefix holds inference-only per-block KDA states after the audio
    prefix, keyed by that block's parameter fingerprint.
    """
    key: str
    kv: list  # per layer post-RoPE (k,v) [B,H,A,hd]
    taps: list  # hidden entering each KDA insertion [B,A,D]
    final: torch.Tensor  # final LayerNorm hidden [B,A,D]
    valid: torch.Tensor  # [B,A]
    kda_prefix: dict = field(default_factory=dict)

    def nbytes(self):
        tensors=[t for kv in self.kv for t in kv]+self.taps+[self.final]
        return sum(t.numel()*t.element_size() for t in tensors)

@dataclass
class DecodeCache:
    context_hash: str
    revision: int
    kv: list
    kda: list
    events: int
    parameter_versions: tuple
    backend: str = 'reference'


def audio_mask(valid):
    """Audio reads valid audio only; invalid queries get harmless self attention."""
    n=valid.shape[1];eye=torch.eye(n,dtype=torch.bool,device=valid.device)[None,None]
    return torch.where(valid[:,None,:,None],valid[:,None,None,:].expand(-1,1,n,-1),eye)


def prefix_mask(n_audio,lengths,audio_valid,valid):
    """True=allowed SDPA for trainable queries [G,R,I,W,E] over keys [A|G,R,I,W,E].

    G->A/G; R->R; I->R/I; W->A/G/R/I/W; E->A/G/R/I/W + causal E. Audio queries
    are not here: they read audio only (audio_mask) and never any trainable row.
    """
    g,r,i,w,e=lengths;n=sum(lengths);p=g+r+i+w;o=n_audio;dev=valid.device
    mask=torch.zeros(n,o+n,dtype=torch.bool,device=dev)
    mask[:g,:o+g]=True
    mask[g:g+r,o+g:o+g+r]=True
    mask[g+r:g+r+i,o+g:o+g+r+i]=True
    mask[g+r+i:,:o+p]=True
    mask[p:,o+p:]=torch.ones(e,e,dtype=torch.bool,device=dev).tril()
    mask=mask[None,None]&torch.cat([audio_valid,valid],1)[:,None,None,:]
    own=torch.zeros(n,o+n,dtype=torch.bool,device=dev);own[torch.arange(n),o+torch.arange(n)]=True
    return torch.where(valid[:,None,:,None],mask,own[None,None])


def shared_attention(block, x, rope, mask, old_kv=None):
    """One module and one QKV/FFN parameter set for audio AND event rows."""
    z=block.norm1(x); b,n,d=z.shape; h=block.attn.heads
    q,k,v=block.attn.qkv(z).reshape(b,n,3,h,-1).permute(2,0,3,1,4).unbind(0)
    if rope is not None:q,k=apply_rope(q,rope),apply_rope(k,rope)  # None=identity
    if old_kv is not None:
        k=torch.cat([old_kv[0],k],dim=2); v=torch.cat([old_kv[1],v],dim=2)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
    x=x+block.attn.proj(out.transpose(1,2).reshape(b,n,d))
    return x+block.mlp(block.norm2(x)), (k,v)


class SharedModel(nn.Module):
    def __init__(self,cfg=Config(),backbone=None):
        super().__init__(); self.cfg=cfg
        self.backbone=backbone if backbone is not None else FrozenLargeTeacher(cfg.dim,cfg.depth,cfg.heads)
        self.backbone.requires_grad_(False)
        self.kda_blocks=nn.ModuleList(KDABlock(cfg.dim,cfg.kda_heads,cfg.ffn_hidden) for _ in cfg.insert_after)
        self.group_embed=nn.Sequential(nn.Linear(cfg.feature_dim,cfg.dim),nn.SiLU(),nn.Linear(cfg.dim,cfg.dim))
        self.reference_embed=nn.Linear(cfg.feature_dim,cfg.dim)
        self.known_embed=nn.Linear(cfg.feature_dim,cfg.dim)
        self.intent=nn.Parameter(torch.randn(cfg.intent_tokens,cfg.dim)*.02)
        self.bos=nn.Parameter(torch.randn(1,1,cfg.dim)*.02)
        self.modality=nn.Embedding(5,cfg.dim)
        self.event_position=nn.Embedding(cfg.max_groups+1,cfg.dim)
        # Bounded outer-step scaffold only. Full GRU/hold/COPY_TIME/Phira deferred.
        self.outer_action=nn.Linear(cfg.dim,2)  # END_GAP / NEW_GROUP
        self.time_coarse=nn.Linear(cfg.dim,800)  # 8s @10ms
        self.time_fine=nn.Linear(cfg.dim,10)
        self.selection_query=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.selection_key=nn.Linear(cfg.dim,cfg.dim,bias=False)
        self.selection_readout=nn.Linear(cfg.dim,1)
        self.revision=0

    @classmethod
    def from_checkpoint(cls,path,*,dtype=torch.bfloat16,device='cpu'):
        # Allocate new modules directly in chosen dtype/device, no huge random base.
        base=load_teacher(path,dtype=dtype,device=device)
        with torch.device(device):
            old=torch.get_default_dtype(); torch.set_default_dtype(dtype)
            try: model=cls(Config(),base)
            finally: torch.set_default_dtype(old)
        return model

    def mark_updated(self):
        """Call after ANY optimizer step/load/edit. Inference caches never survive."""
        self.revision+=1

    def backbone_fingerprint(self):
        return tuple((id(p),p._version,p.dtype,str(p.device)) for p in self.backbone.parameters())

    def audio_key(self,mel,valid):
        h=hashlib.sha256(repr(self.backbone_fingerprint()).encode())
        for x in (mel,valid):
            x=x.detach().cpu().contiguous();h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def _audio_valid(self,mel,audio_valid):
        cfg=self.cfg
        if mel.ndim!=4 or mel.shape[1]!=1 or mel.shape[-1]!=128 or mel.shape[2]<4 or mel.shape[2]%4:
            raise ValueError('S0 requires [B,1,T,128], T>=4, multiple of4; explicit padding upstream')
        if mel.shape[2]>cfg.local_max_frames:raise ValueError('resource_exhausted: bounded audio context')
        n=mel.shape[2]//4*8
        if audio_valid is None:return torch.ones(mel.shape[0],n,dtype=torch.bool,device=mel.device)
        if audio_valid.shape!=(mel.shape[0],n) or audio_valid.dtype!=torch.bool:raise ValueError('invalid mask shape/dtype')
        return audio_valid

    @torch.no_grad()
    def encode_audio(self,mel,audio_valid=None):
        """Frozen local audio stream once per window; see AudioMemory."""
        valid=self._audio_valid(mel,audio_valid)
        x=self.backbone.patch_embed(mel)*valid[...,None]
        rope=make_rope((mel.shape[2]//4,8),self.cfg.dim//self.cfg.heads,device=mel.device)
        mask=audio_mask(valid);kv=[];taps=[]
        for l,block in enumerate(self.backbone.teacher.blocks,1):
            x,pair=shared_attention(block,x,rope,mask);x=x*valid[...,None];kv.append(pair)
            if l in self.cfg.insert_after:taps.append(x)
        final=self.backbone.teacher.norm(x)*valid[...,None]
        return AudioMemory(self.audio_key(mel,valid),kv,taps,final,valid)

    def _memory(self,c,audio):
        valid=self._audio_valid(c.local_mel,c.audio_valid)
        if audio is None:return self.encode_audio(c.local_mel,valid)
        if audio.key!=self.audio_key(c.local_mel,valid):raise ValueError('stale audio memory: mel/mask/backbone changed; encode again')
        return audio

    def _kda_prefix(self,memory,j,backend):
        """Audio-prefix KDA states; recomputed with grad in training, memoized
        in memory.kda_prefix only when autograd is off (fixed weights)."""
        block=self.kda_blocks[j]
        if torch.is_grad_enabled():return block.audio_prefix(memory.taps[j],memory.valid,backend)
        key=(j,backend,self.revision,tuple((id(p),p._version) for p in block.parameters()))
        if key not in memory.kda_prefix:
            for old in [k for k in memory.kda_prefix if k[0]==j]:del memory.kda_prefix[old]
            memory.kda_prefix[key]=block.audio_prefix(memory.taps[j],memory.valid,backend)
        return memory.kda_prefix[key]

    def _prepare(self,c,feedback):
        cfg=self.cfg; b=feedback.shape[0]; device=feedback.device
        if c.global_music.shape[1]>cfg.global_max_tokens: raise ValueError('resource_exhausted: bounded audio context')
        if feedback.shape[1]>cfg.max_groups: raise ValueError('resource_exhausted: groups')
        r=self.reference_embed(c.reference); w=self.known_embed(c.known)
        intent=self.intent[None].expand(b,-1,-1)
        # Caller provides ONLY completed previous groups; output len=previous+1.
        e=torch.cat([self.bos.expand(b,-1,-1),self.group_embed(feedback)],dim=1)
        e=e+self.event_position(torch.arange(e.shape[1],device=device))[None]
        chunks=[c.global_music,r,intent,w,e]; lengths=[x.shape[1] for x in chunks]
        if sum(lengths[:-1])>cfg.max_prefix: raise ValueError('resource_exhausted: prefix; do not truncate')
        x=torch.cat([z+self.modality.weight[j] for j,z in enumerate(chunks)],dim=1)
        def valid(v,n):
            if v is None:return torch.ones(b,n,dtype=torch.bool,device=device)
            if v.shape!=(b,n) or v.dtype!=torch.bool: raise ValueError('invalid mask shape/dtype')
            return v
        vm=torch.cat([valid(c.global_valid,lengths[0]),valid(c.reference_valid,r.shape[1]),valid(None,intent.shape[1]),valid(c.known_valid,w.shape[1]),valid(None,e.shape[1])],dim=1)
        # Abstract summaries/events use identity RoPE (no fictitious coords),
        # learned event index and attribute-derived embeddings.
        return x*vm[...,None],lengths,vm

    def forward(self,c:Context,previous_groups:torch.Tensor,*,audio:AudioMemory|None=None,return_cache=False,backend='reference'):
        """audio: optional reusable AudioMemory for c.local_mel; built if None."""
        if return_cache and backend=='fla':raise ValueError('S0 incremental cache validated only for reference/chunk backends')
        memory=self._memory(c,audio)
        if memory.valid.shape[0]!=previous_groups.shape[0]:raise ValueError('audio memory batch mismatch')
        x,lengths,valid=self._prepare(c,previous_groups)
        na=memory.valid.shape[1];mask=prefix_mask(na,lengths,memory.valid,valid);kvs=[];states=[];j=0
        for l,block in enumerate(self.backbone.teacher.blocks,1):
            # NEVER no_grad here: frozen originals still propagate input/KDA gradients.
            x,kv=shared_attention(block,x,None,mask,memory.kv[l-1]); x=x*valid[...,None]; kvs.append(kv)
            if l in self.cfg.insert_after:
                x,state=self.kda_blocks[j].routed(x,lengths,valid,self._kda_prefix(memory,j,backend),backend=backend)
                states.append(state);j+=1
        h=torch.cat([memory.final,self.backbone.teacher.norm(x)*valid[...,None]],1)
        g,r,i,w,e=lengths; a=na+g; p=a+r+i+w; full=torch.cat([memory.valid,valid],1)
        intent=h[:,a+r:a+r+i]
        # Only target music + isolated intent; no target/event hidden inputs.
        q=self.selection_query(h[:,:a]);k=self.selection_key(intent)
        choice=(q@k.transpose(-1,-2))/(self.cfg.dim**.5)
        selected=choice.softmax(-1)@intent
        selection=self.selection_readout(selected).squeeze(-1)*full[:,:a]
        local_selection=selection[:,:na].reshape(selection.shape[0],-1,8).mean(-1)
        out={'selection_cells':local_selection,'action':self.outer_action(h[:,p:]),'coarse':self.time_coarse(h[:,p:]),'fine':self.time_fine(h[:,p:]),'hidden':h,'intent':intent,'selection':selection,'lengths':(a,r,i,w,e),'audio':memory}
        if return_cache:
            if self.training or torch.is_grad_enabled():raise ValueError('cache is inference-only; use eval()+no_grad()')
            # Store legal mask: invalid prefix keys must remain invisible on steps.
            out['cache']=DecodeCache(c.fingerprint(),self.revision,[(k,v,full.clone()) for k,v in kvs],states,e,tuple((id(p),p._version) for p in self.parameters()),backend)
        return out

    def step(self,c:Context,completed_group:torch.Tensor,cache:DecodeCache):
        if self.training or torch.is_grad_enabled():raise ValueError('step is inference-only')
        if cache.context_hash!=c.fingerprint() or cache.revision!=self.revision or cache.parameter_versions!=tuple((id(p),p._version) for p in self.parameters()):
            raise ValueError('stale cache: context/weights changed; prefill again')
        if cache.events>self.cfg.max_groups:raise ValueError('resource_exhausted: groups')
        x=self.group_embed(completed_group[:,None])+self.modality.weight[4]+self.event_position.weight[cache.events]
        newkv=[];states=[];j=0
        for l,block in enumerate(self.backbone.teacher.blocks,1):
            k,v,valid=cache.kv[l-1];valid=torch.cat([valid,torch.ones(x.shape[0],1,dtype=torch.bool,device=x.device)],dim=1)
            x,(k,v)=shared_attention(block,x,None,valid[:,None,None,:],(k,v));newkv.append((k,v,valid))
            if l in self.cfg.insert_after:
                x,state=self.kda_blocks[j].stream(x,cache.kda[j],cache.backend);states.append(state);j+=1
        h=self.backbone.teacher.norm(x)
        return {'action':self.outer_action(h),'coarse':self.time_coarse(h),'fine':self.time_fine(h),'cache':DecodeCache(cache.context_hash,self.revision,newkv,states,cache.events+1,cache.parameter_versions,cache.backend)}

    def inventory(self):
        modules={}
        for name,p in self.named_parameters():
            group='backbone' if name.startswith('backbone.') else ('.'.join(name.split('.')[:2]) if name.startswith('kda_blocks.') else name.split('.')[0])
            z=modules.setdefault(group,{'total':0,'trainable':0,'frozen':0})
            z['total']+=p.numel();z['trainable' if p.requires_grad else 'frozen']+=p.numel()
        total=sum(p.numel() for p in self.parameters())
        return {'identity':'shared-pupu24-kda4-s0-v1','modules':modules,'total':total,'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),'frozen':sum(p.numel() for p in self.parameters() if not p.requires_grad),'bf16_weight_bytes':2*total,'fp32_weight_bytes':4*total}
