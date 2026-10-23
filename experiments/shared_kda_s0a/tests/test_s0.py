import copy
import unittest
import torch
from torch.nn import functional as F
from beatfcn_s0.model import SharedModel,Config,Context
from beatfcn_s0.kda import recurrence,chunk_recurrence,KDAMixer,KDABlock

torch.set_num_threads(2)

def fixture(batch=1):
    torch.manual_seed(7)
    cfg=Config(dim=32,depth=24,heads=4,kda_heads=4,ffn_hidden=128,intent_tokens=2)
    m=SharedModel(cfg)
    c=Context(torch.randn(batch,1,4,128),torch.randn(batch,2,32),torch.randn(batch,3,8),torch.randn(batch,2,8))
    y=torch.randn(batch,3,8)
    return m,c,y

class S0Tests(unittest.TestCase):
    def test_exact_inventory(self):
        with torch.device('meta'):m=SharedModel()
        inv=m.inventory()
        self.assertEqual(inv['frozen'],403141632)
        self.assertEqual(inv['total'],476739917)
        for n in range(4):self.assertEqual(inv['modules'][f'kda_blocks.{n}']['total'],17328264)

    def test_recurrence_matches_independent_matrix_equation_and_grad(self):
        torch.manual_seed(3)
        q,k,v=[torch.randn(2,5,2,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        qn=F.normalize(q,dim=-1);kn=F.normalize(k,dim=-1)
        g=-F.softplus(torch.randn_like(q));beta=torch.rand(2,5,2,dtype=torch.float64)
        out,s=recurrence(qn,kn,v,g,beta)
        state=torch.zeros(2,2,4,4,dtype=torch.float64);outs=[]
        for t in range(5):
            # Independent matrix form (I-beta kkT) diag(alpha) S + beta kvT.
            kt=kn[:,t]; vt=v[:,t]; eye=torch.eye(4,dtype=torch.float64)
            delta=eye-beta[:,t,:,None,None]*kt[...,None]*kt[...,None,:]
            state=delta@torch.diag_embed(g[:,t].exp())@state+beta[:,t,:,None,None]*kt[...,None]*vt[...,None,:]
            outs.append((qn[:,t,None,:] if False else qn[:,t].unsqueeze(-2)@state).squeeze(-2)/2)
        ref=torch.stack(outs,1)
        torch.testing.assert_close(out,ref,atol=1e-12,rtol=1e-12)
        a=torch.autograd.grad(out.square().sum(),(q,k,v),retain_graph=True)
        b=torch.autograd.grad(ref.square().sum(),(q,k,v))
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)

    def test_kda_streaming_conv_state(self):
        torch.manual_seed(3);m=KDAMixer(32,4);x=torch.randn(2,7,32)
        whole,sw=m(x);state=None;parts=[]
        for t in range(7):
            y,state=m(x[:,t:t+1],state);parts.append(y)
        torch.testing.assert_close(torch.cat(parts,1),whole,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(state.recurrent,sw.recurrent,atol=1e-6,rtol=1e-5)

    def test_shared_weight_identity_and_all_new_gradients(self):
        m,c,y=fixture();seen={};handles=[]
        for j,b in enumerate(m.backbone.teacher.blocks):
            def hook(mod,args,out,j=j):seen.setdefault(j,[]).append((id(mod.weight),args[0].shape[1]))
            handles.append(b.attn.qkv.register_forward_hook(hook))
        out=m(c,y)
        loss=sum(out[n].square().mean() for n in ['action','coarse','fine','selection'])
        loss.backward()
        for j,entries in seen.items():
            # Frozen audio stream (8 patches) and trainable rows (13) use one module.
            self.assertEqual([n for _,n in entries],[8,13])
            self.assertTrue(all(w==id(m.backbone.teacher.blocks[j].attn.qkv.weight) for w,_ in entries))
        missing=[]
        for n,p in m.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:missing.append(n)
            else:self.assertIsNone(p.grad,n)
        self.assertEqual(missing,[])
        for h in handles:h.remove()

    def test_hidden_future_no_leak_and_workspace_isolation(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            before=m(c,y);y2=y.clone();y2[:,1:]+=100;after=m(c,y2)
            for n in ['intent','selection']:torch.testing.assert_close(before[n],after[n],atol=0,rtol=0)
            for n in ['action','coarse','fine']:torch.testing.assert_close(before[n][:,:2],after[n][:,:2],atol=0,rtol=0)
            c2=copy.deepcopy(c);c2.known+=500;c2.global_music-=20;c2.local_mel+=10
            changed=m(c2,y)
            torch.testing.assert_close(before['intent'],changed['intent'],atol=0,rtol=0)
            a,r,i,w,e=before['lengths']
            torch.testing.assert_close(before['hidden'][:,a:a+r+i],changed['hidden'][:,a:a+r+i],atol=0,rtol=0)
            self.assertGreater((before['action']-changed['action']).abs().max().item(),1e-5)
        # Forbidden Jacobian: intent cannot use known/workspace or suffix.
        c.known.requires_grad_();y.requires_grad_();out=m(c,y)
        gy,gk=torch.autograd.grad(out['intent'].square().sum(),(y,c.known),allow_unused=True)
        for g in (gy,gk):self.assertTrue(g is None or torch.count_nonzero(g)==0)

    def test_step_teacher_forcing_and_context_invalidation(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            full=m(c,y);first=m(c,y[:,:0],return_cache=True);cache=first['cache']
            for n in ['action','coarse','fine']:torch.testing.assert_close(first[n],full[n][:,:1],atol=3e-6,rtol=3e-5)
            for t in range(y.shape[1]):
                out=m.step(c,y[:,t],cache);cache=out['cache']
                for n in ['action','coarse','fine']:torch.testing.assert_close(out[n],full[n][:,t+1:t+2],atol=3e-6,rtol=3e-5)
            c2=copy.deepcopy(c);c2.known[:,0,0]+=1
            with self.assertRaisesRegex(ValueError,'stale cache'):m.step(c2,y[:,0],cache)
            fresh=m(c2,y[:,:0],return_cache=True);fresh2=m(c2,y[:,:0],return_cache=True)
            torch.testing.assert_close(fresh['action'],fresh2['action'],atol=0,rtol=0)
            m.mark_updated()
            with self.assertRaisesRegex(ValueError,'stale cache'):m.step(c2,y[:,0],fresh['cache'])

    def test_cache_model_ownership_and_automatic_update_detection(self):
        m,c,y=fixture();m.eval();other=SharedModel(m.cfg).eval()
        with torch.no_grad():
            cache=m(c,y[:,:0],return_cache=True)['cache']
            with self.assertRaisesRegex(ValueError,'stale cache'):other.step(c,y[:,0],cache)
            m.bos.add_(.1)
            with self.assertRaisesRegex(ValueError,'stale cache'):m.step(c,y[:,0],cache)

    def test_padding_and_batch_reset(self):
        m,c,y=fixture(batch=2);m.eval()
        c.reference_valid=torch.tensor([[True,False,True],[True,True,True]])
        c.audio_valid=torch.ones(2,8,dtype=torch.bool);c.audio_valid[0,-1]=False
        with torch.no_grad():
            a=m(c,y);cp=copy.deepcopy(c);cp.reference[0,1]=99999
            # Patch is independent convolution tile; masked patch must not advance KDA.
            cp.local_mel[0,:,:,112:]=99999;b=m(cp,y)
            torch.testing.assert_close(a['action'][0],b['action'][0],atol=0,rtol=0)
            single=Context(**{n:(None if getattr(c,n) is None else getattr(c,n)[1:]) for n in c.__dataclass_fields__})
            one=m(single,y[1:])
            torch.testing.assert_close(one['action'],a['action'][1:],atol=3e-6,rtol=3e-5)

    def test_global_future_music_visible_bounded(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            a=m(c,y);c.global_music[:,1]+=torch.randn(32)*10;b=m(c,y)
        self.assertGreater((a['action']-b['action']).abs().max().item(),1e-5)
        c.global_music=torch.zeros(1,601,32)
        with self.assertRaisesRegex(ValueError,'bounded audio'):m(c,y)

    def test_chunk_recurrence_exact_forward_state_grad(self):
        torch.manual_seed(5)
        for t,chunk,scale in [(37,16,1.),(64,16,8.),(5,8,1.)]:
            q,k,v,raw=[torch.randn(2,t,3,8,dtype=torch.float64,requires_grad=True) for _ in range(4)]
            beta=torch.rand(2,t,3,dtype=torch.float64,requires_grad=True);s0=torch.randn(2,3,8,8,dtype=torch.float64,requires_grad=True)
            run=lambda fn,**kw:fn(F.normalize(q,dim=-1),F.normalize(k,dim=-1),v,-F.softplus(raw)*scale,beta,s0,**kw)
            (o1,s1),(o2,s2)=run(recurrence),run(chunk_recurrence,chunk=chunk)
            torch.testing.assert_close(o2,o1,atol=1e-12,rtol=1e-12);torch.testing.assert_close(s2,s1,atol=1e-12,rtol=1e-12)
            w=torch.randn_like(o1);ws=torch.randn_like(s1);xs=(q,k,v,raw,beta,s0)
            for x,y in zip(torch.autograd.grad((o1*w).sum()+(s1*ws).sum(),xs),torch.autograd.grad((o2*w).sum()+(s2*ws).sum(),xs)):
                torch.testing.assert_close(y,x,atol=1e-11,rtol=1e-11)

    def test_two_scan_routing_equals_independent_routes(self):
        torch.manual_seed(9);blk=KDABlock(32,4,64).double()
        lengths=(2,3,2,2,4);audio=torch.randn(2,6,32,dtype=torch.float64);x=torch.randn(2,13,32,dtype=torch.float64)
        av=torch.tensor([[1,1,0,1,1,1],[1,1,1,1,1,1]],dtype=torch.bool);valid=torch.ones(2,13,dtype=torch.bool);valid[0,3]=False
        y,state=blk.routed(x,lengths,valid,blk.audio_prefix(audio,av))
        g,r,i,w,e=lengths;off=[0,g,g+r,g+r+i,g+r+i+w,13]
        # Spec routes, each a fresh scan: G:A,G  R:R  I:R,I  W:A,G,R,I,W  E:A,...,E
        routes={0:(True,[0]),1:(False,[1]),2:(False,[1,2]),3:(True,[0,1,2,3]),4:(True,[0,1,2,3,4])}
        for b in range(2):
            for role,(use_audio,classes) in routes.items():
                ids=[t for c in classes for t in range(off[c],off[c+1]) if valid[b,t]]
                z=torch.cat([audio[b:b+1,av[b]] if use_audio else audio[b:b+1,:0],x[b:b+1,ids]],1)
                mixed,last=blk.mixer(blk.norm1(z))
                own=[n for n,t in enumerate(ids) if off[role]<=t<off[role+1]]
                rows=[ids[n] for n in own];shift=z.shape[1]-len(ids)
                # Mixer rounds q/k/gates through FP32 by design; BLAS shape-dependent
                # last bits can flip one FP32 rounding, hence 1e-6 not 1e-12.
                torch.testing.assert_close(y[b:b+1,rows],blk.finish(x[b:b+1,rows],mixed[:,[shift+n for n in own]]),atol=1e-6,rtol=1e-6)
                if role==4:torch.testing.assert_close(state.recurrent[b:b+1],last.recurrent,atol=1e-6,rtol=1e-6)
            self.assertEqual(torch.count_nonzero(y[0,3]).item(),0)

    def test_audio_stream_is_exact_pretrained_pupu(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            mem=m.encode_audio(c.local_mel);ref=m.backbone(c.local_mel,return_tokens=True)
        torch.testing.assert_close(mem.final.reshape(ref.shape),ref,atol=0,rtol=0)

    def test_audio_memory_reused_across_updates_and_requests(self):
        m,c,y=fixture();mem=m.encode_audio(c.local_mel)
        self.assertFalse(any(t.requires_grad for kv in mem.kv for t in kv))
        out=m(c,y,audio=mem);sum(out[n].square().mean() for n in ['action','coarse','fine','selection']).backward()
        with torch.no_grad():
            for prm in m.parameters():
                if prm.requires_grad:prm.add_(prm.grad,alpha=-.5)
        m.mark_updated();m.eval()
        with torch.no_grad():
            fresh=m.encode_audio(c.local_mel)
            for (k0,v0),(k1,v1) in zip(mem.kv,fresh.kv):torch.testing.assert_close(k0,k1,atol=0,rtol=0);torch.testing.assert_close(v0,v1,atol=0,rtol=0)
            reused=m(c,y,audio=mem);rebuilt=m(c,y)
            for n in ['action','coarse','fine','selection','intent']:torch.testing.assert_close(reused[n],rebuilt[n],atol=0,rtol=0)
            # Changing only workspace/reference keeps the same audio memory valid.
            c2=copy.deepcopy(c);c2.known+=3;c2.reference-=1;m(c2,y,audio=mem)
            c3=copy.deepcopy(c);c3.local_mel[:,:,0,0]+=1
            with self.assertRaisesRegex(ValueError,'stale audio memory'):m(c3,y,audio=mem)
            m.backbone.teacher.blocks[0].attn.qkv.weight.add_(0)
            with self.assertRaisesRegex(ValueError,'stale audio memory'):m(c,y,audio=mem)

    def test_inference_kda_prefix_memo_and_invalidation(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            mem=m.encode_audio(c.local_mel);first=m(c,y,audio=mem);self.assertEqual(len(mem.kda_prefix),4)
            again=m(c,y,audio=mem);torch.testing.assert_close(first['action'],again['action'],atol=0,rtol=0)
            m.kda_blocks[1].mixer.A_log.add_(.3);changed=m(c,y,audio=mem);clean=m(c,y)
            torch.testing.assert_close(changed['action'],clean['action'],atol=0,rtol=0);self.assertEqual(len(mem.kda_prefix),4)
        out=m(c,y,audio=mem)  # autograd on: fresh prefix with grad, memo untouched
        self.assertTrue(out['action'].requires_grad)

    def test_chunk_backend_matches_reference_model_and_decode(self):
        m,c,y=fixture();m.eval()
        with torch.no_grad():
            ref=m(c,y);chk=m(c,y,backend='chunk')
            for n in ['action','coarse','fine','selection']:torch.testing.assert_close(chk[n],ref[n],atol=2e-5,rtol=2e-4)
            cache=m(c,y[:,:0],return_cache=True,backend='chunk')['cache']
            for t in range(y.shape[1]):
                out=m.step(c,y[:,t],cache);cache=out['cache']
                torch.testing.assert_close(out['action'],ref['action'][:,t+1:t+2],atol=2e-5,rtol=2e-4)

if __name__=='__main__':unittest.main(verbosity=2)
