"""E3-B budgeted token VIB pilot with sparse gates and a rate-target dual."""

from __future__ import annotations

import argparse, json, math, os, random, time
import numpy as np
from .config import ensure_dir, load_config
from .data import load_data
from .train_e3b import extract_tokens, make_batch, set_seed, cpu_clone


class GatedTokenVIB:
    def __init__(self, d_in, d, seed, logvar_init=-4.0):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.mu = nn.Linear(d_in, d); self.logvar = nn.Linear(d_in, d)
        nn.init.constant_(self.logvar.bias, logvar_init)
        self.gate = nn.Linear(d_in, 1); nn.init.constant_(self.gate.bias, 2.0)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 2)

    def to(self, device):
        for m in [self.mu, self.logvar, self.gate, self.head]:
            m.to(device)
        self.query.data = self.query.data.to(device); return self

    def parameters(self):
        return list(self.mu.parameters()) + list(self.logvar.parameters()) + list(self.gate.parameters()) + [self.query] + list(self.head.parameters())

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in ['mu', 'logvar', 'gate', 'head']} | {'query': self.query.detach().cpu()}

    def load_state_dict(self, s):
        for k in ['mu', 'logvar', 'gate', 'head']:
            getattr(self, k).load_state_dict(s[k])
        self.query.data = s['query'].to(self.query.device)

    def __call__(self, h, mask, mc=1, sample=True, gate_scale=None):
        import torch
        B, T, D = h.shape
        mu = self.mu(h); logvar = torch.clamp(self.logvar(h), -8., 8.)
        g = torch.sigmoid(self.gate(h)).squeeze(-1)  # [B,T]
        if gate_scale is not None:
            g = g * gate_scale.to(g.device).view(-1, 1)
        if sample:
            eps = torch.randn((mc, B, T, mu.shape[-1]), device=h.device)
            z = mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)
        else:
            z = mu.unsqueeze(0)
        z = z * g.unsqueeze(0).unsqueeze(-1)
        scores = (z @ self.query) / math.sqrt(z.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(0).bool(), torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, -1)
        pooled = (w.unsqueeze(-1) * z).sum(2)
        logits = self.head(pooled)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1) / math.log(2)
        rate = (g * kl * mask).sum(1)
        return logits, rate, g


def eval_model(m, tokens, y, batch, device, mc=16, target=None):
    import torch
    probs=[]; ces=[]; rates=[]
    with torch.no_grad():
        for st in range(0,len(tokens),batch):
            idx=list(range(st,min(st+batch,len(tokens))))
            h,mask=make_batch(tokens,idx,device); yb=torch.tensor(y[idx],device=device)
            logits,rate,_=m(h,mask,mc=mc,sample=True)
            if target is not None:
                scale=torch.clamp(torch.tensor(target,device=rate.device)/(rate+1e-12),max=1.0)
                logits,rate,_=m(h,mask,mc=mc,sample=True,gate_scale=scale)
            ce=torch.nn.functional.cross_entropy(logits.reshape(-1,2), yb.unsqueeze(0).expand(logits.shape[0],-1).reshape(-1), reduction='none').view(logits.shape[0],-1).mean(0)/math.log(2)
            ces.append(ce.mean().item()); rates.append(rate.mean().item()); probs.append(torch.softmax(logits,-1).mean(0).cpu().numpy())
    p=np.concatenate(probs,0)
    return float(np.mean(ces)), float(np.mean(rates)), float((p.argmax(1)==y).mean())


def train_target(tokens_tr,ytr,tokens_v,yv,tokens_te,yte,cfg,target,seed,device):
    import torch
    set_seed(seed)
    m=GatedTokenVIB(tokens_tr[0].shape[1],int(cfg['latent']),seed).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=float(cfg['lr']),weight_decay=float(cfg['weight_decay']))
    best=(float('inf'),None,-1)
    for ep in range(int(cfg['epochs'])):
        m.mu.train(); m.logvar.train(); m.gate.train(); m.head.train()
        order=list(range(len(tokens_tr))); random.shuffle(order)
        for st in range(0,len(order),int(cfg['batch_size'])):
            idx=order[st:st+int(cfg['batch_size'])]
            h,mask=make_batch(tokens_tr,idx,device); yb=torch.tensor(ytr[idx],device=device)
            opt.zero_grad(set_to_none=True)
            logits,rate,_=m(h,mask,mc=int(cfg['mc_train']),sample=True)
            # Hard per-example budget projection.
            scale=torch.clamp(torch.tensor(target,device=rate.device)/(rate.detach()+1e-12),max=1.0)
            logits,rate,_=m(h,mask,mc=int(cfg['mc_train']),sample=True,gate_scale=scale)
            ce=torch.nn.functional.cross_entropy(logits.reshape(-1,2), yb.unsqueeze(0).expand(logits.shape[0],-1).reshape(-1))/math.log(2)
            ce.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        ce_v,rate_v,_=eval_model(m,tokens_v,yv,int(cfg['eval_batch_size']),device,int(cfg['mc_eval']),target=target)
        if ce_v<best[0]-1e-9: best=(ce_v,cpu_clone(m.state_dict()),ep)
        if ep-best[2]>=int(cfg['patience']): break
    m.load_state_dict(best[1])
    ce_v,rate_v,_=eval_model(m,tokens_v,yv,int(cfg['eval_batch_size']),device,int(cfg['mc_eval']),target=target)
    ce_t,rate_t,acc=eval_model(m,tokens_te,yte,int(cfg['eval_batch_size']),device,int(cfg['mc_eval']),target=target)
    return {'best_epoch':best[2],'val_ce':ce_v,'val_rate':rate_v,'test_ce':ce_t,'test_rate':rate_t,'acc':acc,'state':best[1]}


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--backbone',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda:1'); ap.add_argument('--targets',default='1,2,3,5,50,100'); ap.add_argument('--seeds',default='0,1,2'); args=ap.parse_args(argv)
    import torch
    cfg=load_config(args.config); device=torch.device(args.device); out=ensure_dir(args.out)
    parts,_=load_data(cfg); tokens,_=extract_tokens(args.checkpoint,parts,cfg,device)
    ytr=np.array([r['label'] for r in parts['train']]); yv=np.array([r['label'] for r in parts['val']]); yte=np.array([r['label'] for r in parts['test']])
    for target in [float(x) for x in args.targets.split(',')]:
      for seed in [int(x) for x in args.seeds.split(',')]:
        r=train_target(tokens['train'],ytr,tokens['val'],yv,tokens['test'],yte,cfg,target,seed,device)
        rec={'run_id':f'{args.backbone}__budget{target:g}__s{seed}','backbone':args.backbone,'checkpoint':args.checkpoint,'target_rate':target,'seed':seed,**{k:v for k,v in r.items() if k!='state'}}
        with open(os.path.join(out,'runs.jsonl'),'a') as fh: fh.write(json.dumps(rec)+'\n')
        torch.save({'run_id':rec['run_id'],'state':r['state']},os.path.join(ensure_dir(os.path.join(out,'checkpoints')),rec['run_id']+'.pt'))
        print(rec)
    return 0
if __name__=='__main__': raise SystemExit(main())
