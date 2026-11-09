"""Deterministic token-level bottleneck controls for E3-B."""
import argparse, json, math, os, random
import numpy as np
from .config import ensure_dir, load_config
from .data import load_data
from .train_e3b import TokenVIB, extract_tokens, make_batch, set_seed, cpu_clone


def eval_det(m, tokens, y, batch, device):
    import torch
    m.mu.eval(); m.head.eval(); probs=[]; ces=[]
    with torch.no_grad():
        for start in range(0, len(tokens), batch):
            idx = list(range(start, min(start + batch, len(tokens))))
            h, mask = make_batch(tokens, idx, device); yb = torch.tensor(y[idx], device=device)
            logits, _ = m(h, mask, mc=1, sample=False)
            ces.append(float(torch.nn.functional.cross_entropy(logits[0], yb) / math.log(2)))
            probs.append(torch.softmax(logits[0], -1).cpu().numpy())
    p = np.concatenate(probs, 0)
    return float(np.mean(ces)), float((p.argmax(1) == y).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ['--config', '--backbone', '--checkpoint', '--out']:
        ap.add_argument(k, required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--poolings', default='last,mean,attention,last_gate')
    ap.add_argument('--latent', type=int, default=None)
    ap.add_argument('--identity', action='store_true')
    ap.add_argument('--freeze-mu', action='store_true')
    args = ap.parse_args(argv)
    import torch
    cfg = load_config(args.config); device = torch.device(args.device); out = ensure_dir(args.out)
    parts, _ = load_data(cfg)
    tokens, _ = extract_tokens(args.checkpoint, parts, cfg, device)
    ytr = np.array([r['label'] for r in parts['train']])
    yv = np.array([r['label'] for r in parts['val']])
    yte = np.array([r['label'] for r in parts['test']])
    for pooling in args.poolings.split(','):
      for seed in [0, 1, 2]:
        set_seed(seed)
        m = TokenVIB(tokens['train'][0].shape[1], int(args.latent or cfg['latent']), seed, pooling=pooling, identity_init=args.identity).to(device)
        if args.freeze_mu:
            m.mu.weight.requires_grad_(False); m.mu.bias.requires_grad_(False)
        opt = torch.optim.AdamW(m.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
        best = (float('inf'), None, -1)
        for ep in range(int(cfg['epochs'])):
            m.mu.train(); m.head.train()
            order = list(range(len(tokens['train']))); random.shuffle(order)
            for st in range(0, len(order), int(cfg['batch_size'])):
                idx = order[st:st + int(cfg['batch_size'])]
                h, mask = make_batch(tokens['train'], idx, device)
                yb = torch.tensor(ytr[idx], device=device)
                opt.zero_grad(set_to_none=True)
                logits, _ = m(h, mask, mc=1, sample=False)
                loss = torch.nn.functional.cross_entropy(logits[0], yb)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            val, _ = eval_det(m, tokens['val'], yv, int(cfg['eval_batch_size']), device)
            if val < best[0] - 1e-9:
                best = (val, cpu_clone(m.state_dict()), ep)
            if ep - best[2] >= int(cfg['patience']):
                break
        state = best[1]
        m.mu.load_state_dict(state['mu']); m.logvar.load_state_dict(state['logvar'])
        m.head.load_state_dict(state['head']); m.query.data = state['query'].to(device)
        if m.gate is not None and state.get('gate') is not None:
            m.gate.load_state_dict(state['gate'])
        ce, acc = eval_det(m, tokens['test'], yte, int(cfg['eval_batch_size']), device)
        rec = {'run_id': f'{args.backbone}__e3b_det__{pooling}__s{seed}',
               'backbone': args.backbone, 'checkpoint': args.checkpoint,
               'pooling': pooling, 'seed': seed, 'best_epoch': best[2],
               'val_ce': best[0], 'test_ce': ce, 'test_acc': acc}
        with open(os.path.join(out, 'runs.jsonl'), 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(rec)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
