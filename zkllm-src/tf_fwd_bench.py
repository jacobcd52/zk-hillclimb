#!/usr/bin/env python3
"""Plain fp8 forward-pass timing of the canonical layer at given dims.

The forward pass = the SAME op decomposition the composed proof covers:
7 Hawkeye Triton fp8 matmuls (Wq/Wk/Wv per token grid; QK^T and P.V per
(batch, head) instance) + the nonlinearities (rmsnorm, rope, causal softmax,
silu*up, residual adds) as standard torch bf16 GPU ops + per-row pow2 fp8
quantization at every quantize site.  CUDA-event timed over many iterations.

  python3 tf_fwd_bench.py <seq> <d> <nh> <dh> <dff> <batch> [iters] [batched]

batched (optional 8th arg = 1): BATCHED attention -- all batch*nh instances'
rope/QK^T/softmax/P.V run as single bmm/tensor ops (bf16 matmuls; the 7
projection GEMMs stay real Hawkeye fp8).  The per-instance python loop of the
canonical mode is LAUNCH-BOUND at large batch (32+ instances of tiny GEMMs),
which would understate a production forward; the batched mode is the honest
well-utilized baseline for the overhead headline.

Prints:  FWD seq=.. d=.. ... fwd_ms=<mean per-layer forward milliseconds>
"""
import sys, math, torch
sys.path.insert(0, '/workspace/projects/int-model-approximation/src')
from int_model_approximation.hawkeye import hawkeye_fp8_sum

def quant_rows(x):                       # per-row pow2 scale e4m3 quantization
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    e = torch.clamp(torch.floor(torch.log2(amax + 1e-30)) - 8, min=-126)
    scale = torch.pow(2.0, e)
    codes = (x.float() / scale).to(torch.float8_e4m3fn)
    return codes, scale.squeeze(-1)

def main():
    S, d, nh, dh, dff, B = map(int, sys.argv[1:7])
    iters = int(sys.argv[7]) if len(sys.argv) > 7 else 50
    batched = len(sys.argv) > 8 and sys.argv[8] == '1'
    dev = 'cuda'
    torch.manual_seed(7)
    T = S * B
    x = (torch.randn(T, d, device=dev) * 0.7).to(torch.bfloat16)
    g1 = torch.rand(d, device=dev).to(torch.bfloat16) + 0.5
    g2 = torch.rand(d, device=dev).to(torch.bfloat16) + 0.5
    Wq, Wk, Wv, Wo = (quant_rows((torch.randn(d, d, device=dev) * 0.1).to(torch.bfloat16)) for _ in range(4))
    Wg, Wu = (quant_rows((torch.randn(dff, d, device=dev) * 0.1).to(torch.bfloat16)) for _ in range(2))
    Wd = quant_rows((torch.randn(d, dff, device=dev) * 0.1).to(torch.bfloat16))
    pos = torch.arange(S, device=dev).float()
    j = torch.arange(dh // 2, device=dev).float()
    ang = pos[:, None] * torch.pow(torch.tensor(10000.0, device=dev), -2 * j / dh)[None, :]
    cos = torch.cos(ang).to(torch.bfloat16)
    sin = torch.sin(ang).to(torch.bfloat16)
    causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))

    def mm(xc, xs, W):                   # hawkeye fp8 matmul (bf16 out)
        out, _ = hawkeye_fp8_sum(xc, xs, W[0], W[1])
        return out

    def rope(t):                         # t: (S, dh) bf16
        a, b = t[:, :dh // 2], t[:, dh // 2:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def attn_batched(q, k, v):
        # all A = B*nh instances in single tensor ops: rope, scores, causal
        # softmax, P.V as bf16 bmm (the well-utilized production shape; per-row
        # fp8 quantization is applied at the same sites as the canonical mode)
        A = B * nh
        qh = q.view(B, S, nh, dh).permute(0, 2, 1, 3).reshape(A, S, dh)
        kh = k.view(B, S, nh, dh).permute(0, 2, 1, 3).reshape(A, S, dh)
        vh = v.view(B, S, nh, dh).permute(0, 2, 1, 3).reshape(A, S, dh)
        a1, b1 = qh[..., :dh // 2], qh[..., dh // 2:]
        qh = torch.cat([a1 * cos - b1 * sin, b1 * cos + a1 * sin], dim=-1)
        a1, b1 = kh[..., :dh // 2], kh[..., dh // 2:]
        kh = torch.cat([a1 * cos - b1 * sin, b1 * cos + a1 * sin], dim=-1)
        qc, qs = quant_rows(qh); kc, ks = quant_rows(kh)
        s = torch.bmm((qc.to(torch.bfloat16) * qs[..., None]),
                      (kc.to(torch.bfloat16) * ks[..., None]).transpose(1, 2))
        s = s.float().masked_fill(~causal[None], float('-inf'))
        p = torch.softmax(s, dim=-1).to(torch.bfloat16)
        pc, ps = quant_rows(p)
        vt = vh.transpose(1, 2).contiguous()
        vc, vs = quant_rows(vt)
        o = torch.bmm(pc.to(torch.bfloat16) * ps[..., None],
                      (vc.to(torch.bfloat16) * vs[..., None]).transpose(1, 2))
        return o.view(B, nh, S, dh).permute(0, 2, 1, 3).reshape(T, d)

    def fwd():
        h = x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6).to(torch.bfloat16) * g1
        hc, hs = quant_rows(h)
        q = mm(hc, hs, Wq); k = mm(hc, hs, Wk); v = mm(hc, hs, Wv)
        if batched:
            attn = attn_batched(q, k, v)
            ac, as_ = quant_rows(attn)
            r1 = x + mm(ac, as_, Wo)
            h2 = r1 * torch.rsqrt((r1.float() ** 2).mean(-1, keepdim=True) + 1e-6).to(torch.bfloat16) * g2
            h2c, h2s = quant_rows(h2)
            gate = mm(h2c, h2s, Wg); up = mm(h2c, h2s, Wu)
            m = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up
            mc, ms = quant_rows(m)
            return r1 + mm(mc, ms, Wd)
        attn = torch.empty(T, d, device=dev, dtype=torch.bfloat16)
        for b in range(B):
            rows = slice(b * S, (b + 1) * S)
            for hh in range(nh):
                colsl = slice(hh * dh, (hh + 1) * dh)
                qh = rope(q[rows, colsl]); kh = rope(k[rows, colsl])
                qc, qs = quant_rows(qh); kc, ks = quant_rows(kh)
                s = mm(qc, qs, (kc, ks))                    # S x S scores
                s = s.float().masked_fill(~causal, float('-inf'))
                p = torch.softmax(s, dim=-1).to(torch.bfloat16)
                pc, ps = quant_rows(p)
                vt = v[rows, colsl].t().contiguous()        # dh x S
                vc, vs = quant_rows(vt)
                attn[rows, colsl] = mm(pc, ps, (vc, vs))
        ac, as_ = quant_rows(attn)
        r1 = x + mm(ac, as_, Wo)
        h2 = r1 * torch.rsqrt((r1.float() ** 2).mean(-1, keepdim=True) + 1e-6).to(torch.bfloat16) * g2
        h2c, h2s = quant_rows(h2)
        gate = mm(h2c, h2s, Wg); up = mm(h2c, h2s, Wu)
        m = torch.nn.functional.silu(gate.float()).to(torch.bfloat16) * up
        mc, ms = quant_rows(m)
        return r1 + mm(mc, ms, Wd)

    for _ in range(10): fwd()            # warmup + triton autotune
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fwd()
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    print(f"FWD seq={S} d={d} nh={nh} dh={dh} dff={dff} batch={B} tokens={T} "
          f"iters={iters} batched={1 if batched else 0} fwd_ms={ms:.4f}")

if __name__ == '__main__':
    main()
