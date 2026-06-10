"""Experiment A.1/A.2/A.3a: accepted-set distribution, baseline greedy tokens +
top-2 margins, and coarse all-rows perturbation attacks."""
import json
import torch
from transformers import AutoTokenizer
import capacity_lib as L

tok = AutoTokenizer.from_pretrained(L.MODEL_CARD, cache_dir=L.CACHE_DIR)
model, norms = L.build_model()
inputs = L.load_inputs(tok)

out = {"sites": L.SITE_ORDER, "per_input": {}}
accset_hist = {}     # accepted-set size -> count (pooled over sites,rows,inputs)
plus_acc = 0; minus_acc = 0; tot_rows = 0

for name, ids in inputs.items():
    L.clear_overrides(norms)
    L.set_record(norms, True)
    z0 = L.forward_logits(model, ids)            # baseline logits [seq, V]
    L.set_record(norms, False)

    top2 = z0.topk(2, dim=-1)
    argmax = top2.indices[:, 0]                  # greedy token per position
    margin = (top2.values[:, 0] - top2.values[:, 1])   # top-2 logit margin [seq]

    rec = {"argmax": argmax.cpu().tolist(),
           "margin": margin.cpu().tolist(),
           "accset": {}}

    # ---- A.1 accepted-set sizes per site ----
    for sid in L.SITE_ORDER:
        m = norms[sid]
        size = (m.acc_hi - m.acc_lo + 1)         # [seq]
        plus_ok = (m.acc_hi >= m.R_round + 1)
        minus_ok = (m.acc_lo <= m.R_round - 1)
        rec["accset"][sid] = {
            "size_mean": float(size.double().mean()),
            "size_counts": {int(s): int((size == s).sum()) for s in torch.unique(size).tolist()},
            "frac_plus_accepted": float(plus_ok.double().mean()),
            "frac_minus_accepted": float(minus_ok.double().mean()),
            "R_star_min": float(m.R_star.min()), "R_star_max": float(m.R_star.max()),
            "R_star_median": float(m.R_star.median()),
        }
        for s, c in rec["accset"][sid]["size_counts"].items():
            accset_hist[s] = accset_hist.get(s, 0) + c
        plus_acc += int(plus_ok.sum()); minus_acc += int(minus_ok.sum()); tot_rows += size.numel()

    out["per_input"][name] = rec
    print(f"[{name}] baseline done. margin: min={margin.min():.4g} "
          f"median={margin.median():.4g} | argmax[:8]={argmax[:8].cpu().tolist()}")

# ---- A.3a coarse attacks ----
# variants: per-site all-rows +1 / -1 (where accepted), and all-sites combined +1 / -1.
def apply_variant(variant):
    """Set R_override on the relevant sites to push +1 or -1 where accepted."""
    L.clear_overrides(norms)
    for sid, direction in variant.items():
        m = norms[sid]
        R = m.R_round.clone()
        if direction == +1:
            ok = (m.acc_hi >= m.R_round + 1)
            R = torch.where(ok, m.R_round + 1, m.R_round)
        else:
            ok = (m.acc_lo <= m.R_round - 1)
            R = torch.where(ok, m.R_round - 1, m.R_round)
        m.R_override = R

coarse = {}
for name, ids in inputs.items():
    L.clear_overrides(norms); L.set_record(norms, True)
    z0 = L.forward_logits(model, ids)
    L.set_record(norms, False)
    arg0 = z0.argmax(-1)
    res = {}
    variants = {}
    for sid in L.SITE_ORDER:
        variants[f"{sid}:+1"] = {sid: +1}
        variants[f"{sid}:-1"] = {sid: -1}
    variants["ALL:+1"] = {s: +1 for s in L.SITE_ORDER}
    variants["ALL:-1"] = {s: -1 for s in L.SITE_ORDER}
    # exclude the final norm from a separate "ALL-but-final" (final cannot flip argmax)
    variants["ALLnoFinal:+1"] = {s: +1 for s in L.SITE_ORDER if s != "final"}
    variants["ALLnoFinal:-1"] = {s: -1 for s in L.SITE_ORDER if s != "final"}

    for vname, variant in variants.items():
        apply_variant(variant)
        z = L.forward_logits(model, ids)
        L.clear_overrides(norms)
        dlogit = (z - z0)
        arg = z.argmax(-1)
        flips = int((arg != arg0).sum())
        res[vname] = {
            "argmax_flips": flips,
            "max_abs_logit_delta": float(dlogit.abs().max()),
            "mean_abs_logit_delta": float(dlogit.abs().mean()),
            "flip_positions": (arg != arg0).nonzero().squeeze(-1).cpu().tolist()[:50],
        }
    coarse[name] = res
    print(f"[{name}] coarse: "
          + ", ".join(f"{k}={v['argmax_flips']}flips/maxd={v['max_abs_logit_delta']:.3g}"
                      for k, v in res.items() if k in ("ALL:+1", "ALL:-1", "ALLnoFinal:+1", "ALLnoFinal:-1")))

out["accset_hist_pooled"] = accset_hist
out["frac_plus_accepted_pooled"] = plus_acc / tot_rows
out["frac_minus_accepted_pooled"] = minus_acc / tot_rows
out["mean_bits_per_row_ceiling"] = None
out["coarse"] = coarse

with open("exp_a_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\n=== accepted-set size histogram (pooled over sites/rows/inputs) ===")
for s in sorted(accset_hist):
    print(f"  size {s}: {accset_hist[s]} rows ({100*accset_hist[s]/tot_rows:.1f}%)")
print(f"frac +1 accepted: {plus_acc/tot_rows:.3f}   frac -1 accepted: {minus_acc/tot_rows:.3f}")
print("wrote exp_a_results.json")
