"""Experiment A.3b: targeted greedy attack on the smallest-margin positions.

For the 50 smallest-margin positions across the 3 inputs, greedily search per-row
R choices to flip the position's argmax.  Per the task: rank candidate rows by
single-row influence on the target logit gap (probe one row at a time among the 64
most-attended source rows + the position itself, across the 4 non-final norm sites
-- the final norm is a per-row positive scalar on the whole logit vector and cannot
change argmax), then accumulate the helpful flips and verify with a real forward.

Reports per target: baseline top-2 margin, best achievable gap reduction, whether the
argmax flipped, and (for the lower bound) whether flips are simultaneously achievable
from ONE shared R vector.
"""
import json
import torch
from transformers import AutoTokenizer
import capacity_lib as L

NON_FINAL = ["L0.input", "L0.post_attn", "L1.input", "L1.post_attn"]
TOPK_ATTN = 64

tok = AutoTokenizer.from_pretrained(L.MODEL_CARD, cache_dir=L.CACHE_DIR)
model, norms = L.build_model()
inputs = L.load_inputs(tok)


def base_state(ids):
    """Baseline logits + attention-aggregated source influence."""
    L.clear_overrides(norms); L.set_record(norms, True)
    with torch.no_grad():
        o = model(input_ids=ids, use_cache=False, output_attentions=True)
    L.set_record(norms, False)
    z0 = o.logits.float().squeeze(0)                       # [seq, V]
    # attention received by source r from query t, summed over layers & heads
    att = torch.stack([a.squeeze(0).sum(0) for a in o.attentions]).sum(0)  # [seq_q, seq_k]
    return z0, att


def override_from(choices):
    """choices: dict (site,row)->R_value. Build per-site R_override vectors."""
    L.clear_overrides(norms)
    by_site = {}
    for (sid, r), val in choices.items():
        by_site.setdefault(sid, {})[r] = val
    for sid, m in norms.items():
        if sid in by_site:
            R = m.R_round.clone()
            for r, val in by_site[sid].items():
                R[r] = val
            m.R_override = R


def alts(m, r):
    """Accepted alternative R values at row r (the +-1 neighbours that are accepted)."""
    out = []
    if m.acc_hi[r] >= m.R_round[r] + 1:
        out.append(int(m.R_round[r] + 1))
    if m.acc_lo[r] <= m.R_round[r] - 1:
        out.append(int(m.R_round[r] - 1))
    return out


# ---- collect 50 smallest-margin (input, position) targets ----
allcand = []
states = {}
for name, ids in inputs.items():
    z0, att = base_state(ids)
    states[name] = (z0, att)
    top2 = z0.topk(2, dim=-1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).cpu()
    for pos in range(L.SEQ):
        allcand.append((float(margin[pos]), name, pos,
                        int(top2.indices[pos, 0]), int(top2.indices[pos, 1])))
allcand.sort(key=lambda x: x[0])
targets = allcand[:50]

results = []
flip_choices = {}     # name -> list of (pos, choices) for simultaneity check
_recorded = None      # which input's baseline R is currently stored in the norms


def refresh_baseline(ids):
    """Re-record R_round/acc_lo/acc_hi for THIS input. The IntRMSNorm modules hold
    ONE global baseline (the last input recorded); processing targets from several
    inputs requires refreshing it so alts()/override_from() use the right rows."""
    L.clear_overrides(norms); L.set_record(norms, True)
    L.forward_logits(model, ids)
    L.set_record(norms, False)


# process targets grouped by input so the stored baseline always matches.
targets.sort(key=lambda t: (t[1], t[2]))
for marg, name, pos, top1, comp in targets:
    ids = inputs[name]
    z0, att = states[name]
    if _recorded != name:
        refresh_baseline(ids)
        _recorded = name
    g0 = float(z0[pos, top1] - z0[pos, comp])      # baseline gap (>0); flip wants <=0

    # candidate rows: top-K attended sources (<=pos) + the position itself
    infl = att[pos].clone(); infl[pos+1:] = -1
    rows = set(torch.topk(infl, min(TOPK_ATTN, pos+1)).indices.cpu().tolist())
    rows.add(pos)
    rows = sorted(r for r in rows if r <= pos)

    # probe each (site,row,alt) once; record gap delta at this target
    probe = []   # (dgap, sid, row, Rval)
    for sid in NON_FINAL:
        m = norms[sid]
        for r in rows:
            for Rval in alts(m, r):
                override_from({(sid, r): Rval})
                z = L.forward_logits(model, ids)
                dgap = float((z[pos, top1] - z[pos, comp]) - g0)
                probe.append((dgap, sid, r, Rval))
    L.clear_overrides(norms)

    # greedy accumulate helpful single-row flips (most negative first), one per (site,row)
    probe.sort(key=lambda x: x[0])
    chosen = {}
    for dgap, sid, r, Rval in probe:
        if dgap < 0 and (sid, r) not in chosen:
            chosen[(sid, r)] = Rval
    # verify combined effect with a real forward
    override_from(chosen)
    zc = L.forward_logits(model, ids)
    L.clear_overrides(norms)
    g_final = float(zc[pos, top1] - zc[pos, comp])
    flipped = int(zc[pos].argmax()) != top1
    sum_neg = sum(d for d, *_ in probe if d < 0)   # linear (additive) estimate

    results.append({
        "input": name, "pos": pos, "baseline_margin": marg, "baseline_gap": g0,
        "best_single_dgap": float(probe[0][0]) if probe else 0.0,
        "sum_negative_dgap": sum_neg,
        "combined_gap": g_final, "achieved_gap_reduction": g0 - g_final,
        "flipped": flipped, "n_flips_used": len(chosen),
    })
    if flipped:
        flip_choices.setdefault(name, []).append((pos, dict(chosen)))
    print(f"[{name} pos{pos}] marg={marg:.4g} gap={g0:.4g} "
          f"best1={probe[0][0]:.3g} sumneg={sum_neg:.3g} -> gapfinal={g_final:.4g} "
          f"flip={flipped} ({len(chosen)} flips)")

# ---- simultaneity check: can all flips for one input come from ONE R vector? ----
simul = {}
for name, lst in flip_choices.items():
    merged = {}
    conflict = 0
    for pos, choices in lst:
        for k, v in choices.items():
            if k in merged and merged[k] != v:
                conflict += 1
            merged[k] = v
    # apply merged, count how many targets still flip
    refresh_baseline(inputs[name])      # stored baseline must match this input
    override_from(merged)
    z = L.forward_logits(model, inputs[name])
    L.clear_overrides(norms)
    still = 0
    for pos, choices in lst:
        t1 = [t[3] for t in targets if t[1] == name and t[2] == pos][0]
        if int(z[pos].argmax()) != t1:
            still += 1
    simul[name] = {"n_flippable_individually": len(lst),
                   "merge_conflicts": conflict,
                   "n_flip_simultaneously": still}

n_flip = sum(1 for r in results if r["flipped"])
n_tie = sum(1 for r in results if r["baseline_margin"] == 0.0)
surv = [r["baseline_margin"] for r in results if not r["flipped"]]
out = {"n_targets": len(results), "n_flipped": n_flip, "n_exact_ties": n_tie,
       "min_surviving_margin": min(surv) if surv else None,
       "max_achieved_gap_reduction": max(r["achieved_gap_reduction"] for r in results),
       "simultaneity": simul, "targets": results}
with open("exp_a_targeted_results.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\n=== {n_flip}/{len(results)} flipped ({n_tie} were exact ties) ===")
print(f"max achieved gap reduction over all targets: {out['max_achieved_gap_reduction']:.4g}")
print(f"min surviving (un-flipped) margin: {out['min_surviving_margin']}")
print("simultaneity:", json.dumps(simul))
print("wrote exp_a_targeted_results.json")
