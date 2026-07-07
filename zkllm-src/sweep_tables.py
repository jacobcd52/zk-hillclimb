#!/usr/bin/env python3
"""Render sweep_results.json into the section-13 markdown tables."""
import json

rows = json.load(open('sweep_results.json'))

def fmt(r, key):
    z = r.get(key)
    if not z: return ('-',) * 5
    ov = r.get(f'{key}_overhead')
    return (f"{z['prove']:.2f}", f"{z['verify']:.2f}", f"{z['proof_mb']:.1f}",
            f"{z['rss_gb']:.2f}", f"{ov:,.0f}x" if ov else '-')

def attn_split(r, key):
    z = r.get(key)
    if not z or 'stages' not in z: return '-'
    s = z['stages']
    attn = s['rope'] + s['smx']
    return f"attn(rope+smx)={attn:.2f} mm={s['mm']:.2f} lug={s['lug']:.2f} bo={s['batch']:.2f}"

def table(sel, title):
    print(f'\n#### {title}')
    print('| config | tokens | fwd ms | zk prove s | zk verify s | zk proof MB | zk RSS GB | zk OVERHEAD | nozk prove s | nozk OVERHEAD |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if not r['label'].startswith(sel): continue
        f = r.get('fwd_ms')
        zp, zv, zm, zr, zo = fmt(r, 'zk')
        np_, _, _, _, no = fmt(r, 'nozk')
        cfg = f"S={r['seq']} B={r['batch']} d={r['d']} nh={r['nh']} dff={r['dff']}"
        print(f"| {cfg} | {r['tokens']} | {f:.2f} | {zp} | {zv} | {zm} | {zr} | {zo} | {np_} | {no} |")
    print('\nstage split (zk):')
    for r in rows:
        if not r['label'].startswith(sel): continue
        print(f"  {r['label']}: {attn_split(r, 'zk')}   [nozk: {attn_split(r, 'nozk')}]")

table('params', 'params sweep (tokens=8: seq=8, batch=1; dff=4d, dh=32)')
table('tokens', 'tokens sweep (d=64, nh=2, dff=256, batch=1)')
table('split', 'seq-vs-batch split at fixed tokens=64 (d=64, nh=2, dff=256)')
