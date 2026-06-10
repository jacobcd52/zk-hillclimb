#!/usr/bin/env bash
# Prover entrypoint: produce a SERIALIZED proof directory for the q_proj obligations.
# Usage: prove.sh <workdir> <proof_out_dir>
# <workdir> must contain (from setup_testdata.py): q_proj-int.bin q_proj-input.bin
#   q_proj-pp.bin q_proj-commitment.bin public.json seed.hex
set -euo pipefail
ZK=/root/zkllm
WORK="${1:?workdir}"
OUT="${2:?proof_out_dir}"
SF="${3:-65536}"   # 2^16 rescale

SEED=$(cat "$WORK/seed.hex")

# obligation subdirs (ids EXACTLY from manifest_llama68m.json)
DM="$OUT/layer0.attn.q_proj.matmul"
DR="$OUT/layer0.attn.q_proj.rescaling"
DC="$OUT/layer0.attn.q_proj.commitment_opening"
mkdir -p "$DM" "$DR" "$DC"

# the verifier needs the public generators to reproduce the generator fold
cp "$WORK/q_proj-pp.bin" "$DC/pp.bin"
cp "$WORK/public.json" "$OUT/public.json"
cp "$WORK/seed.hex" "$OUT/seed.hex"

cd "$ZK"
./zkprove_dump \
  "$WORK/q_proj-int.bin" "$WORK/q_proj-input.bin" 768 768 \
  "$SEED" "$DM" "$DR" "$DC" \
  "$WORK/q_proj-pp.bin" "$WORK/q_proj-commitment.bin" "$SF"

echo "proof written to $OUT"
