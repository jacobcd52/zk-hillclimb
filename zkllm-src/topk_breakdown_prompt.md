# Task: top-K (Rinberg) capacity rule — find the optimal b*, report the FULL term breakdown, sweep K

The covert-capacity b-sweep is already done (/workspace/projects/zk-hillclimb/CAPACITY_SWEEP.md,
scripts under capacity/ and measure/, per-position margin + N_b dumps already generated for
schemes: baseline, faithful, codebook). The top-K refined formula is implemented:
  C_topK(b) = H(p) + (1-p)*E_t[log2 N_b]  +  p * ( H(q) + (1-q)*log2 K + q*log2(V-K) )
  where p(b) = fraction with margin>b; q = fraction of VIOLATING tokens whose served token
  is outside the teacher's top-K (measured from data); V=32000; K the top-K parameter.
The existing report gives only the headline min (faithful 0.363, codebook 0.228, baseline
12.07) — NOT the term breakdown at the top-K optimum. Produce that.

Do, for each of the 3 schemes (reuse the existing dumps; do NOT re-run the model):
1. Sweep b finely (>=200 points) under C_topK at K=16; find b*_topK = argmin and the min value.
   Report b*_topK, p(b*), q(b*), E[log2 N_b](b*).
2. At b*_topK, give the FULL FIVE-component breakdown with absolute bits AND % of total:
   (a) H(p)                          [which positions violate]
   (b) (1-p)*E[log2 N_b]             [within-margin multiplicity, compliant]
   (c) p*H(q)                        [violate: tail-vs-topK choice]
   (d) p*(1-q)*log2 K                [violate into top-K]
   (e) p*q*log2(V-K)                 [violate into tail]
   Confirm (a)+(b)+(c)+(d)+(e) == C_topK(b*) to 3 decimals.
3. Sweep K over {1,2,4,8,16,32,64,256,1024} (K=1 means "must serve teacher argmax on a
   violation"): for each K, the min-over-b C_topK and the b* — so we see how the bound
   depends on the top-K modeling choice. Note the adversary-vs-defender reading: smaller K
   = tighter (more realistic) bound but larger tail fraction q; identify the K that gives
   the LOWEST min-over-b capacity and whether it's monotone.
4. Compare to the SIMPLE rule's optimum (the existing min_b C numbers) — how much does the
   top-K refinement buy at the optimum, and which term shrank.

Deliverable: /workspace/projects/zk-hillclimb/CAPACITY_TOPK.md — the per-scheme breakdown
tables (the five components at b*_topK), the K-sweep table + a plot (min-capacity vs K per
scheme), and a 4-line summary: the new best entropy per scheme and which term dominates
after the refinement. Honest about any approximation (e.g. how q is estimated, ties).
Scripts under capacity/. Use /root/int-model-env/bin/python. No git commits; no pushes;
int-model-approximation READ ONLY.
