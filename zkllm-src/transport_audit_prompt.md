# Task: adversarial soundness audit of the batched-opening protocol DESIGN (before any code is written)

OUR OWN defensive ZK codebase. A design for a proof-transport rebuild was just written:
/workspace/projects/zk-hillclimb/TRANSPORT_REBUILD_DESIGN.md. It replaces ~1535 inline
IPA openings with a claim accumulator + ONE batch-evaluation sumcheck (RLC-of-eq
reduction) + one IPA per generator domain. Before we spend ~4-6 weeks implementing it,
audit the PROTOCOL DESIGN for soundness holes — a flaw here voids the whole rebuild.

Read: TRANSPORT_REBUILD_DESIGN.md (esp. §2.1 claim objects, §2.2 the reduction, §2.3 the
soundness argument, §2.4 what's untouched, §4 weight-privacy); PHASE0_NOTES.md §10-13
(the EXISTING IPA/commitment/FS conventions the batch must preserve); vrf_common.cuh +
zkob_lookup.cuh (the current open_prove/open_verify the batch replaces — does the batch
actually prove the SAME statement?).

Hunt (be adversarial — try to construct a cheating prover the batched protocol accepts
but the current per-opening protocol would reject):
1. RLC batching soundness: is the random-linear-combination challenge derived AFTER all
   claims are absorbed? Could a prover, knowing the RLC challenge, craft inconsistent
   per-claim evals that cancel in the combination? Walk the Schwartz-Zippel argument —
   degree, field size, soundness error. Is each claim's (commitment-id, point, eval)
   bound BEFORE the batch challenge?
2. The batch-evaluation sumcheck: does reducing many (point, eval) claims to one random
   point r actually preserve binding to EACH original commitment? Check the eq-polynomial
   construction and that no claim can be silently dropped or aliased to another's domain.
3. Cross-domain correctness: 4 generator domains (gen64/1024/4096/32768) each get one
   IPA — are claims correctly partitioned by domain, and can a prover misattribute a
   claim to the wrong-size domain to evade binding?
4. Homomorphic-link & byte-equality preservation: §2.4 claims affine limb links,
   skip-connection point checks, and chain edges are untouched. VERIFY the batched
   openings still anchor the same commitments those links/edges rely on — could batching
   break the anchoring that makes a homomorphic link meaningful?
5. FS transcript: the new global accumulation phase — is every claim absorbed before the
   batch challenge, every domain's IPA bound, nothing a prover supplies left unabsorbed?
   Compare to the §13 "challenge only after the message it binds" rule.
6. Zero-new-advice claim (§2.3): does the batch introduce any prover-chosen value that
   isn't forced? (The whole project closes covert channels — a leaky batch protocol would
   reopen one.)
7. Weight-privacy forward-compat (§4): does the Stage A-C accumulator format actually
   leave room for the Stage D hiding/masking without a redesign, or is that optimistic?
8. The new forgery battery §5.2 (BO-1..BO-10): do the proposed tests actually catch the
   attacks in 1-6, or are there gaps — attacks no proposed test would detect?

Deliverable: /workspace/projects/zk-hillclimb/TRANSPORT_REVIEW.md — VERDICT
(SOUND-AS-DESIGNED / FIXABLE-ISSUES / BROKEN), then per finding: the precise gap, a
concrete cheating-prover sketch if one exists, and the fix. For clean categories state
what you checked and why it holds. If you find the design sound, say so plainly; if you
find a hole, a concrete forgery sketch is worth more than vague doubt. READ-ONLY except
the review file. No code. No git; no pushes.
