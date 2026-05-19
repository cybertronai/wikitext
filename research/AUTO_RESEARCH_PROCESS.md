# Auto-research process — what I tried, and what kept tripping the assistant

A short retrospective on the *meta-research process* used to find gradient-free
methods for the WikiText char-LM task on the `gradfree-methods` branch. Aimed at
other researchers in the group who orchestrate Claude Code on open-ended research.

Each section is one orchestration approach I tried, in chronological order, with
quoted evidence from the transcripts in `~/.claude/projects/-home-seneca-wikitext/`.
Quotes are trimmed to ≤2 sentences and tagged by session UUID + role.

---

## 1. Pre-built `/survey-wikitext` Claude skill — 2-pass × 5-method round-robin

The starting point. A skill that enumerated 5 candidate methods, dispatched
design+execute waves in parallel, and produced a single REPORT.md at the end.
Output lives in `research/gradfree-survey/`.

> *(skill listing, 387b6772)* "Autonomous gradient-free method survey for
> ~/wikitext. Runs a literature-review investigation, then `loop_count`
> round-robin (design + execute) passes across N methods in parallel, then a
> final report."

**What it bought me.** Coverage and parallelism for free, plus a built-in
no-design-iteration rule. The skill is the right primitive for a first sweep.

**What it didn't buy me.** Novelty. Two passes over 5 methods is structurally
biased toward incrementalism — pass 2 mostly tunes pass 1. The skill's strongest
"win" was a ridge readout swapped onto FF features:

> *(assistant, 28cd7a73)* "The only pass-2 *improvement* came from putting a
> ridge readout on FF features (0.235 → 0.279)."

A +0.044 absolute gain that doesn't generalise to a different paradigm. Useful
for ranking variants, useless for escaping the local basin.

---

## 2. Focused CPU diagnostic with an *analytical* kill report

Rather than re-run NBB at scale and watch it fail, I scoped a small CPU sweep
plus a derivation of why it must fail. The artefact (`nbb-bigram-diagnostic/REPORT.md`)
demoted NBB to Tier C on principled grounds: `E[ΔW/W] = p·η − λ` can't be
balanced across bytes with varying modal-byte probability.

**Process point.** "Closed-discard with mechanism" is much cheaper than "ran
it on Modal and it didn't work" — and it produces a citable artefact rather
than a number.

---

## 3. Single-agent Manning collaborator-graph literature survey

Side-channel for novelty. I asked one agent to scan Manning's papers and rank
them for char-LM relevance, with explicit framing that "different" beats
"better-at-the-same-thing":

> *(user, 391d6c59)* "Produce a list of all papers by Christopher Manning…
> It's fine or even expected for the approaches to be substantially different
> in what they predict… We're looking for novel approaches that have not been
> tried in the field over hill climbing existing work."

This is the first time I had to spell out the novelty bias *to the prompt*.
The skill in §1 implicitly assumed hill-climbing was the goal; here I made the
opposite assumption explicit. Output: `research/manning-bibliography/REPORT.md`
with seven branches (kNN-LM/RETRO, Hyena, Mamba, Pointer-Sentinel, Backpack, …).

**Cost.** ~1 session, no Modal spend.

---

## 4. Cross-source taxonomy with Tier A/B/C/H + named diagnostic gates

Merge step. `catalog/RESEARCH_DIRECTIONS.md` folds the survey results, the NBB
diagnostic, and the Manning report into one taxonomy with:

- a 6-family mechanism axis (layer-local, Hebbian/fast-weight, hierarchical,
  sparse, program-search, adversarial codes)
- a tier filter (A = this week, B = read deeper, C = don't pursue, H = hybrids)
- five named diagnostic gates D1–D5 (e.g. D1 = empirical surprise rate before
  committing to a chunker port)

The synthesis itself surfaced a structural pattern that no single sub-agent had
seen:

> *(assistant, 28cd7a73)* "The structural insight from the survey — *gradient-free
> representation + closed-form readout is the only thing that worked* — points
> away from any further neural-feature experiments unless they explicitly slot
> into that shape."

**Process point.** The taxonomy is the document I'd write *first* if I did this
again. Without it the next step (specs) is just a wishlist.

---

## 5. Hypothesis-driven, individually-gated specs per direction

For each Tier A/B candidate I wrote a short spec (`catalog/new_directions/spec_*.md`)
with a single hypothesis, a single first experiment, an effort budget, and a
go/no-go criterion. Specs 7–16. Each is small enough that an agent can pick it
up and submit without further design conversation:

> *(user, dbd64177)* "write the specs and dispatch in parallel"

**What this fixed.** Stops the agent from improvising scope at submission time.
The spec is the contract; deviations need a new spec, not a comment in chat.

---

## 6. Phased plan with per-phase kill criteria + random-projection control

When Forward-Forward warranted deeper investigation, I rewrote the spec from
"run FF wider on a fresh seed" into an 8-phase investigation plan
(`research/forward-forward-deep/PLAN.md`). The critical addition is a Phase 2
random-projection baseline — *measure the floor before you measure the
candidate*. Every later phase has an explicit kill criterion against that floor.

> *(user, dca25f20)* "implement `.survey/FF_INVESTIGATION.md` / go, $30 is
> fine, swap mono-forward for cosine-goodness."

**Process point.** The push-back here is the most important one in the whole
run. The assistant's instinct was to iterate within FF; my correction was that
FF cannot claim representational value until it beats random projection + ridge.
That gate inverts the default — *measure the floor before you measure the
candidate* — and is the cheapest way to keep a phased plan honest.

---

## 7. Parallel sub-agent fan-out — one spec per agent, no design changes

Once specs and gates were in place I dispatched four submission implementations
concurrently (Hyena, Mamba, Pointer-Sentinel, Chunker), each owned by an
independent agent with the rule that execution errors can be retried but
underperformance cannot trigger a design change:

> *(user, 90dcb5d8)* "Three Modal submissions launched in parallel: lwta_k2
> (bg `bo6bergki`), lwta_k4 (bg `bjpuw5dz5`), ppm_c (bg `bvdn6snew`)… If
> finished, summarize results (val char-acc, energy, DQ status). If still
> running, schedule another wakeup."

**What this bought me.** Real leaderboard rows fast (most DQ'd; see README
table). DQs are evidence, not failures — they tell me which specs were
unrealistic on the 300 s budget.

**Foot-gun.** Agents *will* try to silently widen the design when they hit a
wall. Spec wording matters: "implement this spec exactly, retry on execution
errors only" is the load-bearing sentence.

---

## 8. Triage on empirical evidence — one outlier gets a follow-up

After everything ran, one method (PPM context tree) was the clear outlier:
0.63 val acc at 633 J on 2% of the train budget. Everything else either DQ'd
or didn't generalise. So PPM alone got promoted to a dedicated follow-up
(`research/ppm-c-extension/`) — same algorithm, faster substrate — while the
rest were closed as discard reports with mechanistic kill notes.

> *(assistant, 28cd7a73)* "**PPM in C/Cython/CUDA, full-data.** Same algorithm,
> fast substrate, order-7 with the full 220 MB train… **Likely path to 0.70 at
> ~1–5 kJ.** Lowest research risk in the entire space."

**Process point.** Triage is a separate step from sweep. I had to consciously
stop *generating new directions* and start *investing in the one that worked*.
The catalog tempts you to keep adding rows.

---

## Cross-cutting: assistant failure modes I had to correct for

These showed up repeatedly enough to call out separately.

**Marginal hill-climbing.** Default mode for the assistant is "make the last
number better". The ridge-on-FF swap (§1) is the canonical example. Counter:
explicit novelty framing in the prompt (§3), and a control measurement that
makes the marginal gain illegible (§6).

> *(user, dbd64177)* "Next experiments must prioritize ambitious novel
> architectures over cheap incremental experiments."

**Scope drift at submission time.** Agents widen a spec when a run fails.
Counter: short single-hypothesis specs (§5) and explicit "execution errors
only" retry semantics (§7).

**Closing investigations as "didn't work" instead of "didn't work *because*".**
The cheap version is a DQ row; the useful version is a mechanism. NBB (§2) is
the template: state the rule the method violates, not just the number it
missed.

**Catalog inflation vs. investment.** Easier to spec a new direction than to
go deep on an existing one. Triage (§8) was a deliberate stop to that bias.

---

## What I'd do differently next time

1. **Write the taxonomy *before* the first skill run.** The skill in §1 spent
   compute on five methods I would have ranked lower after writing §4.
2. **Bake the random-projection baseline into the skill itself.** Right now
   it's a downstream correction (§6); it should be Phase 0 of any survey.
3. **Spec the triage step.** §8 happened because I noticed; a discipline like
   "after N submissions, force a 1-method follow-up" would make it
   automatic.
4. **Phrase novelty as a constraint, not a preference.** "It's fine or even
   expected for the approaches to be substantially different" (§3) worked
   better than "look for novel methods". Constraint-style prompts survive
   the agent's hill-climbing default.
