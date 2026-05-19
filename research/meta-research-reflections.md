# Process

1. **Invoked a pre-built `/survey-wikitext` Claude skill** that ran a fixed 2-pass × 5-method round-robin (Phase A enumerate → B1/B2 parallel design+execute → C report), producing the `research/gradfree-survey/` artifacts.
2. **Ran a focused CPU diagnostic with an analytical kill report** to demote a single method (NBB) on principled grounds rather than just on a worse number — `research/nbb-bigram-diagnostic/REPORT.md`.
3. **Dispatched a single-agent literature survey of the Manning collaborator graph** (kNN-LM/RETRO, Pointer-Sentinel, Backpack, etc.), producing `research/manning-bibliography/REPORT.md`.
4. **Merged the skill output, NBB diagnostic, and Manning report into one cross-source taxonomy** with explicit Tier A/B/C/Hybrid filtering and named diagnostic gates D1–D5 (`catalog/RESEARCH_DIRECTIONS.md`).
5. **Wrote hypothesis-driven, individually-gated specs per direction** with effort budget, single first experiment, and go/no-go criterion (`catalog/new_directions/spec_7_*.md` … `spec_16_*.md`).
6. **Replaced the "one more run" reflex with a phased investigation plan for a single method**, with per-phase kill criteria and an explicit random-projection control before iterating (`research/forward-forward-deep/PLAN.md`, 8 phases).
7. **Fanned out parallel sub-agents to implement and submit independent specs concurrently** (Hyena, Mamba, Pointer-Sentinel, Chunker), one spec per agent, no permission to change the design on underperformance.
8. **Triaged on empirical evidence by promoting the one outlier result to a dedicated follow-up investigation** (PPM at 0.63/633 J on 2% of train → `research/ppm-c-extension/` C-port), while leaving the rest as closed-discard reports.

# Footguns

1. Taking a single subset of a method and claiming that the entire method doesn't work
	a. e.g. look at the range of performance at PPM
	b. same for FFN
	c. SOLUTION: ask it to scope how large of the idea
2. Actively avoiding ambitious/open-ended methods
	(i.e. agent makes reasonable decisions within the context of human writing code for a competition)
3. Multiple runs of the same BG research is likely to produce different results




# Solutions

1. Given a spark, claude code is great at iterating and getting it to work.
	a. footgun: even within a single method it's unlikely to try adjecent ones
2. Traversing Google scholar citation graphs
3. High level idea formats that work (particular method FF) > researcher (e.g. Schmidhuber, Manning) >> direction (e.g. alternatives to gradients) >> "just make it better
4. Split research and implementation
5. Actively emphasize approaches that scale with data and compute 
6. Define clear experiment criteria
7. ! creative human ideas are very valuable
