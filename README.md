# Can you come up with a better[^1] language model?

Language models are very expensive to train.
This repo is an ongoing effort to come up with alternative language model architectures that are cheaper to train.
It's sufficiently accessible that you can make progress without prior ML knowledge.

The goal of the wikitext challenge is to find an algorithm that learns the character probability distribution over Wikipedia in the most energy efficient way.
We care about learning a probability distribution because that's the main way LLMs get their knowledge of language.
This task makes an assumption that any algorithm that can predict text well will learn general linguistic intelligence.

**What you'll need:**
1. Claude Code subscription
2. Modal account ($30 of free credits is more than enough to get started).
3. A Python interpreter (version 3.11)

**What you won't need:**
1. Access to a GPU cluster
2. Knowledge of linear algebra or multivariate calculus
3. Previous background in ML or DL

## Quickstart

First, make sure you can replicate the existing state-of-the-art model.

**Prerequisites:**
1. [Modal](https://modal.com) account.
2. Python 3.11

```bash
# From wip-wikitext/

python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

modal token new

python submit.py submissions/modded_nanogpt
```

Well done, you just replicated the existing state-of-the-art[^2] [nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) submission!

## Research Setup

Running existing work is all well and good, but we want to discover new stuff!
Here's how you do that.
We'll set up the Claude Code agent harness to iteratively research new approaches and test them.

1. Ask Claude Code to "set up github.com/cybertronai/sutroana's slash commands in the current project".
    That's the agentic harness.
2. Now, restart Claude Code to make `/optimize` slash command available.
    Run `/optimize .`
3. You can watch the progress by running `python monitor.py . --watch` (from your virtualenv).


(Recommended) While the agent is researching solutions, pick one of them that sounds interesting and ask Claude Code to explain it to you.
Keep asking questions until it makes sense.
Asking Claude to draw HTML visualizations as outputs instead of Markdown helps a lot here.

**Feeling confused?**
That's great!
Understanding comes from resolving confusion.
Just keep asking the coding agent to explain what's going on until you feel like you have a grasp of it.


This is just a suggested starting point.
Customizing the agentic harness is a good next step!

## Record History

| Date | Energy (J) | Char-acc | Config | Submission | Contributor |
|------|-----------:|---------:|--------|------------|-------------|
| 2026-05-11 |     53,337 | 0.7300 | modded_nanogpt | [dir](submissions/modded_nanogpt) | @ab-10 |


## Rules

Train a character-level language model from scratch on **WikiText-103**.
Submissions that meet the constraints below are ranked by **training energy (joules)**, lower wins.
Greedy-argmax char-accuracy is computed on the first 60,000 chars of each split; val is gated by rule 5, test is reported alongside but not gated.

**Submissions must:**

1. Train from scratch. (No pre-trained weights — WikiText overlaps WebText, so pre-trained init poisons the comparison.)
2. Use the standard WikiText-103 train/valid/test split. (You can change batch size, sequence length, attention structure, etc.; just don't change the underlying streams of characters.)
3. Expose a streaming next-character distribution via the `CharModel` API. (The runner calls `predict()` for position `i` strictly before `observe()` commits the ground-truth at position `i` — within-document future-peeking is structurally impossible.)
    a. Implementing `CharModel` ABC from `wikitext.py` is the most straightforward way to do this.
4. Finish training in **< 300 s wall-clock** on the pinned Modal A100-40GB SXM4, measured from the first call into `train()` to its return. (Eval is not charged against this budget.)
5. Attain **val char-acc ≥ 0.70** on the first 60,000 chars of the val split.


## Notes

[^1]: More energy efficient
[^2]: As of writing this
