# wikitext

Train a character-level language model from scratch on WikiText-103.
Submissions are scored by lowest energy to reach 0.7 character prediction accuracy. 
Scorer is agnostic to backprop.
Submissions of novel learning algorithms are encouraged!

## Quickstart

**Prerequisites:**
1. [Modal](https://modal.com) account.
2. Python 3.11

```bash
# From wip-wikitext/

python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

modal token new

python submit.py submissions/modded_nanogpt
```

## Record History

| Date | Energy (J) | Val char-acc | Config | Submission | Contributor |
|------|-----------:|-------------:|--------|------------|-------------|


## Rules

Train a character-level language model from scratch on [WikiText-103-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext).
Submissions that meet the constraints below are ranked by **training energy (joules)**, lower wins.
Greedy-argmax char-accuracy is computed on the first 60,000 chars of each split; val is gated by rule 5, test is reported alongside but not gated.

**Submissions must:**

1. Train from scratch. (No pre-trained weights — WikiText overlaps WebText, so pre-trained init poisons the comparison.)
2. Use the standard WikiText-103 train/valid/test split. (You can change batch size, sequence length, attention structure, etc.; just don't change the underlying streams of characters.)
3. Expose a streaming next-character distribution via the `CharModel` API. (The runner calls `predict()` for position `i` strictly before `observe()` commits the ground-truth at position `i` — within-document future-peeking is structurally impossible.)
    a. Implementing `CharModel` ABC from `wikitext.py` is the most straightforward way to do this.
4. Finish training in **< 300 s wall-clock** on the pinned Modal A100-40GB SXM4, measured from the first call into `train()` to its return. (Eval is not charged against this budget.)
5. Attain **val char-acc ≥ 0.70** on the first 60,000 chars of the val split.


[^1]: More energy efficient
[^2]: As of writing this
