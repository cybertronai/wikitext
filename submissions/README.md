# Submissions

> ⚠️ **2026-05-28 — `CharModel.predict()` contract change.** `predict()` now
> returns a single committed character (`str`) instead of a distribution
> (`dict[str, float]`). Submissions are responsible for their own sampling
> strategy. See [`OUTDATED.md`](OUTDATED.md) for the list of submissions
> not yet ported, the mechanical transform for porting, and the re-run TODO
> before `bugfix/sampling` can promote to `main`.

One subdirectory per submission. Submitter authors `submission.py`;
`submit.py` populates the rest.

```
submissions/
└── <name>/
    ├── submission.py     # train(train_text, valid_text=None) -> CharModel
    ├── result.json       # canonical (training_energy_J, training_duration_s,
    │                     #   val_char_accuracy, gpu_name, _nvml, …) tuple
    ├── nvml.json         # NVML probe summary from the same Modal host
    │                     #   (same data as result.json["_nvml"], split out
    │                     #   as a standalone evidence file)
    └── run.log           # captured stdout from `submit.py`
```

`<name>` is whatever directory name the submitter picks; the same
string ends up in the `Config` column of the Record History table. Pick
something short and snake_case (e.g. `modded_nanogpt`).

To submit:

```bash
python3 wip-wikitext/submit.py wip-wikitext/submissions/<name>
```

`submit.py` writes `result.json`, `nvml.json`, and `run.log`
automatically and appends the row to the Record History in
[`../README.md`](../README.md). For manual runs, drop the files in by
hand and append the row yourself.
