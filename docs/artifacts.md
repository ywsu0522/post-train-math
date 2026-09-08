# Artifact and cache policy

## Three different kinds of state

### Source code

GitHub contains source, tests, documentation, and dependency metadata. Large model/data/run files are not committed.

### Re-creatable external resources

`models/` and `data/` are local materializations of pinned upstream Hugging Face
resources. The download manifests record the resolved immutable commit.

Pinning protects against upstream file movement on `main` and other drift. Once
content exists in the Hugging Face cache it can also be reused during a
temporary network outage. No repository can guarantee access to an uncached
third-party host during an outage, so critical long-lived experiments should
prefetch resources or maintain an organization-controlled mirror.

### Experiment artifacts

All generated experiment state belongs under:

```text
runs/<experiment>/
```

Keep together:

- `provenance.json` (written automatically by successful SFT/GRPO entrypoints);
- `run_config.json`;
- training logs and plots;
- `checkpoint-*` needed for resume;
- `checkpoint_manifest.json`;
- `train_summary.json`;
- `final-model/`;
- evaluation `predictions.jsonl` and `metrics.json`.

`provenance.json` snapshots the code commit/dirty state, `uv.lock` hash,
model/base-model source identity, raw-data download manifest,
processed-data manifest, and hardware/runtime report. LoRA adapter directories
also contain `base_model_source.json`; their `adapter_config.json` records the
canonical Hugging Face base repo and exact revision instead of a machine-local
training pathname.

## Local workstation

`runs/` is ordinary persistent local storage. Back it up according to the
importance of the experiment.

## Hosted notebooks

Colab and Kaggle interactive filesystems can disappear after the session.
Training code still writes to an ordinary `--output-dir`; the notebook decides
how that directory is persisted.

- Colab: archive/copy the run to Google Drive, or upload the publication subset
  to a remote artifact registry.
- Kaggle: use `/kaggle/working`, save a notebook version with outputs, and
  optionally publish the final model elsewhere.

This separation keeps provider-specific storage APIs out of the Python package.

## What to publish

For a portfolio/reproducibility model release, publish:

1. final LoRA adapter or final model;
2. `provenance.json`;
3. `run_config.json` and `train_summary.json`;
4. evaluation metrics;
5. `base_model_source.json` for LoRA artifacts.

The provenance file already includes the exact base-model and dataset source
identity, split metadata, code revision, lock hash, and environment report.

Intermediate checkpoints are usually operational artifacts. Publish them only
when they are required to reproduce a learning-dynamics claim, resume a public
run, or compare checkpoints.

## Hugging Face availability

The project currently uses Hugging Face as the canonical upstream for OLMo and
a pinned consolidated mirror of Hendrycks MATH. Exact revisions provide reproducibility, while the local/HF
cache provides reuse. For stronger availability guarantees, add a separately
managed mirror only after confirming redistribution licenses and documenting
which registry is authoritative.
