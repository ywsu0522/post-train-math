# Google Colab workflow

Colab is treated as a hosted Linux GPU machine. The package contains no
Colab-specific runtime branches.

## 1. Start a GPU runtime and install once

In notebook cells:

```python
!git clone https://github.com/ywsu0522/post-train-math.git
%cd post-train-math
!python -m pip install -q "uv>=0.12.6"
!uv sync --locked --group dev
!uv run --locked --no-sync posttrain-math doctor
```

A T4 is SM75 and should pass the architecture check. `--precision auto` will
select FP16 on a T4.

## 2. Acquire pinned resources

```python
!uv run --locked --no-sync posttrain-math model download
!uv run --locked --no-sync posttrain-math data download
!uv run --locked --no-sync posttrain-math data prepare
```

If you have persistent Drive storage available for Hugging Face cache reuse,
set `HF_HOME` before downloading. Keep the project code independent of this
choice.

## 3. Run training on fast local runtime storage

```python
!bash scripts/launch_gpu.sh 1 train sft \
    --output-dir runs/olmo2-1b-lora-sft-v1 \
    --peft lora \
    --precision auto \
    --gradient-checkpointing
```

Writing Trainer checkpoints directly to mounted Drive can be slower than
Colab-local storage. A practical workflow is to train under `/content/.../runs`
and persist selected artifacts after milestones or at the end.

## 4. Persist before the runtime is recycled

### Full run archive to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!tar -C runs -czf \
  /content/drive/MyDrive/posttrain-math-olmo2-sft-v1.tgz \
  olmo2-1b-lora-sft-v1
```

This preserves checkpoints, logs, metrics, and the final adapter together.

### Publish the reproducibility subset to Hugging Face

Authentication is a user action. Do not commit tokens to the repository.

```python
from huggingface_hub import HfApi, login

login()  # enter a write-capable token interactively
api = HfApi()
repo_id = "YOUR_HF_USER/olmo2-1b-math-sft-v1"
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path="runs/olmo2-1b-lora-sft-v1/final-model",
    path_in_repo=".",
)
```

Also upload/copy the small `run_config.json`, `train_summary.json`, evaluation
`metrics.json`, and the model/data source manifests. Large intermediate
checkpoints are optional unless needed for resume or analysis.

## Reusing cache

Colab's local filesystem is ephemeral. Persisting `HF_HOME` on Drive can avoid
re-downloading an already cached model/dataset, although mounted Drive I/O is
slower. This is a notebook storage decision; the package itself remains
unchanged.
