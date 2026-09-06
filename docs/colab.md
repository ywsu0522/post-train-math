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

The project model downloader materializes a concrete `--output-dir`.
Setting `HF_HOME` alone does not make that repository-local model directory
survive a Colab runtime recycle. To reuse the base model across sessions, either
download it to a persistent Drive path explicitly or archive/copy the model
directory yourself. Keep this storage choice in notebook cells rather than core
Python code.

For example, after mounting Drive:

```python
!uv run --locked --no-sync posttrain-math model download \
    --output-dir /content/drive/MyDrive/posttrain-math/models/olmo-2-0425-1b
```

Then pass that path as `--model`. Loading directly from mounted Drive can be
slower than copying the snapshot back to Colab-local storage before training.

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

## Reusing resources

Colab's local filesystem is ephemeral. Hugging Face cache placement can help
with dataset/cache reuse, while this project's explicit model `--output-dir`
must itself live on persistent storage (or be copied there) if you want the
materialized model snapshot to survive the session. Mounted Drive I/O is slower,
so a common pattern is persistent storage for retention plus Colab-local copies
for active training. The package itself remains provider-independent.
