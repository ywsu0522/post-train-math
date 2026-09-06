# Kaggle Notebook workflow

Kaggle is treated as a hosted Linux GPU machine. The package contains no
Kaggle-specific runtime branches.

## 1. Enable a GPU and internet access

Internet access is needed for the initial GitHub clone and uncached Hugging Face
downloads.

```python
!git clone https://github.com/ywsu0522/post-train-math.git
%cd post-train-math
!python -m pip install -q "uv>=0.12.6"
!uv sync --locked --group dev
!uv run --locked --no-sync posttrain-math doctor
```

For a dual-GPU notebook, `doctor` should list both devices.

## 2. Acquire pinned resources

```python
!uv run --locked --no-sync posttrain-math model download
!uv run --locked --no-sync posttrain-math data download
!uv run --locked --no-sync posttrain-math data prepare
```

## 3. Train using all visible GPUs

```python
!bash scripts/launch_gpu.sh auto train sft \
    --output-dir /kaggle/working/runs/olmo2-1b-lora-sft-v1 \
    --peft lora \
    --precision auto \
    --gradient-checkpointing
```

For GRPO:

```python
!bash scripts/launch_grpo.sh auto \
    --model /kaggle/working/runs/olmo2-1b-lora-sft-v1/final-model \
    --output-dir /kaggle/working/runs/olmo2-1b-grpo-v1 \
    --max-steps 100
```

## 4. Persist outputs

Keep experiment outputs under `/kaggle/working`. Before leaving the notebook,
save a Kaggle notebook version with outputs so the working artifacts are not
lost with the interactive session.

For public model reproducibility, also publish the final adapter/model and small
manifests to a versioned artifact registry such as Hugging Face Hub.

If you store `HF_TOKEN` as a Kaggle secret, retrieve it in notebook code and
authenticate without printing it:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=token)
```

Never place the token in source files, notebook output, or git history.

## Cache policy

The package uses the same pinned upstream revisions as local and Colab runs.
`HF_HOME` may point to any persistent cache you provide. Cache placement is a
notebook/deployment concern and does not change the training code.
