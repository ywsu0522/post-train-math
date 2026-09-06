#!/usr/bin/env python3
"""Apply the runtime/reproducibility refactor discussed for post-train-math.

Run this from the repository root. It intentionally leaves pyproject.toml and
uv.lock unchanged.
"""

from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "src/posttrain_math").is_dir():
    raise SystemExit("Run this script from the post-train-math repository root.")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"write  {path}")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match in {path}, found {count}:\n{old}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patch  {path}")


def replace_all(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected token not found in {path}: {old}")
    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patch  {path}")


write("src/posttrain_math/environment.py", 'from __future__ import annotations\n\nimport json\nimport platform\nimport subprocess\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\n\nimport torch\n\nREFERENCE_CUDA_MAJOR = 13\nREFERENCE_MIN_DRIVER_MAJOR = 580\nREFERENCE_MIN_COMPUTE_CAPABILITY = (7, 5)\n\n\ndef native_bf16_supported() -> bool:\n    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(\n        including_emulation=False\n    )\n\n\ndef _driver_version() -> str | None:\n    try:\n        result = subprocess.run(\n            [\n                "nvidia-smi",\n                "--query-gpu=driver_version",\n                "--format=csv,noheader",\n            ],\n            check=True,\n            capture_output=True,\n            text=True,\n        )\n    except (OSError, subprocess.CalledProcessError):\n        return None\n\n    versions = {\n        line.strip()\n        for line in result.stdout.splitlines()\n        if line.strip()\n    }\n    if len(versions) != 1:\n        return None\n    return versions.pop()\n\n\ndef _driver_major(version: str | None) -> int | None:\n    if not version:\n        return None\n    try:\n        return int(version.split(".", 1)[0])\n    except ValueError:\n        return None\n\n\ndef _compiled_cuda_arches() -> tuple[str, ...]:\n    try:\n        return tuple(torch.cuda.get_arch_list())\n    except (AttributeError, RuntimeError):\n        return ()\n\n\n@dataclass(frozen=True)\nclass GPUReport:\n    index: int\n    name: str\n    compute_capability: str\n    total_vram_gib: float\n    bf16_supported: bool\n    architecture_supported: bool\n\n\n@dataclass(frozen=True)\nclass EnvironmentReport:\n    os: str\n    machine: str\n    python_version: str\n    torch_version: str\n    torch_cuda_version: str | None\n    compiled_cuda_arches: tuple[str, ...]\n    cuda_available: bool\n    driver_version: str | None\n    driver_supported: bool\n    gpus: tuple[GPUReport, ...]\n\n    train_dataset_exists: bool\n    test_dataset_exists: bool\n    model_exists: bool\n\n    @property\n    def hardware_ok(self) -> bool:\n        return (\n            self.cuda_available\n            and self.driver_supported\n            and bool(self.gpus)\n            and all(gpu.architecture_supported for gpu in self.gpus)\n        )\n\n    @property\n    def ok(self) -> bool:\n        # Resource files are informational: `doctor` is useful on a fresh clone\n        # before model/data acquisition.\n        return self.hardware_ok\n\n\ndef inspect_environment(\n    train_dataset_path: Path,\n    test_dataset_path: Path,\n    model_path: Path,\n) -> EnvironmentReport:\n    cuda_available = torch.cuda.is_available()\n    driver_version = _driver_version() if cuda_available else None\n    driver_major = _driver_major(driver_version)\n\n    torch_cuda_version = torch.version.cuda\n    cuda_major = None\n    if torch_cuda_version:\n        try:\n            cuda_major = int(torch_cuda_version.split(".", 1)[0])\n        except ValueError:\n            cuda_major = None\n\n    # The committed reference lock resolves Linux torch to CUDA 13.x.\n    # Fail closed if a future dependency update changes the CUDA major without\n    # updating this explicit contract.\n    driver_supported = bool(\n        cuda_available\n        and cuda_major == REFERENCE_CUDA_MAJOR\n        and driver_major is not None\n        and driver_major >= REFERENCE_MIN_DRIVER_MAJOR\n    )\n\n    gpus: list[GPUReport] = []\n    if cuda_available:\n        for index in range(torch.cuda.device_count()):\n            major, minor = torch.cuda.get_device_capability(index)\n            props = torch.cuda.get_device_properties(index)\n            gpus.append(\n                GPUReport(\n                    index=index,\n                    name=torch.cuda.get_device_name(index),\n                    compute_capability=f"{major}.{minor}",\n                    total_vram_gib=props.total_memory / (1024**3),\n                    # Native BF16 starts with Ampere-class NVIDIA devices for\n                    # the GPU families supported by this project.\n                    bf16_supported=major >= 8,\n                    architecture_supported=(\n                        (major, minor) >= REFERENCE_MIN_COMPUTE_CAPABILITY\n                    ),\n                )\n            )\n\n    return EnvironmentReport(\n        os=platform.system(),\n        machine=platform.machine(),\n        python_version=platform.python_version(),\n        torch_version=torch.__version__,\n        torch_cuda_version=torch_cuda_version,\n        compiled_cuda_arches=_compiled_cuda_arches(),\n        cuda_available=cuda_available,\n        driver_version=driver_version,\n        driver_supported=driver_supported,\n        gpus=tuple(gpus),\n        train_dataset_exists=train_dataset_path.is_file(),\n        test_dataset_exists=test_dataset_path.is_file(),\n        model_exists=model_path.is_dir(),\n    )\n\n\ndef environment_manifest(report: EnvironmentReport) -> dict:\n    return {\n        "reference_contract": {\n            "cuda_major": REFERENCE_CUDA_MAJOR,\n            "minimum_driver_major": REFERENCE_MIN_DRIVER_MAJOR,\n            "minimum_compute_capability": (\n                f"{REFERENCE_MIN_COMPUTE_CAPABILITY[0]}."\n                f"{REFERENCE_MIN_COMPUTE_CAPABILITY[1]}"\n            ),\n        },\n        **asdict(report),\n        "hardware_ok": report.hardware_ok,\n    }\n\n\ndef write_environment_manifest(\n    report: EnvironmentReport,\n    path: Path,\n) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(\n        json.dumps(environment_manifest(report), indent=2, ensure_ascii=False)\n        + "\\n",\n        encoding="utf-8",\n    )\n\n\ndef print_environment_report(\n    report: EnvironmentReport,\n    train_dataset_path: Path,\n    test_dataset_path: Path,\n    model_path: Path,\n) -> None:\n    print("Reference GPU contract")\n    print(f"  CUDA major:            {REFERENCE_CUDA_MAJOR}.x")\n    print(f"  NVIDIA driver:         R{REFERENCE_MIN_DRIVER_MAJOR}+")\n    print(\n        "  Minimum GPU compute:   "\n        f"{REFERENCE_MIN_COMPUTE_CAPABILITY[0]}."\n        f"{REFERENCE_MIN_COMPUTE_CAPABILITY[1]} (Turing / SM75)"\n    )\n\n    print("\\nSoftware")\n    print(f"  OS / machine:          {report.os} / {report.machine}")\n    print(f"  Python:                {report.python_version}")\n    print(f"  PyTorch:               {report.torch_version}")\n    print(f"  Torch CUDA build:      {report.torch_cuda_version}")\n    print(\n        "  Compiled CUDA archs:   "\n        + (", ".join(report.compiled_cuda_arches) or "<unavailable>")\n    )\n    print(f"  CUDA usable:           {report.cuda_available}")\n    print(f"  NVIDIA driver:         {report.driver_version}")\n    print(\n        "  Driver contract:       "\n        f"{\'PASS\' if report.driver_supported else \'FAIL\'}"\n    )\n\n    print("\\nGPUs")\n    if not report.gpus:\n        print("  <none>")\n    for gpu in report.gpus:\n        recommended_dtype = "bf16" if gpu.bf16_supported else "fp16"\n        print(\n            f"  [{gpu.index}] {gpu.name} | sm={gpu.compute_capability} | "\n            f"vram={gpu.total_vram_gib:.1f} GiB | "\n            f"dtype={recommended_dtype} | "\n            f"arch={\'PASS\' if gpu.architecture_supported else \'FAIL\'}"\n        )\n\n    print("\\nExternal resources (informational)")\n    print(\n        f"  Train dataset: {train_dataset_path} "\n        f"[{report.train_dataset_exists}]"\n    )\n    print(\n        f"  Test dataset:  {test_dataset_path} "\n        f"[{report.test_dataset_exists}]"\n    )\n    print(f"  Model:         {model_path} [{report.model_exists}]")\n\n    print()\n    print(f"GPU training status: {\'OK\' if report.ok else \'FAILED\'}")\n')
write("src/posttrain_math/resources.py", 'from __future__ import annotations\n\nimport json\nimport shutil\nfrom pathlib import Path\n\nfrom huggingface_hub import HfApi, snapshot_download\n\nDEFAULT_MODEL_REPO = "allenai/OLMo-2-0425-1B"\n# Immutable upstream revision. `download_model` resolves this to the full SHA\n# again and records the result in source.json.\nDEFAULT_MODEL_REVISION = "13cece9360d59bb9db636273ea8d000b67fcc27b"\nDEFAULT_MODEL_DIR = Path("models/olmo-2-0425-1b")\n\n\ndef download_model(\n    *,\n    repo_id: str = DEFAULT_MODEL_REPO,\n    output_dir: Path = DEFAULT_MODEL_DIR,\n    revision: str = DEFAULT_MODEL_REVISION,\n    force: bool = False,\n) -> Path:\n    """Download a Hub model once, then use the local directory everywhere else."""\n    output_dir = Path(output_dir)\n\n    if output_dir.exists() and any(output_dir.iterdir()):\n        if not force:\n            manifest_path = output_dir / "source.json"\n            config_path = output_dir / "config.json"\n            if manifest_path.is_file() and config_path.is_file():\n                print("Model already present; skipping download")\n                print(f"- local directory: {output_dir}")\n                return output_dir\n            raise FileExistsError(\n                f"Model directory is non-empty but incomplete: {output_dir}. "\n                "Use --force to replace it."\n            )\n        shutil.rmtree(output_dir)\n\n    output_dir.parent.mkdir(parents=True, exist_ok=True)\n\n    info = HfApi().model_info(repo_id=repo_id, revision=revision)\n    commit_sha = info.sha\n    if not commit_sha:\n        raise RuntimeError(f"Could not resolve a commit SHA for {repo_id}@{revision}")\n\n    snapshot_download(\n        repo_id=repo_id,\n        revision=commit_sha,\n        local_dir=output_dir,\n    )\n\n    manifest = {\n        "repo_id": repo_id,\n        "requested_revision": revision,\n        "resolved_commit": commit_sha,\n        "local_dir": str(output_dir),\n    }\n    (output_dir / "source.json").write_text(\n        json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n",\n        encoding="utf-8",\n    )\n\n    print("Model downloaded")\n    print(f"- repo: {repo_id}")\n    print(f"- revision: {revision}")\n    print(f"- commit: {commit_sha}")\n    print(f"- local directory: {output_dir}")\n\n    return output_dir\n')
write("README.md", '# posttrain-math\n\nReproducible post-training experiments for mathematical language models. The\ncurrent pipeline uses OLMo 2 1B and Hendrycks MATH for deterministic data\npreparation, completion-only SFT, GRPO, and boxed-answer evaluation.\n\n## Reference environment\n\nThe reference GPU runtime is Linux x86_64 with:\n\n- Python 3.12;\n- the committed `uv.lock`;\n- PyTorch 2.13.0, whose locked Linux dependency set uses CUDA 13.x;\n- NVIDIA driver R580 or newer;\n- NVIDIA compute capability 7.5 or newer (Turing / SM75+).\n\nA T4 (SM75) is supported and uses FP16. Newer GPUs use BF16 automatically when\nPyTorch reports native BF16 support. Multi-GPU runs use one process per visible\nGPU and NCCL through `torchrun`.\n\nThe CUDA 13 architecture and driver requirements are documented in\n[`docs/environment.md`](docs/environment.md).\n\n## Install\n\nInstallation is an explicit user action. Runtime scripts never run `uv sync`.\n\n```bash\ngit clone https://github.com/ywsu0522/post-train-math.git\ncd post-train-math\n\nuv sync --locked --group dev\nuv run --locked --no-sync pytest -q\nuv run --locked --no-sync posttrain-math doctor\n```\n\n`doctor` checks the installed Torch CUDA build, NVIDIA driver, visible GPUs,\ncompute capability, VRAM, BF16 support, and the presence of local model/data\nresources. Model/data absence is informational so the command is useful on a\nfresh clone.\n\nThe legacy `posttrain-math environment` command remains as an alias.\n\n## Pinned external resources\n\nReference experiments do not follow mutable Hugging Face `main` branches.\n\n- model: `allenai/OLMo-2-0425-1B` at pinned revision `13cece9360d59bb9db636273ea8d000b67fcc27b`;\n- dataset: `EleutherAI/hendrycks_math` at\n  `21a5633873b6a120296cce3e2df9d5550074f4a3`.\n\nDownload and prepare once:\n\n```bash\nuv run --locked --no-sync posttrain-math model download\nuv run --locked --no-sync posttrain-math data download\nuv run --locked --no-sync posttrain-math data prepare\n```\n\nEach download records the upstream resolved commit in a local manifest.\n`models/`, `data/`, and `runs/` remain gitignored.\n\nPinning protects experiments from upstream drift. It cannot make an uncached\nremote service available during an outage. For resilient hosted runs, keep\n`HF_HOME` on persistent storage or prefetch before training. See\n[`docs/artifacts.md`](docs/artifacts.md).\n\n## Single- and multi-GPU execution\n\nThe Python core is platform-independent. The thin launch scripts only select\none process per GPU; they do not install dependencies.\n\n```bash\n# one GPU\nbash scripts/launch_gpu.sh 1 eval --split dev --prompt boxed --batch-size 2\n\n# two GPUs\nbash scripts/launch_gpu.sh 2 eval --split dev --prompt boxed --batch-size 2\n\n# all visible GPUs\nbash scripts/launch_gpu.sh auto eval --split dev --prompt boxed --batch-size 2\n```\n\nSFT uses Hugging Face Trainer/Accelerate DDP when launched with multiple\nprocesses. The target global batch is preserved through gradient accumulation.\n\n## SFT\n\nFirst validate the completion-only data contract:\n\n```bash\nuv run --locked --no-sync posttrain-math train inspect\n```\n\nThen run a one-batch LoRA overfit sanity check:\n\n```bash\nuv run --locked --no-sync posttrain-math train overfit-one-batch \\\n  --peft lora \\\n  --precision auto\n```\n\nRun SFT:\n\n```bash\nbash scripts/launch_gpu.sh auto train sft \\\n  --output-dir runs/olmo2-1b-lora-sft-v1 \\\n  --peft lora \\\n  --precision auto \\\n  --gradient-checkpointing\n```\n\nSFT writes Trainer checkpoints plus `final-model/`, configuration, logs,\nsummaries, and plots under the selected run directory.\n\n## Evaluation\n\n```bash\nbash scripts/launch_gpu.sh auto eval \\\n  --model runs/olmo2-1b-lora-sft-v1/final-model \\\n  --split dev \\\n  --prompt boxed \\\n  --output-dir runs/evals/olmo2-1b-lora-sft-v1\n```\n\nEvaluation writes `predictions.jsonl` and `metrics.json`. The current benchmark\nmetric requires a parseable final `\\boxed{...}` answer and verifies its\nmathematical equivalence with the locked `math-verify` version.\n\n## GRPO\n\nGRPO remains a separate module and uses the same answer-verification contract.\n\n```bash\nbash scripts/launch_grpo.sh auto \\\n  --model runs/olmo2-1b-lora-sft-v1/final-model \\\n  --output-dir runs/olmo2-1b-grpo-v1 \\\n  --max-steps 100\n```\n\nEvaluate a sequence of checkpoints with:\n\n```bash\nbash scripts/eval_checkpoints.sh auto runs/olmo2-1b-grpo-v1\n```\n\n## Artifact contract\n\nLocal generated artifacts belong under `runs/<experiment>/`. Keep source\nmanifests, configuration, logs, metrics, checkpoints needed for resume, and the\nfinal model/adapter together.\n\nA local workstation keeps `runs/` on its normal filesystem. Hosted notebook\nfilesystems are ephemeral, so persistence is handled by the notebook workflow,\nwithout Colab/Kaggle branches in the Python package:\n\n- [Colab workflow](docs/colab.md)\n- [Kaggle workflow](docs/kaggle.md)\n- [Artifact and cache policy](docs/artifacts.md)\n\nFor public reproducibility, publish the final adapter/model and the small\nconfiguration/metric manifests to a versioned model repository (for example,\nHugging Face Hub). Preserve large intermediate checkpoints only when they are\nneeded for resume or for a specific analysis.\n\n## Repository directories\n\n| Path | Purpose | Git policy |\n| --- | --- | --- |\n| `src/posttrain_math/` | reusable Python implementation | committed |\n| `tests/` | CPU-testable contracts | committed |\n| `scripts/` | thin generic launch/orchestration helpers | committed |\n| `docs/` | environment and hosted-notebook procedures | committed |\n| `data/` | downloaded and derived dataset files | ignored |\n| `models/` | downloaded base-model snapshot | ignored |\n| `runs/` | training/evaluation artifacts | ignored |\n| `examples/reference_runs/` | small historical reference outputs | committed |\n\nDependency maintenance uses `uv lock`. Installation uses\n`uv sync --locked`. Normal execution uses `uv run --locked --no-sync`, which\nkeeps experiment execution from mutating the environment.\n')
write("docs/environment.md", "# GPU environment contract\n\n## Reference runtime\n\nThis repository's reproducible GPU target is Linux x86_64, Python 3.12, and the\ncommitted `uv.lock`. The current lock resolves `torch==2.13.0` on Linux with a\nCUDA 13.x runtime dependency set.\n\nCUDA 13 requires NVIDIA Turing or newer for the toolkit/library support used by\nthis project. The practical minimum is compute capability 7.5 (SM75). CUDA 13.x\nminor-version compatibility requires an NVIDIA R580+ driver.\n\nReferences:\n\n- NVIDIA CUDA 13.0 release notes:\n  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/\n- uv PyTorch integration:\n  https://docs.astral.sh/uv/guides/integration/pytorch/\n\n## Compatibility policy\n\nThe project checks capabilities instead of cloud-provider names.\n\nRequired for GPU training/evaluation:\n\n1. `torch.cuda.is_available()` is true.\n2. `torch.version.cuda` reports CUDA 13.x for the reference lock.\n3. `nvidia-smi` reports driver major version 580 or newer.\n4. Every visible training GPU has compute capability >= 7.5.\n\nT4 / SM75 runs in FP16. `--precision auto` selects BF16 only when PyTorch\nreports native BF16 support.\n\nRun:\n\n```bash\nuv run --locked --no-sync posttrain-math doctor\n```\n\nTo save a machine-readable report:\n\n```bash\nuv run --locked --no-sync posttrain-math doctor \\\n  --output runs/environment.json\n```\n\n`doctor` also prints `torch.cuda.get_arch_list()` so a reviewer can see the CUDA\narchitectures compiled into the installed Torch binary.\n\n## Multi-GPU\n\n`scripts/launch_gpu.sh` and `scripts/launch_grpo.sh` use `torchrun` with one\nprocess per selected GPU. The reference distributed backend is NCCL on Linux.\n\nA Windows CUDA installation may work for single-GPU Python execution, but the\nportfolio's tested/reference multi-GPU contract is Linux + NCCL. macOS can run\nCPU-side data/verifier tests but is outside the GPU training contract.\n\n## Changing the Torch/CUDA build\n\nTreat a Torch/CUDA change as an environment-contract change:\n\n1. edit `pyproject.toml` if the Torch source/version changes;\n2. regenerate `uv.lock`;\n3. update the constants/checks in `posttrain_math.environment`;\n4. rerun unit tests and `doctor` on a real GPU;\n5. update this document with the new driver and architecture floor.\n\nDo not add Colab/Kaggle conditionals to core Python code. Hosted platforms are\ndeployment procedures around the same hardware contract.\n")
write("docs/colab.md", '# Google Colab workflow\n\nColab is treated as a hosted Linux GPU machine. The package contains no\nColab-specific runtime branches.\n\n## 1. Start a GPU runtime and install once\n\nIn notebook cells:\n\n```python\n!git clone https://github.com/ywsu0522/post-train-math.git\n%cd post-train-math\n!python -m pip install -q "uv>=0.12.6"\n!uv sync --locked --group dev\n!uv run --locked --no-sync posttrain-math doctor\n```\n\nA T4 is SM75 and should pass the architecture check. `--precision auto` will\nselect FP16 on a T4.\n\n## 2. Acquire pinned resources\n\n```python\n!uv run --locked --no-sync posttrain-math model download\n!uv run --locked --no-sync posttrain-math data download\n!uv run --locked --no-sync posttrain-math data prepare\n```\n\nIf you have persistent Drive storage available for Hugging Face cache reuse,\nset `HF_HOME` before downloading. Keep the project code independent of this\nchoice.\n\n## 3. Run training on fast local runtime storage\n\n```python\n!bash scripts/launch_gpu.sh 1 train sft \\\n    --output-dir runs/olmo2-1b-lora-sft-v1 \\\n    --peft lora \\\n    --precision auto \\\n    --gradient-checkpointing\n```\n\nWriting Trainer checkpoints directly to mounted Drive can be slower than\nColab-local storage. A practical workflow is to train under `/content/.../runs`\nand persist selected artifacts after milestones or at the end.\n\n## 4. Persist before the runtime is recycled\n\n### Full run archive to Google Drive\n\n```python\nfrom google.colab import drive\ndrive.mount("/content/drive")\n```\n\n```python\n!tar -C runs -czf \\\n  /content/drive/MyDrive/posttrain-math-olmo2-sft-v1.tgz \\\n  olmo2-1b-lora-sft-v1\n```\n\nThis preserves checkpoints, logs, metrics, and the final adapter together.\n\n### Publish the reproducibility subset to Hugging Face\n\nAuthentication is a user action. Do not commit tokens to the repository.\n\n```python\nfrom huggingface_hub import HfApi, login\n\nlogin()  # enter a write-capable token interactively\napi = HfApi()\nrepo_id = "YOUR_HF_USER/olmo2-1b-math-sft-v1"\napi.create_repo(repo_id, repo_type="model", exist_ok=True)\napi.upload_folder(\n    repo_id=repo_id,\n    repo_type="model",\n    folder_path="runs/olmo2-1b-lora-sft-v1/final-model",\n    path_in_repo=".",\n)\n```\n\nAlso upload/copy the small `run_config.json`, `train_summary.json`, evaluation\n`metrics.json`, and the model/data source manifests. Large intermediate\ncheckpoints are optional unless needed for resume or analysis.\n\n## Reusing cache\n\nColab\'s local filesystem is ephemeral. Persisting `HF_HOME` on Drive can avoid\nre-downloading an already cached model/dataset, although mounted Drive I/O is\nslower. This is a notebook storage decision; the package itself remains\nunchanged.\n')
write("docs/kaggle.md", '# Kaggle Notebook workflow\n\nKaggle is treated as a hosted Linux GPU machine. The package contains no\nKaggle-specific runtime branches.\n\n## 1. Enable a GPU and internet access\n\nInternet access is needed for the initial GitHub clone and uncached Hugging Face\ndownloads.\n\n```python\n!git clone https://github.com/ywsu0522/post-train-math.git\n%cd post-train-math\n!python -m pip install -q "uv>=0.12.6"\n!uv sync --locked --group dev\n!uv run --locked --no-sync posttrain-math doctor\n```\n\nFor a dual-GPU notebook, `doctor` should list both devices.\n\n## 2. Acquire pinned resources\n\n```python\n!uv run --locked --no-sync posttrain-math model download\n!uv run --locked --no-sync posttrain-math data download\n!uv run --locked --no-sync posttrain-math data prepare\n```\n\n## 3. Train using all visible GPUs\n\n```python\n!bash scripts/launch_gpu.sh auto train sft \\\n    --output-dir /kaggle/working/runs/olmo2-1b-lora-sft-v1 \\\n    --peft lora \\\n    --precision auto \\\n    --gradient-checkpointing\n```\n\nFor GRPO:\n\n```python\n!bash scripts/launch_grpo.sh auto \\\n    --model /kaggle/working/runs/olmo2-1b-lora-sft-v1/final-model \\\n    --output-dir /kaggle/working/runs/olmo2-1b-grpo-v1 \\\n    --max-steps 100\n```\n\n## 4. Persist outputs\n\nKeep experiment outputs under `/kaggle/working`. Before leaving the notebook,\nsave a Kaggle notebook version with outputs so the working artifacts are not\nlost with the interactive session.\n\nFor public model reproducibility, also publish the final adapter/model and small\nmanifests to a versioned artifact registry such as Hugging Face Hub.\n\nIf you store `HF_TOKEN` as a Kaggle secret, retrieve it in notebook code and\nauthenticate without printing it:\n\n```python\nfrom kaggle_secrets import UserSecretsClient\nfrom huggingface_hub import login\n\ntoken = UserSecretsClient().get_secret("HF_TOKEN")\nlogin(token=token)\n```\n\nNever place the token in source files, notebook output, or git history.\n\n## Cache policy\n\nThe package uses the same pinned upstream revisions as local and Colab runs.\n`HF_HOME` may point to any persistent cache you provide. Cache placement is a\nnotebook/deployment concern and does not change the training code.\n')
write("docs/artifacts.md", '# Artifact and cache policy\n\n## Three different kinds of state\n\n### Source code\n\nGitHub contains source, tests, documentation, dependency metadata, and small\nreference outputs. Large model/data/run files are not committed.\n\n### Re-creatable external resources\n\n`models/` and `data/` are local materializations of pinned upstream Hugging Face\nresources. The download manifests record the resolved immutable commit.\n\nPinning protects against upstream file movement on `main` and other drift. Once\ncontent exists in the Hugging Face cache it can also be reused during a\ntemporary network outage. No repository can guarantee access to an uncached\nthird-party host during an outage, so critical long-lived experiments should\nprefetch resources or maintain an organization-controlled mirror.\n\n### Experiment artifacts\n\nAll generated experiment state belongs under:\n\n```text\nruns/<experiment>/\n```\n\nKeep together:\n\n- `run_config.json`;\n- training logs and plots;\n- `checkpoint-*` needed for resume;\n- `checkpoint_manifest.json`;\n- `train_summary.json`;\n- `final-model/`;\n- evaluation `predictions.jsonl` and `metrics.json`;\n- a saved `doctor` environment report when publishing a result;\n- the model `source.json` and data `download_manifest.json` used by the run.\n\n## Local workstation\n\n`runs/` is ordinary persistent local storage. Back it up according to the\nimportance of the experiment.\n\n## Hosted notebooks\n\nColab and Kaggle interactive filesystems can disappear after the session.\nTraining code still writes to an ordinary `--output-dir`; the notebook decides\nhow that directory is persisted.\n\n- Colab: archive/copy the run to Google Drive, or upload the publication subset\n  to a remote artifact registry.\n- Kaggle: use `/kaggle/working`, save a notebook version with outputs, and\n  optionally publish the final model elsewhere.\n\nThis separation keeps provider-specific storage APIs out of the Python package.\n\n## What to publish\n\nFor a portfolio/reproducibility model release, publish:\n\n1. final LoRA adapter or final model;\n2. exact base model repo/revision;\n3. exact dataset repo/revision and split seed;\n4. git commit of this repository;\n5. dependency/lock identity;\n6. run config;\n7. environment report;\n8. evaluation metrics.\n\nIntermediate checkpoints are usually operational artifacts. Publish them only\nwhen they are required to reproduce a learning-dynamics claim, resume a public\nrun, or compare checkpoints.\n\n## Hugging Face availability\n\nThe project currently uses Hugging Face as the canonical upstream for OLMo and\nHendrycks MATH. Exact revisions provide reproducibility, while the local/HF\ncache provides reuse. For stronger availability guarantees, add a separately\nmanaged mirror only after confirming redistribution licenses and documenting\nwhich registry is authoritative.\n')
write("tests/test_environment.py", 'from posttrain_math.environment import (\n    REFERENCE_CUDA_MAJOR,\n    REFERENCE_MIN_COMPUTE_CAPABILITY,\n    REFERENCE_MIN_DRIVER_MAJOR,\n    _driver_major,\n)\n\n\ndef test_reference_gpu_contract_is_explicit() -> None:\n    assert REFERENCE_CUDA_MAJOR == 13\n    assert REFERENCE_MIN_DRIVER_MAJOR == 580\n    assert REFERENCE_MIN_COMPUTE_CAPABILITY == (7, 5)\n\n\ndef test_driver_major_parser() -> None:\n    assert _driver_major("580.82.07") == 580\n    assert _driver_major("610.43.02") == 610\n    assert _driver_major(None) is None\n    assert _driver_major("unknown") is None\n')

# Pin Hendrycks MATH without rewriting the rest of data.py.
replace_once(
    "src/posttrain_math/data.py",
    'HENDRYCKS_MATH_REPO = "EleutherAI/hendrycks_math"\nHENDRYCKS_MATH_CONFIGS = (',
    'HENDRYCKS_MATH_REPO = "EleutherAI/hendrycks_math"\n'
    'HENDRYCKS_MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"\n'
    'HENDRYCKS_MATH_CONFIGS = (',
)
replace_once(
    "src/posttrain_math/data.py",
    '    revision: str = "main",\n',
    '    revision: str = HENDRYCKS_MATH_REVISION,\n',
)

# Wire immutable defaults + doctor into the existing CLI.
replace_once(
    "src/posttrain_math/cli.py",
    '    HENDRYCKS_MATH_REPO,\n',
    '    HENDRYCKS_MATH_REPO,\n    HENDRYCKS_MATH_REVISION,\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    inspect_environment,\n    print_environment_report,\n',
    '    inspect_environment,\n    print_environment_report,\n'
    '    write_environment_manifest,\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    DEFAULT_MODEL_DIR,\n    DEFAULT_MODEL_REPO,\n',
    '    DEFAULT_MODEL_DIR,\n    DEFAULT_MODEL_REPO,\n'
    '    DEFAULT_MODEL_REVISION,\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    model_download.add_argument(\n'
    '        "--revision",\n'
    '        default="main",\n'
    '    )\n',
    '    model_download.add_argument(\n'
    '        "--revision",\n'
    '        default=DEFAULT_MODEL_REVISION,\n'
    '    )\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    # environment\n'
    '    env_parser = commands.add_parser(\n'
    '        "environment",\n'
    '    )\n\n'
    '    env_parser.add_argument(\n'
    '        "--train-data",\n'
    '        type=Path,\n'
    '        default=DEFAULT_TRAIN_DATASET,\n'
    '    )\n\n'
    '    env_parser.add_argument(\n'
    '        "--test-data",\n'
    '        type=Path,\n'
    '        default=DEFAULT_TEST_DATASET,\n'
    '    )\n\n'
    '    env_parser.add_argument(\n'
    '        "--model",\n'
    '        type=Path,\n'
    '        default=DEFAULT_MODEL,\n'
    '    )\n',
    '    # hardware/runtime doctor; keep `environment` as a compatibility alias.\n'
    '    for environment_command in ("doctor", "environment"):\n'
    '        env_parser = commands.add_parser(environment_command)\n'
    '        env_parser.add_argument(\n'
    '            "--train-data",\n'
    '            type=Path,\n'
    '            default=DEFAULT_TRAIN_DATASET,\n'
    '        )\n'
    '        env_parser.add_argument(\n'
    '            "--test-data",\n'
    '            type=Path,\n'
    '            default=DEFAULT_TEST_DATASET,\n'
    '        )\n'
    '        env_parser.add_argument(\n'
    '            "--model",\n'
    '            type=Path,\n'
    '            default=DEFAULT_MODEL,\n'
    '        )\n'
    '        env_parser.add_argument(\n'
    '            "--output",\n'
    '            type=Path,\n'
    '            default=None,\n'
    '            help="Optional JSON path for a machine-readable environment report.",\n'
    '        )\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    download_parser.add_argument(\n'
    '        "--revision",\n'
    '        default="main",\n'
    '    )\n',
    '    download_parser.add_argument(\n'
    '        "--revision",\n'
    '        default=HENDRYCKS_MATH_REVISION,\n'
    '    )\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '    print_environment_report(\n'
    '        report,\n'
    '        train_dataset_path=(\n'
    '            args.train_data\n'
    '        ),\n'
    '        test_dataset_path=(\n'
    '            args.test_data\n'
    '        ),\n'
    '        model_path=args.model,\n'
    '    )\n\n'
    '    return 0 if report.ok else 1\n',
    '    print_environment_report(\n'
    '        report,\n'
    '        train_dataset_path=(\n'
    '            args.train_data\n'
    '        ),\n'
    '        test_dataset_path=(\n'
    '            args.test_data\n'
    '        ),\n'
    '        model_path=args.model,\n'
    '    )\n\n'
    '    if args.output is not None:\n'
    '        write_environment_manifest(report, args.output)\n'
    '        print(f"Environment JSON: {args.output}")\n\n'
    '    return 0 if report.ok else 1\n',
)
replace_once(
    "src/posttrain_math/cli.py",
    '        elif args.command == "environment":\n',
    '        elif args.command in {"doctor", "environment"}:\n',
)

# Runtime scripts assume the user already installed/synced the environment.
for script in (
    "scripts/launch_gpu.sh",
    "scripts/launch_grpo.sh",
    "scripts/eval_checkpoints.sh",
):
    replace_all(script, "uv run --locked", "uv run --locked --no-sync")

# Tests should follow the pinned defaults.
replace_once(
    "tests/test_resources.py",
    '    assert resources.DEFAULT_MODEL_REPO == "allenai/OLMo-2-0425-1B"\n'
    '    assert resources.DEFAULT_MODEL_DIR == Path("models/olmo-2-0425-1b")\n',
    '    assert resources.DEFAULT_MODEL_REPO == "allenai/OLMo-2-0425-1B"\n'
    '    assert resources.DEFAULT_MODEL_REVISION == "13cece9360d59bb9db636273ea8d000b67fcc27b"\n'
    '    assert resources.DEFAULT_MODEL_DIR == Path("models/olmo-2-0425-1b")\n',
)
replace_once(
    "tests/test_resources.py",
    '            assert revision == "main"\n',
    '            assert revision == resources.DEFAULT_MODEL_REVISION\n',
)
replace_once(
    "tests/test_data.py",
    '            assert revision == "main"\n',
    '            assert revision == data_module.HENDRYCKS_MATH_REVISION\n',
)

# Remove scripts whose responsibility belongs in installation/docs, not runtime.
for rel in (
    "scripts/bootstrap_colab.sh",
    "scripts/setup_vllm.sh",
    "scripts/update_lock.sh",
    "scripts/update_lock.ps1",
):
    path = ROOT / rel
    if path.exists():
        path.unlink()
        print(f"delete {rel}")

print("\nRefactor applied. pyproject.toml and uv.lock were intentionally unchanged.")
