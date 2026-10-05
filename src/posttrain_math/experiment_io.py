"""Small reproducibility and review-bundle helpers for RL experiments."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_files() -> list[Path]:
    return sorted(Path('src/posttrain_math').glob('*.py')) + [Path('pyproject.toml'), Path('uv.lock')]


def code_hash() -> str:
    return hashlib.sha256(json.dumps([(p.as_posix(), file_hash(p)) for p in source_files()]).encode()).hexdigest()


def package_results(root: Path) -> Path:
    target = root / 'analysis_bundle.zip'
    temporary = root / 'analysis_bundle.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if any(p.startswith('checkpoint-') or p == 'final-model' for p in relative.parts):
                continue
            if path.is_file() and (path.suffix in {'.json', '.jsonl', '.md', '.log'}
                                   or path.name == 'code_snapshot.zip'):
                archive.write(path, relative.as_posix())
    temporary.replace(target)
    return target
