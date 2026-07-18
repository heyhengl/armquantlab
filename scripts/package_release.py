#!/usr/bin/env python3
"""Build a deterministic public ArmQuantLab source and evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = (
    ".gitignore",
    ".python-version",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
)
PUBLIC_GLOBS = (
    "evidence/*",
    "models/*.onnx",
    "scripts/*.py",
    "scripts/*.swift",
    "src/armquantlab/*.py",
    "submission/*.md",
    "tests/*.py",
)
FORBIDDEN_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
}
ARCHIVE_NAME = "armquantlab-0.1.0-source-and-evidence.zip"
MANIFEST_NAME = "armquantlab-0.1.0-source-and-evidence-manifest.json"
ZIP_TIMESTAMP = (2026, 7, 18, 0, 0, 0)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(root: Path) -> list[Path]:
    candidates = [root / name for name in ROOT_FILES]
    for pattern in PUBLIC_GLOBS:
        candidates.extend(root.glob(pattern))
    files: list[Path] = []
    for path in sorted(set(candidates)):
        if not path.exists():
            raise FileNotFoundError(path.relative_to(root))
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release input must be a regular file: {relative}")
        if FORBIDDEN_PARTS.intersection(relative.parts):
            raise ValueError(f"forbidden release path: {relative}")
        files.append(path)
    return files


def build(root: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_NAME
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in collect(root):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, data, compresslevel=9)
            entries.append({"path": relative, "bytes": len(data), "sha256": digest(data)})

    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": "ArmQuantLab",
        "version": "0.1.0",
        "archive": ARCHIVE_NAME,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": digest(archive_path.read_bytes()),
        "file_count": len(entries),
        "files": entries,
        "scope": {
            "root_files": list(ROOT_FILES),
            "public_globs": list(PUBLIC_GLOBS),
            "forbidden_parts": sorted(FORBIDDEN_PARTS),
        },
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify(archive_path, manifest)
    return manifest


def verify(archive_path: Path, manifest: dict[str, object]) -> None:
    entries = manifest["files"]
    if not isinstance(entries, list):
        raise AssertionError("manifest files must be a list")
    expected = {entry["path"]: entry for entry in entries if isinstance(entry, dict)}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise AssertionError("archive inventory mismatch")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise AssertionError(f"unsafe path: {name}")
            if FORBIDDEN_PARTS.intersection(pure.parts):
                raise AssertionError(f"forbidden path: {name}")
            data = archive.read(name)
            if len(data) != expected[name]["bytes"] or digest(data) != expected[name]["sha256"]:
                raise AssertionError(f"digest mismatch: {name}")
    if digest(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise AssertionError("archive digest mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dist/release"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    result = build(project_root, args.out.resolve())
    print(
        json.dumps(
            {
                "archive": result["archive"],
                "archive_sha256": result["archive_sha256"],
                "file_count": result["file_count"],
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )
