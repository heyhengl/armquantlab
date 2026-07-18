#!/usr/bin/env python3
"""Verify the generated ArmQuantLab evidence and public-source privacy boundary."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "api_key": re.compile(r"\b(?:sk-|AIza)[A-Za-z0-9_-]{20,}\b"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "credential_url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def iter_public_text(root: Path):
    ignored = {".git", ".venv", ".pytest_cache", ".ruff_cache", "dist"}
    allowed = {".md", ".py", ".swift", ".toml", ".txt", ".json", ".gitignore"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.name == "uv.lock" or path.name == "LICENSE":
            continue
        if path.suffix in allowed or path.name == ".gitignore":
            yield path, path.read_text(encoding="utf-8")


def verify(root: Path) -> dict[str, object]:
    evidence = root / "evidence"
    report_path = evidence / "benchmark-report.json"
    summary_path = evidence / "benchmark-summary.md"
    if not report_path.is_file() or not summary_path.is_file():
        raise AssertionError("generated benchmark evidence is missing")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["environment"]["architecture"] != "arm64":
        raise AssertionError("authoritative demo was not measured on native arm64")
    if report["environment"]["hostname_collected"] is not False:
        raise AssertionError("hostname collection boundary failed")
    if report["environment"]["user_identity_collected"] is not False:
        raise AssertionError("user identity collection boundary failed")
    if report["gates"]["int8_model_smaller"] is not True:
        raise AssertionError("INT8 model is not smaller")
    if report["gates"]["cosine_similarity_at_least_0_999"] is not True:
        raise AssertionError("output parity gate failed")
    if len(report["batches"]) != 3:
        raise AssertionError("authoritative benchmark must include three batch sizes")

    models = root / "models"
    fp32_model = models / "fixture-fp32.onnx"
    int8_model = models / "fixture-int8.onnx"
    source_model = models / "fixture-source.onnx"
    if not all(path.is_file() for path in (source_model, fp32_model, int8_model)):
        raise AssertionError("frozen model evidence is missing")
    if sha256(fp32_model) != report["optimization"]["fp32_sha256"]:
        raise AssertionError("FP32 model hash does not match report")
    if sha256(int8_model) != report["optimization"]["int8_sha256"]:
        raise AssertionError("INT8 model hash does not match report")

    findings: list[dict[str, str]] = []
    scanned = 0
    for path, text in iter_public_text(root):
        scanned += 1
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": str(path.relative_to(root)), "pattern": name})
    if findings:
        raise AssertionError(f"privacy scan findings: {findings}")

    return {
        "architecture": report["environment"]["architecture"],
        "batches": len(report["batches"]),
        "int8_model_smaller": True,
        "output_parity": True,
        "privacy_files_scanned": scanned,
        "privacy_findings": 0,
        "status": "pass",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(verify(project_root), indent=2, sort_keys=True))
