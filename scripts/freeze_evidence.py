#!/usr/bin/env python3
"""Freeze one verified local benchmark as immutable public evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_verified(source: Path, destination: Path, project_root: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    destination.write_bytes(data)
    return {
        "path": destination.relative_to(project_root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def freeze(project_root: Path, source: Path) -> dict[str, object]:
    report_path = source / "evidence" / "benchmark-report.json"
    summary_path = source / "evidence" / "benchmark-summary.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["environment"]["architecture"] != "arm64":
        raise AssertionError("only a native arm64 run may be frozen")
    if report["gates"]["int8_model_smaller"] is not True:
        raise AssertionError("model-size gate failed")
    if report["gates"]["cosine_similarity_at_least_0_999"] is not True:
        raise AssertionError("output-parity gate failed")

    source_models = source / "models"
    destinations = {
        source_models / "fixture-source.onnx": project_root / "models" / "fixture-source.onnx",
        source_models / "fixture-fp32.onnx": project_root / "models" / "fixture-fp32.onnx",
        source_models / "fixture-int8.onnx": project_root / "models" / "fixture-int8.onnx",
        report_path: project_root / "evidence" / "benchmark-report.json",
        summary_path: project_root / "evidence" / "benchmark-summary.md",
    }
    files = [copy_verified(src, dest, project_root) for src, dest in destinations.items()]

    frozen_report = json.loads(
        (project_root / "evidence" / "benchmark-report.json").read_text(encoding="utf-8")
    )
    if (
        sha256(project_root / "models" / "fixture-fp32.onnx")
        != frozen_report["optimization"]["fp32_sha256"]
    ):
        raise AssertionError("frozen FP32 model differs from report")
    if (
        sha256(project_root / "models" / "fixture-int8.onnx")
        != frozen_report["optimization"]["int8_sha256"]
    ):
        raise AssertionError("frozen INT8 model differs from report")

    receipt: dict[str, object] = {
        "schema_version": 1,
        "project": "ArmQuantLab",
        "source_kind": "local_native_arm64_benchmark",
        "files": files,
        "identity_fields_collected": False,
        "credentials_collected": False,
        "status": "pass",
    }
    receipt_path = project_root / "evidence" / "freeze-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dist/demo"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze(root, args.source.resolve()), indent=2, sort_keys=True))
