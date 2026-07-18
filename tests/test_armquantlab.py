from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import onnx
import pytest

from armquantlab.benchmark import benchmark_models
from armquantlab.cli import main
from armquantlab.model import build_fixture_model
from armquantlab.optimize import inspect_single_file_model, preprocess_fp32, quantize_int8


def build_pair(root: Path) -> tuple[Path, Path]:
    source = build_fixture_model(root / "fixture-source.onnx")
    fp32 = preprocess_fp32(source, root / "fixture-fp32.onnx")
    int8 = quantize_int8(fp32, root / "fixture-int8.onnx")
    return fp32, int8


def test_model_generation_and_quantization_are_deterministic(tmp_path: Path) -> None:
    first_fp32, first_int8 = build_pair(tmp_path / "first")
    second_fp32, second_int8 = build_pair(tmp_path / "second")
    assert first_fp32.read_bytes() == second_fp32.read_bytes()
    assert first_int8.read_bytes() == second_int8.read_bytes()
    assert first_int8.stat().st_size < first_fp32.stat().st_size
    onnx.checker.check_model(onnx.load(first_fp32))
    onnx.checker.check_model(onnx.load(first_int8))


def test_benchmark_records_parity_without_identity(tmp_path: Path) -> None:
    fp32, int8 = build_pair(tmp_path / "models")
    report = benchmark_models(
        fp32,
        int8,
        tmp_path / "evidence",
        batch_sizes=(1, 4),
        warmup=1,
        repeats=5,
        iterations_per_sample=2,
    )
    assert report["gates"]["int8_model_smaller"] is True
    assert report["gates"]["cosine_similarity_at_least_0_999"] is True
    assert report["environment"]["hostname_collected"] is False
    assert report["environment"]["user_identity_collected"] is False
    assert len(report["batches"]) == 2
    text = (tmp_path / "evidence" / "benchmark-report.json").read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)


def test_cli_emits_complete_evidence(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "demo",
                "--out",
                str(tmp_path),
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--iterations",
                "2",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "pass"
    assert receipt["int8_model_smaller"] is True
    assert (tmp_path / "models" / "fixture-fp32.onnx").is_file()
    assert (tmp_path / "models" / "fixture-int8.onnx").is_file()
    assert (tmp_path / "models" / "fixture-source.onnx").is_file()
    assert (tmp_path / "evidence" / "benchmark-report.json").is_file()
    assert (tmp_path / "evidence" / "benchmark-summary.md").is_file()


def test_compatible_user_model_path_is_rights_gated_and_path_private(
    tmp_path: Path, capsys
) -> None:
    source = build_fixture_model(tmp_path / "private-source-name.onnx")
    with pytest.raises(SystemExit):
        main(
            [
                "optimize",
                "--model",
                str(source),
                "--out",
                str(tmp_path / "missing-ack"),
            ]
        )

    output = tmp_path / "optimized"
    assert (
        main(
            [
                "optimize",
                "--model",
                str(source),
                "--out",
                str(output),
                "--batch-sizes",
                "1,4",
                "--ops",
                "MatMul",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--iterations",
                "2",
                "--acknowledge-model-rights",
            ]
        )
        == 0
    )
    cli_receipt = json.loads(capsys.readouterr().out)
    assert cli_receipt["status"] == "pass"
    report_path = output / "evidence" / "benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = report_path.read_text(encoding="utf-8")
    assert report["workload"]["kind"] == "rights_acknowledged_user_supplied_onnx"
    assert report["workload"]["model_rights_acknowledged"] is True
    assert report["workload"]["input_values_persisted"] is False
    assert report["workload"]["source_path_persisted"] is False
    assert report["optimization"]["quantized_ops"] == ["MatMul"]
    assert report["optimization"]["source_sha256"]
    assert str(source) not in serialized
    assert source.name not in serialized
    assert (output / "models" / "candidate-fp32.onnx").is_file()
    assert (output / "models" / "candidate-int8.onnx").is_file()


def test_model_inspection_rejects_symlink_and_absent_ops(tmp_path: Path) -> None:
    source = build_fixture_model(tmp_path / "source.onnx")
    link = tmp_path / "linked.onnx"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        inspect_single_file_model(link, ("MatMul",))
    with pytest.raises(ValueError, match="none of the requested"):
        inspect_single_file_model(source, ("Conv",))


def test_public_release_archive_is_reproducible_and_scoped(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        result = subprocess.run(
            [sys.executable, "scripts/package_release.py", "--out", str(output)],
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout)["status"] == "pass"
    first_manifest = json.loads(
        (first / "armquantlab-0.1.0-source-and-evidence-manifest.json").read_text()
    )
    second_manifest = json.loads(
        (second / "armquantlab-0.1.0-source-and-evidence-manifest.json").read_text()
    )
    assert first_manifest["archive_sha256"] == second_manifest["archive_sha256"]
    with ZipFile(first / "armquantlab-0.1.0-source-and-evidence.zip") as archive:
        names = archive.namelist()
    assert "evidence/benchmark-report.json" in names
    assert "models/fixture-int8.onnx" in names
    assert "scripts/build_demo_frames.py" in names
    assert "scripts/render_demo.swift" in names
    assert all("dist" not in Path(name).parts for name in names)
