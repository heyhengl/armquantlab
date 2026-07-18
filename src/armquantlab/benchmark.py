"""Benchmark FP32 and INT8 ONNX models without collecting machine identity."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from armquantlab.model import MODEL_SEED

MAX_GENERATED_INPUT_ELEMENTS = 16_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _latency_samples(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    warmup: int,
    repeats: int,
    iterations_per_sample: int,
) -> list[float]:
    for _ in range(warmup):
        session.run(None, inputs)

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(iterations_per_sample):
            session.run(None, inputs)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        samples.append(elapsed_ms / iterations_per_sample)
    return samples


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64).ravel()
    right_flat = right.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator == 0:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def _input_contract(session: ort.InferenceSession) -> list[dict[str, object]]:
    contract: list[dict[str, object]] = []
    for index, item in enumerate(session.get_inputs()):
        if item.type != "tensor(float)":
            raise ValueError(
                f"only float32 tensor inputs are supported; input {index} uses {item.type}"
            )
        if not item.shape:
            raise ValueError(f"input {index} must have at least one dimension")
        shape: list[int | str] = []
        for dimension_index, dimension in enumerate(item.shape):
            if isinstance(dimension, int) and dimension > 0:
                shape.append(dimension)
            elif dimension_index == 0:
                shape.append("batch")
            else:
                raise ValueError(
                    "only the leading batch dimension may be dynamic; "
                    f"input {index} dimension {dimension_index} is unresolved"
                )
        contract.append(
            {
                "index": index,
                "dtype": "float32",
                "shape": shape,
            }
        )
    if not contract:
        raise ValueError("model must expose at least one input")
    return contract


def _compatible_contracts(
    fp32_session: ort.InferenceSession,
    int8_session: ort.InferenceSession,
) -> list[dict[str, object]]:
    fp32_contract = _input_contract(fp32_session)
    int8_contract = _input_contract(int8_session)
    if fp32_contract != int8_contract:
        raise ValueError("FP32 and INT8 models expose different input contracts")
    return fp32_contract


def _automatic_batch_sizes(contract: list[dict[str, object]]) -> tuple[int, ...]:
    leading = [item["shape"][0] for item in contract]
    if all(dimension == "batch" for dimension in leading):
        return (1, 8, 32)
    if all(isinstance(dimension, int) for dimension in leading) and len(set(leading)) == 1:
        return (int(leading[0]),)
    raise ValueError("model inputs do not share one compatible batch dimension")


def _generated_inputs(
    session: ort.InferenceSession,
    contract: list[dict[str, object]],
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    element_count = 0
    for metadata, spec in zip(session.get_inputs(), contract, strict=True):
        dimensions: list[int] = []
        for index, dimension in enumerate(spec["shape"]):
            resolved = batch_size if dimension == "batch" else int(dimension)
            if index == 0 and dimension != "batch" and resolved != batch_size:
                raise ValueError(
                    f"requested batch {batch_size} conflicts with fixed model batch {resolved}"
                )
            dimensions.append(resolved)
        elements = int(np.prod(np.asarray(dimensions, dtype=np.int64)))
        element_count += elements
        if element_count > MAX_GENERATED_INPUT_ELEMENTS:
            raise ValueError(
                f"generated inputs exceed the {MAX_GENERATED_INPUT_ELEMENTS}-element safety limit"
            )
        arrays[metadata.name] = rng.standard_normal(tuple(dimensions)).astype(np.float32)
    return arrays


def _remap_inputs(
    source: dict[str, np.ndarray],
    source_session: ort.InferenceSession,
    target_session: ort.InferenceSession,
) -> dict[str, np.ndarray]:
    source_values = [source[item.name] for item in source_session.get_inputs()]
    return {
        metadata.name: value
        for metadata, value in zip(target_session.get_inputs(), source_values, strict=True)
    }


def _output_parity(
    fp32_outputs: list[np.ndarray],
    int8_outputs: list[np.ndarray],
) -> dict[str, float]:
    if len(fp32_outputs) != len(int8_outputs):
        raise ValueError("FP32 and INT8 models expose different output counts")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    absolute_sum = 0.0
    maximum = 0.0
    element_count = 0
    for left, right in zip(fp32_outputs, int8_outputs, strict=True):
        if left.shape != right.shape:
            raise ValueError("FP32 and INT8 output shapes differ")
        left_flat = left.astype(np.float64).ravel()
        right_flat = right.astype(np.float64).ravel()
        difference = np.abs(left_flat - right_flat)
        dot += float(np.dot(left_flat, right_flat))
        left_norm += float(np.dot(left_flat, left_flat))
        right_norm += float(np.dot(right_flat, right_flat))
        absolute_sum += float(difference.sum())
        maximum = max(maximum, float(difference.max(initial=0.0)))
        element_count += difference.size
    denominator = float(np.sqrt(left_norm) * np.sqrt(right_norm))
    cosine = dot / denominator if denominator else float(dot == 0 and left_norm == right_norm)
    return {
        "cosine_similarity": round(cosine, 9),
        "max_absolute_error": round(maximum, 9),
        "mean_absolute_error": round(absolute_sum / max(element_count, 1), 9),
    }


def benchmark_models(
    fp32_path: Path,
    int8_path: Path,
    output_dir: Path,
    *,
    batch_sizes: tuple[int, ...] | None = (1, 8, 32),
    warmup: int = 10,
    repeats: int = 40,
    iterations_per_sample: int = 200,
    input_seed: int = MODEL_SEED + 1,
    workload_kind: str = "synthetic_three_layer_mlp",
    quantized_ops: tuple[str, ...] = ("MatMul",),
    source_model_path: Path | None = None,
    model_rights_acknowledged: bool = True,
) -> dict[str, object]:
    """Run paired benchmarks and persist a machine-readable evidence report."""
    if warmup < 1 or repeats < 5:
        raise ValueError("warmup must be >= 1 and repeats must be >= 5")
    if iterations_per_sample < 1:
        raise ValueError("iterations per sample must be positive")

    fp32_session = _session(fp32_path)
    int8_session = _session(int8_path)
    contract = _compatible_contracts(fp32_session, int8_session)
    if batch_sizes is None:
        batch_sizes = _automatic_batch_sizes(contract)
    if not batch_sizes or any(batch < 1 for batch in batch_sizes):
        raise ValueError("batch sizes must be positive")
    rng = np.random.default_rng(input_seed)
    batches: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        inputs = _generated_inputs(fp32_session, contract, batch_size, rng)
        int8_inputs = _remap_inputs(inputs, fp32_session, int8_session)
        fp32_output = fp32_session.run(None, inputs)
        int8_output = int8_session.run(None, int8_inputs)
        fp32_samples = _latency_samples(
            fp32_session, inputs, warmup, repeats, iterations_per_sample
        )
        int8_samples = _latency_samples(
            int8_session, int8_inputs, warmup, repeats, iterations_per_sample
        )
        fp32_p50 = statistics.median(fp32_samples)
        int8_p50 = statistics.median(int8_samples)
        batches.append(
            {
                "batch_size": batch_size,
                "fp32_latency_ms": {
                    "p50": round(fp32_p50, 6),
                    "p95": round(_percentile(fp32_samples, 95), 6),
                },
                "int8_latency_ms": {
                    "p50": round(int8_p50, 6),
                    "p95": round(_percentile(int8_samples, 95), 6),
                },
                "p50_speedup": round(fp32_p50 / int8_p50, 6),
                "fp32_items_per_second": round(batch_size / (fp32_p50 / 1000), 3),
                "int8_items_per_second": round(batch_size / (int8_p50 / 1000), 3),
                "output_parity": _output_parity(fp32_output, int8_output),
            }
        )

    fp32_bytes = fp32_path.stat().st_size
    int8_bytes = int8_path.stat().st_size
    report: dict[str, object] = {
        "schema_version": 1,
        "project": "ArmQuantLab",
        "workload": {
            "kind": workload_kind,
            "input_seed": input_seed,
            "input_contract": contract,
            "model_rights_acknowledged": model_rights_acknowledged,
            "input_values_persisted": False,
            "source_path_persisted": False,
            "truth_boundary": (
                "Measures runtime optimization and numerical parity only; it does not prove "
                "task accuracy or production model quality."
            ),
        },
        "environment": {
            "architecture": platform.machine(),
            "operating_system": platform.system(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "thread_count": 1,
            "hostname_collected": False,
            "user_identity_collected": False,
        },
        "optimization": {
            "method": "ONNX Runtime dynamic per-channel INT8 weight quantization",
            "quantized_ops": list(quantized_ops),
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
            "size_reduction_percent": round((1 - int8_bytes / fp32_bytes) * 100, 3),
            "fp32_sha256": _sha256(fp32_path),
            "int8_sha256": _sha256(int8_path),
        },
        "protocol": {
            "warmup_runs": warmup,
            "measured_runs": repeats,
            "iterations_per_sample": iterations_per_sample,
            "batch_sizes": list(batch_sizes),
            "timer": "time.perf_counter_ns",
        },
        "batches": batches,
        "gates": {
            "native_arm64": platform.machine() == "arm64",
            "int8_model_smaller": int8_bytes < fp32_bytes,
            "cosine_similarity_at_least_0_999": all(
                batch["output_parity"]["cosine_similarity"] >= 0.999 for batch in batches
            ),
            "int8_faster_for_any_measured_batch": any(
                batch["p50_speedup"] > 1 for batch in batches
            ),
        },
    }
    if source_model_path is not None:
        report["optimization"]["source_bytes"] = source_model_path.stat().st_size
        report["optimization"]["source_sha256"] = _sha256(source_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "benchmark-summary.md").write_text(_summary(report), encoding="utf-8")
    return report


def _summary(report: dict[str, object]) -> str:
    optimization = report["optimization"]
    environment = report["environment"]
    rows = [
        "# ArmQuantLab benchmark summary",
        "",
        f"- Architecture: `{environment['architecture']}`",
        f"- Provider: `{environment['provider']}`",
        f"- FP32 model: `{optimization['fp32_bytes']}` bytes",
        f"- INT8 model: `{optimization['int8_bytes']}` bytes",
        f"- Size reduction: `{optimization['size_reduction_percent']}%`",
        "",
        "| Batch | FP32 p50 ms | INT8 p50 ms | Speedup | Cosine similarity |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in report["batches"]:
        rows.append(
            "| {batch_size} | {fp32} | {int8} | {speedup}x | {cosine} |".format(
                batch_size=batch["batch_size"],
                fp32=batch["fp32_latency_ms"]["p50"],
                int8=batch["int8_latency_ms"]["p50"],
                speedup=batch["p50_speedup"],
                cosine=batch["output_parity"]["cosine_similarity"],
            )
        )
    rows.extend(
        [
            "",
            "## Truth boundary",
            "",
            str(report["workload"]["truth_boundary"]),
            (
                "Latency is local evidence for this run and can vary with hardware, load, "
                "and runtime version."
            ),
            "",
        ]
    )
    return "\n".join(rows)
