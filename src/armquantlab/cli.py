"""Command-line interface for the reproducible Arm64 optimization demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from armquantlab.benchmark import benchmark_models
from armquantlab.model import build_fixture_model
from armquantlab.optimize import inspect_single_file_model, preprocess_fp32, quantize_int8


def _positive_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not batches or any(item < 1 for item in batches):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batch sizes must not repeat")
    return batches


def _quantization_ops(value: str) -> tuple[str, ...]:
    ops = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"Conv", "Gemm", "MatMul"}
    if not ops or not set(ops).issubset(allowed):
        raise argparse.ArgumentTypeError(f"ops must be a subset of {sorted(allowed)}")
    if len(set(ops)) != len(ops):
        raise argparse.ArgumentTypeError("ops must not repeat")
    return ops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="armquantlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="generate, quantize, and benchmark the fixture")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--warmup", type=int, default=10)
    demo.add_argument("--repeats", type=int, default=40)
    demo.add_argument("--iterations", type=int, default=200)

    optimize = subparsers.add_parser(
        "optimize",
        help="quantize and benchmark a rights-cleared, compatible single-file ONNX model",
    )
    optimize.add_argument("--model", type=Path, required=True)
    optimize.add_argument("--out", type=Path, required=True)
    optimize.add_argument("--batch-sizes", type=_positive_batches)
    optimize.add_argument("--ops", type=_quantization_ops, default=("MatMul", "Gemm"))
    optimize.add_argument("--warmup", type=int, default=10)
    optimize.add_argument("--repeats", type=int, default=40)
    optimize.add_argument("--iterations", type=int, default=200)
    optimize.add_argument("--seed", type=int, default=20260719)
    optimize.add_argument("--acknowledge-model-rights", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = args.out / "models"
    evidence = args.out / "evidence"
    if args.command == "demo":
        source_path = build_fixture_model(models / "fixture-source.onnx")
        selected_ops = ("MatMul",)
        fp32_path = preprocess_fp32(source_path, models / "fixture-fp32.onnx")
        int8_path = quantize_int8(
            fp32_path,
            models / "fixture-int8.onnx",
            op_types=selected_ops,
        )
        batch_sizes = (1, 8, 32)
        workload_kind = "synthetic_three_layer_mlp"
        input_seed = 20260719
    elif args.command == "optimize":
        source_path = args.model
        selected_ops = inspect_single_file_model(source_path, args.ops)
        fp32_path = preprocess_fp32(source_path, models / "candidate-fp32.onnx")
        int8_path = quantize_int8(
            fp32_path,
            models / "candidate-int8.onnx",
            op_types=selected_ops,
        )
        batch_sizes = args.batch_sizes
        workload_kind = "rights_acknowledged_user_supplied_onnx"
        input_seed = args.seed
    else:
        raise AssertionError("unreachable command")

    report = benchmark_models(
        fp32_path,
        int8_path,
        evidence,
        batch_sizes=batch_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations_per_sample=args.iterations,
        input_seed=input_seed,
        workload_kind=workload_kind,
        quantized_ops=selected_ops,
        source_model_path=source_path,
        model_rights_acknowledged=True,
    )
    print(
        json.dumps(
            {
                "architecture": report["environment"]["architecture"],
                "int8_model_smaller": report["gates"]["int8_model_smaller"],
                "native_arm64": report["gates"]["native_arm64"],
                "report": "evidence/benchmark-report.json",
                "size_reduction_percent": report["optimization"]["size_reduction_percent"],
                "status": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
