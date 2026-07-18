"""Apply ONNX Runtime dynamic INT8 weight quantization."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

MAX_SINGLE_FILE_MODEL_BYTES = 1_000_000_000
ALLOWED_QUANTIZATION_OPS = frozenset({"Conv", "Gemm", "MatMul"})


def inspect_single_file_model(source_path: Path, requested_ops: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a bounded, self-contained ONNX file and return present requested ops."""
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("source model must be a regular non-symlink file")
    if source_path.suffix.lower() != ".onnx":
        raise ValueError("source model must use the .onnx extension")
    if source_path.stat().st_size > MAX_SINGLE_FILE_MODEL_BYTES:
        raise ValueError(
            f"source model exceeds the {MAX_SINGLE_FILE_MODEL_BYTES}-byte safety limit"
        )
    if not requested_ops or not set(requested_ops).issubset(ALLOWED_QUANTIZATION_OPS):
        raise ValueError(
            f"requested ops must be a non-empty subset of {sorted(ALLOWED_QUANTIZATION_OPS)}"
        )
    model = onnx.load(source_path, load_external_data=False)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("models with external tensor data are not supported")
    present = {node.op_type for node in model.graph.node}
    selected = tuple(op for op in requested_ops if op in present)
    if not selected:
        raise ValueError("source model contains none of the requested quantizable ops")
    return selected


def preprocess_fp32(source_path: Path, fp32_path: Path) -> Path:
    """Run ONNX shape inference and graph optimization before quantization."""
    if source_path.resolve() == fp32_path.resolve():
        raise ValueError("preprocessed output must differ from the source model")
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(
        input_model=source_path,
        output_model_path=fp32_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        # Symbolic inference is intended primarily for transformer graphs. This fixture already
        # carries complete ONNX shapes, so keeping it off avoids an unnecessary SymPy dependency.
        skip_symbolic_shape=True,
    )
    return fp32_path


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    *,
    op_types: tuple[str, ...] = ("MatMul",),
) -> Path:
    """Quantize MatMul weights per channel while preserving FP32 activations."""
    if fp32_path.resolve() == int8_path.resolve():
        raise ValueError("INT8 output must differ from the FP32 model")
    if not op_types or not set(op_types).issubset(ALLOWED_QUANTIZATION_OPS):
        raise ValueError(
            f"op_types must be a non-empty subset of {sorted(ALLOWED_QUANTIZATION_OPS)}"
        )
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(op_types),
    )
    return int8_path
