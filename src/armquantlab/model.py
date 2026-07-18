"""Generate a deterministic synthetic ONNX workload for optimization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT_WIDTH = 128
HIDDEN_WIDTH = 512
OUTPUT_WIDTH = 64
MODEL_SEED = 20260718


def _weight(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    scale = np.float32(1.0 / np.sqrt(rows))
    return (rng.standard_normal((rows, columns)).astype(np.float32) * scale).astype(np.float32)


def build_fixture_model(path: Path) -> Path:
    """Create a deterministic three-layer MLP with dynamic batch dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MODEL_SEED)

    initializers = [
        numpy_helper.from_array(_weight(rng, INPUT_WIDTH, HIDDEN_WIDTH), name="weight_1"),
        numpy_helper.from_array(np.zeros(HIDDEN_WIDTH, dtype=np.float32), name="bias_1"),
        numpy_helper.from_array(_weight(rng, HIDDEN_WIDTH, HIDDEN_WIDTH), name="weight_2"),
        numpy_helper.from_array(np.zeros(HIDDEN_WIDTH, dtype=np.float32), name="bias_2"),
        numpy_helper.from_array(_weight(rng, HIDDEN_WIDTH, OUTPUT_WIDTH), name="weight_3"),
        numpy_helper.from_array(np.zeros(OUTPUT_WIDTH, dtype=np.float32), name="bias_3"),
    ]

    nodes = [
        helper.make_node("MatMul", ["input", "weight_1"], ["matmul_1"], name="matmul_1"),
        helper.make_node("Add", ["matmul_1", "bias_1"], ["add_1"], name="add_1"),
        helper.make_node("Relu", ["add_1"], ["relu_1"], name="relu_1"),
        helper.make_node("MatMul", ["relu_1", "weight_2"], ["matmul_2"], name="matmul_2"),
        helper.make_node("Add", ["matmul_2", "bias_2"], ["add_2"], name="add_2"),
        helper.make_node("Relu", ["add_2"], ["relu_2"], name="relu_2"),
        helper.make_node("MatMul", ["relu_2", "weight_3"], ["matmul_3"], name="matmul_3"),
        helper.make_node("Add", ["matmul_3", "bias_3"], ["output"], name="output_add"),
    ]

    graph = helper.make_graph(
        nodes,
        "armquantlab_fixture_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", INPUT_WIDTH])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", OUTPUT_WIDTH])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="armquantlab",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
