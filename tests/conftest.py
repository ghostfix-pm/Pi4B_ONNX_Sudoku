"""Shared fixtures: a tiny parquet dataset, a tiny deterministic ONNX model and helpers.

Everything here avoids the network and the real (1 GB) Sudoku dataset, so the
whole suite runs offline in a couple of seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: A solved grid used as ground truth for the solver tests.
SOLVED_GRID = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

#: The same puzzle as an 81-character string, 51 blanks.
EASY_PUZZLE_STRING = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)


def grid_from_string(text: str) -> list[list[int]]:
    from sudoku_core import parse_grid_string

    return parse_grid_string(text)


def blank_out(grid: list[list[int]], coords) -> list[list[int]]:
    copy = [row[:] for row in grid]
    for r, c in coords:
        copy[r][c] = 0
    return copy


@pytest.fixture(scope="session")
def solved_grid() -> list[list[int]]:
    return [row[:] for row in SOLVED_GRID]


@pytest.fixture
def tiny_parquet(tmp_path: Path) -> Path:
    """A 4-row parquet with the same schema as the HuggingFace Sudoku dataset."""
    pytest.importorskip("pandas")
    import pandas as pd

    puzzles, solutions = [], []
    for row in range(4):
        blanks = sorted({(row * 13 + i * 7) % 81 for i in range(20 + row)})
        cells = [str(value) for grid_row in SOLVED_GRID for value in grid_row]
        for index in blanks:
            cells[index] = "0"
        puzzles.append("".join(cells))
        solutions.append("".join(str(value) for grid_row in SOLVED_GRID for value in grid_row))
    path = tmp_path / "tiny.parquet"
    pd.DataFrame({"puzzle": puzzles, "solution": solutions}).to_parquet(path, engine="pyarrow")
    return path


def build_tiny_onnx(path: Path, seed: int = 0) -> Path:
    """Write a deterministic, tiny ONNX digit classifier (Conv/ReLU/GAP/Gemm)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    conv_w = rng.normal(size=(2, 1, 3, 3)).astype(np.float32)
    conv_b = rng.normal(size=(2,)).astype(np.float32)
    dense_w = rng.normal(size=(2, 10)).astype(np.float32)
    dense_b = rng.normal(size=(10,)).astype(np.float32)

    nodes = [
        helper.make_node(
            "Conv", ["input", "conv_w", "conv_b"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "dense_w", "dense_b"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_digitnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 28, 28])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 10])],
        initializer=[
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(conv_b, "conv_b"),
            numpy_helper.from_array(dense_w, "dense_w"),
            numpy_helper.from_array(dense_b, "dense_b"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19)], producer_name="ghostfix-tests"
    )
    # onnx 1.23 writes IR version 14 by default, which older ONNX Runtime builds
    # (<= 1.31) refuse to load; IR 10 covers opset 19 and every runtime in use.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))

    # Keep the weights next to the file so the numpy reference can reuse them.
    np.savez(path.with_suffix(".npz"), conv_w=conv_w, conv_b=conv_b, dense_w=dense_w, dense_b=dense_b)
    return path


def onnx_reference(path: Path, batch: np.ndarray) -> np.ndarray:
    """NumPy implementation of the tiny model (independent of ONNX Runtime)."""
    weights = np.load(path.with_suffix(".npz"))
    conv_w = weights["conv_w"]
    conv_b = weights["conv_b"]
    holes = np.zeros((batch.shape[0], 1, batch.shape[2] + 2, batch.shape[3] + 2), dtype=np.float32)
    holes[:, :, 1:-1, 1:-1] = batch[:, :1]
    conv = np.zeros((batch.shape[0], 2, batch.shape[2], batch.shape[3]), dtype=np.float32)
    for channel in range(2):
        for dr in range(3):
            for dc in range(3):
                conv[:, channel] += (
                    conv_w[channel, 0, dr, dc]
                    * holes[:, 0, dr : dr + batch.shape[2], dc : dc + batch.shape[3]]
                )
        conv[:, channel] += conv_b[channel]
    pooled = np.maximum(conv, 0).mean(axis=(2, 3))
    return pooled @ weights["dense_w"] + weights["dense_b"]


@pytest.fixture(scope="session")
def tiny_onnx_path(tmp_path_factory) -> Path:
    pytest.importorskip("onnx")
    directory = tmp_path_factory.mktemp("onnx")
    return build_tiny_onnx(directory / "tiny_digitnet.onnx")


class StubSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``."""

    def __init__(self, logits: np.ndarray):
        self._logits = np.asarray(logits, dtype=np.float32)

    def get_inputs(self):  # noqa: D102 - mimics the ORT API
        return [type("Input", (), {"name": "input", "shape": ["N", 1, 28, 28]})()]

    def get_outputs(self):  # noqa: D102 - mimics the ORT API
        return [type("Output", (), {"name": "logits", "shape": ["N", 10]})()]

    def run(self, output_names, input_feed):  # noqa: D102 - mimics the ORT API
        batch = np.asarray(input_feed["input"], dtype=np.float32)
        logits = self._logits
        if logits.shape[0] == 1:
            logits = np.repeat(logits, batch.shape[0], axis=0)
        return [logits]


@pytest.fixture
def stub_session_factory():
    """Return a factory building a stub session that always predicts one class."""

    def make(class_index: int = 3, confidence_logit: float = 20.0) -> StubSession:
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, class_index] = confidence_logit
        return StubSession(logits)

    return make
