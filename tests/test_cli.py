"""End-to-end tests for the command line entry points (subprocess, real files)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import build_tiny_onnx

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
FIXTURE_PUZZLE = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)


def run_script(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run a repository script in *cwd* with a clean environment."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("PYTHONWARNINGS", None)
    return subprocess.run(
        [PYTHON, str(REPO_ROOT / script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture
def dataset_dir(tmp_path: Path, tiny_parquet: Path) -> Path:
    """Directory holding ``train_1.parquet`` (the name the scripts default to)."""
    target = tmp_path / "dataset"
    target.mkdir()
    (target / "train_1.parquet").write_bytes(tiny_parquet.read_bytes())
    return target


# --------------------------------------------------------------------------- #
# prepare_data.py
# --------------------------------------------------------------------------- #
def test_list_variants(dataset_dir: Path) -> None:
    result = run_script("prepare_data.py", "--list-variants", cwd=dataset_dir)
    assert result.returncode == 0
    for name in ("v0", "v1", "v2", "v3", "v4"):
        assert f"  {name}  " in result.stdout
    assert "PrepareData4.py" in result.stdout


def test_cli_generates_one_puzzle(dataset_dir: Path) -> None:
    result = run_script(
        "prepare_data.py",
        "--variant",
        "v4",
        "--n-train",
        "1",
        "--n-val",
        "0",
        "--no-progress",
        cwd=dataset_dir,
    )
    assert result.returncode == 0, result.stderr
    assert "variant v4 (PrepareData4.py) -> data_v4" in result.stdout
    assert (dataset_dir / "data_v4" / "train" / "0").is_dir()
    crops = list((dataset_dir / "data_v4" / "train").rglob("*.png"))
    assert len(crops) == 2 * 81  # clean + camera arm
    assert (dataset_dir / "data_v4" / "grids" / "train" / "000000_clean.png").is_file()


def test_cli_honours_out_dir(dataset_dir: Path) -> None:
    target = dataset_dir / "custom_out"
    result = run_script(
        "prepare_data.py",
        "--variant",
        "v0",
        "--out-dir",
        str(target),
        "--n-train",
        "1",
        "--n-val",
        "0",
        "--no-progress",
        cwd=dataset_dir,
    )
    assert result.returncode == 0, result.stderr
    assert (target / "train" / "9").is_dir()
    # page-scope arm also stores the warped page image
    assert (target / "grids" / "train" / "000000_cam.png").is_file()


def test_cli_reports_an_empty_split(dataset_dir: Path) -> None:
    result = run_script(
        "prepare_data.py", "--variant", "v4", "--n-train", "4", "--n-val", "1", "--no-progress",
        cwd=dataset_dir,
    )
    assert result.returncode == 0
    assert "WARNING: split 'val' is empty" in result.stdout
    assert "empty split(s): val" in result.stderr


def test_cli_strict_mode_fails_on_a_short_dataset(dataset_dir: Path) -> None:
    result = run_script(
        "prepare_data.py", "--variant", "v4", "--n-train", "4", "--n-val", "1", "--strict", "--no-progress",
        cwd=dataset_dir,
    )
    assert result.returncode == 1
    assert "error (--strict)" in result.stderr


def test_cli_missing_dataset_gives_a_hint(dataset_dir: Path) -> None:
    result = run_script("prepare_data.py", "--variant", "v4", "--parquet", "nope.parquet", cwd=dataset_dir)
    assert result.returncode == 2
    assert "parquet dataset not found" in result.stderr
    assert "Ritvik19/Sudoku-Dataset" in result.stderr


def test_cli_rejects_a_bad_variant(dataset_dir: Path) -> None:
    result = run_script("prepare_data.py", "--variant", "v9", cwd=dataset_dir)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_cli_rejects_a_negative_count(dataset_dir: Path) -> None:
    result = run_script("prepare_data.py", "--variant", "v4", "--n-train", "-1", cwd=dataset_dir)
    assert result.returncode == 2
    assert "--n-train must be >= 0" in result.stderr


# --------------------------------------------------------------------------- #
# legacy wrappers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "script, out_dir",
    [
        ("PrepareData.py", "data"),
        ("PrepareData1.py", "data_v1"),
        ("PrepareData2.py", "data_v2"),
        ("PrepareData3.py", "data_v3"),
        ("PrepareData4.py", "data_v4"),
    ],
)
def test_legacy_wrappers_still_work(script: str, out_dir: str, dataset_dir: Path) -> None:
    result = run_script(script, "--n-train", "1", "--n-val", "0", "--no-progress", cwd=dataset_dir)
    assert result.returncode == 0, result.stderr
    assert (dataset_dir / out_dir / "train" / "0").is_dir()
    variant = "v0" if script == "PrepareData.py" else f"v{script[11]}"
    assert f"variant {variant}" in result.stdout


# --------------------------------------------------------------------------- #
# RunSudokuProcessor.py
# --------------------------------------------------------------------------- #
@pytest.fixture
def demo_inputs(tmp_path: Path, dataset_dir: Path) -> tuple[Path, Path]:
    """A generated page image plus a tiny ONNX model path."""
    generation = run_script(
        "prepare_data.py", "--variant", "v0", "--n-train", "1", "--n-val", "0", "--no-progress",
        cwd=dataset_dir,
    )
    assert generation.returncode == 0, generation.stderr
    image = dataset_dir / "data" / "grids" / "train" / "000000_clean.png"
    assert image.is_file()
    model = build_tiny_onnx(tmp_path / "tiny.onnx")
    return image, model


def test_run_processor_with_explicit_inputs(tmp_path: Path, demo_inputs) -> None:
    image, model = demo_inputs
    result = run_script(
        "RunSudokuProcessor.py",
        "--image",
        str(image),
        "--onnx",
        str(model),
        "--out-dir",
        str(tmp_path / "out"),
        cwd=tmp_path,
    )
    assert result.returncode in (0, 1)
    assert "Recognized board" in result.stdout
    assert (tmp_path / "out" / "warped.png").is_file()


def test_run_processor_quiet_mode(tmp_path: Path, demo_inputs) -> None:
    image, model = demo_inputs
    result = run_script(
        "RunSudokuProcessor.py",
        "--image",
        str(image),
        "--onnx",
        str(model),
        "--out-dir",
        str(tmp_path / "out"),
        "--quiet",
        "--no-overlay",
        cwd=tmp_path,
    )
    assert result.returncode in (0, 1)
    assert "Recognized board" not in result.stdout
    assert not (tmp_path / "out" / "overlay_solution.png").exists()


def test_run_processor_reports_a_missing_image(tmp_path: Path) -> None:
    result = run_script(
        "RunSudokuProcessor.py", "--image", str(tmp_path / "nope.png"), cwd=tmp_path
    )
    assert result.returncode == 2
    assert "image not found" in result.stdout


def test_run_processor_reports_a_missing_model(tmp_path: Path, demo_inputs) -> None:
    image, _model = demo_inputs
    result = run_script(
        "RunSudokuProcessor.py", "--image", str(image), "--onnx", str(tmp_path / "nope.onnx"), cwd=tmp_path
    )
    assert result.returncode == 2
    assert "onnx not found" in result.stdout


def test_run_processor_without_arguments_explains_what_is_missing(tmp_path: Path) -> None:
    """No generated data and no model in an empty directory: clear error, exit 2."""
    result = run_script("RunSudokuProcessor.py", cwd=tmp_path)
    assert result.returncode == 2
    assert "--image" in result.stdout


def test_documented_image_candidate_is_used(tmp_path: Path, demo_inputs) -> None:
    """The v4 page image is found without --image, as the README workflow does."""
    _image, model = demo_inputs
    candidate = tmp_path / "data_v4" / "grids" / "val"
    candidate.mkdir(parents=True)
    source = tmp_path / "page.png"
    source.write_bytes(Path(demo_inputs[0]).read_bytes())
    (candidate / "000001_cam.png").write_bytes(source.read_bytes())
    result = run_script(
        "RunSudokuProcessor.py", "--onnx", str(model), "--out-dir", str(tmp_path / "out"), cwd=tmp_path
    )
    assert result.returncode in (0, 1)
    assert "data_v4/grids/val/000001_cam.png" in result.stdout
