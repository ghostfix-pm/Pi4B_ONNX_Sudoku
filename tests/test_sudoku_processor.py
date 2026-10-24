"""Tests for :mod:`sudoku_processor` — preprocessing, inference and the pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from conftest import EASY_PUZZLE_STRING, SOLVED_GRID, build_tiny_onnx, grid_from_string, onnx_reference

from sudoku_core import SIZE, find_conflicts, find_empty, grid_to_string, parse_grid_string
from sudoku_data import render_page
from sudoku_processor import SudokuProcessor, softmax, solve_sudoku, validate_board, write_image


def make_processor(tmp_path: Path, tiny_onnx_path: Path, **kwargs) -> SudokuProcessor:
    options = {
        "save_debug_images": False,
        "verbose": False,
        "debug_dir": tmp_path,
    }
    options.update(kwargs)
    return SudokuProcessor(onnx_path=tiny_onnx_path, **options)


def synthetic_cell(digit: int | None = None, size: int = 50) -> np.ndarray:
    """White cell with an optional black digit in the middle."""
    cell = np.full((size, size, 3), 255, np.uint8)
    if digit is not None:
        cv.putText(
            cell,
            str(digit),
            (int(size * 0.25), int(size * 0.72)),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
    return cell


@pytest.fixture
def page_image(tmp_path: Path) -> Path:
    """A rendered page with an added white margin, saved as a PNG."""
    page, _ = render_page(parse_grid_string(EASY_PUZZLE_STRING))
    padded = cv.copyMakeBorder(page, 40, 40, 40, 40, cv.BORDER_CONSTANT, value=(255, 255, 255))
    path = tmp_path / "page.png"
    assert cv.imwrite(str(path), padded)
    return path


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_missing_model_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        SudokuProcessor(onnx_path=tmp_path / "nope.onnx", save_debug_images=False)


def test_a_session_or_a_model_path_is_required() -> None:
    with pytest.raises(ValueError, match="either onnx_path or session is required"):
        SudokuProcessor(onnx_path=None, save_debug_images=False)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input_size": 0}, "input_size must be positive"),
        ({"warp_size": 4}, "warp_size must be at least 9"),
        ({"blank_conf_threshold": 1.5}, "blank_conf_threshold must be in"),
        ({"blank_class": 12}, "blank_class must be in 0..9"),
        ({"blank_ink_max": -1}, "blank_ink_max must be >= 0"),
    ],
)
def test_constructor_validation(tiny_onnx_path: Path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SudokuProcessor(onnx_path=tiny_onnx_path, save_debug_images=False, **kwargs)


def test_non_multiple_warp_size_warns_about_dropped_pixels(tiny_onnx_path: Path) -> None:
    with pytest.warns(UserWarning, match="not a multiple of 9"):
        processor = SudokuProcessor(onnx_path=tiny_onnx_path, warp_size=100, save_debug_images=False)
    assert processor.warp_size == 100


def test_session_injection_skips_the_model_file(tiny_onnx_path: Path, stub_session_factory) -> None:
    processor = SudokuProcessor(
        onnx_path=None, session=stub_session_factory(4), save_debug_images=False
    )
    assert processor.onnx_path is None
    assert processor.input_name == "input"
    assert processor.output_name == "logits"


def test_model_input_and_output_names_are_read_from_the_session(
    tmp_path: Path, tiny_onnx_path: Path
) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    assert processor.input_name == "input"
    assert processor.output_name == "logits"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_cell_shape_and_range(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    tensor = processor.preprocess_cell(synthetic_cell(7))
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_preprocess_cell_blanks_the_margins(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    tensor = processor.preprocess_cell(synthetic_cell(3))
    # the outer 12 % of the cell is forced to white -> +1 after normalisation
    assert np.allclose(tensor[0, 0, 0, :], 1.0)
    assert np.allclose(tensor[0, 0, :, 0], 1.0)


def test_preprocess_cell_of_a_white_cell_is_all_ones(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    tensor = processor.preprocess_cell(synthetic_cell(None))
    assert np.allclose(tensor, 1.0)


def test_preprocess_cell_of_a_black_cell_keeps_a_dark_centre(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    tensor = processor.preprocess_cell(np.zeros((50, 50, 3), np.uint8))
    # the margins are blanked to white, the centre stays black
    assert tensor.min() == pytest.approx(-1.0)
    assert tensor[0, 0, 14, 14] == pytest.approx(-1.0)
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)


def test_ink_pixels_separates_blank_and_digit_cells(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    assert processor.ink_pixels(synthetic_cell(None)) == 0
    assert processor.ink_pixels(synthetic_cell(8)) > 60


def test_ink_pixels_ignores_the_blanked_margin(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    cell = synthetic_cell(None)
    cell[0:6, :] = 0  # a grid line bleeding into the margin
    assert processor.ink_pixels(cell) == 0


def test_ink_pixels_counts_a_fully_dark_cell(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    assert processor.ink_pixels(np.zeros((50, 50, 3), np.uint8)) > 1000


def test_blank_ink_max_replaces_an_empty_cell_with_white(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_ink_max=10)
    blank = synthetic_cell(None)
    blank[24:26, 24:26] = 0  # a speck the classifier should not be asked about
    assert 0 < processor.ink_pixels(blank) <= 10
    assert np.allclose(processor.preprocess_cell(blank), 1.0)


def test_blank_ink_max_keeps_a_digit_cell(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_ink_max=10)
    tensor = processor.preprocess_cell(synthetic_cell(5))
    assert tensor.min() < 0.0  # the digit is still there


def test_blank_ink_max_keeps_a_fully_dark_cell(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_ink_max=10)
    tensor = processor.preprocess_cell(np.zeros((50, 50, 3), np.uint8))
    assert tensor.min() == pytest.approx(-1.0)


def test_blank_ink_max_is_off_by_default(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    assert processor.blank_ink_max is None


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_infer_cells_matches_the_numpy_reference(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    cells = [(r, c, synthetic_cell((r + c) % 10)) for r in range(9) for c in range(9)]
    classes, confidences = processor.infer_cells(cells)

    batch = np.concatenate([processor.preprocess_cell(cell) for _r, _c, cell in cells], axis=0)
    reference = onnx_reference(tiny_onnx_path, batch)
    expected = softmax(reference, axis=1)

    assert classes.shape == (81,)
    assert confidences.shape == (81,)
    assert np.array_equal(classes, expected.argmax(axis=1))
    assert np.allclose(confidences, expected.max(axis=1), atol=1e-5)


def test_recognize_board_returns_a_9x9_board_and_confidences(
    tmp_path: Path, tiny_onnx_path: Path
) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_conf_threshold=0.0)
    cells = [(r, c, synthetic_cell(1)) for r in range(9) for c in range(9)]
    board, confidence = processor.recognize_board(cells, verbose=False)
    assert len(board) == 9 and all(len(row) == 9 for row in board)
    assert all(0.0 <= value <= 1.0 for row in confidence for value in row)
    assert all(value == board[0][0] for row in board for value in row)


def test_recognize_board_applies_the_confidence_threshold(
    tmp_path: Path, stub_session_factory
) -> None:
    processor = SudokuProcessor(
        onnx_path=None,
        session=stub_session_factory(3, confidence_logit=1.0),  # softmax peak well below 0.65
        save_debug_images=False,
        verbose=False,
        blank_conf_threshold=0.65,
    )
    cells = [(r, c, synthetic_cell(3)) for r in range(9) for c in range(9)]
    board, _confidence = processor.recognize_board(cells, verbose=False)
    assert all(value == processor.blank_class for row in board for value in row)


def test_recognize_board_keeps_confident_predictions(tmp_path: Path, stub_session_factory) -> None:
    processor = SudokuProcessor(
        onnx_path=None,
        session=stub_session_factory(3, confidence_logit=20.0),
        save_debug_images=False,
        verbose=False,
    )
    cells = [(r, c, synthetic_cell(3)) for r in range(9) for c in range(9)]
    board, _confidence = processor.recognize_board(cells, verbose=False)
    assert all(value == 3 for row in board for value in row)


def test_recognize_board_prints_the_grid_when_verbose(
    tmp_path: Path, tiny_onnx_path: Path, capsys
) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, verbose=True)
    cells = [(r, c, synthetic_cell(1)) for r in range(9) for c in range(9)]
    processor.recognize_board(cells, verbose=True)
    captured = capsys.readouterr().out
    assert "Predicted board (9x9)" in captured
    assert captured.count("-" * 21) == 2


def test_recognize_board_is_silent_when_not_verbose(
    tmp_path: Path, tiny_onnx_path: Path, capsys
) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    cells = [(r, c, synthetic_cell(1)) for r in range(9) for c in range(9)]
    processor.recognize_board(cells, verbose=False)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Detection and warping
# --------------------------------------------------------------------------- #
def test_split_cells_returns_eighty_one_square_cells(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    cells = processor.split_cells(np.zeros((450, 450, 3), np.uint8))
    assert len(cells) == 81
    assert [(r, c) for r, c, _cell in cells] == [(r, c) for r in range(9) for c in range(9)]
    assert all(cell.shape == (50, 50, 3) for _r, _c, cell in cells)


def test_split_cells_keeps_the_page_order(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    board = np.zeros((450, 450, 3), np.uint8)
    board[0:50, 0:50] = 255  # top-left cell only
    cells = processor.split_cells(board)
    assert cells[0][2].mean() == 255
    assert all(cell.mean() == 0 for _r, _c, cell in cells[1:])


def test_detect_and_warp_board_finds_the_page(tmp_path: Path, tiny_onnx_path: Path, page_image: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    bgr = cv.imread(str(page_image))
    warped, matrix, quad = processor.detect_and_warp_board(bgr)
    assert warped.shape == (450, 450, 3)
    assert quad.shape == (4, 2)
    assert matrix.shape == (3, 3)
    assert len(np.unique(quad, axis=0)) == 4


def test_detect_and_warp_board_rejects_a_blank_image(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    with pytest.raises(RuntimeError, match="No contours found"):
        processor.detect_and_warp_board(np.full((200, 200, 3), 255, np.uint8))


def test_detect_and_warp_board_requires_a_four_corner_contour(
    tmp_path: Path, tiny_onnx_path: Path
) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    canvas = np.full((300, 300, 3), 255, np.uint8)
    cv.circle(canvas, (150, 150), 80, (0, 0, 0), -1)  # no 4-corner quad to find
    with pytest.raises(RuntimeError, match="4-corner"):
        processor.detect_and_warp_board(canvas)


def test_warped_page_keeps_the_top_left_marker_top_left(
    tmp_path: Path, tiny_onnx_path: Path
) -> None:
    """A black square in the top-left cell must stay in the top-left cell."""
    page = np.full((1200, 800, 3), 255, np.uint8)
    from sudoku_data import draw_grid

    x0, y0, step = draw_grid(page)
    page[y0 + 2 : y0 + step - 2, x0 + 2 : x0 + step - 2] = 0
    path = tmp_path / "marker.png"
    assert cv.imwrite(str(path), page)

    processor = make_processor(tmp_path, tiny_onnx_path)
    warped, _matrix, _quad = processor.detect_and_warp_board(cv.imread(str(path)))
    cells = processor.split_cells(warped)
    means = [float(cell.mean()) for _r, _c, cell in cells]
    assert means[0] < 60  # the marker landed in the first cell
    assert min(means[1:]) > 128  # and nowhere else


# --------------------------------------------------------------------------- #
# Overlay and end-to-end
# --------------------------------------------------------------------------- #
def test_overlay_solution_draws_only_on_empty_cells(tmp_path: Path, tiny_onnx_path: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    page, _geom = render_page(SOLVED_GRID)
    warped, matrix, _quad = processor.detect_and_warp_board(page)
    identity = np.eye(3, dtype=np.float32)  # page -> itself

    full_board = [row[:] for row in SOLVED_GRID]
    unchanged = processor.overlay_solution(page, identity, full_board, full_board)
    assert np.array_equal(unchanged, page)

    empty_board = [[0] * 9 for _ in range(9)]
    annotated = processor.overlay_solution(page, identity, empty_board, SOLVED_GRID)
    assert annotated.shape == page.shape
    assert not np.array_equal(annotated, page)
    assert warped.shape == (450, 450, 3)
    assert matrix.shape == (3, 3)


def test_process_image_end_to_end(tmp_path: Path, tiny_onnx_path: Path, page_image: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_conf_threshold=0.0)
    bgr = cv.imread(str(page_image))
    board, solved, debug, overlay = processor.process_image(bgr)

    assert len(board) == 9
    assert set(debug) >= {"warped", "homography", "quad", "confidence", "solved", "conflicts"}
    assert debug["warped"].shape == (450, 450, 3)
    if solved is not None:
        assert find_empty(solved) is None
        assert find_conflicts(solved) == []
        assert overlay is not None
    assert list(tmp_path.glob("*.png")) == [page_image]


def test_process_image_reports_conflicts_for_a_broken_board(
    tmp_path: Path, stub_session_factory, page_image: Path
) -> None:
    processor = SudokuProcessor(
        onnx_path=None,
        session=stub_session_factory(7, confidence_logit=20.0),
        save_debug_images=False,
        verbose=False,
        blank_conf_threshold=0.0,
        debug_dir=tmp_path,
    )
    _board, solved, debug, overlay = processor.process_image(cv.imread(str(page_image)))
    assert solved is None
    assert debug["solved"] is False
    assert debug["conflicts"]
    assert overlay is None
    assert any("contains 7 more than once" in message for message in debug["conflicts"])


def test_process_image_can_skip_the_overlay(tmp_path: Path, tiny_onnx_path: Path, page_image: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path, blank_conf_threshold=0.0)
    _board, _solved, _debug, overlay = processor.process_image(cv.imread(str(page_image)), overlay=False)
    assert overlay is None


# --------------------------------------------------------------------------- #
# Debug output
# --------------------------------------------------------------------------- #
def test_debug_images_go_to_the_debug_directory(
    tmp_path: Path, tiny_onnx_path: Path, page_image: Path
) -> None:
    debug_dir = tmp_path / "debug"
    processor = make_processor(tmp_path, tiny_onnx_path, save_debug_images=True, debug_dir=debug_dir)
    processor.process_image(cv.imread(str(page_image)))
    assert (debug_dir / "compare_bgr_warped.png").is_file()
    assert (debug_dir / "splitted_cells_grid.png").is_file()
    assert len(list(debug_dir.glob("resized_cell_*.png"))) == 10


def test_no_debug_images_by_default(tmp_path: Path, tiny_onnx_path: Path, page_image: Path) -> None:
    processor = make_processor(tmp_path, tiny_onnx_path)
    processor.process_image(cv.imread(str(page_image)))
    assert list(tmp_path.glob("*.png")) == [page_image]


def test_write_image_reports_failures(tmp_path: Path) -> None:
    target = tmp_path / "dir.png"
    target.mkdir()
    with pytest.raises(OSError, match="could not write image"):
        write_image(target, np.zeros((2, 2), np.uint8))


# --------------------------------------------------------------------------- #
# Backwards compatibility
# --------------------------------------------------------------------------- #
def test_solver_helpers_are_still_importable_from_the_processor() -> None:
    board = parse_grid_string(EASY_PUZZLE_STRING)
    assert solve_sudoku(board) is True
    assert board == SOLVED_GRID
    assert validate_board(board) == []
    assert grid_to_string(board).startswith("534678912")
    assert SIZE == 9


def test_a_complete_but_invalid_board_is_not_reported_as_solved(
    tmp_path: Path, stub_session_factory, page_image: Path
) -> None:
    """The recogniser's errors must surface as "solver failed", not as a solution."""
    processor = SudokuProcessor(
        onnx_path=None,
        session=stub_session_factory(2, confidence_logit=30.0),
        save_debug_images=False,
        verbose=False,
        blank_conf_threshold=0.0,
        debug_dir=tmp_path,
    )
    board, solved, debug, _overlay = processor.process_image(cv.imread(str(page_image)))
    assert solved is None
    assert debug["conflicts"]
    assert all(value == 2 for row in board for value in row)


def test_build_tiny_onnx_fixture_is_reusable(tmp_path: Path) -> None:
    """Guard the fixture itself: the model must load and run."""
    model_path = build_tiny_onnx(tmp_path / "model.onnx")
    processor = SudokuProcessor(onnx_path=model_path, save_debug_images=False, verbose=False)
    classes, confidences = processor.infer_cells([(0, 0, synthetic_cell(1))])
    assert classes.shape == (1,)
    assert 0.0 < float(confidences[0]) <= 1.0
    assert grid_from_string(EASY_PUZZLE_STRING)[0][0] == 5
