"""Tests for :mod:`sudoku_core` — grid rules, solver and corner ordering."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import EASY_PUZZLE_STRING, SOLVED_GRID, blank_out

from sudoku_core import (
    BLANK,
    SIZE,
    candidates,
    check_shape,
    empty_board,
    find_conflicts,
    find_empty,
    grid_to_string,
    is_valid_placement,
    order_quad_points,
    parse_grid_string,
    polygon_area,
    softmax,
    solve_sudoku,
    valid,
    validate_board,
)

#: Same puzzle with one given changed: no contradictory givens, zero solutions
#: (the independent counting solver in this file agrees).
UNSOLVABLE_PUZZLE_STRING = (
    "130070000600195000098000060800060003400803001700020006060000280000419005000080079"
)

#: A complete grid that is invalid: (0, 0) and (0, 1) both hold a 1.
FULL_INVALID_GRID = [
    [1, 1, 4, 6, 7, 8, 9, 2, 3],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

#: Famous "world's hardest" style puzzle with only 17 givens.
HARD_PUZZLE_STRING = (
    "000000010400000000020000000000050407008000300001090000300400200050100000000806000"
)


def count_solutions(board, limit: int = 2) -> int:
    """Independent solution counter (deliberately a separate implementation)."""
    grid = [row[:] for row in board]

    def recurse() -> int:
        best, options_best = None, None
        for r in range(SIZE):
            for c in range(SIZE):
                if grid[r][c]:
                    continue
                used = set(grid[r]) | {grid[k][c] for k in range(SIZE)}
                br, bc = 3 * (r // 3), 3 * (c // 3)
                used |= {
                    grid[br + di][bc + dj] for di in range(3) for dj in range(3)
                }
                options = [v for v in range(1, 10) if v not in used]
                if options_best is None or len(options) < len(options_best):
                    best, options_best = (r, c), options
        if best is None:
            return 1
        r, c = best
        total = 0
        for value in options_best:
            grid[r][c] = value
            total += recurse()
            grid[r][c] = 0
            if total >= limit:
                break
        return total

    return recurse()


# --------------------------------------------------------------------------- #
# Parsing / printing
# --------------------------------------------------------------------------- #
def test_parse_grid_string_reads_zeros_as_blanks() -> None:
    grid = parse_grid_string(EASY_PUZZLE_STRING)
    assert len(grid) == SIZE
    assert grid[0][:4] == [5, 3, 0, 0]
    assert sum(value == BLANK for row in grid for value in row) == 51


def test_parse_grid_string_accepts_alternative_blank_characters() -> None:
    dotted = EASY_PUZZLE_STRING.replace("0", ".")
    starred = EASY_PUZZLE_STRING.replace("0", "*")
    underscored = EASY_PUZZLE_STRING.replace("0", "_")
    assert parse_grid_string(dotted) == parse_grid_string(EASY_PUZZLE_STRING)
    assert parse_grid_string(starred) == parse_grid_string(EASY_PUZZLE_STRING)
    assert parse_grid_string(underscored) == parse_grid_string(EASY_PUZZLE_STRING)


def test_parse_grid_string_ignores_whitespace_and_newlines() -> None:
    pretty = "\n".join(EASY_PUZZLE_STRING[i : i + 9] for i in range(0, 81, 9))
    assert parse_grid_string(pretty) == parse_grid_string(EASY_PUZZLE_STRING)


def test_parse_grid_string_reports_length() -> None:
    with pytest.raises(ValueError, match=r"must contain 81 cells, got 80"):
        parse_grid_string(EASY_PUZZLE_STRING[:80])


def test_parse_grid_string_reports_bad_character_position() -> None:
    broken = "x" + EASY_PUZZLE_STRING[1:]
    with pytest.raises(ValueError) as excinfo:
        parse_grid_string(broken)
    assert "row 0, column 0" in str(excinfo.value)
    assert "'x'" in str(excinfo.value)


def test_parse_grid_string_rejects_non_strings() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        parse_grid_string(12345)  # type: ignore[arg-type]


def test_grid_to_string_round_trips() -> None:
    for text in (EASY_PUZZLE_STRING, HARD_PUZZLE_STRING, "0" * 81):
        assert grid_to_string(parse_grid_string(text)) == text


def test_grid_to_string_pads_single_digits() -> None:
    assert grid_to_string(SOLVED_GRID)[:9] == "534678912"


def test_empty_board_is_blank() -> None:
    board = empty_board()
    assert len(board) == SIZE and all(len(row) == SIZE for row in board)
    assert all(value == BLANK for row in board for value in row)


@pytest.mark.parametrize(
    "board, message",
    [
        ([[0] * 9] * 8, "must have 9 rows"),
        ([[0] * 8] + [[0] * 9] * 8, "row 0 must have 9 cells"),
        ([[10] + [0] * 8] + [[0] * 9] * 8, "out of range"),
        ([["a"] + [0] * 8] + [[0] * 9] * 8, "not an integer"),
    ],
)
def test_check_shape_rejects_bad_boards(board, message) -> None:
    with pytest.raises(ValueError, match=message):
        check_shape(board)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def test_find_empty_returns_first_blank_in_row_major_order() -> None:
    board = parse_grid_string(EASY_PUZZLE_STRING)
    assert find_empty(board) == (0, 2)
    assert find_empty([row[:] for row in SOLVED_GRID]) is None


def test_candidates_for_a_single_missing_cell() -> None:
    board = blank_out(SOLVED_GRID, [(0, 0)])
    assert candidates(board, 0, 0) == [5]


def test_candidates_exclude_row_column_and_box_values() -> None:
    board = parse_grid_string(EASY_PUZZLE_STRING)
    options = candidates(board, 0, 2)
    assert options == [1, 2, 4]
    for value in options:
        assert is_valid_placement(board, 0, 2, value)


def test_is_valid_placement_ignores_the_cell_itself() -> None:
    board = [row[:] for row in SOLVED_GRID]
    # the value already sitting in the cell must not block re-placing it
    assert is_valid_placement(board, 0, 0, 5)
    # a duplicate elsewhere in the row, column or box is still rejected
    assert not is_valid_placement(board, 0, 0, 3)
    board[0][4] = 7
    assert not is_valid_placement(board, 0, 0, 7)


def test_valid_alias_points_at_is_valid_placement() -> None:
    assert valid is is_valid_placement


@pytest.mark.parametrize(
    "unit, coords",
    [
        ("row", [(0, 0), (0, 4)]),
        ("column", [(0, 0), (4, 0)]),
        ("box", [(0, 0), (1, 1)]),
    ],
)
def test_find_conflicts_detects_duplicates(unit, coords) -> None:
    board = [row[:] for row in SOLVED_GRID]
    for r, c in coords[1:]:
        board[r][c] = board[coords[0][0]][coords[0][1]]
    conflicts = find_conflicts(board)
    assert any(conflict["unit"] == unit for conflict in conflicts)


def test_find_conflicts_is_empty_for_a_valid_grid() -> None:
    assert find_conflicts(SOLVED_GRID) == []
    assert validate_board(SOLVED_GRID) == []
    assert find_conflicts(parse_grid_string(EASY_PUZZLE_STRING)) == []


def test_validate_board_describes_the_conflict() -> None:
    messages = validate_board(FULL_INVALID_GRID)
    assert "row 0 contains 1 more than once (r0c0, r0c1)" in messages
    assert len(messages) >= 3


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def test_solve_sudoku_solves_a_classic_puzzle() -> None:
    board = parse_grid_string(EASY_PUZZLE_STRING)
    assert solve_sudoku(board) is True
    assert board == SOLVED_GRID


def test_solve_sudoku_solves_the_hard_puzzle_with_mrv() -> None:
    """A 17-clue puzzle: the row-major scan needs minutes, MRV needs milliseconds."""
    puzzle = parse_grid_string(HARD_PUZZLE_STRING)
    assert count_solutions(puzzle, limit=2) == 1
    board = [row[:] for row in puzzle]
    assert solve_sudoku(board, heuristic="mrv") is True
    assert find_empty(board) is None
    assert validate_board(board) == []


def test_solve_sudoku_accepts_a_complete_valid_board() -> None:
    board = [row[:] for row in SOLVED_GRID]
    assert solve_sudoku(board) is True
    assert board == SOLVED_GRID


def test_solve_sudoku_rejects_a_complete_but_invalid_board() -> None:
    """Regression: a full board with duplicate givens used to report "solved"."""
    board = [row[:] for row in FULL_INVALID_GRID]
    assert solve_sudoku(board) is False


def test_solve_sudoku_without_validation_keeps_the_old_behaviour() -> None:
    board = [row[:] for row in FULL_INVALID_GRID]
    assert solve_sudoku(board, validate=False) is True


def test_solve_sudoku_returns_false_when_no_solution_exists() -> None:
    puzzle = parse_grid_string(UNSOLVABLE_PUZZLE_STRING)
    assert find_conflicts(puzzle) == []  # the givens are not contradictory
    assert count_solutions(puzzle, limit=2) == 0
    board = [row[:] for row in puzzle]
    assert solve_sudoku(board) is False


def test_solve_sudoku_restores_blanks_when_it_fails() -> None:
    board = parse_grid_string(UNSOLVABLE_PUZZLE_STRING)
    original = [row[:] for row in board]
    assert solve_sudoku(board) is False
    assert board == original


@pytest.mark.parametrize("heuristic", ["first", "mrv"])
def test_solve_sudoku_heuristics_agree(heuristic) -> None:
    board = parse_grid_string(EASY_PUZZLE_STRING)
    assert solve_sudoku(board, heuristic=heuristic) is True
    assert board == SOLVED_GRID


def test_solve_sudoku_rejects_an_unknown_heuristic() -> None:
    with pytest.raises(ValueError, match="unknown heuristic"):
        solve_sudoku(parse_grid_string(EASY_PUZZLE_STRING), heuristic="magic")


def test_solve_sudoku_rejects_a_bad_grid_shape() -> None:
    with pytest.raises(ValueError, match="9x9"):
        solve_sudoku([[0] * 9 for _ in range(8)])


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_order_quad_points_normalises_a_shuffled_quad() -> None:
    quad = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
    for permutation in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert np.array_equal(order_quad_points(quad[permutation]), quad)


def test_order_quad_points_handles_a_skewed_quad() -> None:
    skew = np.array([[10, 20], [430, 5], [440, 460], [5, 430]], dtype=np.float32)
    assert np.array_equal(order_quad_points(skew), skew)


def test_order_quad_points_handles_a_rotated_quad() -> None:
    """Regression: the x+y / y-x shortcut returned the same corner twice here."""
    diamond = np.array([[200, 0], [400, 200], [200, 400], [0, 200]], dtype=np.float32)
    ordered = order_quad_points(diamond)
    assert len(np.unique(ordered, axis=0)) == 4
    assert np.array_equal(ordered, diamond)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [10, 0], [10, 10]],  # three points
        [[0, 0], [10, 0], [10, 10], [10, 10]],  # duplicate corner
        [[0, 0], [10, 0], [20, 0], [30, 0]],  # collinear
    ],
)
def test_order_quad_points_rejects_degenerate_input(points) -> None:
    with pytest.raises(ValueError):
        order_quad_points(np.asarray(points, dtype=np.float32))


def test_polygon_area_of_a_unit_square() -> None:
    assert polygon_area(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)) == 100.0


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    probabilities = softmax(logits, axis=1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_softmax_is_stable_for_huge_logits() -> None:
    logits = np.array([[1000.0, 1000.0, 999.0]], dtype=np.float32)
    probabilities = softmax(logits, axis=1)
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 0] == pytest.approx(probabilities[0, 1], rel=1e-6)


def test_softmax_supports_axis_zero() -> None:
    probabilities = softmax(np.ones((2, 2), dtype=np.float32), axis=0)
    assert np.allclose(probabilities, 0.5)


def test_softmax_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        softmax(np.array([], dtype=np.float32), axis=1)


def test_softmax_probabilities_are_below_one() -> None:
    probabilities = softmax(np.array([[5.0, 1.0]], dtype=np.float32), axis=1)
    assert probabilities.max() < 1.0
