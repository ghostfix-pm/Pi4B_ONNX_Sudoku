"""Dependency-light Sudoku logic shared by the data pipeline and the ONNX demo.

Only ``numpy`` is imported here (no OpenCV, no onnxruntime), so the solver, the
grid parsing and the corner ordering can be unit-tested on a bare Python
install and reused from any entry point.

The functions previously lived inline in ``sudoku_processor.py``; they are kept
importable from there for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

SIZE = 9
BOX = 3
BLANK = 0

#: Characters accepted as an empty cell when parsing an 81-character grid.
BLANK_CHARS = frozenset("0._*")

__all__ = [
    "SIZE",
    "BOX",
    "BLANK",
    "BLANK_CHARS",
    "empty_board",
    "check_shape",
    "parse_grid_string",
    "grid_to_string",
    "find_empty",
    "candidates",
    "is_valid_placement",
    "valid",
    "find_conflicts",
    "validate_board",
    "solve_sudoku",
    "order_quad_points",
    "polygon_area",
    "softmax",
]


# --------------------------------------------------------------------------- #
# Grid parsing / printing
# --------------------------------------------------------------------------- #
def empty_board() -> list[list[int]]:
    """Return a 9x9 board filled with :data:`BLANK`."""
    return [[BLANK for _ in range(SIZE)] for _ in range(SIZE)]


def check_shape(board: Sequence[Sequence[int]]) -> list[list[int]]:
    """Validate that *board* is 9x9 and return it as a list of int rows.

    Raises:
        ValueError: if the board is not a 9x9 grid of integers in 0..9.
    """
    if len(board) != SIZE:
        raise ValueError(f"board must have {SIZE} rows, got {len(board)}")
    rows: list[list[int]] = []
    for r, row in enumerate(board):
        if len(row) != SIZE:
            raise ValueError(f"board row {r} must have {SIZE} cells, got {len(row)}")
        values = []
        for c, value in enumerate(row):
            try:
                v = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"board cell ({r}, {c}) is not an integer: {value!r}"
                ) from exc
            if not BLANK <= v <= 9:
                raise ValueError(f"board cell ({r}, {c}) out of range 0..9: {v}")
            values.append(v)
        rows.append(values)
    return rows


def parse_grid_string(text: str) -> list[list[int]]:
    """Parse an 81-character puzzle string into a 9x9 list of ints.

    ``0``, ``.``, ``_`` and ``*`` all mean "blank" (the HuggingFace Sudoku
    dataset ships ``0``).  Raises :class:`ValueError` with the offending
    position instead of a bare ``AssertionError`` / ``ValueError``, and keeps
    working under ``python -O`` where ``assert`` statements are stripped.
    """
    if not isinstance(text, str):
        raise ValueError(f"grid must be a string, got {type(text).__name__}")

    cleaned = "".join(text.split())
    if len(cleaned) != SIZE * SIZE:
        raise ValueError(
            f"grid must contain {SIZE * SIZE} cells, got {len(cleaned)}: {text!r}"
        )

    grid = empty_board()
    for index, char in enumerate(cleaned):
        r, c = divmod(index, SIZE)
        if char in BLANK_CHARS:
            continue
        if not char.isdigit():
            raise ValueError(
                f"grid cell {index} (row {r}, column {c}) is not a digit: {char!r}"
            )
        grid[r][c] = int(char)
    return grid


def grid_to_string(board: Sequence[Sequence[int]]) -> str:
    """Serialise a 9x9 board back to the canonical 81-character form."""
    rows = check_shape(board)
    return "".join(str(value) for row in rows for value in row)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def find_empty(board: Sequence[Sequence[int]]) -> tuple[int, int] | None:
    """Return the first empty cell (row-major) or ``None`` when the board is full."""
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == BLANK:
                return (r, c)
    return None


def candidates(board: Sequence[Sequence[int]], r: int, c: int) -> list[int]:
    """Return every value that may be placed at ``(r, c)`` without conflict."""
    used = set(board[r])
    used.update(board[i][c] for i in range(SIZE))
    br, bc = BOX * (r // BOX), BOX * (c // BOX)
    used.update(
        board[i][j] for i in range(br, br + BOX) for j in range(bc, bc + BOX)
    )
    return [v for v in range(1, SIZE + 1) if v not in used]


def is_valid_placement(board: Sequence[Sequence[int]], r: int, c: int, value: int) -> bool:
    """Return whether *value* can be placed at ``(r, c)`` (the cell itself is ignored)."""
    for j in range(SIZE):
        if j != c and board[r][j] == value:
            return False
    for i in range(SIZE):
        if i != r and board[i][c] == value:
            return False
    br, bc = BOX * (r // BOX), BOX * (c // BOX)
    for i in range(br, br + BOX):
        for j in range(bc, bc + BOX):
            if (i, j) != (r, c) and board[i][j] == value:
                return False
    return True


#: Backwards-compatible alias (the original ``sudoku_processor.valid``).
valid = is_valid_placement


def find_conflicts(board: Sequence[Sequence[int]]) -> list[dict]:
    """Return duplicates that already exist in *board*, ignoring empty cells.

    Each entry is ``{"unit": "row"|"column"|"box", "index": int, "value": int,
    "cells": [(r, c), ...]}``.  A non-empty result means the *givens* are
    contradictory and no solution exists.
    """
    grid = check_shape(board)
    conflicts: list[dict] = []

    def scan(cells: Iterable[tuple[int, int]], unit: str, index: int) -> None:
        seen: dict[int, list[tuple[int, int]]] = {}
        for r, c in cells:
            value = grid[r][c]
            if value == BLANK:
                continue
            seen.setdefault(value, []).append((r, c))
        for value, where in sorted(seen.items()):
            if len(where) > 1:
                conflicts.append(
                    {"unit": unit, "index": index, "value": value, "cells": where}
                )

    for r in range(SIZE):
        scan(((r, c) for c in range(SIZE)), "row", r)
    for c in range(SIZE):
        scan(((r, c) for r in range(SIZE)), "column", c)
    for br in range(0, SIZE, BOX):
        for bc in range(0, SIZE, BOX):
            box_index = (br // BOX) * BOX + bc // BOX
            scan(
                ((br + i, bc + j) for i in range(BOX) for j in range(BOX)),
                "box",
                box_index,
            )
    return conflicts


def validate_board(board: Sequence[Sequence[int]]) -> list[str]:
    """Human-readable descriptions of contradictory givens (empty list == valid)."""
    messages = []
    for conflict in find_conflicts(board):
        cells = ", ".join(f"r{r}c{c}" for r, c in conflict["cells"])
        messages.append(
            f"{conflict['unit']} {conflict['index']} contains {conflict['value']} "
            f"more than once ({cells})"
        )
    return messages


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def solve_sudoku(
    board: list[list[int]],
    *,
    validate: bool = True,
    heuristic: str = "first",
) -> bool:
    """Solve *board* **in place**.

    Args:
        board: mutable 9x9 grid; filled cells are treated as givens.
        validate: when true (default) reject boards whose givens already
            conflict.  Without this check a *complete* but contradictory board
            was reported as solved, because an empty-cell search finds nothing
            left to fill.
        heuristic: ``"first"`` (original behaviour: scan row-major) or
            ``"mrv"`` (always branch on the empty cell with the fewest
            candidates, which explores far fewer dead ends).

    Returns:
        ``True`` when the board is completely and consistently filled.
    """
    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        raise ValueError("board must be 9x9")
    if heuristic not in {"first", "mrv"}:
        raise ValueError(f"unknown heuristic: {heuristic!r}")

    if validate and find_conflicts(board):
        return False

    def step() -> bool:
        if heuristic == "mrv":
            target: tuple[int, int] | None = None
            options_at_target: list[int] = []
            for r in range(SIZE):
                for c in range(SIZE):
                    if board[r][c] != BLANK:
                        continue
                    options = candidates(board, r, c)
                    if not options:
                        return False
                    if target is None or len(options) < len(options_at_target):
                        target, options_at_target = (r, c), options
                        if len(options_at_target) == 1:
                            break
            if target is None:
                return True
            r, c = target
            for value in options_at_target:
                board[r][c] = value
                if step():
                    return True
                board[r][c] = BLANK
            return False

        position = find_empty(board)
        if position is None:
            return True
        r, c = position
        for value in range(1, SIZE + 1):
            if is_valid_placement(board, r, c, value):
                board[r][c] = value
                if step():
                    return True
                board[r][c] = BLANK
        return False

    return step()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def polygon_area(pts: np.ndarray) -> float:
    """Absolute area of a 2D polygon via the shoelace formula."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four image points as ``[top-left, top-right, bottom-right, bottom-left]``.

    The points are sorted by their angle around the centroid and then rotated so
    the corner nearest the top-left of the bounding box comes first.  Unlike the
    "min/max of x+y and y-x" shortcut this cannot hand back the *same* corner
    twice for a strongly rotated or skewed quad.

    Raises:
        ValueError: if the input is not four distinct, convex, non-degenerate
            points.  (The caller can then fall back to another contour instead
            of warping the image with a garbage homography.)
    """
    points = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if points.shape != (4, 2):
        raise ValueError(f"expected 4 points of shape (2,), got shape {points.shape}")
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError(f"quad corners must be distinct: {points.tolist()}")

    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]

    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)

    # Consistent winding: every consecutive turn must have the same sign.
    turns = []
    for i in range(4):
        a = ordered[(i + 1) % 4] - ordered[i]
        b = ordered[(i + 2) % 4] - ordered[(i + 1) % 4]
        turns.append(float(a[0] * b[1] - a[1] * b[0]))
    if any(t == 0.0 for t in turns) or not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError(f"quad is not convex: {ordered.tolist()}")
    if polygon_area(ordered) <= 1.0:
        raise ValueError(f"quad is degenerate (area {polygon_area(ordered):.3f} px)")

    if turns[0] < 0:  # keep the same (clockwise in image coordinates) winding
        ordered = ordered[[0, 3, 2, 1]]

    return ordered.astype(np.float32)


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax over *axis*."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.size == 0:
        raise ValueError("softmax needs a non-empty array")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / (np.sum(exponentiated, axis=axis, keepdims=True) + 1e-12)
