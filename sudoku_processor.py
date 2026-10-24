"""Sudoku grid detection, digit recognition and solution overlay for the ONNX model.

Pipeline: ``detect_and_warp_board`` -> ``split_cells`` -> ``recognize_board`` ->
``solve_sudoku`` -> ``overlay_solution``.

The solver, the grid rules and the corner ordering live in :mod:`sudoku_core`
and are re-exported here, so ``from sudoku_processor import solve_sudoku`` keeps
working for existing code while the logic stays unit-testable without OpenCV or
ONNX Runtime.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort

from sudoku_core import (  # noqa: F401  (re-exported for backwards compatibility)
    BLANK,
    BOX,
    SIZE,
    candidates,
    check_shape,
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

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)

__all__ = [
    "BLANK",
    "BOX",
    "SIZE",
    "DEFAULT_PROVIDERS",
    "SudokuProcessor",
    "candidates",
    "check_shape",
    "find_conflicts",
    "find_empty",
    "grid_to_string",
    "is_valid_placement",
    "order_quad_points",
    "parse_grid_string",
    "polygon_area",
    "softmax",
    "solve_sudoku",
    "valid",
    "validate_board",
    "write_image",
]


def write_image(path: str | Path, img: np.ndarray) -> Path:
    """Write *img* to *path*, creating parents, and raise if OpenCV fails.

    ``cv.imwrite`` returns ``False`` on failure instead of raising, so a call
    that looked successful could leave the file unwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), img):
        raise OSError(f"could not write image: {path}")
    return path


class SudokuProcessor:
    """Recognise, solve and annotate a Sudoku board in a photo."""

    def __init__(
        self,
        onnx_path: str | Path | None,
        input_size: int = 28,
        warp_size: int = 450,
        blank_class: int = 0,
        blank_conf_threshold: float = 0.65,
        providers=DEFAULT_PROVIDERS,
        save_debug_images: bool = True,
        debug_dir: str | Path | None = None,
        session: ort.InferenceSession | None = None,
        verbose: bool = True,
        blank_ink_max: int | None = None,
        solver_heuristic: str = "first",
    ):
        """Create a processor for one ONNX digit classifier.

        Args:
            onnx_path: ONNX model to run; not needed when *session* is given.
            input_size: side of the (square) model input in pixels.
            warp_size: side of the square the detected board is warped to.
            blank_class: class index used for empty cells.
            blank_conf_threshold: predictions below this confidence become blank.
            providers: ONNX Runtime execution providers.
            save_debug_images: write intermediate images for inspection.
            debug_dir: where those images go (default: the current directory).
            session: pre-built ONNX Runtime session (for tests, or when the
                caller already has one).
            verbose: print the recognised board.
            blank_ink_max: opt-in short-circuit — a cell with at most this many
                ink pixels is fed to the model as a white image (the intent of
                the old duplicated branch in ``preprocess_cell``, which did
                nothing at all).
            solver_heuristic: ``"first"`` keeps the original row-major search,
                ``"mrv"`` branches on the most constrained cell first (much
                faster on sparse boards, same solution for a unique puzzle).
        """
        resolved_session: ort.InferenceSession
        if session is not None:
            resolved_session = session
        else:
            if onnx_path is None:
                raise ValueError("either onnx_path or session is required")
            model_path = Path(onnx_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            resolved_session = ort.InferenceSession(
                str(model_path), providers=list(providers)
            )
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if warp_size < SIZE:
            raise ValueError(f"warp_size must be at least {SIZE}, got {warp_size}")
        if warp_size % SIZE:
            warnings.warn(
                f"warp_size {warp_size} is not a multiple of {SIZE}: the last "
                f"{warp_size % SIZE} px of every row and column are dropped",
                stacklevel=2,
            )
        if not 0.0 <= blank_conf_threshold <= 1.0:
            raise ValueError(
                f"blank_conf_threshold must be in [0, 1], got {blank_conf_threshold}"
            )
        if not 0 <= blank_class <= 9:
            raise ValueError(f"blank_class must be in 0..9, got {blank_class}")
        if blank_ink_max is not None and blank_ink_max < 0:
            raise ValueError(f"blank_ink_max must be >= 0, got {blank_ink_max}")
        if solver_heuristic not in {"first", "mrv"}:
            raise ValueError(f"unknown solver_heuristic: {solver_heuristic!r}")

        self.onnx_path = str(onnx_path) if onnx_path is not None else None
        self.input_size = input_size
        self.warp_size = warp_size
        self.blank_class = blank_class
        self.blank_conf_threshold = blank_conf_threshold
        self.save_debug_images = save_debug_images
        self.debug_dir = Path(debug_dir) if debug_dir is not None else Path.cwd()
        self.verbose = verbose
        self.blank_ink_max = blank_ink_max
        self.solver_heuristic = solver_heuristic

        self.sess = resolved_session
        self.input_name = resolved_session.get_inputs()[0].name
        self.output_name = resolved_session.get_outputs()[0].name

        self._preprocess_save_count = 0
        self._preprocess_save_limit = 10

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def process_image(self, bgr: np.ndarray, overlay: bool = True):
        """Recognise and solve a board image.

        Returns:
            ``(board, solved_or_None, debug, overlay_or_None)``; *solved* is
            ``None`` when the recognised board cannot be solved, and *debug*
            additionally carries the solver flag and any conflicting givens.
        """
        warped, matrix, quad = self.detect_and_warp_board(bgr)
        cells = self.split_cells(warped)
        board, confidence = self.recognize_board(cells)

        solved = [row[:] for row in board]
        solved_ok = solve_sudoku(solved, heuristic=self.solver_heuristic)

        overlay_img = None
        if overlay and solved_ok:
            overlay_img = self.overlay_solution(bgr, matrix, board, solved)

        debug = {
            "warped": warped,
            "homography": matrix,
            "quad": quad,
            "confidence": confidence,
            "solved": solved_ok,
            "conflicts": validate_board(board),
        }
        return board, (solved if solved_ok else None), debug, overlay_img

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect_and_warp_board(self, bgr: np.ndarray):
        """Find the board, order its corners and warp it to a square."""
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        thresholded = cv.adaptiveThreshold(
            blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7
        )

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thresholded = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found.")

        contours = sorted(contours, key=cv.contourArea, reverse=True)
        quad = None
        for contour in contours[:20]:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            try:
                quad = order_quad_points(approx.reshape(4, 2).astype(np.float32))
            except ValueError:
                # Degenerate or duplicated corners: try the next contour rather
                # than warping the image with a garbage homography.
                continue
            break

        if quad is None:
            raise RuntimeError("Could not find a 4-corner Sudoku grid.")

        destination = np.array(
            [
                [0, 0],
                [self.warp_size - 1, 0],
                [self.warp_size - 1, self.warp_size - 1],
                [0, self.warp_size - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv.getPerspectiveTransform(quad, destination)
        warped = cv.warpPerspective(bgr, matrix, (self.warp_size, self.warp_size))

        if self.save_debug_images:
            self._save_side_by_side(bgr, warped)

        return warped, matrix, quad

    def split_cells(self, warped_bgr: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Cut the warped board into 81 ``(row, column, cell)`` tuples."""
        cells = []
        step = self.warp_size // SIZE
        for r in range(SIZE):
            for c in range(SIZE):
                y0, y1 = r * step, (r + 1) * step
                x0, x1 = c * step, (c + 1) * step
                cells.append((r, c, warped_bgr[y0:y1, x0:x1].copy()))

        if self.save_debug_images:
            self._save_cell_grid(cells)

        return cells

    # ------------------------------------------------------------------ #
    # Preprocessing / inference
    # ------------------------------------------------------------------ #
    def ink_pixels(self, cell_bgr: np.ndarray) -> int:
        """Count pixels darker than mid-grey in the centre of *cell_bgr*.

        The outer 12 % (the margin that :meth:`preprocess_cell` blanks) is
        ignored, so a grid line bleeding into the crop does not make an empty
        cell look like a digit.  A fixed mid-grey cut keeps the answer
        predictable: a white cell scores 0, a printed digit hundreds, and a
        fully dark crop scores everything (it is not empty).
        """
        gray = cv.cvtColor(cell_bgr, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        margin = int(0.12 * min(h, w))
        interior = gray[margin : h - margin, margin : w - margin] if margin > 0 else gray
        if interior.size == 0:
            interior = gray
        return int((interior < 128).sum())

    def preprocess_cell(self, cell_bgr: np.ndarray) -> np.ndarray:
        """Turn one cell into a normalized ``(1, 1, input_size, input_size)`` tensor.

        The cell margins are blanked, the image is resized to ``input_size`` and
        scaled to ``[-1, 1]``, matching ``Training.py`` / ``EvaluateOnnxMode.py``.

        The cell is *not* augmented here: the generated dataset already carries
        the camera-like jitter and blur.  When ``blank_ink_max`` is set, an
        (almost) empty cell is replaced by a white image so the classifier is
        not asked to read noise.
        """
        gray = cv.cvtColor(cell_bgr, cv.COLOR_BGR2GRAY)

        h, w = gray.shape
        margin = int(0.12 * min(h, w))

        if margin > 0:
            gray[:margin, :] = 255
            gray[-margin:, :] = 255
            gray[:, :margin] = 255
            gray[:, -margin:] = 255

        if self.blank_ink_max is not None and self.ink_pixels(cell_bgr) <= self.blank_ink_max:
            resized = np.full((self.input_size, self.input_size), 255, np.uint8)
        else:
            resized = cv.resize(
                gray, (self.input_size, self.input_size), interpolation=cv.INTER_AREA
            )

        self._save_resized(resized)

        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor[None, None, :, :]

    def infer_cells(self, cells) -> tuple[np.ndarray, np.ndarray]:
        """Classify all 81 cells; return ``(classes, confidences)``."""
        tensors = [self.preprocess_cell(cell) for _r, _c, cell in cells]
        batch = np.concatenate(tensors, axis=0).astype(np.float32)
        logits = self.sess.run([self.output_name], {self.input_name: batch})[0]
        probabilities = softmax(logits, axis=1)
        return probabilities.argmax(axis=1), probabilities.max(axis=1)

    def recognize_board(self, cells, verbose: bool | None = None):
        """Classify every cell; return ``(board, confidence_grid)``.

        Predictions below ``blank_conf_threshold`` become ``blank_class``; the
        confidence grid always keeps the raw maxima.
        """
        if verbose is None:
            verbose = self.verbose

        predicted, confidences = self.infer_cells(cells)

        if verbose:
            self._print_prediction(predicted)

        board = [[BLANK for _ in range(SIZE)] for _ in range(SIZE)]
        confidence_grid = [[0.0 for _ in range(SIZE)] for _ in range(SIZE)]
        for index, (r, c, _cell) in enumerate(cells):
            value = int(predicted[index])
            confidence = float(confidences[index])
            if confidence < self.blank_conf_threshold:
                value = self.blank_class
            board[r][c] = value
            confidence_grid[r][c] = confidence

        return board, confidence_grid

    # ------------------------------------------------------------------ #
    # Overlay
    # ------------------------------------------------------------------ #
    def overlay_solution(self, original_bgr, matrix, board, solved):
        """Draw the solved digits onto the original photo."""
        inverse = np.linalg.inv(matrix)
        overlay = original_bgr.copy()

        step = self.warp_size // SIZE
        layer = np.zeros((self.warp_size, self.warp_size, 3), dtype=np.uint8)

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != BLANK:
                    continue
                digit = solved[r][c]
                x = int(c * step + step * 0.32)
                y = int(r * step + step * 0.72)
                cv.putText(
                    layer,
                    str(digit),
                    (x, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 200, 0),
                    2,
                    cv.LINE_AA,
                )

        h0, w0 = original_bgr.shape[:2]
        back = cv.warpPerspective(layer, inverse, (w0, h0))

        mask = (back.sum(axis=2) > 0).astype(np.uint8) * 255
        mask3 = cv.merge([mask, mask, mask])
        return np.where(mask3 > 0, cv.addWeighted(overlay, 0.6, back, 0.4, 0), overlay)

    # ------------------------------------------------------------------ #
    # Debug output
    # ------------------------------------------------------------------ #
    def _print_prediction(self, predicted: np.ndarray) -> None:
        print("Predicted board (9x9):")
        grid = predicted.reshape(SIZE, SIZE)
        for r in range(SIZE):
            row = ""
            for c in range(SIZE):
                row += str(int(grid[r, c]))
                row += " | " if c in (2, 5) else " "
            print(row)
            if r in (2, 5):
                print("-" * 21)

    def _save_side_by_side(self, original_bgr: np.ndarray, warped: np.ndarray) -> None:
        """Save ``original | warped`` side by side for visual inspection."""
        try:
            h_warped = warped.shape[0]
            h_original, w_original = original_bgr.shape[:2]
            scaled = cv.resize(
                original_bgr,
                (int(w_original * h_warped / h_original), h_warped),
                interpolation=cv.INTER_AREA,
            )
            combined = cv.hconcat([scaled, warped])
            path = write_image(self.debug_dir / "compare_bgr_warped.png", combined)
            print(f"Saved comparison image: {path}")
        except Exception as exc:  # debug output must never break the pipeline
            print(f"Could not save the comparison image: {exc}")

    def _save_cell_grid(self, cells) -> None:
        """Save the 81 crops as one padded contact sheet."""
        try:
            cell_h, cell_w = cells[0][2].shape[:2]
            padding = 4
            grid_img = np.ones(
                (
                    SIZE * cell_h + (SIZE + 1) * padding,
                    SIZE * cell_w + (SIZE + 1) * padding,
                    3,
                ),
                dtype=np.uint8,
            ) * 255
            for r, c, cell in cells:
                y0 = padding + r * (cell_h + padding)
                x0 = padding + c * (cell_w + padding)
                grid_img[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
            path = write_image(self.debug_dir / "splitted_cells_grid.png", grid_img)
            print(f"Saved cell grid: {path}")
        except Exception as exc:
            print(f"Could not save the cell grid: {exc}")

    def _save_resized(self, resized: np.ndarray) -> None:
        """Save the first few preprocessed cells with the other debug output."""
        if not self.save_debug_images or self._preprocess_save_count >= self._preprocess_save_limit:
            return
        try:
            path = write_image(
                self.debug_dir / f"resized_cell_{self._preprocess_save_count:03d}.png", resized
            )
            print(f"Saved resized cell: {path}")
        except Exception as exc:
            print(f"Could not save the resized cell: {exc}")
        self._preprocess_save_count += 1
