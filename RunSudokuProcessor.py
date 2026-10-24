"""Run the ONNX digit recogniser on one Sudoku photo and print the solution.

Usage::

    python3 RunSudokuProcessor.py                                  # first generated page found
    python3 RunSudokuProcessor.py --image photo.jpg
    python3 RunSudokuProcessor.py --image photo.jpg --onnx artifacts_v4/sudoku_digitnet.onnx

The script used to hard-code ``artifacts/sudoku_digitnet.onnx``, a path no script
in this repository produces (``Training.py`` writes to ``artifacts_v4/``), so the
demo could not run as shipped.  Both the image and the model are now discovered
from the usual locations and can be overridden on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv

from sudoku_core import SIZE
from sudoku_processor import SudokuProcessor, write_image

#: Places a generated board image may live, in preference order.
IMAGE_CANDIDATES = (
    "data_v4/grids/val/000001_cam.png",
    "data/grids/val/000001_cam.png",
    "data_v1/grids/val/000001_cam.png",
    "data_v2/grids/val/000001_cam.png",
    "data_v3/grids/val/000001_cam.png",
)

#: Places the exported model may live, in preference order.
ONNX_CANDIDATES = (
    "artifacts_v4/sudoku_digitnet.onnx",
    "artifacts/sudoku_digitnet.onnx",
    "artifacts_v1/sudoku_digitnet.onnx",
    "artifacts_v2/sudoku_digitnet.onnx",
    "artifacts_v3/sudoku_digitnet.onnx",
)


def find_first(candidates, explicit: str | None, kind: str) -> Path:
    """Return *explicit* if given, else the first existing candidate."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{kind} not found: {path}")
        return path
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    searched = "\n  ".join(candidates)
    raise FileNotFoundError(
        f"no {kind} found, searched:\n  {searched}\n"
        f"pass one explicitly with --{'image' if kind == 'image' else 'onnx'}"
    )


def print_board(board, title: str = "Board") -> None:
    """Print a 9x9 board with box separators (``.`` for empty cells)."""
    print("\n" + title)
    for r in range(SIZE):
        row = ""
        for c in range(SIZE):
            value = board[r][c]
            row += ". " if value == 0 else f"{value} "
            if c % 3 == 2 and c != SIZE - 1:
                row += "| "
        print(row.strip())
        if r % 3 == 2 and r != SIZE - 1:
            print("-" * 21)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Recognise and solve a Sudoku board in a photo with the ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", default=None, help="input photo or generated page")
    parser.add_argument("--onnx", default=None, help="ONNX digit classifier")
    parser.add_argument(
        "--out-dir",
        default="artifacts",
        help="directory for warped.png / overlay_solution.png",
    )
    parser.add_argument("--warp-size", type=int, default=450, help="square side after warping")
    parser.add_argument(
        "--blank-conf-threshold",
        type=float,
        default=0.65,
        help="predictions below this confidence are treated as empty cells",
    )
    parser.add_argument(
        "--blank-ink-max",
        type=int,
        default=None,
        help="treat cells with at most this many ink pixels as empty",
    )
    parser.add_argument("--no-overlay", action="store_true", help="skip the solved-board overlay")
    parser.add_argument("--debug-dir", default=None, help="where debug images are written")
    parser.add_argument("--quiet", action="store_true", help="do not print the recognised board")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the recogniser on one image; returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        image_path = find_first(IMAGE_CANDIDATES, args.image, "image")
        onnx_path = find_first(ONNX_CANDIDATES, args.onnx, "onnx")
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    bgr = cv.imread(str(image_path))
    if bgr is None:
        print(f"error: could not read image: {image_path}")
        return 2

    proc = SudokuProcessor(
        onnx_path=onnx_path,
        warp_size=args.warp_size,
        blank_conf_threshold=args.blank_conf_threshold,
        blank_ink_max=args.blank_ink_max,
        save_debug_images=False,
        debug_dir=args.debug_dir,
        verbose=not args.quiet,
    )

    print(f"image: {image_path}\nmodel: {onnx_path}")
    board, solved, debug, overlay = proc.process_image(bgr, overlay=not args.no_overlay)

    if not args.quiet:
        print_board(board, "Recognized board")

    if solved is None:
        print("\nSolver failed.")
        for conflict in debug["conflicts"]:
            print(f"  conflicting givens: {conflict}")
        print("  the board probably contains recognition errors")
    else:
        print_board(solved, "Solved board")

    out_dir = Path(args.out_dir)
    print(f"\nSaved: {write_image(out_dir / 'warped.png', debug['warped'])}")
    if overlay is not None:
        print(f"Saved: {write_image(out_dir / 'overlay_solution.png', overlay)}")

    return 0 if solved is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
