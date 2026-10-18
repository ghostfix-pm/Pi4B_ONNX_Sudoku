import os
import cv2 as cv

from sudoku_processor import SudokuProcessor

def print_board(board, title="Board"):
    print("\n" + title)
    for r in range(9):
        row = ""
        for c in range(9):
            v = board[r][c]
            row += (". " if v == 0 else f"{v} ")
            if c % 3 == 2 and c != 8:
                row += "| "
        print(row.strip())
        if r % 3 == 2 and r != 8:
            print("-" * 21)


def main():
    # Use any image path you like:
    # - a real photo
    # - a synthetic grid, e.g. data/grids/val/000001_cam.png
    img_path = "data/grids/val/000001_cam.png"
    onnx_path = os.path.join("artifacts", "sudoku_digitnet.onnx")

    bgr = cv.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    proc = SudokuProcessor(onnx_path=onnx_path, warp_size=450, blank_conf_threshold=0.65)

    board, solved, dbg, overlay = proc.process_image(bgr, overlay=True)

    print_board(board, "Recognized board")
    if solved is None:
        print("\nSolver failed (board might contain recognition errors).")
    else:
        print_board(solved, "Solved board")

    # Save debug outputs
    cv.imwrite("artifacts/warped.png", dbg["warped"])
    if overlay is not None:
        cv.imwrite("artifacts/overlay_solution.png", overlay)
        print("\nSaved: artifacts/overlay_solution.png")
    print("Saved: artifacts/warped.png")

if __name__ == "__main__":
    main()
