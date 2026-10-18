import cv2 as cv
import numpy as np
import onnxruntime as ort
import os

class SudokuProcessor:
    def __init__(
        self,
        onnx_path: str,
        input_size: int = 28,
        warp_size: int = 450,
        blank_class: int = 0,
        blank_conf_threshold: float = 0.65,
        providers=("CPUExecutionProvider",),
        save_debug_images: bool = True,  # === 新增：控制是否保存中间过程图的 flag ===
    ):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.warp_size = warp_size
        self.blank_class = blank_class
        self.blank_conf_threshold = blank_conf_threshold
        self.save_debug_images = save_debug_images  # === 保存标志位 ===

        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        self._preprocess_save_count = 0
        self._preprocess_save_limit = 10

    def process_image(self, bgr: np.ndarray, overlay: bool = True):
        warped, H, quad = self.detect_and_warp_board(bgr)
        cells = self.split_cells(warped)
        board, conf = self.recognize_board(cells)

        solved = [row[:] for row in board]
        ok = solve_sudoku(solved)

        overlay_img = None
        if overlay and ok:
            overlay_img = self.overlay_solution(bgr, H, board, solved)

        debug = {
            "warped": warped,
            "homography": H,
            "quad": quad,
            "confidence": conf,
        }
        return board, (solved if ok else None), debug, overlay_img

    def detect_and_warp_board(self, bgr: np.ndarray):
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        thr = cv.adaptiveThreshold(
            blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7
        )

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found.")

        contours = sorted(contours, key=cv.contourArea, reverse=True)
        quad = None
        for c in contours[:20]:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break

        if quad is None:
            raise RuntimeError("Could not find a 4-corner Sudoku grid.")

        quad = order_quad_points(quad)

        dst = np.array(
            [[0, 0], [self.warp_size - 1, 0], [self.warp_size - 1, self.warp_size - 1], [0, self.warp_size - 1]],
            dtype=np.float32,
        )
        H = cv.getPerspectiveTransform(quad, dst)
        warped = cv.warpPerspective(bgr, H, (self.warp_size, self.warp_size))

        # === 修改：加入 flag 控制 ===
        if self.save_debug_images:
            try:
                h_warped, w_warped = warped.shape[:2]
                h_bgr, w_bgr = bgr.shape[:2]
                scale = h_warped / h_bgr
                bgr_resized = cv.resize(bgr, (int(w_bgr * scale), h_warped), interpolation=cv.INTER_AREA)

                combined = cv.hconcat([bgr_resized, warped])
                save_path = os.path.join(os.getcwd(), "compare_bgr_warped.png")
                cv.imwrite(save_path, combined)
                print(f"✅ 已保存对比图: {save_path}")
            except Exception as e:
                print(f"⚠️ 保存对比图失败: {e}")

        return warped, H, quad

    def split_cells(self, warped_bgr: np.ndarray):
        cells = []
        step = self.warp_size // 9
        for r in range(9):
            for c in range(9):
                y0, y1 = r * step, (r + 1) * step
                x0, x1 = c * step, (c + 1) * step
                cell = warped_bgr[y0:y1, x0:x1].copy()
                cells.append((r, c, cell))

        # === 修改：加入 flag 控制 ===
        if self.save_debug_images:
            try:
                cell_h, cell_w = cells[0][2].shape[:2]
                padding = 4
                
                grid_h = 9 * cell_h + 10 * padding
                grid_w = 9 * cell_w + 10 * padding
                
                grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

                for r, c, cell in cells:
                    y0 = padding + r * (cell_h + padding)
                    x0 = padding + c * (cell_w + padding)
                    grid_img[y0:y0+cell_h, x0:x0+cell_w] = cell

                save_path = os.path.join(os.getcwd(), "splitted_cells_grid.png")
                cv.imwrite(save_path, grid_img)
                print(f"✅ 已保存切分的 cells 网格图: {save_path}")
            except Exception as e:
                print(f"⚠️ 保存 cells 网格图失败: {e}")

        return cells

    def preprocess_cell(self, cell_bgr: np.ndarray):
        g = cv.cvtColor(cell_bgr, cv.COLOR_BGR2GRAY)

        h, w = g.shape
        m = int(0.12 * min(h, w))
        
        if m > 0:
            g[:m, :] = 255
            g[-m:, :] = 255
            g[:, :m] = 255
            g[:, -m:] = 255

        g_blur = cv.GaussianBlur(g, (3, 3), 0)
        bw = cv.adaptiveThreshold(g_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5)
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

        if (bw > 0).sum() < 15:
            resized = cv.resize(g, (self.input_size, self.input_size), interpolation=cv.INTER_AREA)
        else:
            resized = cv.resize(g, (self.input_size, self.input_size), interpolation=cv.INTER_AREA)

        # === 修改：加入 flag 控制 ===
        if self.save_debug_images and self._preprocess_save_count < self._preprocess_save_limit:
            save_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(
                save_dir,
                f"resized_cell_{self._preprocess_save_count:03d}.png"
            )

            cv.imwrite(save_path, resized)
            
            print(f"已保存 resized 图像: {save_path}")
            self._preprocess_save_count += 1

        x = resized.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x[None, None, :, :]
        return x

    def recognize_board(self, cells):
        xs = []
        coords = []
        for r, c, cell in cells:
            coords.append((r, c))
            xs.append(self.preprocess_cell(cell))

        X = np.concatenate(xs, axis=0).astype(np.float32)
        logits = self.sess.run([self.output_name], {self.input_name: X})[0]
        probs = softmax(logits, axis=1)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        print("Predicted board (9x9):")
        pred_grid = pred.reshape(9, 9)
        for r in range(9):
            row_str = ""
            for c in range(9):
                row_str += str(int(pred_grid[r, c]))
                if c in (2, 5):
                    row_str += " | "
                else:
                    row_str += " "
            print(row_str)
            if r in (2, 5):
                print("-" * 21)

        board = [[0 for _ in range(9)] for _ in range(9)]
        conf_grid = [[0.0 for _ in range(9)] for _ in range(9)]
        for i, (r, c) in enumerate(coords):
            p = int(pred[i])
            cf = float(conf[i])
            if cf < self.blank_conf_threshold:
                p = self.blank_class
            board[r][c] = p
            conf_grid[r][c] = cf

        return board, conf_grid

    def overlay_solution(self, original_bgr, H, board, solved):
        invH = np.linalg.inv(H)
        overlay = original_bgr.copy()

        step = self.warp_size // 9
        layer = np.zeros((self.warp_size, self.warp_size, 3), dtype=np.uint8)

        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    continue
                d = solved[r][c]
                x = int(c * step + step * 0.32)
                y = int(r * step + step * 0.72)
                cv.putText(layer, str(d), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2, cv.LINE_AA)

        h0, w0 = original_bgr.shape[:2]
        back = cv.warpPerspective(layer, invH, (w0, h0))

        mask = (back.sum(axis=2) > 0).astype(np.uint8) * 255
        mask3 = cv.merge([mask, mask, mask])
        overlay = np.where(mask3 > 0, cv.addWeighted(overlay, 0.6, back, 0.4, 0), overlay)
        return overlay


def solve_sudoku(board):
    pos = find_empty(board)
    if pos is None:
        return True
    r, c = pos
    for v in range(1, 10):
        if valid(board, r, c, v):
            board[r][c] = v
            if solve_sudoku(board):
                return True
            board[r][c] = 0
    return False


def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None


def valid(board, r, c, v):
    for j in range(9):
        if board[r][j] == v:
            return False
    for i in range(9):
        if board[i][c] == v:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == v:
                return False
    return True


def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)
