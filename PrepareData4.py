import os, random, pathlib
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

random.seed(0)

# Parameters
PARQUET_PATH = "train_1.parquet"   # path to your downloaded HF Parquet
# OUT_DIR      = pathlib.Path("data")
OUT_DIR      = pathlib.Path("data_v4")
N_TRAIN      = 1000               # how many puzzles to render for training
N_VAL        = 100                # how many for validation
IMG_H, IMG_W = 1200, 800          # page size (portrait-ish)
GRID_MARGIN  = 60                 # outer margin, px
CELL_SIZE    = 28                 # output crop size for classifier (MNIST-like)
FONT         = cv.FONT_HERSHEY_SIMPLEX
#

def str_to_grid(s: str):
    """81-char '012345678' string -> 9x9 list of ints."""
    s = s.strip()
    assert len(s) == 81, f"bad length: {len(s)}"
    return [[int(s[9*r+c]) for c in range(9)] for r in range(9)]

def load_puzzles(parquet_path, n_train, n_val):
    """Load puzzles/solutions; return two lists of 9x9 int grids for train/val."""
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    # Shuffle reproducibly
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    # Keep only needed columns if present
    need_cols = [c for c in ["puzzle", "solution"] if c in df.columns]
    if not need_cols or "puzzle" not in need_cols:
        raise ValueError(f"Expected 'puzzle' (and optionally 'solution') columns; got: {list(df.columns)}")

    # Slice train/val partitions
    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_train+n_val]

    puzzles_train = [str_to_grid(p) for p in df_train["puzzle"].astype(str)]
    puzzles_val   = [str_to_grid(p) for p in df_val["puzzle"].astype(str)]

    # Solutions are optional (useful later for solver validation)
    solutions_train = [str_to_grid(s) for s in df_train["solution"].astype(str)] if "solution" in df_train else None
    solutions_val   = [str_to_grid(s) for s in df_val["solution"].astype(str)]   if "solution" in df_val   else None

    return (puzzles_train, solutions_train), (puzzles_val, solutions_val)

def draw_grid(img, size=9, margin=GRID_MARGIN):
    H, W = img.shape[:2]
    step = (min(H, W) - 2*margin) // size
    x0 = (W - size*step) // 2
    y0 = (H - size*step) // 2
    for i in range(size+1):
        thickness = 3 if i % 3 == 0 else 1
        # vertical
        cv.line(img, (x0 + i*step, y0), (x0 + i*step, y0 + size*step), (0, 0, 0), thickness)
        # horizontal
        cv.line(img, (x0, y0 + i*step), (x0 + size*step, y0 + i*step), (0, 0, 0), thickness)
    return (x0, y0, step)

def put_digit(img, r, c, d, x0, y0, step):
    if d == 0:
        return  # blank cell
    text = str(d)
    scale = step / 60.0
    thickness = 2
    (tw, th), base = cv.getTextSize(text, FONT, scale, thickness)
    cx = x0 + c*step + (step - tw)//2
    cy = y0 + r*step + (step + th)//2 - th//4
    cv.putText(img, text, (cx, cy), FONT, scale, (0, 0, 0), thickness, cv.LINE_AA)

def render_page(puzzle9x9):
    page = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
    x0, y0, step = draw_grid(page, 9, GRID_MARGIN)
    for r in range(9):
        for c in range(9):
            put_digit(page, r, c, puzzle9x9[r][c], x0, y0, step)
    return page, (x0, y0, step)

# 注意：这里删除了原来的 aug_camera 函数，因为我们将增强逻辑放到了 save_crops 中

def ensure_dirs(split):
    for cls in range(10):  # 0..9 (0 == blank)
        (OUT_DIR / split / str(cls)).mkdir(parents=True, exist_ok=True)

def save_crops(page, geom, puzzle9x9, split, base_id, is_cam=False):
    """
    x0, y0, step: 网格的左上角坐标及单元格边长
    is_cam: 如果为 True，则对切割出来的单个小图进行局部形变和模糊
    """
    x0, y0, step = geom
    idx = 0
    for r in range(9):
        for c in range(9):
            x1, y1 = x0 + c*step, y0 + r*step
            
            # 【关键修改】这里必须加 .copy()，防止修改影响原图
            roi = page[y1:y1+step, x1:x1+step].copy()
            
            # 【关键修改】局部透视抖动与模糊
            if is_cam:
                h_roi, w_roi = roi.shape[:2]
                def jitter(pt, s=20):
                    return (pt[0] + random.randint(-s, s), pt[1] + random.randint(-s, s))
                
                src_pts = np.float32([(0, 0), (w_roi, 0), (w_roi, h_roi), (0, h_roi)])
                dst_pts = np.float32([jitter((0,0)), jitter((w_roi,0)), 
                                      jitter((w_roi,h_roi)), jitter((0,h_roi))])
                
                M = cv.getPerspectiveTransform(src_pts, dst_pts)
                # 用纯白色填充边缘
                roi = cv.warpPerspective(roi, M, (w_roi, h_roi), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
                
                # 随机轻微高斯模糊
                if random.random() < 0.5:
                    k = random.choice([1, 2])
                    roi = cv.GaussianBlur(roi,  (2*k+1, 2*k+1), 0)
        
            # 转灰度并 resize
            g = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            g = cv.resize(g, (CELL_SIZE, CELL_SIZE), interpolation=cv.INTER_AREA)
            label = puzzle9x9[r][c]  # 0 for blank, 1..9 digits
            out_path = OUT_DIR / split / str(label) / f"{base_id}_{idx:02d}.png"
            cv.imwrite(str(out_path), g)
            idx += 1

def process_split(puzzles, split_name, n_limit):
    ensure_dirs(split_name)
    grid_dir = OUT_DIR / "grids" / split_name
    grid_dir.mkdir(parents=True, exist_ok=True)

    N = min(n_limit, len(puzzles))
    for i in tqdm(range(N), desc=f"render {split_name}"):
        puzzle = puzzles[i]

        # Clean page
        page, geom = render_page(puzzle)
        save_crops(page, geom, puzzle, split_name, base_id=f"{i:06d}_clean")
        cv.imwrite(str(grid_dir / f"{i:06d}_clean.png"), page)

        # 2. 保存局部的 camera-like 切图
        # 【注意】大图不再变形，只保存干净的大图。变异只在局部格子发生。
        save_crops(page, geom, puzzle, split_name, base_id=f"{i:06d}_cam", is_cam=True)

def main():
    (p_train, _s_train), (p_val, _s_val) = load_puzzles(PARQUET_PATH, N_TRAIN, N_VAL)
    process_split(p_train, "train", N_TRAIN)
    process_split(p_val,   "val",   N_VAL)
    print("Done. Output under:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()