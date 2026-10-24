"""Synthetic Sudoku dataset synthesis (the core behind ``PrepareData*.py``).

The repository ships five near-identical generators — ``PrepareData.py`` and
``PrepareData1..4.py`` — that differ only in the output directory, the number of
puzzles and how the camera-like augmentation is applied:

===========  ===========  ==============  ===============  ====================
script       output dir   train / val     augmentation     perspective jitter
===========  ===========  ==============  ===============  ====================
PrepareData  data         1000 / 100      whole page       +/- 20 px
PrepareData1 data_v1      1000 / 100      whole page       +/- 20 px
PrepareData2 data_v2      2000 / 200      inside each cell +/- 4 px
PrepareData3 data_v3      2000 / 200      inside each cell +/- 20 px
PrepareData4 data_v4      1000 / 100      inside each cell +/- 20 px
===========  ===========  ==============  ===============  ====================

``PrepareData.py`` reproduces the *original* learn.arm.com recipe (augment the
page, then crop on the un-warped grid, which is why its crops can sit off the
digit), while ``PrepareData1..4.py`` are the follow-up experiments that moved the
augmentation into the cell to get rid of the black grid lines that made the
model confuse 6 with 9.  All of that is captured here as five :class:`Variant`
profiles, so the arms cannot drift apart and the pipeline becomes testable.

Every profile reproduces its original script byte for byte — see
``tests/test_sudoku_data.py::test_variants_reproduce_original_scripts``.

Rendering details are unchanged: page 1200x800, 60 px margin, cells resized to
28x28 grayscale, blank cells saved as class ``0``.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from sudoku_core import BLANK, SIZE, parse_grid_string

#: A 9x9 grid of digit values (``0`` == blank).
Grid = list[list[int]]

# Page / cell geometry
IMG_H, IMG_W = 1200, 800
GRID_MARGIN = 60
CELL_SIZE = 28
FONT = cv.FONT_HERSHEY_SIMPLEX

#: Class directory for blank cells (the model has 10 outputs).
CLASSES = tuple(range(10))

#: Corner order used for the perspective jitter (top-left, top-right,
#: bottom-right, bottom-left).  The ``random`` draw order matters: it is what
#: makes a seeded run reproduce the original scripts.
_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Augmentation:
    """Camera-like augmentation profile.

    Args:
        scope: ``"page"`` warps/blurs the rendered page *before* cropping
            (the original recipe), ``"cell"`` warps/blurs every crop
            individually after cropping (the improved recipe).
        jitter_px: maximum corner displacement of the perspective jitter.
        blur_prob: probability that a Gaussian blur is applied.
        blur_kernels: kernel sizes to choose from; a single entry means a fixed
            kernel (the original fixed-blur scripts never draw from ``random``
            there, which matters for reproducibility).
        border_value: fill colour of the warped area outside the source quad.
    """

    scope: str
    jitter_px: int = 20
    blur_prob: float = 0.5
    blur_kernels: tuple[int, ...] = (3, 5)
    border_value: int = 255

    def __post_init__(self) -> None:
        if self.scope not in {"page", "cell"}:
            raise ValueError(
                f"augmentation scope must be 'page' or 'cell', got {self.scope!r}"
            )
        if self.jitter_px < 0:
            raise ValueError(f"jitter_px must be >= 0, got {self.jitter_px}")
        if not 0.0 <= self.blur_prob <= 1.0:
            raise ValueError(f"blur_prob must be in [0, 1], got {self.blur_prob}")
        if not self.blur_kernels:
            raise ValueError("blur_kernels must not be empty")
        if any(k <= 0 or k % 2 == 0 for k in self.blur_kernels):
            raise ValueError(
                f"blur_kernels must be positive odd sizes, got {self.blur_kernels}"
            )


@dataclass(frozen=True)
class Variant:
    """One experiment arm: output directory, split sizes and augmentation."""

    name: str
    out_dir: str
    n_train: int
    n_val: int
    augmentation: Augmentation
    description: str = ""

    @property
    def legacy_script(self) -> str:
        """Name of the original script this profile replaces."""
        return "PrepareData.py" if self.name == "v0" else f"PrepareData{self.name[1:]}.py"


PAGE_AUG = Augmentation(
    scope="page", jitter_px=20, blur_prob=0.5, blur_kernels=(3, 5), border_value=220
)
CELL_AUG_JITTER4 = Augmentation(
    scope="cell", jitter_px=4, blur_prob=0.3, blur_kernels=(3,), border_value=255
)
CELL_AUG_JITTER20_FIXED = Augmentation(
    scope="cell", jitter_px=20, blur_prob=0.3, blur_kernels=(3,), border_value=255
)
CELL_AUG_JITTER20_CHOICE = Augmentation(
    scope="cell", jitter_px=20, blur_prob=0.5, blur_kernels=(3, 5), border_value=255
)

#: All experiment arms, in the order they were added to the repository.
VARIANTS: dict[str, Variant] = {
    "v0": Variant(
        "v0",
        "data",
        1000,
        100,
        PAGE_AUG,
        "original learn.arm.com recipe: page warped before cropping, 20 px jitter",
    ),
    "v1": Variant(
        "v1",
        "data_v1",
        1000,
        100,
        PAGE_AUG,
        "same as v0, separate output directory",
    ),
    "v2": Variant(
        "v2",
        "data_v2",
        2000,
        200,
        CELL_AUG_JITTER4,
        "per-cell augmentation, mild 4 px jitter, fixed 3x3 blur",
    ),
    "v3": Variant(
        "v3",
        "data_v3",
        2000,
        200,
        CELL_AUG_JITTER20_FIXED,
        "per-cell augmentation, 20 px jitter, fixed 3x3 blur",
    ),
    "v4": Variant(
        "v4",
        "data_v4",
        1000,
        100,
        CELL_AUG_JITTER20_CHOICE,
        "per-cell augmentation, 20 px jitter, blur kernel drawn from 3 or 5",
    ),
}

#: Default variant used when none is given (matches ``PrepareData.py``).
DEFAULT_VARIANT = "v0"


def get_variant(name: str) -> Variant:
    """Look up a :class:`Variant` by name, with a helpful error message."""
    try:
        return VARIANTS[name]
    except KeyError:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"unknown variant {name!r}; known variants: {known}") from None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def draw_grid(
    img: np.ndarray, size: int = SIZE, margin: int = GRID_MARGIN
) -> tuple[int, int, int]:
    """Draw the puzzle grid on *img*; return the ``(x0, y0, step)`` geometry."""
    height, width = img.shape[:2]
    step = (min(height, width) - 2 * margin) // size
    if step <= 0:
        raise ValueError(
            f"image {width}x{height} is too small for a {size}x{size} grid "
            f"with a {margin} px margin"
        )
    x0 = (width - size * step) // 2
    y0 = (height - size * step) // 2
    for i in range(size + 1):
        thickness = 3 if i % 3 == 0 else 1
        cv.line(
            img, (x0 + i * step, y0), (x0 + i * step, y0 + size * step), (0, 0, 0), thickness
        )
        cv.line(
            img, (x0, y0 + i * step), (x0 + size * step, y0 + i * step), (0, 0, 0), thickness
        )
    return (x0, y0, step)


def put_digit(
    img: np.ndarray, r: int, c: int, digit: int, x0: int, y0: int, step: int
) -> None:
    """Draw *digit* centred in cell ``(r, c)``; blank cells are skipped."""
    if digit == BLANK:
        return
    text = str(digit)
    scale = step / 60.0
    thickness = 2
    (text_w, text_h), _ = cv.getTextSize(text, FONT, scale, thickness)
    cx = x0 + c * step + (step - text_w) // 2
    cy = y0 + r * step + (step + text_h) // 2 - text_h // 4
    cv.putText(img, text, (cx, cy), FONT, scale, (0, 0, 0), thickness, cv.LINE_AA)


def render_page(
    puzzle: Grid,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
    margin: int = GRID_MARGIN,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Render a clean page for *puzzle*; return ``(page, (x0, y0, step))``."""
    page = np.full((img_h, img_w, 3), 255, np.uint8)
    x0, y0, step = draw_grid(page, SIZE, margin)
    for r in range(SIZE):
        for c in range(SIZE):
            put_digit(page, r, c, puzzle[r][c], x0, y0, step)
    return page, (x0, y0, step)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _perspective_jitter(img: np.ndarray, jitter_px: int, border_value: int) -> np.ndarray:
    """Warp *img* with a small random perspective perturbation.

    Draws from the global :mod:`random` state — two ``randint`` calls per corner,
    x before y, corners top-left, top-right, bottom-right, bottom-left — so a
    seeded run reproduces the original scripts exactly.
    """
    if jitter_px <= 0:
        return img
    height, width = img.shape[:2]
    src = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
    dst = np.float32(
        [
            (
                fx * width + random.randint(-jitter_px, jitter_px),
                fy * height + random.randint(-jitter_px, jitter_px),
            )
            for fx, fy in _CORNERS
        ]
    )
    matrix = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(
        img,
        matrix,
        (width, height),
        flags=cv.INTER_LINEAR,
        borderValue=(border_value, border_value, border_value),
    )


def _maybe_blur(img: np.ndarray, aug: Augmentation) -> np.ndarray:
    """Apply the profile's Gaussian blur (random kernel choice only when >1 option)."""
    if random.random() >= aug.blur_prob:
        return img
    kernels = aug.blur_kernels
    kernel = random.choice(kernels) if len(kernels) > 1 else kernels[0]
    return cv.GaussianBlur(img, (kernel, kernel), 0)


def augment_page(page: np.ndarray, aug: Augmentation) -> np.ndarray:
    """Apply whole-page augmentation (the original recipe), then return the page."""
    return _maybe_blur(_perspective_jitter(page, aug.jitter_px, aug.border_value), aug)


def augment_crop(crop: np.ndarray, aug: Augmentation) -> np.ndarray:
    """Apply per-cell augmentation (the improved recipe), then return the crop."""
    return _maybe_blur(_perspective_jitter(crop, aug.jitter_px, aug.border_value), aug)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_image(path: Path, img: np.ndarray) -> Path:
    """Write *img* to *path* (creating parents) and fail loudly on error.

    ``cv.imwrite`` returns ``False`` instead of raising, which used to drop
    crops silently whenever the output directory was not writable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), img):
        raise OSError(f"could not write image: {path}")
    return path


def ensure_dirs(out_dir: Path, split: str) -> None:
    """Create the ``out_dir/<split>/<class>`` directories for one split."""
    for cls in CLASSES:
        (Path(out_dir) / split / str(cls)).mkdir(parents=True, exist_ok=True)


def save_crops(
    page: np.ndarray,
    geom: tuple[int, int, int],
    puzzle: Grid,
    out_dir: Path,
    split: str,
    base_id: str,
    aug: Augmentation | None = None,
    cell_size: int = CELL_SIZE,
) -> int:
    """Save the 81 cells of *page* into ``out_dir/<split>/<label>/``.

    Args:
        page: the rendered (or page-augmented) 9x9 page.
        geom: ``(x0, y0, step)`` geometry of the grid inside *page*.
        puzzle: the puzzle used to pick the label directory (0..9).
        aug: per-cell augmentation profile, or ``None`` for clean crops.

    Returns:
        Number of crops written.
    """
    x0, y0, step = geom
    written = 0
    index = 0
    for r in range(SIZE):
        for c in range(SIZE):
            y1, x1 = y0 + r * step, x0 + c * step
            roi = page[y1 : y1 + step, x1 : x1 + step].copy()
            if aug is not None:
                roi = augment_crop(roi, aug)
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, (cell_size, cell_size), interpolation=cv.INTER_AREA)
            label = puzzle[r][c]
            write_image(
                Path(out_dir) / split / str(label) / f"{base_id}_{index:02d}.png", gray
            )
            written += 1
            index += 1
    return written


@dataclass
class SplitStats:
    """What one split actually produced."""

    split: str
    requested: int
    rendered: int
    crops: int = 0
    pages: int = 0

    @property
    def empty(self) -> bool:
        return self.rendered == 0


def process_split(
    puzzles: list[Grid],
    split: str,
    n_limit: int,
    out_dir: Path,
    aug: Augmentation,
    *,
    progress: bool = True,
) -> SplitStats:
    """Render up to *n_limit* puzzles of one split and write crops and pages.

    ``min(n_limit, len(puzzles))`` silently capped the count before; the cap is
    now reported (and warns) instead of shrinking the dataset quietly.
    """
    out_dir = Path(out_dir)
    ensure_dirs(out_dir, split)
    grid_dir = out_dir / "grids" / split
    grid_dir.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(split=split, requested=n_limit, rendered=min(n_limit, len(puzzles)))
    if stats.rendered < n_limit:
        warnings.warn(
            f"{split}: only {stats.rendered} of {n_limit} requested puzzles are available",
            stacklevel=2,
        )

    for i in tqdm(range(stats.rendered), desc=f"render {split}", disable=not progress):
        puzzle = puzzles[i]

        page, geom = render_page(puzzle)
        stats.crops += save_crops(page, geom, puzzle, out_dir, split, f"{i:06d}_clean")
        write_image(grid_dir / f"{i:06d}_clean.png", page)
        stats.pages += 1

        if aug.scope == "page":
            warped = augment_page(page, aug)
            stats.crops += save_crops(warped, geom, puzzle, out_dir, split, f"{i:06d}_cam")
            write_image(grid_dir / f"{i:06d}_cam.png", warped)
            stats.pages += 1
        else:
            stats.crops += save_crops(
                page, geom, puzzle, out_dir, split, f"{i:06d}_cam", aug
            )

    return stats


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_puzzles(
    parquet_path: str | Path,
    n_train: int,
    n_val: int,
) -> tuple[tuple[list[Grid], list[Grid] | None], tuple[list[Grid], list[Grid] | None]]:
    """Load a train/val split of 9x9 grids from the HuggingFace parquet.

    The frame is shuffled with a fixed ``random_state`` (as before) and sliced
    ``[:n_train]`` / ``[n_train:n_train + n_val]``.

    Warns:
        UserWarning: when the dataset holds fewer rows than requested, which
            used to end with a *silently empty* validation split followed by
            ``Best val acc: 0.0000`` in ``Training.py``.  Turn warnings into
            errors with ``--strict`` (or ``-W error::UserWarning``) to fail fast.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet dataset not found: {parquet_path}")

    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)

    if "puzzle" not in frame.columns:
        raise ValueError(f"expected a 'puzzle' column, got: {list(frame.columns)}")

    available = len(frame)
    if available < n_train + n_val:
        warnings.warn(
            f"dataset has {available} rows but {n_train + n_val} were requested "
            f"(train={n_train}, val={n_val}): the validation split may be empty",
            stacklevel=2,
        )

    def parse_all(slice_frame: pd.DataFrame) -> list[Grid]:
        return [parse_grid_string(str(value)) for value in slice_frame["puzzle"]]

    def parse_solutions(slice_frame: pd.DataFrame) -> list[Grid] | None:
        if "solution" not in slice_frame.columns:
            return None
        return [parse_grid_string(str(value)) for value in slice_frame["solution"]]

    train_frame = frame.iloc[:n_train]
    val_frame = frame.iloc[n_train : n_train + n_val]
    return (
        (parse_all(train_frame), parse_solutions(train_frame)),
        (parse_all(val_frame), parse_solutions(val_frame)),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class GenerationReport:
    """Summary of one generation run (also used for the CLI output)."""

    variant: Variant
    out_dir: Path
    parquet: Path
    seed: int
    splits: list[SplitStats] = field(default_factory=list)

    @property
    def total_crops(self) -> int:
        return sum(s.crops for s in self.splits)

    def lines(self) -> list[str]:
        aug = self.variant.augmentation
        out = [
            f"variant {self.variant.name} ({self.variant.legacy_script}) -> {self.out_dir}",
            f"  dataset: {self.parquet}   seed: {self.seed}",
            f"  augmentation: {aug.scope}-level, jitter +/-{aug.jitter_px} px, "
            f"blur p={aug.blur_prob} kernels={aug.blur_kernels}",
        ]
        for stats in self.splits:
            note = f"  (requested {stats.requested})" if stats.rendered < stats.requested else ""
            out.append(
                f"  {stats.split}: {stats.rendered} puzzles{note}, "
                f"{stats.crops} crops, {stats.pages} page images"
            )
            if stats.empty:
                out.append(f"    WARNING: split '{stats.split}' is empty")
        out.append(f"  total: {self.total_crops} crops")
        return out


def generate(
    out_dir: str | Path,
    parquet_path: str | Path,
    n_train: int,
    n_val: int,
    aug: Augmentation,
    variant: Variant | None = None,
    seed: int = 0,
    progress: bool = True,
) -> GenerationReport:
    """Generate a full dataset (train + val) for one augmentation profile."""
    random.seed(seed)
    out_dir = Path(out_dir)
    (p_train, _s_train), (p_val, _s_val) = load_puzzles(parquet_path, n_train, n_val)
    splits = [
        process_split(p_train, "train", n_train, out_dir, aug, progress=progress),
        process_split(p_val, "val", n_val, out_dir, aug, progress=progress),
    ]
    return GenerationReport(
        variant=variant or Variant("custom", str(out_dir), n_train, n_val, aug),
        out_dir=out_dir,
        parquet=Path(parquet_path),
        seed=seed,
        splits=splits,
    )


def generate_variant(
    variant_name: str,
    *,
    out_dir: str | Path | None = None,
    parquet_path: str | Path = "train_1.parquet",
    n_train: int | None = None,
    n_val: int | None = None,
    seed: int = 0,
    progress: bool = True,
) -> GenerationReport:
    """Generate one named variant, using its own defaults for anything omitted."""
    variant = get_variant(variant_name)
    return generate(
        out_dir if out_dir is not None else variant.out_dir,
        parquet_path,
        variant.n_train if n_train is None else n_train,
        variant.n_val if n_val is None else n_val,
        variant.augmentation,
        variant=variant,
        seed=seed,
        progress=progress,
    )
