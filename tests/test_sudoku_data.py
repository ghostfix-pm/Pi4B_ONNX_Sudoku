"""Tests for :mod:`sudoku_data` — the five experiment arms and the renderer."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from conftest import EASY_PUZZLE_STRING, SOLVED_GRID, grid_from_string

from sudoku_data import (
    CELL_AUG_JITTER20_CHOICE,
    CELL_SIZE,
    CLASSES,
    DEFAULT_VARIANT,
    VARIANTS,
    Augmentation,
    SplitStats,
    augment_crop,
    augment_page,
    draw_grid,
    ensure_dirs,
    generate,
    generate_variant,
    get_variant,
    load_puzzles,
    process_split,
    put_digit,
    render_page,
    save_crops,
    write_image,
)

PAGE_SCOPE_VARIANTS = [name for name, variant in VARIANTS.items() if variant.augmentation.scope == "page"]
CELL_SCOPE_VARIANTS = [name for name, variant in VARIANTS.items() if variant.augmentation.scope == "cell"]


def tree_hashes(root: Path) -> dict[str, str]:
    """Map every file below *root* to its sha256."""
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def read_gray(path: Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    assert image is not None, f"unreadable image: {path}"
    return image


# --------------------------------------------------------------------------- #
# Variant configuration
# --------------------------------------------------------------------------- #
def test_variants_match_the_documented_table() -> None:
    expected = {
        "v0": ("data", 1000, 100, "page", 20, 0.5, (3, 5)),
        "v1": ("data_v1", 1000, 100, "page", 20, 0.5, (3, 5)),
        "v2": ("data_v2", 2000, 200, "cell", 4, 0.3, (3,)),
        "v3": ("data_v3", 2000, 200, "cell", 20, 0.3, (3,)),
        "v4": ("data_v4", 1000, 100, "cell", 20, 0.5, (3, 5)),
    }
    assert set(VARIANTS) == set(expected)
    for name, (out_dir, n_train, n_val, scope, jitter, blur_prob, kernels) in expected.items():
        variant = VARIANTS[name]
        assert variant.out_dir == out_dir
        assert (variant.n_train, variant.n_val) == (n_train, n_val)
        assert variant.augmentation.scope == scope
        assert variant.augmentation.jitter_px == jitter
        assert variant.augmentation.blur_prob == blur_prob
        assert variant.augmentation.blur_kernels == kernels


def test_default_variant_is_the_original_script() -> None:
    assert DEFAULT_VARIANT == "v0"
    assert VARIANTS["v0"].out_dir == "data"


def test_legacy_script_names() -> None:
    assert VARIANTS["v0"].legacy_script == "PrepareData.py"
    assert VARIANTS["v1"].legacy_script == "PrepareData1.py"
    assert VARIANTS["v4"].legacy_script == "PrepareData4.py"


def test_get_variant_reports_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown variant 'v9'"):
        get_variant("v9")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"scope": "page", "jitter_px": -1}, "jitter_px"),
        ({"scope": "page", "blur_prob": 1.5}, "blur_prob"),
        ({"scope": "page", "blur_kernels": ()}, "blur_kernels"),
        ({"scope": "page", "blur_kernels": (2,)}, "odd sizes"),
        ({"scope": "sideways"}, "scope must be"),
    ],
)
def test_augmentation_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Augmentation(**kwargs)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_page_geometry_matches_the_original_constants() -> None:
    page, (x0, y0, step) = render_page(SOLVED_GRID)
    assert page.shape == (1200, 800, 3)
    assert step == 75
    assert (x0, y0) == (62, 262)
    assert page.dtype == np.uint8
    assert not (page == 255).all()  # digits and grid lines were drawn


def test_render_page_draws_every_given() -> None:
    page, _ = render_page(SOLVED_GRID)
    blank_page, _ = render_page([[0] * 9 for _ in range(9)])
    assert (page < 128).sum() > (blank_page < 128).sum()


def test_draw_grid_rejects_images_that_are_too_small() -> None:
    with pytest.raises(ValueError, match="too small"):
        draw_grid(np.full((50, 50, 3), 255, np.uint8))


def test_put_digit_skips_blank_cells() -> None:
    img = np.full((1200, 800, 3), 255, np.uint8)
    before = img.copy()
    put_digit(img, 0, 0, 0, 62, 262, 75)
    assert np.array_equal(img, before)


def test_put_digit_draws_six_and_nine_differently() -> None:
    six = np.full((1200, 800, 3), 255, np.uint8)
    nine = np.full((1200, 800, 3), 255, np.uint8)
    put_digit(six, 0, 0, 6, 62, 262, 75)
    put_digit(nine, 0, 0, 9, 62, 262, 75)
    assert not np.array_equal(six, nine)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def test_augmentation_is_deterministic_for_a_seed() -> None:
    page, _ = render_page(SOLVED_GRID)
    random.seed(7)
    first = augment_page(page, CELL_AUG_JITTER20_CHOICE)
    random.seed(7)
    second = augment_page(page, CELL_AUG_JITTER20_CHOICE)
    assert np.array_equal(first, second)


def test_augmentation_changes_with_the_seed() -> None:
    page, _ = render_page(SOLVED_GRID)
    random.seed(1)
    first = augment_page(page, CELL_AUG_JITTER20_CHOICE)
    random.seed(2)
    second = augment_page(page, CELL_AUG_JITTER20_CHOICE)
    assert not np.array_equal(first, second)


def test_zero_jitter_and_zero_blur_is_a_no_op() -> None:
    page, _ = render_page(SOLVED_GRID)
    profile = Augmentation(scope="page", jitter_px=0, blur_prob=0.0)
    random.seed(0)
    assert np.array_equal(augment_page(page, profile), page)


def test_page_augmentation_fills_the_border_with_the_profile_colour() -> None:
    page, _ = render_page(SOLVED_GRID)
    random.seed(3)
    warped = augment_page(page, Augmentation(scope="page", jitter_px=60, blur_prob=0.0, border_value=220))
    corner = warped[0, 0]
    assert tuple(int(v) for v in corner) == (220, 220, 220)


def test_cell_augmentation_uses_the_white_border() -> None:
    crop = np.full((75, 75, 3), 0, np.uint8)
    random.seed(5)
    warped = augment_crop(crop, Augmentation(scope="cell", jitter_px=60, blur_prob=0.0, border_value=255))
    assert tuple(int(v) for v in warped[0, 0]) == (255, 255, 255)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_write_image_creates_parents(tmp_path: Path) -> None:
    path = write_image(tmp_path / "a" / "b" / "cell.png", np.zeros((4, 4), np.uint8))
    assert path.is_file()


def test_write_image_raises_when_opencv_cannot_write(tmp_path: Path) -> None:
    """Regression: cv.imwrite returns False instead of raising."""
    target = tmp_path / "not-a-file.png"
    target.mkdir()
    with pytest.raises(OSError, match="could not write image"):
        write_image(target, np.zeros((4, 4), np.uint8))


def test_ensure_dirs_creates_every_class_directory(tmp_path: Path) -> None:
    ensure_dirs(tmp_path, "train")
    for cls in CLASSES:
        assert (tmp_path / "train" / str(cls)).is_dir()


def test_save_crops_writes_81_files_with_the_right_labels(tmp_path: Path) -> None:
    puzzle = grid_from_string(EASY_PUZZLE_STRING)
    page, geom = render_page(puzzle)
    written = save_crops(page, geom, puzzle, tmp_path, "train", "000000_clean")
    assert written == 81
    assert len(list(tmp_path.rglob("*.png"))) == 81
    # 51 blanks -> class 0, 30 givens spread over classes 1..9
    assert len(list((tmp_path / "train" / "0").glob("*.png"))) == 51
    assert sum(len(list((tmp_path / "train" / str(cls)).glob("*.png"))) for cls in range(1, 10)) == 30
    for path in (tmp_path / "train" / "5").glob("*.png"):
        image = read_gray(path)
        assert image.shape == (CELL_SIZE, CELL_SIZE)


def test_save_crops_with_augmentation_differs_from_clean(tmp_path: Path) -> None:
    puzzle = grid_from_string(EASY_PUZZLE_STRING)
    page, geom = render_page(puzzle)
    random.seed(11)
    save_crops(page, geom, puzzle, tmp_path, "train", "clean")
    random.seed(11)
    save_crops(page, geom, puzzle, tmp_path, "train", "cam", CELL_AUG_JITTER20_CHOICE)
    clean = read_gray(tmp_path / "train" / "5" / "clean_00.png")
    cam = read_gray(tmp_path / "train" / "5" / "cam_00.png")
    assert not np.array_equal(clean, cam)


def test_save_crops_uses_the_requested_size(tmp_path: Path) -> None:
    puzzle = grid_from_string(EASY_PUZZLE_STRING)
    page, geom = render_page(puzzle)
    save_crops(page, geom, puzzle, tmp_path, "train", "small", cell_size=14)
    assert read_gray(next((tmp_path / "train" / "0").glob("small_*.png"))).shape == (14, 14)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_process_split_reports_what_it_rendered(tmp_path: Path) -> None:
    puzzles = [grid_from_string(EASY_PUZZLE_STRING), grid_from_string(EASY_PUZZLE_STRING)]
    stats = process_split(puzzles, "train", 2, tmp_path, VARIANTS["v0"].augmentation, progress=False)
    assert isinstance(stats, SplitStats)
    assert stats.rendered == 2
    assert stats.crops == 4 * 81  # clean + camera crops for both puzzles
    assert stats.pages == 4  # clean + warped page image per puzzle
    assert not stats.empty


def test_process_split_cell_scope_saves_one_page_per_puzzle(tmp_path: Path) -> None:
    puzzles = [grid_from_string(EASY_PUZZLE_STRING)]
    stats = process_split(puzzles, "train", 1, tmp_path, VARIANTS["v4"].augmentation, progress=False)
    assert stats.pages == 1
    assert stats.crops == 2 * 81


def test_process_split_warns_when_fewer_puzzles_are_available(tmp_path: Path) -> None:
    puzzles = [grid_from_string(EASY_PUZZLE_STRING)]
    with pytest.warns(UserWarning, match="only 1 of 5 requested puzzles are available"):
        stats = process_split(puzzles, "train", 5, tmp_path, VARIANTS["v4"].augmentation, progress=False)
    assert stats.rendered == 1
    assert stats.requested == 5


def test_process_split_marks_an_empty_split(tmp_path: Path) -> None:
    stats = process_split([], "val", 10, tmp_path, VARIANTS["v0"].augmentation, progress=False)
    assert stats.empty and stats.rendered == 0 and stats.crops == 0


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def test_load_puzzles_returns_train_and_val_grids(tiny_parquet: Path) -> None:
    (train, train_solutions), (val, val_solutions) = load_puzzles(tiny_parquet, 3, 1)
    assert len(train) == 3 and len(val) == 1
    assert all(len(row) == 9 for grid in train for row in grid)
    assert train_solutions is not None and len(train_solutions) == 3
    assert val_solutions is not None and len(val_solutions) == 1


def test_load_puzzles_handles_a_dataset_without_solutions(tiny_parquet: Path, tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.read_parquet(tiny_parquet, engine="pyarrow")[["puzzle"]]
    path = tmp_path / "no_solutions.parquet"
    frame.to_parquet(path, engine="pyarrow")
    (train, train_solutions), _ = load_puzzles(path, 2, 1)
    assert len(train) == 2
    assert train_solutions is None


def test_load_puzzles_warns_when_the_dataset_is_too_small(tiny_parquet: Path) -> None:
    with pytest.warns(UserWarning, match="dataset has 4 rows but 1100 were requested"):
        load_puzzles(tiny_parquet, 1000, 100)


def test_load_puzzles_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="parquet dataset not found"):
        load_puzzles(tmp_path / "nope.parquet", 1, 1)


def test_load_puzzles_requires_a_puzzle_column(tiny_parquet: Path, tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "wrong.parquet"
    pd.DataFrame({"p": ["0" * 81] * 3}).to_parquet(path, engine="pyarrow")
    with pytest.raises(ValueError, match="expected a 'puzzle' column"):
        load_puzzles(path, 1, 1)


def test_load_puzzles_split_is_deterministic(tiny_parquet: Path) -> None:
    first = load_puzzles(tiny_parquet, 2, 2)[0][0]
    second = load_puzzles(tiny_parquet, 2, 2)[0][0]
    assert first == second


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generate_reports_totals(tmp_path: Path, tiny_parquet: Path) -> None:
    report = generate(
        tmp_path / "out", tiny_parquet, 2, 2, VARIANTS["v4"].augmentation, progress=False
    )
    assert report.total_crops == 4 * 2 * 81  # 4 puzzles x (clean + camera) x 81 cells
    assert report.splits[0].split == "train"
    lines = "\n".join(report.lines())
    assert "variant custom" in lines or "custom" in lines
    assert "train: 2 puzzles" in lines


def test_generate_is_reproducible_for_a_seed(tmp_path: Path, tiny_parquet: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first, tiny_parquet, 2, 1, VARIANTS["v4"].augmentation, seed=0, progress=False)
    generate(second, tiny_parquet, 2, 1, VARIANTS["v4"].augmentation, seed=0, progress=False)
    assert tree_hashes(first) == tree_hashes(second)


def test_generate_with_a_different_seed_changes_only_the_camera_arm(
    tmp_path: Path, tiny_parquet: Path
) -> None:
    first = tmp_path / "seed0"
    second = tmp_path / "seed1"
    generate(first, tiny_parquet, 1, 0, VARIANTS["v4"].augmentation, seed=0, progress=False)
    generate(second, tiny_parquet, 1, 0, VARIANTS["v4"].augmentation, seed=1, progress=False)
    clean_first = {k: v for k, v in tree_hashes(first).items() if "_clean" in k or "grids" in k}
    clean_second = {k: v for k, v in tree_hashes(second).items() if "_clean" in k or "grids" in k}
    camera_first = {k: v for k, v in tree_hashes(first).items() if "_cam" in k}
    camera_second = {k: v for k, v in tree_hashes(second).items() if "_cam" in k}
    assert clean_first == clean_second
    assert camera_first != camera_second


def test_generate_variant_uses_the_profile_defaults(tmp_path: Path, tiny_parquet: Path) -> None:
    report = generate_variant(
        "v4", out_dir=tmp_path / "v4", parquet_path=tiny_parquet, n_train=1, n_val=1, progress=False
    )
    assert report.variant.name == "v4"
    assert (tmp_path / "v4" / "train" / "9").is_dir()


def test_generate_variant_default_output_directory(tmp_path: Path, tiny_parquet: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = generate_variant("v2", parquet_path=tiny_parquet, n_train=1, n_val=0, progress=False)
    assert report.out_dir == Path("data_v2")
    assert (tmp_path / "data_v2" / "train" / "0").is_dir()


def test_generate_requires_a_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate(
            tmp_path / "out", tmp_path / "missing.parquet", 1, 1, VARIANTS["v0"].augmentation, progress=False
        )


# --------------------------------------------------------------------------- #
# Cross-variant behaviour (the point of the experiment)
# --------------------------------------------------------------------------- #
def test_all_variants_share_the_same_clean_arm(tmp_path: Path, tiny_parquet: Path) -> None:
    """Every arm crops the clean page, so ``*_clean`` output must be identical."""
    clean_hashes = {}
    for name in VARIANTS:
        out_dir = tmp_path / name
        generate(out_dir, tiny_parquet, 1, 0, VARIANTS[name].augmentation, progress=False)
        clean_hashes[name] = {
            key.split("/", 1)[1]: value
            for key, value in tree_hashes(out_dir).items()
            if "/train/" in key and "_clean" in key
        }
    reference = clean_hashes["v0"]
    for name, hashes in clean_hashes.items():
        assert hashes == reference, f"{name} clean crops differ from v0"


def test_page_scope_variants_keep_the_camera_page_image(tmp_path: Path, tiny_parquet: Path) -> None:
    for name in PAGE_SCOPE_VARIANTS:
        out_dir = tmp_path / f"page_{name}"
        generate(out_dir, tiny_parquet, 1, 0, VARIANTS[name].augmentation, progress=False)
        assert list((out_dir / "grids" / "train").glob("*_cam.png")), name
    for name in CELL_SCOPE_VARIANTS:
        out_dir = tmp_path / f"cell_{name}"
        generate(out_dir, tiny_parquet, 1, 0, VARIANTS[name].augmentation, progress=False)
        assert not list((out_dir / "grids" / "train").glob("*_cam.png")), name


def test_larger_jitter_moves_the_crops_further(tmp_path: Path, tiny_parquet: Path) -> None:
    """v2 (+-4 px) must distort a cell less than v4 (+-20 px)."""

    def mean_deviation(variant_name: str) -> float:
        out_dir = tmp_path / variant_name
        generate(
            out_dir,
            tiny_parquet,
            1,
            0,
            VARIANTS[variant_name].augmentation,
            seed=0,
            progress=False,
        )
        deviations = []
        for clean_path in (out_dir / "train" / "5").glob("000000_clean_*.png"):
            cam_path = clean_path.with_name(clean_path.name.replace("_clean_", "_cam_"))
            clean = read_gray(clean_path).astype(np.float64)
            camera = read_gray(cam_path).astype(np.float64)
            deviations.append(np.abs(camera - clean).mean())
        return float(np.mean(deviations))

    mild = mean_deviation("v2")
    strong = mean_deviation("v4")
    assert mild > 0.0
    assert strong > mild, f"v4 ({strong:.2f}) should distort more than v2 ({mild:.2f})"
