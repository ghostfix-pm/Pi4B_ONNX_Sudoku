"""CLI for the synthetic Sudoku dataset generator.

One command replaces the five ``PrepareData*.py`` scripts::

    python3 prepare_data.py --variant v4          # == python3 PrepareData4.py
    python3 prepare_data.py --list-variants
    python3 prepare_data.py --variant v2 --n-train 5000 --n-val 500 --out-dir /data/sudoku_v2

The variant profiles (output directory, train/val sizes, augmentation recipe)
live in :mod:`sudoku_data`; anything passed on the command line overrides the
profile default.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from sudoku_data import (
    DEFAULT_VARIANT,
    VARIANTS,
    generate_variant,
    get_variant,
)


def build_parser(variant_name: str = DEFAULT_VARIANT) -> argparse.ArgumentParser:
    """Build the argument parser, with *variant_name* providing the defaults."""
    variant = get_variant(variant_name)
    parser = argparse.ArgumentParser(
        prog="prepare_data.py",
        description=(
            "Render the synthetic Sudoku dataset (train/val crops + page images) "
            "for one experiment variant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        default=variant.name,
        choices=sorted(VARIANTS),
        help="experiment arm to generate",
    )
    parser.add_argument(
        "--parquet",
        default="train_1.parquet",
        help="HuggingFace parquet with a 'puzzle' (and optional 'solution') column",
    )
    parser.add_argument(
        "--out-dir",
        default=variant.out_dir,
        help="directory to write <split>/<class>/*.png into",
    )
    parser.add_argument("--n-train", type=int, default=variant.n_train, help="training puzzles")
    parser.add_argument("--n-val", type=int, default=variant.n_val, help="validation puzzles")
    parser.add_argument("--seed", type=int, default=0, help="seed for the augmentation RNG")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings (empty/short splits) as errors",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable the tqdm progress bars"
    )
    parser.add_argument(
        "--list-variants", action="store_true", help="show the variants and exit"
    )
    return parser


def print_variants() -> None:
    """Print the variant table."""
    print("variants (profile defaults):")
    for name, variant in sorted(VARIANTS.items()):
        aug = variant.augmentation
        print(
            f"  {name}  {variant.legacy_script:15s} out={variant.out_dir:8s} "
            f"train/val={variant.n_train}/{variant.n_val:<4d} "
            f"aug={aug.scope}(jitter {aug.jitter_px}px, blur p={aug.blur_prob}, "
            f"kernels {aug.blur_kernels})"
        )
        print(f"        {variant.description}")


def run(variant_name: str, argv: list[str] | None = None, *, progress: bool = True) -> int:
    """Run the generator for one variant. Returns a process exit code."""
    parser = build_parser(variant_name)
    args = parser.parse_args(argv)

    if args.list_variants:
        print_variants()
        return 0

    if args.strict:
        warnings.simplefilter("error", UserWarning)

    for label, value in (("--n-train", args.n_train), ("--n-val", args.n_val)):
        if value < 0:
            parser.error(f"{label} must be >= 0, got {value}")

    try:
        report = generate_variant(
            args.variant,
            out_dir=args.out_dir,
            parquet_path=args.parquet,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=args.seed,
            progress=progress and not args.no_progress,
        )
    except UserWarning as exc:
        print(f"error (--strict): {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: download the dataset with\n"
            "  wget https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset/resolve/main/train_1.parquet",
            file=sys.stderr,
        )
        return 2

    for line in report.lines():
        print(line)
    empty = [stats.split for stats in report.splits if stats.empty]
    if empty:
        print(
            f"warning: empty split(s): {', '.join(empty)} "
            f"(dataset too small for --n-train {args.n_train} / --n-val {args.n_val})",
            file=sys.stderr,
        )
    print(f"done, output under {Path(args.out_dir).resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: variant comes from ``--variant`` (default v0)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known = parser.parse_known_args(argv)[0]
    return run(known.variant, argv)


if __name__ == "__main__":
    raise SystemExit(main())
