"""Experiment arm v3 — see :mod:`sudoku_data` for the profile.

Kept as a thin wrapper so the documented workflow (python3 PrepareData3.py) and any
existing scripts keep working; the implementation now lives in one place.
Extra flags from `prepare_data.py` are accepted, e.g.::

    python3 PrepareData3.py --n-train 500 --out-dir /data/sudoku_v3
"""

from prepare_data import run

if __name__ == "__main__":
    raise SystemExit(run("v3"))
