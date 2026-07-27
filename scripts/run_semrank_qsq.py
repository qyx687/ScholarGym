#!/usr/bin/env python3
"""Dedicated process entry point for SemRank runs.

Keeping the process command distinct prevents unrelated baseline watchdogs
that target ``code/eval.py`` from terminating an active SemRank evaluation.
The experiment logic remains the exact ``eval.main`` implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from eval import main  # noqa: E402


if __name__ == "__main__":
    main()
