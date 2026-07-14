"""Test import setup for the src-layout package."""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
