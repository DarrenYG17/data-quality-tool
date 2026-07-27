"""Ensures the project root is on sys.path so `from src...` imports work
regardless of how/where pytest is invoked from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
