"""PyInstaller runtime hook: ensure sys.stdout/stderr are never None (--windowed mode)."""
import sys, os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
