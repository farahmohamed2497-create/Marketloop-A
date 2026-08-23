"""Case-safe import shim for the repository's ``RAG`` source directory.

Python resolves package names case-sensitively even on Windows. Keeping the
existing uppercase directory avoids a disruptive case-only Git rename, while
this module exposes it as the lowercase ``rag`` package used by the app.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "RAG")]
