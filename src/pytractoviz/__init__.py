"""pyTractoViz package.

Python tools for diffusion tractography visualization
"""

from __future__ import annotations

from pytractoviz._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
