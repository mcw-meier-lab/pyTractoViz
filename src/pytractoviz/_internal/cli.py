# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m pytractoviz` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `pytractoviz.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `pytractoviz.__main__` in `sys.modules`.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pytractoviz._internal import debug
from pytractoviz.viz import (
    InvalidInputError,
    TractographyVisualizationError,
    TractographyVisualizer,
)


class _DebugInfo(argparse.Action):
    def __init__(self, nargs: int | str | None = 0, **kwargs: Any) -> None:
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        debug._print_debug_info()
        sys.exit(0)


def get_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser.

    Returns
    -------
        An argparse parser.
    """
    parser = argparse.ArgumentParser(prog="pytractoviz")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {debug._get_version()}")
    parser.add_argument("--debug-info", action=_DebugInfo, help="Print debug information.")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive QC subcommand
    qc_parser = subparsers.add_parser(
        "qc-interactive",
        help="Interactive quality check: view tracts one by one in 3D",
        description=(
            "Walk through tracts interactively for quality checking. "
            "Opens each tract in a 3D viewer that you can rotate and examine. "
            "Close the window to move to the next tract."
        ),
    )
    qc_parser.add_argument(
        "tracts",
        nargs="+",
        help="Tract files (.trk) or directories containing .trk files",
    )
    qc_parser.add_argument(
        "--ref",
        "--reference",
        dest="ref_img",
        required=True,
        help="Reference T1-weighted image (required)",
    )
    qc_parser.add_argument(
        "--no-glass-brain",
        dest="show_glass_brain",
        action="store_false",
        help="Disable glass brain outline",
    )
    qc_parser.add_argument(
        "--max-streamlines",
        type=int,
        help="Maximum number of streamlines to render (subsample if more)",
    )
    qc_parser.add_argument(
        "--subsample-factor",
        type=float,
        help="Fraction of streamlines to keep (0.0 to 1.0)",
    )
    qc_parser.add_argument(
        "--max-points-per-streamline",
        type=int,
        help="Maximum points per streamline (resample if more)",
    )
    qc_parser.add_argument(
        "--resample-streamlines",
        action="store_true",
        help="Resample all streamlines to reduce point count",
    )
    qc_parser.add_argument(
        "--flip-lr",
        action="store_true",
        help="Flip left-right (X-axis) - useful for MNI space tracts",
    )
    qc_parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=[800, 800],
        metavar=("WIDTH", "HEIGHT"),
        help="Window size in pixels (default: 800 800)",
    )
    qc_parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of tracts to skip at the start (default: 0)",
    )

    return parser


def _collect_tract_files(paths: list[str]) -> list[Path]:
    """Collect all .trk files from given paths (files or directories).

    Parameters
    ----------
    paths : list[str]
        List of file paths or directory paths.

    Returns
    -------
    list[Path]
        Sorted list of .trk file paths.
    """
    tract_files: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            if path.suffix.lower() == ".trk":
                tract_files.append(path)
            else:
                print(f"Warning: Skipping non-.trk file: {path}", file=sys.stderr)
        elif path.is_dir():
            # Find all .trk files in directory
            found_files = sorted(path.glob("*.trk"))
            if not found_files:
                print(f"Warning: No .trk files found in directory: {path}", file=sys.stderr)
            tract_files.extend(found_files)
        else:
            print(f"Warning: Path is neither file nor directory: {path}", file=sys.stderr)

    return sorted(set(tract_files))  # Remove duplicates and sort


def main(args: list[str] | None = None) -> int:
    """Run the main program.

    This function is executed when you type `pytractoviz` or `python -m pytractoviz`.

    Parameters
    ----------
        args: Arguments passed from the command line.

    Returns
    -------
        An exit code.
    """
    parser = get_parser()
    opts = parser.parse_args(args=args)

    if opts.command == "qc-interactive":
        # Collect all tract files
        tract_files = _collect_tract_files(opts.tracts)

        if not tract_files:
            print("Error: No .trk files found.", file=sys.stderr)
            return 1

        # Skip tracts if requested
        if opts.skip > 0:
            if opts.skip >= len(tract_files):
                print(f"Error: Cannot skip {opts.skip} tracts (only {len(tract_files)} found)", file=sys.stderr)
                return 1
            tract_files = tract_files[opts.skip :]
            print(f"Skipping first {opts.skip} tracts. Starting with: {tract_files[0].name}")

        print(f"\nFound {len(tract_files)} tract file(s) to review.")
        print("Instructions:")
        print("  - Rotate: Click and drag with left mouse button")
        print("  - Zoom: Scroll wheel or right-click and drag")
        print("  - Pan: Middle-click and drag (or Shift+left-click)")
        print("  - Close window to move to next tract")
        print("  - Press 'q' or close window to exit current tract\n")

        # Initialize visualizer
        visualizer = TractographyVisualizer(reference_image=Path(opts.ref_img))

        # Process each tract
        for i, tract_file in enumerate(tract_files, start=1):
            print(f"\n[{i}/{len(tract_files)}] Loading: {tract_file.name}")
            print(f"  Full path: {tract_file}")

            try:
                visualizer.view_tract_interactive(
                    tract_file,
                    ref_img=opts.ref_img,
                    show_glass_brain=opts.show_glass_brain,
                    max_streamlines=opts.max_streamlines,
                    subsample_factor=opts.subsample_factor,
                    max_points_per_streamline=opts.max_points_per_streamline,
                    resample_streamlines=opts.resample_streamlines,
                    flip_lr=opts.flip_lr,
                    window_size=tuple(opts.window_size),
                )
                print(f"  ✓ Completed review of {tract_file.name}")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting.")
                return 130
            except (FileNotFoundError, InvalidInputError, TractographyVisualizationError) as e:
                print(f"  ✗ Error viewing {tract_file.name}: {e}", file=sys.stderr)
                response = input("  Continue to next tract? [Y/n]: ").strip().lower()
                if response and response != "y":
                    return 1

        print(f"\n✓ Completed review of all {len(tract_files)} tract(s).")
        return 0

    # Default behavior (no command specified)
    if opts.command is None:
        parser.print_help()
        return 0

    return 0
