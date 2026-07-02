from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pytractoviz.group_interactive_report import generate_group_interactive_html, parse_group_spec
from pytractoviz.qc_config import load_qc_path_config
from pytractoviz.qc_viz import generate_group_summary_html, generate_subject_qc_html
from pytractoviz.qc_viz import main as run_subject_qc

if TYPE_CHECKING:
    import argparse

_REPORT_COMMANDS = frozenset({"run", "html", "group-html", "group-interactive"})


def _add_report_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register batch QC / report subcommands on the main parser."""
    run_parser = subparsers.add_parser("run", help="Run full QC pipeline for a subject.")
    run_parser.add_argument("subject", help="Subject ID (e.g. sub-0000).")
    run_parser.add_argument("project", help="Project name (e.g. UNC).")
    run_parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel tract workers (default: 1).")
    run_parser.add_argument(
        "--render-backend",
        choices=["osmesa", "egl"],
        default="osmesa",
        help="Rendering backend for VTK/FURY (use egl on GPU nodes).",
    )
    run_parser.add_argument(
        "--no-streamline-caps",
        action="store_true",
        help="Disable all streamline count caps (same as 0 for each --max-streamlines-*).",
    )
    run_parser.add_argument(
        "--max-streamlines-view",
        type=int,
        default=5000,
        help="Cap streamlines for tract/atlas view rendering; 0 = no cap.",
    )
    run_parser.add_argument(
        "--max-streamlines-cci",
        type=int,
        default=5000,
        help="Cap streamlines for CCI; 0 = no cap.",
    )
    run_parser.add_argument(
        "--max-streamlines-afq",
        type=int,
        default=6000,
        help="Cap streamlines for AFQ profiles; 0 = no cap.",
    )
    run_parser.add_argument(
        "--max-streamlines-shape",
        type=int,
        default=4000,
        help="Cap streamlines for shape similarity; 0 = no cap.",
    )
    run_parser.add_argument(
        "--tractography-root",
        default=None,
        help="Top-level tractography directory (or set $PYTRACTOVIZ_TRACTOGRAPHY_ROOT / $TRACTOGRAPHY_ROOT).",
    )
    run_parser.add_argument(
        "--atlas-dir",
        default=None,
        help="Atlas bundle .trk directory (default: <tractography-root>/hcp_atlas_16_bundles/.../bundles).",
    )
    run_parser.add_argument(
        "--atlas-ref-img",
        default=None,
        help="MNI brain mask for atlas rendering (default: <tractography-root>/templates/...).",
    )
    run_parser.add_argument(
        "--qsiprep-root",
        default=None,
        help="Root containing qsiprep/<project>/<subject>/anat (default: tractography-root).",
    )
    run_parser.add_argument(
        "--qsirecon-root",
        default=None,
        help="Root containing qsirecon/<project>/derivatives (default: tractography-root).",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="QC output directory for this project (default: <qsirecon-root>/qsirecon/<project>/derivatives/QC).",
    )

    html_parser = subparsers.add_parser("html", help="(Re)generate HTML report from existing qc_data.json.")
    html_parser.add_argument("qc_json", help="Path to qc_data.json.")

    group_parser = subparsers.add_parser(
        "group-html",
        help="Generate group summary HTML with links to each subject's report.",
    )
    group_parser.add_argument("group_dir", help="Path to group QC directory (contains one subdir per subject).")
    group_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: group_dir/group_summary.html).",
    )
    group_parser.add_argument(
        "--skip-pairwise-shape",
        action="store_true",
        help="Do not compute subject x subject shape similarity (can be slow for many subjects).",
    )
    group_parser.add_argument(
        "--pairwise-shape-max-streamlines",
        type=int,
        default=4000,
        help="Cap streamlines for pairwise shape similarity (default: 4000; 0 = no cap).",
    )
    group_parser.add_argument(
        "--pairwise-shape-jobs",
        type=int,
        default=1,
        help="Parallel workers for pairwise shape (one tract per worker). Use 0 for os.cpu_count(). Default: 1 (serial).",
    )

    group_interactive_parser = subparsers.add_parser(
        "group-interactive",
        help="Generate interactive HTML report from combined tract/profile stats (supports multiple groups).",
    )
    group_interactive_parser.add_argument(
        "groups",
        nargs="+",
        help="Group QC directory, or label=path (e.g. UNC=/data/QC/UNC ADNI=/data/QC/ADNI).",
    )
    group_interactive_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: group_interactive.html or group_interactive_comparison.html).",
    )
    group_interactive_parser.add_argument(
        "--refresh-csv",
        action="store_true",
        help="Rescan subject subdirs and rewrite combined_*.csv before building the report.",
    )


def _run_report_command(opts: argparse.Namespace) -> int:
    """Execute a parsed report subcommand."""
    if opts.command == "html":
        if Path(opts.qc_json).suffix != ".json":
            print("Error: qc_json must be a .json file (e.g. qc_data.json).", file=sys.stderr)
            return 2
        generate_subject_qc_html(opts.qc_json)
        print(f"HTML report written next to {opts.qc_json}")
        return 0

    if opts.command == "group-html":
        psm = int(opts.pairwise_shape_max_streamlines)
        pj = int(opts.pairwise_shape_jobs)
        if pj <= 0:
            pj = int(os.cpu_count() or 1)
        out = generate_group_summary_html(
            opts.group_dir,
            opts.output,
            pairwise_shape=not opts.skip_pairwise_shape,
            pairwise_shape_max_streamlines=psm,
            pairwise_shape_jobs=pj,
        )
        print(f"Group summary written to {out}")
        interactive_path = Path(opts.group_dir) / "group_interactive.html"
        if interactive_path.exists():
            print(f"Interactive stats report: {interactive_path}")
        return 0

    if opts.command == "group-interactive":
        specs = [parse_group_spec(g) for g in opts.groups]
        out = generate_group_interactive_html(
            specs,
            opts.output,
            refresh_csv=bool(opts.refresh_csv),
        )
        print(f"Interactive group report written to {out}")
        return 0

    if opts.command == "run":
        if opts.no_streamline_caps:
            mv = mc = ma = ms = 0
        else:
            mv, mc, ma, ms = (
                int(opts.max_streamlines_view),
                int(opts.max_streamlines_cci),
                int(opts.max_streamlines_afq),
                int(opts.max_streamlines_shape),
            )
        try:
            path_config = load_qc_path_config(
                tractography_root=opts.tractography_root,
                atlas_dir=opts.atlas_dir,
                atlas_ref_img=opts.atlas_ref_img,
                qsiprep_root=opts.qsiprep_root,
                qsirecon_root=opts.qsirecon_root,
                qc_output_dir=opts.output_dir,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        run_subject_qc(
            opts.subject,
            opts.project,
            path_config=path_config,
            n_jobs=max(1, int(opts.jobs)),
            render_backend=opts.render_backend,
            max_streamlines_view=mv,
            max_streamlines_cci=mc,
            max_streamlines_afq=ma,
            max_streamlines_shape=ms,
        )
        return 0

    print(f"Error: unknown report subcommand: {opts.command!r}", file=sys.stderr)
    return 2
