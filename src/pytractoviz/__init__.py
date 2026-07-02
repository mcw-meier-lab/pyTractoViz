"""pyTractoViz package.

Python tools for diffusion tractography visualization
"""

from __future__ import annotations

from pytractoviz._internal.cli import get_parser, main
from pytractoviz.group_interactive_report import (
    generate_group_interactive_html,
    parse_group_spec,
)
from pytractoviz.html import create_quality_check_html, create_summary_csv
from pytractoviz.qc_config import QCPathConfig, load_qc_path_config
from pytractoviz.qc_viz import (
    collect_group_qc_data,
    generate_group_summary_html,
    generate_subject_qc_html,
    write_combined_group_csvs,
)

__all__: list[str] = [
    "QCPathConfig",
    "collect_group_qc_data",
    "create_quality_check_html",
    "create_summary_csv",
    "generate_group_interactive_html",
    "generate_group_summary_html",
    "generate_subject_qc_html",
    "get_parser",
    "load_qc_path_config",
    "main",
    "parse_group_spec",
    "write_combined_group_csvs",
]
