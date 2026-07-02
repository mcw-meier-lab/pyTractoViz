"""Tests for QC report helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pytractoviz import main
from pytractoviz._internal.report_cli import _run_report_command
from pytractoviz.group_interactive_report import parse_group_spec


class TestParseGroupSpec:
    """Tests for :func:`pytractoviz.group_interactive_report.parse_group_spec`."""

    def test_path_only(self, tmp_path: Path) -> None:
        """Test that the group spec is parsed correctly when only a path is provided."""
        label, path = parse_group_spec(str(tmp_path))
        assert label == tmp_path.name
        assert path == tmp_path

    def test_label_equals_path(self, tmp_path: Path) -> None:
        """Test that the group spec is parsed correctly when the label equals the path."""
        label, path = parse_group_spec(f"UNC={tmp_path}")
        assert label == "UNC"
        assert path == tmp_path


class TestReportCli:
    """Tests for report CLI handlers in :mod:`pytractoviz._internal.report_cli`."""

    def test_html_requires_json_suffix(self, tmp_path: Path) -> None:
        """Test that the HTML report is rejected if the JSON file does not have the .json suffix."""
        bad_path = tmp_path / "qc_data.txt"
        bad_path.touch()
        opts = type("Opts", (), {"command": "html", "qc_json": str(bad_path)})()
        assert _run_report_command(opts) == 2

    @patch("pytractoviz._internal.report_cli.generate_subject_qc_html")
    def test_html_success(self, mock_generate: MagicMock, tmp_path: Path) -> None:
        """Test that the HTML report is generated correctly."""
        qc_json = tmp_path / "qc_data.json"
        qc_json.write_text("{}")
        opts = type("Opts", (), {"command": "html", "qc_json": str(qc_json)})()
        assert _run_report_command(opts) == 0

    @patch("pytractoviz._internal.report_cli.generate_group_interactive_html")
    @patch("pytractoviz._internal.report_cli.parse_group_spec")
    def test_group_interactive(self, mock_parse: MagicMock, mock_generate: MagicMock, tmp_path: Path) -> None:
        """Test that the group interactive report is generated correctly."""
        mock_parse.return_value = ("UNC", tmp_path)
        mock_generate.return_value = str(tmp_path / "group_interactive.html")
        opts = type(
            "Opts",
            (),
            {"command": "group-interactive", "groups": [str(tmp_path)], "output": None, "refresh_csv": False},
        )()
        assert _run_report_command(opts) == 0
        mock_generate.assert_called_once()

    def test_run_help_available(self, capsys: pytest.CaptureFixture) -> None:
        """Test that the help message is displayed when -h is passed to the run command."""
        with pytest.raises(SystemExit):
            main(["run", "-h"])
        captured = capsys.readouterr()
        assert "--tractography-root" in captured.out
