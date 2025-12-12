"""Tests for the CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pytractoviz import main
from pytractoviz._internal import debug
from pytractoviz._internal.cli import _collect_tract_files
from pytractoviz.viz import TractographyVisualizationError


def test_main() -> None:
    """Basic CLI test."""
    assert main([]) == 0


def test_show_help(capsys: pytest.CaptureFixture) -> None:
    """Show help.

    Parameters
    ----------
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["-h"])
    captured = capsys.readouterr()
    assert "pytractoviz" in captured.out


def test_show_version(capsys: pytest.CaptureFixture) -> None:
    """Show version.

    Parameters
    ----------
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["-V"])
    captured = capsys.readouterr()
    assert debug._get_version() in captured.out


def test_show_debug_info(capsys: pytest.CaptureFixture) -> None:
    """Show debug information.

    Parameters
    ----------
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["--debug-info"])
    captured = capsys.readouterr().out.lower()
    assert "python" in captured
    assert "system" in captured
    assert "environment" in captured
    assert "packages" in captured


class TestCollectTractFiles:
    """Test _collect_tract_files helper function."""

    def test_collect_tract_files_from_directory(self, tmp_path: Path) -> None:
        """Test collecting .trk files from directory."""
        tract_dir = tmp_path / "tracts"
        tract_dir.mkdir()
        tract1 = tract_dir / "tract1.trk"
        tract2 = tract_dir / "tract2.trk"
        tract1.touch()
        tract2.touch()

        result = _collect_tract_files([str(tract_dir)])

        assert len(result) == 2
        assert tract1 in result
        assert tract2 in result
        assert result == sorted(result)  # Should be sorted

    def test_collect_tract_files_from_file(self, tmp_path: Path) -> None:
        """Test collecting single .trk file."""
        tract_file = tmp_path / "tract.trk"
        tract_file.touch()

        result = _collect_tract_files([str(tract_file)])

        assert len(result) == 1
        assert tract_file in result

    def test_collect_tract_files_mixed(self, tmp_path: Path) -> None:
        """Test collecting from mix of files and directories."""
        tract_dir = tmp_path / "tracts"
        tract_dir.mkdir()
        tract1 = tract_dir / "tract1.trk"
        tract2 = tmp_path / "tract2.trk"
        tract1.touch()
        tract2.touch()

        result = _collect_tract_files([str(tract_dir), str(tract2)])

        assert len(result) == 2
        assert tract1 in result
        assert tract2 in result

    def test_collect_tract_files_empty_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Test collecting from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = _collect_tract_files([str(empty_dir)])

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "No .trk files found" in captured.err

    def test_collect_tract_files_nonexistent_path(self, capsys: pytest.CaptureFixture) -> None:
        """Test collecting from non-existent path."""
        result = _collect_tract_files(["/nonexistent/path"])

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "does not exist" in captured.err

    def test_collect_tract_files_non_trk_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Test skipping non-.trk files."""
        other_file = tmp_path / "file.txt"
        other_file.touch()

        result = _collect_tract_files([str(other_file)])

        assert len(result) == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "non-.trk file" in captured.err

    def test_collect_tract_files_duplicates(self, tmp_path: Path) -> None:
        """Test removing duplicate files."""
        tract_file = tmp_path / "tract.trk"
        tract_file.touch()

        result = _collect_tract_files([str(tract_file), str(tract_file)])

        assert len(result) == 1
        assert tract_file in result


class TestQcInteractiveCommand:
    """Test qc-interactive CLI command."""

    @patch("pytractoviz._internal.cli.TractographyVisualizer")
    def test_qc_interactive_success_single_file(
        self,
        mock_visualizer_class: Mock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test qc-interactive with single file."""
        tract_file = tmp_path / "tract.trk"
        tract_file.touch()
        ref_file = tmp_path / "t1w.nii.gz"
        ref_file.touch()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        result = main(["qc-interactive", str(tract_file), "--ref", str(ref_file)])

        assert result == 0
        # CLI converts ref_img string to Path, so check it was called with Path
        call_kwargs = mock_visualizer_class.call_args[1]
        assert call_kwargs["reference_image"] == Path(ref_file)
        mock_visualizer.view_tract_interactive.assert_called_once()
        captured = capsys.readouterr()
        assert "Found 1 tract file(s)" in captured.out
        assert "Loading:" in captured.out

    @patch("pytractoviz._internal.cli.TractographyVisualizer")
    def test_qc_interactive_success_multiple_files(
        self,
        mock_visualizer_class: Mock,
        tmp_path: Path,
    ) -> None:
        """Test qc-interactive with multiple files."""
        tract1 = tmp_path / "tract1.trk"
        tract2 = tmp_path / "tract2.trk"
        ref_file = tmp_path / "t1w.nii.gz"
        tract1.touch()
        tract2.touch()
        ref_file.touch()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        result = main(["qc-interactive", str(tract1), str(tract2), "--ref", str(ref_file)])

        assert result == 0
        assert mock_visualizer.view_tract_interactive.call_count == 2

    @patch("pytractoviz._internal.cli.TractographyVisualizer")
    def test_qc_interactive_with_skip(
        self,
        mock_visualizer_class: Mock,
        tmp_path: Path,
    ) -> None:
        """Test qc-interactive with --skip option."""
        tract1 = tmp_path / "tract1.trk"
        tract2 = tmp_path / "tract2.trk"
        ref_file = tmp_path / "t1w.nii.gz"
        tract1.touch()
        tract2.touch()
        ref_file.touch()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        result = main(["qc-interactive", str(tract1), str(tract2), "--ref", str(ref_file), "--skip", "1"])

        assert result == 0
        assert mock_visualizer.view_tract_interactive.call_count == 1

    def test_qc_interactive_missing_ref(self, tmp_path: Path) -> None:
        """Test qc-interactive without --ref argument."""
        tract_file = tmp_path / "tract.trk"
        tract_file.touch()

        with pytest.raises(SystemExit):
            main(["qc-interactive", str(tract_file)])

    def test_qc_interactive_no_files_found(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Test qc-interactive when no .trk files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ref_file = tmp_path / "t1w.nii.gz"
        ref_file.touch()

        result = main(["qc-interactive", str(empty_dir), "--ref", str(ref_file)])

        assert result == 1
        captured = capsys.readouterr()
        assert "No .trk files found" in captured.err

    @patch("pytractoviz._internal.cli.TractographyVisualizer")
    @patch("builtins.input", return_value="y")
    def test_qc_interactive_error_continue(
        self,
        mock_input: Mock,
        mock_visualizer_class: Mock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test qc-interactive continues on error when user confirms."""
        tract1 = tmp_path / "tract1.trk"
        tract2 = tmp_path / "tract2.trk"
        ref_file = tmp_path / "t1w.nii.gz"
        tract1.touch()
        tract2.touch()
        ref_file.touch()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer
        # First call raises error, second succeeds
        mock_visualizer.view_tract_interactive.side_effect = [
            TractographyVisualizationError("Test error"),
            None,
        ]

        result = main(["qc-interactive", str(tract1), str(tract2), "--ref", str(ref_file)])

        assert result == 0
        assert mock_visualizer.view_tract_interactive.call_count == 2
        captured = capsys.readouterr()
        assert "Error viewing" in captured.err
        mock_input.assert_called_once()

    @patch("pytractoviz._internal.cli.TractographyVisualizer")
    def test_qc_interactive_keyboard_interrupt(
        self,
        mock_visualizer_class: Mock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test qc-interactive handles KeyboardInterrupt."""
        tract_file = tmp_path / "tract.trk"
        ref_file = tmp_path / "t1w.nii.gz"
        tract_file.touch()
        ref_file.touch()

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.view_tract_interactive.side_effect = KeyboardInterrupt()

        result = main(["qc-interactive", str(tract_file), "--ref", str(ref_file)])

        assert result == 130
        captured = capsys.readouterr()
        assert "Interrupted by user" in captured.out
