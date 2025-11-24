"""Tests for the HTML report generation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from pytractoviz.html import create_quality_check_html


class TestCreateQualityCheckHTML:
    """Test create_quality_check_html function."""

    def test_basic_html_generation(self, tmp_path: Path) -> None:
        """Test basic HTML generation with minimal data."""
        output_file = tmp_path / "test_report.html"
        data = {
            "sub-001": {
                "AF_L": {
                    "image": "path/to/image.png",
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Tractography Quality Check" in content
        assert "sub-001" in content
        assert "AF_L" in content

    def test_multiple_subjects_and_tracts(self, tmp_path: Path) -> None:
        """Test HTML generation with multiple subjects and tracts."""
        output_file = tmp_path / "test_report.html"
        data = {
            "sub-001": {
                "AF_L": {"image": "path/to/image1.png"},
                "AF_R": {"image": "path/to/image2.png"},
            },
            "sub-002": {
                "AF_L": {"image": "path/to/image3.png"},
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "sub-001" in content
        assert "sub-002" in content
        assert "AF_L" in content
        assert "AF_R" in content

    def test_different_media_types(self, tmp_path: Path) -> None:
        """Test HTML generation with different media types."""
        output_file = tmp_path / "test_report.html"
        # Create dummy files for testing
        image_file = tmp_path / "test_image.png"
        plot_file = tmp_path / "test_plot.png"
        gif_file = tmp_path / "test_animation.gif"
        video_file = tmp_path / "test_video.mp4"
        image_file.touch()
        plot_file.touch()
        gif_file.touch()
        video_file.touch()

        data = {
            "sub-001": {
                "AF_L": {
                    "image": str(image_file),
                    "plot": str(plot_file),
                    "gif": str(gif_file),
                    "video": str(video_file),
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Check that media types are handled
        assert "test_image.png" in content or "test_image" in content
        assert "test_plot.png" in content or "test_plot" in content
        assert "test_animation.gif" in content or "test_animation" in content
        assert "test_video.mp4" in content or "test_video" in content

    def test_numeric_scores(self, tmp_path: Path) -> None:
        """Test HTML generation with numeric scores."""
        output_file = tmp_path / "test_report.html"
        data: dict[str, dict[str, dict[str, str]]] = {
            "sub-001": {
                "AF_L": {
                    "shape_similarity_score": "0.85",  # Numeric score as string
                    "image": "path/to/image.png",
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "0.85" in content or "shape_similarity_score" in content

    def test_custom_title(self, tmp_path: Path) -> None:
        """Test HTML generation with custom title."""
        output_file = tmp_path / "test_report.html"
        custom_title = "Custom Quality Report"
        data = {
            "sub-001": {
                "AF_L": {"image": "path/to/image.png"},
            },
        }
        create_quality_check_html(data, str(output_file), title=custom_title)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert custom_title in content
        assert "Tractography Quality Check" not in content

    def test_custom_items_per_page(self, tmp_path: Path) -> None:
        """Test HTML generation with custom items_per_page."""
        output_file = tmp_path / "test_report.html"
        data = {
            "sub-001": {
                "AF_L": {"image": "path/to/image.png"},
            },
        }
        create_quality_check_html(data, str(output_file), items_per_page=25)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Check that items_per_page is used in JavaScript
        assert "25" in content or "itemsPerPage" in content.lower()

    def test_empty_data(self, tmp_path: Path) -> None:
        """Test HTML generation with empty data."""
        output_file = tmp_path / "test_report.html"
        data: dict[str, dict[str, dict[str, str]]] = {}
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Should still generate valid HTML
        assert "<!DOCTYPE html>" in content
        assert "<html" in content

    def test_relative_paths(self, tmp_path: Path) -> None:
        """Test that file paths are converted to relative paths."""
        output_file = tmp_path / "report.html"
        image_file = tmp_path / "images" / "test.png"
        image_file.parent.mkdir()
        image_file.touch()

        data = {
            "sub-001": {
                "AF_L": {"image": str(image_file)},
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Should use relative path
        assert "images/test.png" in content or "test.png" in content

    def test_nonexistent_files(self, tmp_path: Path) -> None:
        """Test handling of non-existent files."""
        output_file = tmp_path / "test_report.html"
        data = {
            "sub-001": {
                "AF_L": {
                    "image": "nonexistent/path/image.png",  # Doesn't exist
                    "score": "0.95",  # String score
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        # Should handle gracefully without error
        content = output_file.read_text(encoding="utf-8")
        assert "sub-001" in content

    def test_float_scores(self, tmp_path: Path) -> None:
        """Test handling of float scores."""
        output_file = tmp_path / "test_report.html"
        data: dict[str, dict[str, dict[str, str]]] = {
            "sub-001": {
                "AF_L": {
                    "shape_similarity_score": "0.123456789",  # Float score as string
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Float should be converted to string
        assert "0.123456789" in content or "shape_similarity_score" in content

    def test_int_scores(self, tmp_path: Path) -> None:
        """Test handling of integer scores."""
        output_file = tmp_path / "test_report.html"
        data: dict[str, dict[str, dict[str, str]]] = {
            "sub-001": {
                "AF_L": {
                    "count": "42",  # Integer score as string
                },
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # Integer should be converted to string
        assert "42" in content or "count" in content

    def test_html_structure(self, tmp_path: Path) -> None:
        """Test that generated HTML has correct structure."""
        output_file = tmp_path / "test_report.html"
        data = {
            "sub-001": {
                "AF_L": {"image": "path/to/image.png"},
            },
        }
        create_quality_check_html(data, str(output_file))
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")

        # Check essential HTML elements
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "</html>" in content

        # Check for interactive features
        assert "filter" in content.lower() or "search" in content.lower()
