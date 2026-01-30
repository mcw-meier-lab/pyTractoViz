"""Tests for the utility functions module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

try:
    import numpy as np
    from dipy.tracking.streamline import Streamlines
    from fury import window
except ImportError:
    pytest.skip("Required dependencies (numpy, dipy, fury) not available", allow_module_level=True)

from pytractoviz.utils import (
    ANATOMICAL_VIEW_NAMES,
    calculate_bbox_size,
    calculate_centroid,
    calculate_combined_bbox_size,
    calculate_combined_centroid,
    calculate_direction_colors,
    set_anatomical_camera,
)


@pytest.fixture
def sample_streamlines() -> Streamlines:
    """Create sample streamlines for testing."""
    return Streamlines(
        [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]),
        ],
    )


@pytest.fixture
def single_streamline() -> Streamlines:
    """Create a single streamline for testing."""
    return Streamlines(
        [
            np.array([[0, 0, 0], [10, 10, 10]]),
        ],
    )


@pytest.fixture
def empty_streamlines() -> Streamlines:
    """Create empty streamlines for testing."""
    return Streamlines([])


class TestAnatomicalViewNames:
    """Test ANATOMICAL_VIEW_NAMES constant."""

    def test_anatomical_view_names_exists(self) -> None:
        """Test that ANATOMICAL_VIEW_NAMES is defined."""
        assert ANATOMICAL_VIEW_NAMES is not None
        assert isinstance(ANATOMICAL_VIEW_NAMES, tuple)

    def test_anatomical_view_names_keys(self) -> None:
        """Test that all expected view names exist."""
        expected = ("coronal", "axial", "sagittal")
        assert expected == ANATOMICAL_VIEW_NAMES

    def test_anatomical_view_names_membership(self) -> None:
        """Test that view names are strings and valid."""
        for name in ANATOMICAL_VIEW_NAMES:
            assert isinstance(name, str)
        assert "coronal" in ANATOMICAL_VIEW_NAMES
        assert "axial" in ANATOMICAL_VIEW_NAMES
        assert "sagittal" in ANATOMICAL_VIEW_NAMES


class TestCalculateCentroid:
    """Test calculate_centroid function."""

    def test_calculate_centroid_basic(self, sample_streamlines: Streamlines) -> None:
        """Test basic centroid calculation."""
        centroid = calculate_centroid(sample_streamlines)
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (3,)
        # Points: [0,0,0], [1,1,1], [2,2,2], [0,0,0], [1,0,0], [2,0,0], [0,0,0], [0,1,0], [0,2,0]
        # Mean: X=(0+1+2+0+1+2+0+0+0)/9=6/9=2/3, Y=(0+1+2+0+0+0+0+1+2)/9=6/9=2/3, Z=(0+1+2+0+0+0+0+0+0)/9=3/9=1/3
        assert np.allclose(centroid, np.array([2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]))

    def test_calculate_centroid_single_streamline(self, single_streamline: Streamlines) -> None:
        """Test centroid calculation with single streamline."""
        centroid = calculate_centroid(single_streamline)
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (3,)
        # Centroid should be the mean of all points
        expected = np.array([5.0, 5.0, 5.0])  # Mean of [0,0,0] and [10,10,10]
        assert np.allclose(centroid, expected)

    def test_calculate_centroid_offset(self) -> None:
        """Test centroid calculation with offset streamlines."""
        streamlines = Streamlines(
            [
                np.array([[10, 20, 30], [11, 21, 31]]),
                np.array([[10, 20, 30], [12, 22, 32]]),
            ],
        )
        centroid = calculate_centroid(streamlines)
        expected = np.array([10.75, 20.75, 30.75])
        assert np.allclose(centroid, expected)

    def test_calculate_centroid_empty(self, empty_streamlines: Streamlines) -> None:
        """Test centroid calculation with empty streamlines."""
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            calculate_centroid(empty_streamlines)


class TestCalculateBboxSize:
    """Test calculate_bbox_size function."""

    def test_calculate_bbox_size_basic(self, sample_streamlines: Streamlines) -> None:
        """Test basic bounding box size calculation."""
        bbox_size = calculate_bbox_size(sample_streamlines)
        assert isinstance(bbox_size, np.ndarray)
        assert bbox_size.shape == (3,)
        # Max points: [2, 2, 2], Min points: [0, 0, 0]
        expected = np.array([2.0, 2.0, 2.0])
        assert np.allclose(bbox_size, expected)

    def test_calculate_bbox_size_single_streamline(self, single_streamline: Streamlines) -> None:
        """Test bbox calculation with single streamline."""
        bbox_size = calculate_bbox_size(single_streamline)
        assert isinstance(bbox_size, np.ndarray)
        assert bbox_size.shape == (3,)
        expected = np.array([10.0, 10.0, 10.0])
        assert np.allclose(bbox_size, expected)

    def test_calculate_bbox_size_rectangular(self) -> None:
        """Test bbox calculation with rectangular region."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [10, 0, 0]]),
                np.array([[0, 0, 0], [0, 20, 0]]),
                np.array([[0, 0, 0], [0, 0, 30]]),
            ],
        )
        bbox_size = calculate_bbox_size(streamlines)
        expected = np.array([10.0, 20.0, 30.0])
        assert np.allclose(bbox_size, expected)

    def test_calculate_bbox_size_empty(self, empty_streamlines: Streamlines) -> None:
        """Test bbox calculation with empty streamlines."""
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            calculate_bbox_size(empty_streamlines)


class TestCalculateDirectionColors:
    """Test calculate_direction_colors function."""

    def test_calculate_direction_colors_basic(self, sample_streamlines: Streamlines) -> None:
        """Test basic direction color calculation."""
        colors = calculate_direction_colors(sample_streamlines)
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (len(sample_streamlines), 3)
        # All colors should be in [0, 1] range
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

    def test_calculate_direction_colors_x_direction(self) -> None:
        """Test direction colors for X-axis streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [10, 0, 0]]),  # Pure X direction
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (1, 3)
        # Should be mostly red (X -> Red)
        assert colors[0, 0] > colors[0, 1]  # Red > Green
        assert colors[0, 0] > colors[0, 2]  # Red > Blue
        assert colors[0, 0] == 1.0  # Normalized to max

    def test_calculate_direction_colors_y_direction(self) -> None:
        """Test direction colors for Y-axis streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [0, 10, 0]]),  # Pure Y direction
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (1, 3)
        # Should be mostly green (Y -> Green)
        assert colors[0, 1] > colors[0, 0]  # Green > Red
        assert colors[0, 1] > colors[0, 2]  # Green > Blue
        assert colors[0, 1] == 1.0  # Normalized to max

    def test_calculate_direction_colors_z_direction(self) -> None:
        """Test direction colors for Z-axis streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [0, 0, 10]]),  # Pure Z direction
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (1, 3)
        # Should be mostly blue (Z -> Blue)
        assert colors[0, 2] > colors[0, 0]  # Blue > Red
        assert colors[0, 2] > colors[0, 1]  # Blue > Green
        assert colors[0, 2] == 1.0  # Normalized to max

    def test_calculate_direction_colors_degenerate_short(self) -> None:
        """Test direction colors with degenerate (too short) streamline."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0]]),  # Only one point
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (1, 3)
        # Should use default gray color
        assert np.allclose(colors[0], [0.5, 0.5, 0.5])

    def test_calculate_direction_colors_degenerate_zero_length(self) -> None:
        """Test direction colors with zero-length streamline."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [0, 0, 0]]),  # Same start and end
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (1, 3)
        # Should use default gray color
        assert np.allclose(colors[0], [0.5, 0.5, 0.5])

    def test_calculate_direction_colors_mixed_directions(self) -> None:
        """Test direction colors with mixed direction streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [1, 0, 0]]),  # X direction
                np.array([[0, 0, 0], [0, 1, 0]]),  # Y direction
                np.array([[0, 0, 0], [0, 0, 1]]),  # Z direction
                np.array([[0, 0, 0], [1, 1, 1]]),  # Diagonal
            ],
        )
        colors = calculate_direction_colors(streamlines)
        assert colors.shape == (4, 3)
        # All colors should be valid RGB values
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

    def test_calculate_direction_colors_empty(self, empty_streamlines: Streamlines) -> None:
        """Test direction colors with empty streamlines."""
        colors = calculate_direction_colors(empty_streamlines)
        assert isinstance(colors, np.ndarray)
        # Empty array can be shape (0,) or (0, 3) depending on numpy version
        assert colors.shape in {(0, 3), (0,)}


class TestCalculateCombinedCentroid:
    """Test calculate_combined_centroid function."""

    def test_calculate_combined_centroid_single_group(self, sample_streamlines: Streamlines) -> None:
        """Test combined centroid with single group."""
        centroid = calculate_combined_centroid(sample_streamlines)
        expected = calculate_centroid(sample_streamlines)
        assert np.allclose(centroid, expected)

    def test_calculate_combined_centroid_multiple_groups(self) -> None:
        """Test combined centroid with multiple groups."""
        group1 = Streamlines(
            [
                np.array([[0, 0, 0], [1, 1, 1]]),
            ],
        )
        group2 = Streamlines(
            [
                np.array([[10, 10, 10], [11, 11, 11]]),
            ],
        )
        centroid = calculate_combined_centroid(group1, group2)
        # Combined centroid should be mean of all points
        expected = np.array([5.5, 5.5, 5.5])
        assert np.allclose(centroid, expected)

    def test_calculate_combined_centroid_three_groups(self) -> None:
        """Test combined centroid with three groups."""
        group1 = Streamlines([np.array([[0, 0, 0], [2, 0, 0]])])
        group2 = Streamlines([np.array([[0, 0, 0], [0, 2, 0]])])
        group3 = Streamlines([np.array([[0, 0, 0], [0, 0, 2]])])
        centroid = calculate_combined_centroid(group1, group2, group3)
        # Mean of all points: [0,0,0], [2,0,0], [0,0,0], [0,2,0], [0,0,0], [0,0,2]
        # X=(0+2+0+0+0+0)/6=1/3, Y=(0+0+0+2+0+0)/6=1/3, Z=(0+0+0+0+0+2)/6=1/3
        expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        assert np.allclose(centroid, expected)

    def test_calculate_combined_centroid_empty_group(self) -> None:
        """Test combined centroid with empty group."""
        group1 = Streamlines([np.array([[0, 0, 0], [1, 1, 1]])])
        group2 = Streamlines([])
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            calculate_combined_centroid(group1, group2)


class TestCalculateCombinedBboxSize:
    """Test calculate_combined_bbox_size function."""

    def test_calculate_combined_bbox_size_single_group(self, sample_streamlines: Streamlines) -> None:
        """Test combined bbox with single group."""
        bbox_size = calculate_combined_bbox_size(sample_streamlines)
        expected = calculate_bbox_size(sample_streamlines)
        assert np.allclose(bbox_size, expected)

    def test_calculate_combined_bbox_size_multiple_groups(self) -> None:
        """Test combined bbox with multiple groups."""
        group1 = Streamlines(
            [
                np.array([[0, 0, 0], [5, 0, 0]]),
            ],
        )
        group2 = Streamlines(
            [
                np.array([[10, 0, 0], [15, 0, 0]]),
            ],
        )
        bbox_size = calculate_combined_bbox_size(group1, group2)
        # Combined bbox should span from 0 to 15 in X
        expected = np.array([15.0, 0.0, 0.0])
        assert np.allclose(bbox_size, expected)

    def test_calculate_combined_bbox_size_three_groups(self) -> None:
        """Test combined bbox with three groups."""
        group1 = Streamlines([np.array([[0, 0, 0], [10, 0, 0]])])
        group2 = Streamlines([np.array([[0, 0, 0], [0, 20, 0]])])
        group3 = Streamlines([np.array([[0, 0, 0], [0, 0, 30]])])
        bbox_size = calculate_combined_bbox_size(group1, group2, group3)
        expected = np.array([10.0, 20.0, 30.0])
        assert np.allclose(bbox_size, expected)

    def test_calculate_combined_bbox_size_empty_group(self) -> None:
        """Test combined bbox with empty group."""
        group1 = Streamlines([np.array([[0, 0, 0], [1, 1, 1]])])
        group2 = Streamlines([])
        with pytest.raises(ValueError, match="need at least one array to concatenate"):
            calculate_combined_bbox_size(group1, group2)


class TestSetAnatomicalCamera:
    """Test set_anatomical_camera function."""

    @pytest.fixture
    def mock_scene(self) -> Mock:
        """Create a mock FURY scene."""
        return Mock(spec=window.Scene)

    def test_set_anatomical_camera_coronal(self, mock_scene: Mock) -> None:
        """Test setting camera for coronal view."""
        centroid = np.array([0, 0, 0])
        bbox_size = np.array([100, 100, 100])
        set_anatomical_camera(mock_scene, centroid, "coronal", bbox_size=bbox_size)

        mock_scene.set_camera.assert_called_once()
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert "position" in call_kwargs
        assert "focal_point" in call_kwargs
        assert "view_up" in call_kwargs
        assert np.allclose(call_kwargs["focal_point"], centroid)
        # Camera should be offset in -Y direction
        assert call_kwargs["position"][1] < centroid[1]

    def test_set_anatomical_camera_axial(self, mock_scene: Mock) -> None:
        """Test setting camera for axial view."""
        centroid = np.array([0, 0, 0])
        bbox_size = np.array([100, 100, 100])
        set_anatomical_camera(mock_scene, centroid, "axial", bbox_size=bbox_size)

        mock_scene.set_camera.assert_called_once()
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert np.allclose(call_kwargs["focal_point"], centroid)
        # Camera should be offset in +Z direction
        assert call_kwargs["position"][2] > centroid[2]

    def test_set_anatomical_camera_sagittal(self, mock_scene: Mock) -> None:
        """Test setting camera for sagittal view."""
        centroid = np.array([0, 0, 0])
        bbox_size = np.array([100, 100, 100])
        set_anatomical_camera(mock_scene, centroid, "sagittal", bbox_size=bbox_size)

        mock_scene.set_camera.assert_called_once()
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert np.allclose(call_kwargs["focal_point"], centroid)
        # Camera should be offset in -X direction
        assert call_kwargs["position"][0] < centroid[0]

    def test_set_anatomical_camera_with_camera_distance(self, mock_scene: Mock) -> None:
        """Test setting camera with explicit camera distance."""
        centroid = np.array([0, 0, 0])
        camera_distance = 500.0
        set_anatomical_camera(mock_scene, centroid, "coronal", camera_distance=camera_distance)

        mock_scene.set_camera.assert_called_once()
        call_kwargs = mock_scene.set_camera.call_args[1]
        # Camera position should be at specified distance
        position = call_kwargs["position"]
        distance = np.linalg.norm(position - centroid)
        assert np.isclose(distance, camera_distance)

    def test_set_anatomical_camera_without_bbox(self, mock_scene: Mock) -> None:
        """Test setting camera without bbox_size (uses default)."""
        centroid = np.array([0, 0, 0])
        set_anatomical_camera(mock_scene, centroid, "coronal")

        mock_scene.set_camera.assert_called_once()
        call_kwargs = mock_scene.set_camera.call_args[1]
        # Should use default max_dim = 100.0, so distance = 100 * 2.5 = 250
        position = call_kwargs["position"]
        distance = np.linalg.norm(position - centroid)
        assert np.isclose(distance, 250.0)

    def test_set_anatomical_camera_invalid_view(self, mock_scene: Mock) -> None:
        """Test setting camera with invalid view name."""
        centroid = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="Invalid view name"):
            set_anatomical_camera(mock_scene, centroid, "invalid_view")

    def test_set_anatomical_camera_view_up_vectors(self, mock_scene: Mock) -> None:
        """Test that view_up vectors are set correctly for each view."""
        centroid = np.array([0, 0, 0])
        bbox_size = np.array([100, 100, 100])

        # Test coronal view_up
        set_anatomical_camera(mock_scene, centroid, "coronal", bbox_size=bbox_size)
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert np.allclose(call_kwargs["view_up"], np.array([0, 0, 1]))

        # Test axial view_up
        set_anatomical_camera(mock_scene, centroid, "axial", bbox_size=bbox_size)
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert np.allclose(call_kwargs["view_up"], np.array([0, -1, 0]))

        # Test sagittal view_up
        set_anatomical_camera(mock_scene, centroid, "sagittal", bbox_size=bbox_size)
        call_kwargs = mock_scene.set_camera.call_args[1]
        assert np.allclose(call_kwargs["view_up"], np.array([0, 0, 1]))

    def test_set_anatomical_camera_camera_position_directions(self, mock_scene: Mock) -> None:
        """Test that camera positions are in correct directions."""
        centroid = np.array([50, 50, 50])
        camera_distance = 200.0

        # Coronal: camera should be in -Y direction
        set_anatomical_camera(mock_scene, centroid, "coronal", camera_distance=camera_distance)
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert np.isclose(position[0], centroid[0])
        assert position[1] < centroid[1]  # -Y
        assert np.isclose(position[2], centroid[2])

        # Axial: camera should be in +Z direction
        set_anatomical_camera(mock_scene, centroid, "axial", camera_distance=camera_distance)
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert np.isclose(position[0], centroid[0])
        assert np.isclose(position[1], centroid[1])
        assert position[2] > centroid[2]  # +Z

        # Sagittal: camera should be in -X direction
        set_anatomical_camera(mock_scene, centroid, "sagittal", camera_distance=camera_distance)
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert position[0] < centroid[0]  # -X
        assert np.isclose(position[1], centroid[1])
        assert np.isclose(position[2], centroid[2])

    def test_set_anatomical_camera_mni_template_space(self, mock_scene: Mock) -> None:
        """Test that template_space='mni' flips camera positions for MNI convention."""
        centroid = np.array([0, 0, 0])
        camera_distance = 200.0

        # MNI coronal: camera from +Y (anterior)
        set_anatomical_camera(
            mock_scene,
            centroid,
            "coronal",
            camera_distance=camera_distance,
            template_space="mni",
        )
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert position[1] > centroid[1]  # +Y

        # MNI axial: camera from -Z (inferior)
        set_anatomical_camera(
            mock_scene,
            centroid,
            "axial",
            camera_distance=camera_distance,
            template_space="mni",
        )
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert position[2] > centroid[2]  # +Z

        # MNI sagittal: camera from +X (right)
        set_anatomical_camera(
            mock_scene,
            centroid,
            "sagittal",
            camera_distance=camera_distance,
            template_space="mni",
        )
        call_kwargs = mock_scene.set_camera.call_args[1]
        position = call_kwargs["position"]
        assert position[0] > centroid[0]  # +X
