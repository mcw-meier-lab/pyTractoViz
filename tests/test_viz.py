"""Tests for the visualization module."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import nibabel as nib
    import numpy as np
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.tracking.streamline import Streamlines
except ImportError:
    pytest.skip("Required dependencies (numpy, nibabel, dipy) not available", allow_module_level=True)

from pytractoviz.viz import (
    InvalidInputError,
    TractographyVisualizationError,
    TractographyVisualizer,
)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_t1w_file(tmp_dir: Path) -> Path:
    """Create a mock T1-weighted image file."""
    t1w_file = tmp_dir / "t1w.nii.gz"
    t1w_file.touch()
    return t1w_file


@pytest.fixture
def mock_tract_file(tmp_dir: Path) -> Path:
    """Create a mock tractography file."""
    tract_file = tmp_dir / "tract.trk"
    tract_file.touch()
    return tract_file


@pytest.fixture
def mock_nibabel_image() -> Mock:
    """Create a mock nibabel image object."""
    mock_img = Mock(spec=nib.Nifti1Image)
    mock_img.get_fdata.return_value = np.random.rand(100, 100, 100)
    mock_img.affine = np.eye(4)
    # Add header that supports subscripting for StatefulTractogram
    # The header needs to support header["dim"][1:4] access
    # Use MagicMock to support subscripting
    mock_header = MagicMock(spec=nib.Nifti1Header)
    mock_header.get_best_affine.return_value = np.eye(4)
    # Create dim array that will be returned when accessing header["dim"]
    dim_array = np.array([3, 100, 100, 100, 1, 1, 1, 1], dtype=np.int16)
    # Configure MagicMock to return dim_array when subscripted with "dim"
    mock_header.__getitem__.return_value = dim_array
    mock_img.header = mock_header
    return mock_img


@pytest.fixture
def mock_stateful_tractogram() -> Mock:
    """Create a mock StatefulTractogram."""
    mock_tract = Mock(spec=StatefulTractogram)
    # Create some mock streamlines
    streamlines = Streamlines(
        [
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]),
        ],
    )
    mock_tract.streamlines = streamlines
    return mock_tract


@pytest.fixture
def visualizer(mock_t1w_file: Path, tmp_dir: Path) -> TractographyVisualizer:
    """Create a TractographyVisualizer instance for testing."""
    return TractographyVisualizer(
        reference_image=mock_t1w_file,
        output_directory=tmp_dir,
    )


class TestExceptions:
    """Test exception classes."""

    def test_tractography_visualization_error(self) -> None:
        """Test TractographyVisualizationError can be raised."""
        with pytest.raises(TractographyVisualizationError, match="test error"):
            raise TractographyVisualizationError("test error")

    def test_file_not_found_error(self) -> None:
        """Test FileNotFoundError can be raised."""
        # FileNotFoundError is a built-in exception, not a subclass of our custom exception
        with pytest.raises(FileNotFoundError, match="file not found"):
            raise FileNotFoundError("file not found")

    def test_invalid_input_error(self) -> None:
        """Test InvalidInputError is a subclass of TractographyVisualizationError."""
        with pytest.raises(InvalidInputError, match="invalid input"):
            raise InvalidInputError("invalid input")
        assert issubclass(InvalidInputError, TractographyVisualizationError)


class TestTractographyVisualizerInit:
    """Test TractographyVisualizer initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        viz = TractographyVisualizer()
        assert viz._reference_image is None
        assert viz._output_directory is None
        assert viz.gif_size == (608, 608)
        assert viz.gif_duration == 0.2
        assert viz.gif_palette_size == 64
        assert viz.gif_frames == 60
        assert viz.min_streamline_length == 40.0
        assert viz.cci_threshold == 1.0
        assert viz.afq_resample_points == 100

    def test_init_with_reference_image(self, mock_t1w_file: Path) -> None:
        """Test initialization with reference image."""
        viz = TractographyVisualizer(reference_image=mock_t1w_file)
        assert viz._reference_image == mock_t1w_file

    def test_init_with_output_directory(self, tmp_dir: Path) -> None:
        """Test initialization with output directory."""
        viz = TractographyVisualizer(output_directory=tmp_dir)
        assert viz._output_directory == tmp_dir

    def test_init_with_custom_parameters(self, mock_t1w_file: Path, tmp_dir: Path) -> None:
        """Test initialization with custom parameters."""
        viz = TractographyVisualizer(
            reference_image=mock_t1w_file,
            output_directory=tmp_dir,
            gif_size=(1024, 1024),
            gif_duration=0.3,
            gif_palette_size=128,
            gif_frames=120,
            min_streamline_length=50.0,
            cci_threshold=1.5,
            afq_resample_points=200,
        )
        assert viz.gif_size == (1024, 1024)
        assert viz.gif_duration == 0.3
        assert viz.gif_palette_size == 128
        assert viz.gif_frames == 120
        assert viz.min_streamline_length == 50.0
        assert viz.cci_threshold == 1.5
        assert viz.afq_resample_points == 200


class TestTractographyVisualizerBasicMethods:
    """Test basic methods of TractographyVisualizer."""

    def test_set_reference_image(self, visualizer: TractographyVisualizer, mock_t1w_file: Path) -> None:
        """Test setting reference image."""
        new_t1w = mock_t1w_file.parent / "new_t1w.nii.gz"
        new_t1w.touch()
        visualizer.set_reference_image(new_t1w)
        assert visualizer._reference_image == new_t1w

    def test_set_reference_image_not_found(self, visualizer: TractographyVisualizer) -> None:
        """Test setting reference image with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Reference image not found"):
            visualizer.set_reference_image("nonexistent.nii.gz")

    def test_set_output_directory(self, visualizer: TractographyVisualizer, tmp_dir: Path) -> None:
        """Test setting output directory."""
        new_dir = tmp_dir / "new_output"
        visualizer.set_output_directory(new_dir)
        assert visualizer._output_directory == new_dir
        assert new_dir.exists()

    def test_reference_image_property(self, visualizer: TractographyVisualizer, mock_t1w_file: Path) -> None:
        """Test reference_image property."""
        assert visualizer.reference_image == mock_t1w_file

    def test_output_directory_property(self, visualizer: TractographyVisualizer, tmp_dir: Path) -> None:
        """Test output_directory property."""
        assert visualizer.output_directory == tmp_dir

    def test_tract_name_from_path_stem(self) -> None:
        """Test that tract name is derived from Path.stem for output filenames."""
        assert Path("path/to/tract.trk").stem == "tract"
        assert Path("other/subject_AF_L.trk").stem == "subject_AF_L"


class TestGetGlassBrain:
    """Test get_glass_brain method."""

    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.actor.contour_from_roi")
    def test_get_glass_brain_success(
        self,
        mock_contour: Mock,
        mock_load: Mock,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful glass brain creation."""
        mock_load.return_value = mock_nibabel_image
        mock_actor = Mock()
        mock_contour.return_value = mock_actor

        result = visualizer.get_glass_brain(mock_t1w_file)
        assert result == mock_actor
        mock_load.assert_called_once()
        mock_contour.assert_called_once()

    def test_get_glass_brain_no_reference(self) -> None:
        """Test get_glass_brain without reference image."""
        viz = TractographyVisualizer()
        with pytest.raises(InvalidInputError, match="No reference image provided"):
            viz.get_glass_brain()

    @patch("pytractoviz.viz.nib.load")
    def test_get_glass_brain_load_error(
        self,
        mock_load: Mock,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
    ) -> None:
        """Test get_glass_brain with load error."""
        # Use OSError which is caught by the exception handler
        mock_load.side_effect = OSError("Load error")
        with pytest.raises(TractographyVisualizationError, match="Failed to load glass brain"):
            visualizer.get_glass_brain(mock_t1w_file)


class TestCreateScene:
    """Test _create_scene method."""

    @patch("pytractoviz.viz.window.Scene")
    @patch.object(TractographyVisualizer, "get_glass_brain")
    def test_create_scene_with_glass_brain(
        self,
        mock_get_glass_brain: Mock,
        mock_scene_class: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test scene creation with glass brain."""
        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene
        mock_brain_actor = Mock()
        mock_get_glass_brain.return_value = mock_brain_actor

        scene, brain_actor = visualizer._create_scene(show_glass_brain=True)
        assert scene == mock_scene
        assert brain_actor == mock_brain_actor
        mock_scene.SetBackground.assert_called_once_with(1, 1, 1)
        mock_scene.add.assert_called_once_with(mock_brain_actor)

    @patch("pytractoviz.viz.window.Scene")
    def test_create_scene_without_glass_brain(
        self,
        mock_scene_class: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test scene creation without glass brain."""
        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene

        scene, brain_actor = visualizer._create_scene(show_glass_brain=False)
        assert scene == mock_scene
        assert brain_actor is None

    @patch.object(TractographyVisualizer, "get_glass_brain")
    @patch("pytractoviz.viz.window.Scene")
    def test_create_scene_with_custom_ref_img(
        self,
        mock_scene_class: Mock,
        mock_get_glass_brain: Mock,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
    ) -> None:
        """Test scene creation with custom reference image."""
        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene
        mock_brain_actor = Mock()
        mock_get_glass_brain.return_value = mock_brain_actor

        scene, brain_actor = visualizer._create_scene(ref_img=mock_t1w_file, show_glass_brain=True)
        assert scene == mock_scene
        assert brain_actor == mock_brain_actor
        mock_get_glass_brain.assert_called_once_with(mock_t1w_file)


class TestCreateStreamlineActor:
    """Test _create_streamline_actor method."""

    @patch("pytractoviz.viz.actor.line")
    def test_create_streamline_actor_no_colors(
        self,
        mock_line: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test creating streamline actor without colors."""
        streamlines = Streamlines([np.array([[0, 0, 0], [1, 1, 1]])])
        mock_actor = Mock()
        mock_line.return_value = mock_actor

        result = visualizer._create_streamline_actor(streamlines)
        assert result == mock_actor
        mock_line.assert_called_once_with(streamlines)

    @patch("pytractoviz.viz.actor.line")
    def test_create_streamline_actor_with_matching_colors(
        self,
        mock_line: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test creating streamline actor with matching colors."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [1, 1, 1]]),
                np.array([[0, 0, 0], [2, 2, 2]]),
            ],
        )
        colors = np.array([[1, 0, 0], [0, 1, 0]])
        mock_actor = Mock()
        mock_line.return_value = mock_actor

        result = visualizer._create_streamline_actor(streamlines, colors)
        assert result == mock_actor
        mock_line.assert_called_once_with(streamlines, colors=colors)

    @patch("pytractoviz.viz.actor.line")
    def test_create_streamline_actor_with_fewer_colors(
        self,
        mock_line: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test creating streamline actor with fewer colors than streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [1, 1, 1]]),
                np.array([[0, 0, 0], [2, 2, 2]]),
                np.array([[0, 0, 0], [3, 3, 3]]),
            ],
        )
        colors = np.array([[1, 0, 0], [0, 1, 0]])
        mock_actor = Mock()
        mock_line.return_value = mock_actor

        result = visualizer._create_streamline_actor(streamlines, colors)
        assert result == mock_actor
        # Should have extended colors to match streamlines
        call_args = mock_line.call_args
        assert call_args is not None
        assert len(call_args.kwargs["colors"]) == 3

    @patch("pytractoviz.viz.actor.line")
    def test_create_streamline_actor_with_more_colors(
        self,
        mock_line: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test creating streamline actor with more colors than streamlines."""
        streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [1, 1, 1]]),
                np.array([[0, 0, 0], [2, 2, 2]]),
            ],
        )
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        mock_actor = Mock()
        mock_line.return_value = mock_actor

        result = visualizer._create_streamline_actor(streamlines, colors)
        assert result == mock_actor
        # Should truncate colors to match streamlines
        call_args = mock_line.call_args
        assert call_args is not None
        assert len(call_args.kwargs["colors"]) == 2


class TestSetAnatomicalCamera:
    """Test _set_anatomical_camera method."""

    @patch("pytractoviz.viz.set_anatomical_camera")
    def test_set_anatomical_camera_success(
        self,
        mock_set_camera: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test setting anatomical camera successfully."""
        mock_scene = Mock()
        centroid = np.array([0, 0, 0])
        visualizer._set_anatomical_camera(mock_scene, centroid, "coronal")
        mock_set_camera.assert_called_once()

    @patch("pytractoviz.viz.set_anatomical_camera")
    def test_set_anatomical_camera_with_custom_distance(
        self,
        mock_set_camera: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test setting anatomical camera with custom camera distance."""
        mock_scene = Mock()
        centroid = np.array([0, 0, 0])
        visualizer._set_anatomical_camera(mock_scene, centroid, "axial", camera_distance=100.0)
        mock_set_camera.assert_called_once()
        # Check that camera_distance was passed
        call_kwargs = mock_set_camera.call_args[1]
        assert call_kwargs["camera_distance"] == 100.0

    @patch("pytractoviz.viz.set_anatomical_camera")
    def test_set_anatomical_camera_with_bbox_size(
        self,
        mock_set_camera: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test setting anatomical camera with bbox_size."""
        mock_scene = Mock()
        centroid = np.array([0, 0, 0])
        bbox_size = np.array([100, 100, 100])
        visualizer._set_anatomical_camera(mock_scene, centroid, "sagittal", bbox_size=bbox_size)
        mock_set_camera.assert_called_once()
        # Check that bbox_size was passed
        call_kwargs = mock_set_camera.call_args[1]
        assert np.array_equal(call_kwargs["bbox_size"], bbox_size)

    @patch("pytractoviz.viz.set_anatomical_camera")
    def test_set_anatomical_camera_invalid_view(
        self,
        mock_set_camera: Mock,
        visualizer: TractographyVisualizer,
    ) -> None:
        """Test setting anatomical camera with invalid view."""
        mock_scene = Mock()
        centroid = np.array([0, 0, 0])
        mock_set_camera.side_effect = ValueError("Invalid view")
        with pytest.raises(InvalidInputError):
            visualizer._set_anatomical_camera(mock_scene, centroid, "invalid_view")


class TestLoadTract:
    """Test load_tract method."""

    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    @patch("pytractoviz.viz.actor.line")
    def test_load_tract_success(
        self,
        mock_actor_line: Mock,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful tract loading."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_actor = Mock()
        mock_actor_line.return_value = mock_actor

        result = visualizer.load_tract(mock_tract_file, mock_t1w_file)
        assert result == mock_actor
        mock_load_trk.assert_called_once()
        mock_tract.to_rasmm.assert_called_once()
        mock_nib_load.assert_called_once()
        mock_transform.assert_called_once()
        mock_actor_line.assert_called_once()

    def test_load_tract_no_reference(self, visualizer: TractographyVisualizer, mock_tract_file: Path) -> None:
        """Test load_tract without reference image."""
        # Remove reference image if set
        visualizer._reference_image = None
        with pytest.raises(InvalidInputError, match="No reference image provided"):
            visualizer.load_tract(mock_tract_file)

    @patch("pytractoviz.viz.load_trk")
    def test_load_tract_file_not_found(
        self,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
    ) -> None:
        """Test load_tract with non-existent file."""
        mock_load_trk.side_effect = FileNotFoundError("Tractography file not found")
        with pytest.raises(TractographyVisualizationError, match="Failed to load tract"):
            visualizer.load_tract("nonexistent.trk", mock_t1w_file)

    @patch("pytractoviz.viz.load_trk")
    def test_load_tract_load_error(
        self,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test load_tract with load error."""
        # Use OSError which is caught by the exception handler
        mock_load_trk.side_effect = OSError("Load error")
        with pytest.raises(TractographyVisualizationError, match="Failed to load tract"):
            visualizer.load_tract(mock_tract_file, mock_t1w_file)


class TestCalcCCI:
    """Test calc_cci method."""

    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.cluster_confidence")
    @patch("pytractoviz.viz.length")
    @patch("pytractoviz.viz.load_trk")
    def test_calc_cci_success(
        self,
        mock_load_trk: Mock,
        mock_length: Mock,
        mock_cluster_confidence: Mock,
        mock_nib_load: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful CCI calculation."""
        long_streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [50, 0, 0]]),
                np.array([[0, 0, 0], [0, 50, 0]]),
            ],
        )
        mock_stateful_tractogram.streamlines = long_streamlines
        mock_stateful_tractogram.to_rasmm = Mock()
        mock_load_trk.return_value = mock_stateful_tractogram
        mock_length.return_value = [50.0, 50.0]
        cci_array = np.array([1.5, 2.0], dtype=np.float64)
        mock_cluster_confidence.side_effect = lambda x: cci_array
        mock_nib_load.return_value = mock_nibabel_image

        cci, keep_cci, keep_tract, long_sl = visualizer.calc_cci(mock_tract_file)
        assert isinstance(cci, np.ndarray)
        assert len(cci) == 2
        assert isinstance(keep_cci, np.ndarray)
        assert all(c >= visualizer.cci_threshold for c in keep_cci)
        assert isinstance(keep_tract, StatefulTractogram) or hasattr(keep_tract, "streamlines")
        assert len(long_sl) == 2
        mock_load_trk.assert_called_once()
        mock_length.assert_called_once()
        mock_cluster_confidence.assert_called_once()

    @patch("pytractoviz.viz.load_trk")
    def test_calc_cci_empty_tractogram(
        self,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test calc_cci with empty tractogram (no long streamlines)."""
        mock_tract = Mock(spec=StatefulTractogram)
        mock_tract.streamlines = Streamlines()
        mock_tract.to_rasmm = Mock()
        mock_load_trk.return_value = mock_tract
        with pytest.raises(InvalidInputError, match="No streamlines longer than"):
            visualizer.calc_cci(mock_tract_file, ref_img=mock_t1w_file)

    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.length")
    @patch("pytractoviz.viz.load_trk")
    def test_calc_cci_no_long_streamlines(
        self,
        mock_load_trk: Mock,
        mock_length: Mock,
        _mock_nib_load: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
    ) -> None:
        """Test calc_cci with no streamlines longer than threshold."""
        short_streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [10, 0, 0]]),
                np.array([[0, 0, 0], [20, 0, 0]]),
            ],
        )
        mock_stateful_tractogram.streamlines = short_streamlines
        mock_stateful_tractogram.to_rasmm = Mock()
        mock_load_trk.return_value = mock_stateful_tractogram
        mock_length.return_value = [10.0, 20.0]
        with pytest.raises(InvalidInputError, match="No streamlines longer than"):
            visualizer.calc_cci(mock_tract_file)

    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.cluster_confidence")
    @patch("pytractoviz.viz.length")
    @patch("pytractoviz.viz.load_trk")
    def test_calc_cci_filters_by_threshold(
        self,
        mock_load_trk: Mock,
        mock_length: Mock,
        mock_cluster_confidence: Mock,
        mock_nib_load: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test calc_cci filters streamlines by CCI threshold."""
        long_streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [50, 0, 0]]),
                np.array([[0, 0, 0], [0, 50, 0]]),
                np.array([[0, 0, 0], [0, 0, 50]]),
            ],
        )
        mock_stateful_tractogram.streamlines = long_streamlines
        mock_stateful_tractogram.to_rasmm = Mock()
        mock_load_trk.return_value = mock_stateful_tractogram
        mock_length.return_value = [50.0, 50.0, 50.0]
        cci_array = np.array([1.5, 2.0, 0.5], dtype=np.float64)
        mock_cluster_confidence.side_effect = lambda x: cci_array
        mock_nib_load.return_value = mock_nibabel_image

        cci, keep_cci, _kt, _long_sl = visualizer.calc_cci(mock_tract_file)
        assert isinstance(cci, np.ndarray)
        assert len(cci) == 3
        assert isinstance(keep_cci, np.ndarray)
        assert len(keep_cci) == 2
        assert all(c >= visualizer.cci_threshold for c in keep_cci)


class TestWeightedAFQ:
    """Test weighted_afq method."""

    @patch("pytractoviz.viz.afq_profile")
    @patch("pytractoviz.viz.gaussian_weights")
    @patch("pytractoviz.viz.orient_by_streamline")
    @patch("pytractoviz.viz.QuickBundles")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_weighted_afq_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_qb_class: Mock,
        mock_orient: Mock,
        mock_weights: Mock,
        mock_afq_profile: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful weighted AFQ calculation."""
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_atlas_tract, mock_tract]
        mock_nib_load.return_value = mock_nibabel_image

        mock_cluster = Mock()
        mock_centroid = np.array([[0, 0, 0], [1, 1, 1]])
        mock_cluster.centroids = [mock_centroid]
        mock_qb = Mock()
        mock_qb.cluster.return_value = mock_cluster
        mock_qb_class.return_value = mock_qb

        mock_orient.return_value = mock_stateful_tractogram.streamlines
        mock_weights.return_value = np.array([0.5, 0.5, 0.5])
        mock_afq_profile.return_value = np.array([1.0, 2.0, 3.0])

        result = visualizer.weighted_afq(mock_tract_file, mock_t1w_file, mock_t1w_file)
        assert isinstance(result, np.ndarray)
        mock_load_trk.assert_called()
        mock_afq_profile.assert_called_once()

    def test_weighted_afq_file_not_found(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test weighted_afq with non-existent file."""
        # FileNotFoundError gets wrapped in TractographyVisualizationError
        with pytest.raises(TractographyVisualizationError, match="Failed to calculate weighted AFQ profile"):
            visualizer.weighted_afq(mock_tract_file, "nonexistent.trk", mock_t1w_file)

    @patch("pytractoviz.viz.QuickBundles")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_weighted_afq_no_centroids(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_qb_class: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test weighted_afq with no centroids."""
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_atlas_tract, mock_tract]
        mock_nib_load.return_value = mock_nibabel_image

        mock_cluster = Mock()
        mock_cluster.centroids = []
        mock_qb = Mock()
        mock_qb.cluster.return_value = mock_cluster
        mock_qb_class.return_value = mock_qb

        with pytest.raises(InvalidInputError, match="No centroids found"):
            visualizer.weighted_afq(mock_tract_file, mock_t1w_file, mock_t1w_file)


class TestGenerateAnatomicalViews:
    """Test generate_anatomical_views method."""

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_anatomical_views_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful generation of anatomical views."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines

        views = visualizer.generate_anatomical_views(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(views, dict)
        assert "coronal" in views
        assert "axial" in views
        assert "sagittal" in views
        assert all(isinstance(path, Path) for path in views.values())

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_anatomical_views_specific_views(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test generating specific views only."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines

        views = visualizer.generate_anatomical_views(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
            views=["coronal", "axial"],
        )
        assert len(views) == 2
        assert "coronal" in views
        assert "axial" in views
        assert "sagittal" not in views

    def test_generate_anatomical_views_invalid_view(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test generating views with invalid view name."""
        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.generate_anatomical_views(
                mock_tract_file,
                ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )

    def test_generate_anatomical_views_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test generating views without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.generate_anatomical_views(mock_tract_file, ref_img=mock_t1w_file)


class TestGenerateAtlasViews:
    """Test generate_atlas_views method."""

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_atlas_views_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful generation of atlas views."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines

        views = visualizer.generate_atlas_views(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(views, dict)
        assert "coronal" in views
        assert "axial" in views
        assert "sagittal" in views

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_atlas_views_with_flip(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test generating atlas views with left-right flip."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines

        views = visualizer.generate_atlas_views(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
            flip_lr=True,
        )
        assert isinstance(views, dict)
        assert len(views) == 3

    def test_generate_atlas_views_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test generating atlas views without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.generate_atlas_views(mock_tract_file, ref_img=mock_t1w_file)

    def test_generate_atlas_views_invalid_view(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test generating atlas views with invalid view name."""
        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.generate_atlas_views(
                mock_tract_file,
                ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_atlas_views_with_custom_name(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,  # Keep fixture name
        mock_nibabel_image: Mock,
    ) -> None:
        """Test generating atlas views with custom atlas name."""
        mock_tract = Mock()
        # Access streamlines from the fixture's return value
        mock_stateful = mock_stateful_tractogram
        mock_tract.streamlines = mock_stateful.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful.streamlines

        views = visualizer.generate_atlas_views(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
            atlas_name="custom_atlas",
        )
        assert isinstance(views, dict)
        # Check that output files use custom name
        assert all("custom_atlas_atlas" in str(path) for path in views.values())


class TestPlotAFQ:
    """Test plot_afq method."""

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.window.Scene")
    @patch("pytractoviz.viz.actor.line")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "weighted_afq")
    @patch("pytractoviz.viz.plt.subplots")
    @patch("pytractoviz.viz.plt.close")
    def test_plot_afq_success(
        self,
        _mock_plt_close: Mock,
        mock_subplots: Mock,
        mock_weighted_afq: Mock,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_actor_line: Mock,
        mock_scene_class: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful AFQ plotting."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_weighted_afq.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = visualizer.plot_afq(
            mock_t1w_file,
            "FA",
            mock_tract_file,
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert "profile_plot" in result

    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "weighted_afq")
    def test_plot_afq_no_output_dir(
        self,
        mock_weighted_afq: Mock,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
    ) -> None:
        """Test plot_afq without output directory."""
        visualizer._output_directory = None
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_weighted_afq.return_value = np.array([1.0, 2.0, 3.0])

        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.plot_afq(
                mock_t1w_file,
                "FA",
                mock_tract_file,
                mock_tract_file,
                ref_img=mock_t1w_file,
            )

    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "weighted_afq")
    def test_plot_afq_invalid_view(
        self,
        mock_weighted_afq: Mock,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
    ) -> None:
        """Test plot_afq with invalid view name."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_weighted_afq.return_value = np.array([1.0, 2.0, 3.0])

        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.plot_afq(
                mock_t1w_file,
                "FA",
                mock_tract_file,
                mock_tract_file,
                ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )


class TestGenerateGIF:
    """Test generate_gif method."""

    @patch("pytractoviz.viz.imageio.mimsave")
    @patch("pytractoviz.viz.window.snapshot")
    @patch("pytractoviz.viz.actor.line")
    @patch("pytractoviz.viz.window.Scene")
    @patch.object(TractographyVisualizer, "get_glass_brain")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_generate_gif_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_get_glass_brain: Mock,
        mock_scene_class: Mock,
        _mock_actor_line: Mock,
        mock_snapshot: Mock,
        mock_mimsave: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful GIF generation."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines

        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene
        mock_brain_actor = Mock()
        mock_get_glass_brain.return_value = mock_brain_actor
        mock_snapshot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        gif_path = visualizer.generate_gif(
            "test_tract",
            mock_tract_file,
            ref_file=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(gif_path, Path)
        assert gif_path.name == "test_tract.gif"
        mock_mimsave.assert_called_once()

    def test_generate_gif_no_reference(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test generate_gif without reference image."""
        visualizer._reference_image = None
        with pytest.raises(InvalidInputError, match="No reference image provided"):
            visualizer.generate_gif("test", mock_tract_file, output_dir=tmp_dir)

    def test_generate_gif_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test generate_gif without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.generate_gif("test", mock_tract_file, ref_file=mock_t1w_file)


class TestConvertGifToMp4:
    """Test convert_gif_to_mp4 method."""

    @pytest.mark.xfail(reason="Complex mocking of imageio reader iterator")
    @patch("pytractoviz.viz.imageio.get_reader")
    @patch("pytractoviz.viz.imageio.get_writer")
    def test_convert_gif_to_mp4_success(
        self,
        mock_get_writer: Mock,
        mock_get_reader: Mock,
        visualizer: TractographyVisualizer,
        tmp_dir: Path,
    ) -> None:
        """Test successful GIF to MP4 conversion."""
        gif_file = tmp_dir / "test.gif"
        gif_file.touch()
        mp4_file = tmp_dir / "test.mp4"

        # Create a proper iterable mock reader
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_reader = Mock()

        # Make it iterable by setting __iter__ as a method
        def iter_mock(_self: Mock) -> Iterator[Any]:
            return iter([frame])

        mock_reader.__iter__ = iter_mock
        mock_reader.close = Mock()
        mock_reader_context = Mock()
        mock_reader_context.__enter__ = Mock(return_value=mock_reader)
        mock_reader_context.__exit__ = Mock(return_value=None)
        mock_get_reader.return_value = mock_reader_context

        mock_writer = Mock()
        mock_writer.append_data = Mock()
        mock_writer.close = Mock()
        mock_writer_context = Mock()
        mock_writer_context.__enter__ = Mock(return_value=mock_writer)
        mock_writer_context.__exit__ = Mock(return_value=None)
        mock_get_writer.return_value = mock_writer_context

        result = visualizer.convert_gif_to_mp4(gif_file, mp4_path=mp4_file)
        assert result == mp4_file

    @pytest.mark.xfail(reason="Complex mocking of imageio reader iterator")
    @patch("pytractoviz.viz.imageio.get_reader")
    @patch("pytractoviz.viz.imageio.get_writer")
    def test_convert_gif_to_mp4_auto_path(
        self,
        mock_get_writer: Mock,
        mock_get_reader: Mock,
        visualizer: TractographyVisualizer,
        tmp_dir: Path,
    ) -> None:
        """Test GIF to MP4 conversion with auto-generated path."""
        gif_file = tmp_dir / "test.gif"
        gif_file.touch()

        # Create a proper iterable mock reader
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_reader = Mock()

        # Make it iterable by setting __iter__ as a method
        def iter_mock(_self: Mock) -> Iterator[Any]:
            return iter([frame])

        mock_reader.__iter__ = iter_mock
        mock_reader.close = Mock()
        mock_reader_context = Mock()
        mock_reader_context.__enter__ = Mock(return_value=mock_reader)
        mock_reader_context.__exit__ = Mock(return_value=None)
        mock_get_reader.return_value = mock_reader_context

        mock_writer = Mock()
        mock_writer.append_data = Mock()
        mock_writer.close = Mock()
        mock_writer_context = Mock()
        mock_writer_context.__enter__ = Mock(return_value=mock_writer)
        mock_writer_context.__exit__ = Mock(return_value=None)
        mock_get_writer.return_value = mock_writer_context

        result = visualizer.convert_gif_to_mp4(gif_file)
        assert result.suffix == ".mp4"
        assert result.stem == "test"


class TestCalculateShapeSimilarity:
    """Test calculate_shape_similarity method."""

    @patch("pytractoviz.viz.bundle_shape_similarity")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_calculate_shape_similarity_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        _mock_similarity: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,  # Keep fixture name, but unused in test
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful shape similarity calculation."""
        _ = mock_t1w_file  # Mark as intentionally unused
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image

        similarity = visualizer.calculate_shape_similarity(
            mock_tract_file,
            mock_tract_file,  # Using same file as atlas for simplicity
        )
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    @patch("pytractoviz.viz.bundle_shape_similarity")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_calculate_shape_similarity_empty_subject_tract(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        _mock_similarity: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test calculate_shape_similarity with empty subject tract."""
        mock_tract = Mock()
        mock_tract.streamlines = Streamlines([])  # Empty streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image

        with pytest.raises(InvalidInputError, match="Subject tract is empty"):
            visualizer.calculate_shape_similarity(mock_tract_file, mock_tract_file)

    @patch("pytractoviz.viz.bundle_shape_similarity")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_calculate_shape_similarity_empty_atlas_tract(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        _mock_similarity: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test calculate_shape_similarity with empty atlas tract."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = Streamlines([])  # Empty streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image

        with pytest.raises(InvalidInputError, match="Atlas tract is empty"):
            visualizer.calculate_shape_similarity(mock_tract_file, mock_tract_file)

    @patch("pytractoviz.viz.bundle_shape_similarity")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_calculate_shape_similarity_with_atlas_ref_img(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_similarity: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test calculate_shape_similarity with atlas reference image."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_similarity.return_value = 0.85

        similarity = visualizer.calculate_shape_similarity(
            mock_tract_file,
            mock_tract_file,
            atlas_ref_img=mock_t1w_file,
        )
        assert isinstance(similarity, float)
        mock_transform.assert_called_once()

    @patch("pytractoviz.viz.bundle_shape_similarity")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_calculate_shape_similarity_with_flip_lr(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_similarity: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test calculate_shape_similarity with left-right flip."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image
        mock_similarity.return_value = 0.85

        similarity = visualizer.calculate_shape_similarity(
            mock_tract_file,
            mock_tract_file,
            flip_lr=True,
        )
        assert isinstance(similarity, float)
        # Should not call transform when no atlas_ref_img
        mock_transform.assert_not_called()


class TestVisualizeShapeSimilarity:
    """Test visualize_shape_similarity method."""

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.window.Scene")
    @patch("pytractoviz.viz.actor.line")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "calculate_shape_similarity")
    def test_visualize_shape_similarity_success(
        self,
        mock_calc_similarity: Mock,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_actor_line: Mock,
        mock_scene_class: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful shape similarity visualization."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_calc_similarity.return_value = 0.85

        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene

        result = visualizer.visualize_shape_similarity(
            mock_tract_file,
            mock_tract_file,
            atlas_ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert "coronal" in result or "axial" in result or "sagittal" in result

    def test_visualize_shape_similarity_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test visualize_shape_similarity without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.visualize_shape_similarity(
                mock_tract_file,
                mock_tract_file,
                atlas_ref_img=mock_t1w_file,
            )

    def test_visualize_shape_similarity_invalid_view(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test visualize_shape_similarity with invalid view name."""
        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.visualize_shape_similarity(
                mock_tract_file,
                mock_tract_file,
                atlas_ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )


class TestCompareBeforeAfterCCI:
    """Test compare_before_after_cci method (histogram + before/after CCI views)."""

    @patch("pytractoviz.viz.plt.close")
    @patch("pytractoviz.viz.plt.subplots")
    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.window.Scene")
    @patch("pytractoviz.viz.actor.line")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch.object(TractographyVisualizer, "calc_cci")
    def test_compare_before_after_cci_success(
        self,
        mock_calc_cci: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        _mock_actor_line: Mock,
        mock_scene_class: Mock,
        _mock_record: Mock,
        mock_subplots: Mock,
        _mock_plt_close: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful before/after CCI comparison (histogram + CCI-colored views)."""
        long_streamlines = Streamlines(
            [
                np.array([[0, 0, 0], [50, 0, 0]]),
                np.array([[0, 0, 0], [0, 50, 0]]),
            ],
        )
        cci_full = np.array([1.5, 2.0], dtype=np.float64)
        keep_cci = np.array([2.0], dtype=np.float64)
        mock_filtered_tract = Mock(spec=StatefulTractogram)
        mock_filtered_tract.streamlines = Streamlines([long_streamlines[1]])
        mock_calc_cci.return_value = (cci_full, keep_cci, mock_filtered_tract, long_streamlines)

        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = long_streamlines
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene

        result = visualizer.compare_before_after_cci(
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert "histogram" in result
        assert result["histogram"] == tmp_dir / "tract_cci_histogram.png"
        assert "coronal" in result or "axial" in result or "sagittal" in result

        any_view_key = "coronal" if "coronal" in result else next(iter(k for k in result if k != "histogram"))
        any_view = result[any_view_key]
        assert isinstance(any_view, dict)
        assert "before" in any_view
        assert "after" in any_view

    def test_compare_before_after_cci_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test compare_before_after_cci without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.compare_before_after_cci(mock_tract_file, ref_img=mock_t1w_file)

    def test_compare_before_after_cci_invalid_view(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test compare_before_after_cci with invalid view name."""
        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.compare_before_after_cci(
                mock_tract_file,
                ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )


class TestVisualizeBundleAssignment:
    """Test visualize_bundle_assignment method."""

    @patch("pytractoviz.viz.window.record")
    @patch("pytractoviz.viz.window.Scene")
    @patch("pytractoviz.viz.actor.line")
    @patch("pytractoviz.viz.assignment_map")
    @patch.object(TractographyVisualizer, "_downsample_streamlines")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_visualize_bundle_assignment_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_downsample_streamlines: Mock,
        mock_assignment_map: Mock,
        _mock_actor_line: Mock,
        mock_scene_class: Mock,
        _mock_record: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful bundle assignment visualization."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_atlas_tract = Mock()
        mock_atlas_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.side_effect = [mock_tract, mock_atlas_tract]
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        # Return streamlines unchanged so point count matches assignment count
        mock_downsample_streamlines.side_effect = lambda sl: sl
        # assignment_map returns one assignment per point
        # mock_stateful_tractogram has 3 streamlines with 3 points each = 9 total points
        mock_assignment_map.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

        mock_scene = Mock()
        mock_scene_class.return_value = mock_scene

        result = visualizer.visualize_bundle_assignment(
            mock_tract_file,
            mock_tract_file,
            ref_img=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert "coronal" in result or "axial" in result or "sagittal" in result

    def test_visualize_bundle_assignment_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test visualize_bundle_assignment without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.visualize_bundle_assignment(
                mock_tract_file,
                mock_tract_file,
                ref_img=mock_t1w_file,
            )

    def test_visualize_bundle_assignment_invalid_view(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test visualize_bundle_assignment with invalid view name."""
        with pytest.raises(InvalidInputError, match="Invalid view names"):
            visualizer.visualize_bundle_assignment(
                mock_tract_file,
                mock_tract_file,
                ref_img=mock_t1w_file,
                output_dir=tmp_dir,
                views=["invalid_view"],
            )


class TestGenerateVideos:
    """Test generate_videos method."""

    @pytest.mark.xfail(reason="Complex mocking of generate_gif and convert_gif_to_mp4 chain")
    @patch.object(TractographyVisualizer, "convert_gif_to_mp4")
    @patch.object(TractographyVisualizer, "generate_gif")
    def test_generate_videos_success(
        self,
        mock_generate_gif: Mock,
        mock_convert_to_mp4: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test successful video generation."""
        # Use the actual filename stem from mock_tract_file
        tract_stem = mock_tract_file.stem
        gif_path = tmp_dir / f"{tract_stem}.gif"
        mp4_path = tmp_dir / f"{tract_stem}.mp4"
        mock_generate_gif.return_value = gif_path
        mock_convert_to_mp4.return_value = mp4_path

        result = visualizer.generate_videos(
            tract_files=[mock_tract_file],
            ref_file=mock_t1w_file,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert len(result) == 1
        # The key is the stem of the tract file
        assert tract_stem in result
        mock_generate_gif.assert_called_once()
        mock_convert_to_mp4.assert_called_once()

    def test_generate_videos_no_reference(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test generate_videos without reference image."""
        visualizer._reference_image = None
        with pytest.raises(InvalidInputError, match="No reference image provided"):
            visualizer.generate_videos([mock_tract_file], output_dir=tmp_dir)

    def test_generate_videos_empty_list(
        self,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
        tmp_dir: Path,
    ) -> None:
        """Test generate_videos with empty tract files list."""
        with pytest.raises(InvalidInputError, match="No tract files provided"):
            visualizer.generate_videos([], ref_file=mock_t1w_file, output_dir=tmp_dir)

    def test_generate_videos_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test generate_videos without output directory."""
        visualizer._output_directory = None
        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.generate_videos([mock_tract_file], ref_file=mock_t1w_file)


class TestRunQualityCheckWorkflow:
    """Test run_quality_check_workflow method."""

    @patch("pytractoviz.viz.create_quality_check_html")
    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "visualize_bundle_assignment")
    @patch.object(TractographyVisualizer, "visualize_shape_similarity")
    @patch.object(TractographyVisualizer, "plot_afq")
    @patch.object(TractographyVisualizer, "compare_before_after_cci")
    @patch.object(TractographyVisualizer, "calc_cci")
    @patch.object(TractographyVisualizer, "generate_atlas_views")
    @patch.object(TractographyVisualizer, "generate_anatomical_views")
    def test_run_quality_check_workflow_basic(
        self,
        mock_anatomical: Mock,
        mock_atlas: Mock,
        mock_calc_cci: Mock,
        mock_compare_cci: Mock,
        mock_plot_afq: Mock,
        mock_shape_sim: Mock,
        mock_bundle: Mock,
        mock_load_trk: Mock,
        _mock_html: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,  # Keep fixture name, but unused in test
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
    ) -> None:
        """Test basic quality check workflow."""
        _ = mock_t1w_file  # Mark as intentionally unused
        subjects_original: dict[str, dict[str, str | Path]] = {
            "sub-001": {"AF_L": mock_tract_file},
        }

        # Mock load_trk for initial metrics loading
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract

        # Mock return values (calc_cci returns 4 values; compare_before_after_cci returns histogram + views)
        mock_calc_cci.return_value = (
            np.array([1.5, 2.0]),
            np.array([2.0]),
            Mock(),
            Mock(spec=Streamlines),
        )
        mock_compare_cci.return_value = {
            "histogram": tmp_dir / "AF_L_cci_histogram.png",
            "coronal": {
                "before": tmp_dir / "cci_before_AF_L_coronal.png",
                "after": tmp_dir / "cci_after_AF_L_coronal.png",
            },
            "axial": {"before": tmp_dir / "cci_before_AF_L_axial.png", "after": tmp_dir / "cci_after_AF_L_axial.png"},
            "sagittal": {
                "before": tmp_dir / "cci_before_AF_L_sagittal.png",
                "after": tmp_dir / "cci_after_AF_L_sagittal.png",
            },
        }
        mock_anatomical.return_value = {"coronal": tmp_dir / "coronal.png"}
        mock_atlas.return_value = {"coronal": tmp_dir / "atlas_coronal.png"}
        mock_plot_afq.return_value = tmp_dir / "afq.png"
        mock_shape_sim.return_value = {"similarity_score": 0.85, "image": tmp_dir / "shape.png"}
        mock_bundle.return_value = {"image": tmp_dir / "bundle.png"}

        result = visualizer.run_quality_check_workflow(
            subjects_original_space=subjects_original,
            output_dir=tmp_dir,
        )
        assert isinstance(result, dict)
        assert "sub-001" in result
        mock_anatomical.assert_called()

    def test_run_quality_check_workflow_no_output_dir(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
    ) -> None:
        """Test workflow without output directory."""
        visualizer._output_directory = None
        subjects_original: dict[str, dict[str, str | Path]] = {"sub-001": {"AF_L": mock_tract_file}}

        with pytest.raises(InvalidInputError, match="No output directory provided"):
            visualizer.run_quality_check_workflow(
                subjects_original_space=subjects_original,
            )

    @patch("pytractoviz.viz.load_trk")
    @patch.object(TractographyVisualizer, "generate_anatomical_views")
    def test_run_quality_check_workflow_skip_checks(
        self,
        mock_anatomical: Mock,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        tmp_dir: Path,
        mock_stateful_tractogram: Mock,
    ) -> None:
        """Test workflow with skipped checks."""
        subjects_original: dict[str, dict[str, str | Path]] = {"sub-001": {"AF_L": mock_tract_file}}
        mock_anatomical.return_value = {"coronal": tmp_dir / "coronal.png"}

        # Mock load_trk to avoid HeaderError
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract

        result = visualizer.run_quality_check_workflow(
            subjects_original_space=subjects_original,
            output_dir=tmp_dir,
            skip_checks=[
                "cci",
                "before_after_cci",
                "afq_profile",
                "shape_similarity",
                "bundle_assignment",
                "atlas_comparison",
            ],
        )
        assert isinstance(result, dict)
        # Should still generate anatomical views
        mock_anatomical.assert_called()


class TestViewTractInteractive:
    """Test view_tract_interactive method."""

    @patch("pytractoviz.viz.window.show")
    @patch("pytractoviz.viz.calculate_bbox_size")
    @patch("pytractoviz.viz.calculate_centroid")
    @patch("pytractoviz.viz.calculate_direction_colors")
    @patch.object(TractographyVisualizer, "_set_anatomical_camera")
    @patch.object(TractographyVisualizer, "_create_streamline_actor")
    @patch.object(TractographyVisualizer, "_downsample_streamlines")
    @patch.object(TractographyVisualizer, "_create_scene")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_view_tract_interactive_success(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_create_scene: Mock,
        mock_downsample_streamlines: Mock,
        mock_create_actor: Mock,
        mock_set_camera: Mock,
        mock_calc_colors: Mock,
        mock_calc_centroid: Mock,
        mock_calc_bbox: Mock,
        mock_window_show: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test successful interactive viewing."""
        # Setup mocks
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_downsample_streamlines.return_value = mock_stateful_tractogram.streamlines
        mock_calc_colors.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_calc_centroid.return_value = np.array([0, 0, 0])
        mock_calc_bbox.return_value = np.array([100, 100, 100])
        mock_actor = Mock()
        mock_create_actor.return_value = mock_actor
        mock_scene = Mock()
        mock_create_scene.return_value = (mock_scene, None)

        # Call method
        visualizer.view_tract_interactive(mock_tract_file, ref_img=mock_t1w_file)

        # Assertions
        mock_load_trk.assert_called_once()
        mock_nib_load.assert_called_once()
        mock_transform.assert_called_once()
        mock_downsample_streamlines.assert_called_once()
        mock_create_scene.assert_called_once()
        mock_create_actor.assert_called_once()
        mock_scene.add.assert_called_once_with(mock_actor)
        mock_set_camera.assert_called_once()
        mock_window_show.assert_called_once()
        # Verify window.show was called with scene
        assert mock_window_show.call_args[0][0] == mock_scene

    def test_view_tract_interactive_no_reference(
        self,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
    ) -> None:
        """Test view_tract_interactive without reference image."""
        visualizer._reference_image = None
        with pytest.raises(InvalidInputError, match="No reference image provided"):
            visualizer.view_tract_interactive(mock_tract_file)

    def test_view_tract_interactive_tract_file_not_found(
        self,
        visualizer: TractographyVisualizer,
        mock_t1w_file: Path,
    ) -> None:
        """Test view_tract_interactive with non-existent tract file."""
        with pytest.raises(FileNotFoundError, match="Tract file not found"):
            visualizer.view_tract_interactive("nonexistent.trk", ref_img=mock_t1w_file)

    @patch("pytractoviz.viz.load_trk")
    def test_view_tract_interactive_load_error(
        self,
        mock_load_trk: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
    ) -> None:
        """Test view_tract_interactive with load error."""
        mock_load_trk.side_effect = OSError("Load error")
        with pytest.raises(TractographyVisualizationError, match="Failed to display tract"):
            visualizer.view_tract_interactive(mock_tract_file, ref_img=mock_t1w_file)

    @patch("pytractoviz.viz.window.show")
    @patch("pytractoviz.viz.calculate_bbox_size")
    @patch("pytractoviz.viz.calculate_centroid")
    @patch("pytractoviz.viz.calculate_direction_colors")
    @patch.object(TractographyVisualizer, "_set_anatomical_camera")
    @patch.object(TractographyVisualizer, "_create_streamline_actor")
    @patch.object(TractographyVisualizer, "_downsample_streamlines")
    @patch.object(TractographyVisualizer, "_create_scene")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_view_tract_interactive_without_glass_brain(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_create_scene: Mock,
        mock_downsample_streamlines: Mock,
        mock_create_actor: Mock,
        mock_set_camera: Mock,
        mock_calc_colors: Mock,
        mock_calc_centroid: Mock,
        mock_calc_bbox: Mock,
        mock_window_show: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test view_tract_interactive with show_glass_brain=False."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_downsample_streamlines.return_value = mock_stateful_tractogram.streamlines
        mock_calc_colors.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_calc_centroid.return_value = np.array([0, 0, 0])
        mock_calc_bbox.return_value = np.array([100, 100, 100])
        mock_actor = Mock()
        mock_create_actor.return_value = mock_actor
        mock_scene = Mock()
        mock_create_scene.return_value = (mock_scene, None)

        visualizer.view_tract_interactive(
            mock_tract_file,
            ref_img=mock_t1w_file,
            show_glass_brain=False,
        )

        # Verify show_glass_brain=False was passed
        mock_create_scene.assert_called_once()
        call_kwargs = mock_create_scene.call_args[1]
        assert call_kwargs["show_glass_brain"] is False

    @patch("pytractoviz.viz.window.show")
    @patch("pytractoviz.viz.calculate_bbox_size")
    @patch("pytractoviz.viz.calculate_centroid")
    @patch("pytractoviz.viz.calculate_direction_colors")
    @patch.object(TractographyVisualizer, "_set_anatomical_camera")
    @patch.object(TractographyVisualizer, "_create_streamline_actor")
    @patch.object(TractographyVisualizer, "_downsample_streamlines")
    @patch.object(TractographyVisualizer, "_create_scene")
    @patch("pytractoviz.viz.transform_streamlines")
    @patch("pytractoviz.viz.nib.load")
    @patch("pytractoviz.viz.load_trk")
    def test_view_tract_interactive_with_custom_window_size(
        self,
        mock_load_trk: Mock,
        mock_nib_load: Mock,
        mock_transform: Mock,
        mock_create_scene: Mock,
        mock_downsample_streamlines: Mock,
        mock_create_actor: Mock,
        mock_set_camera: Mock,
        mock_calc_colors: Mock,
        mock_calc_centroid: Mock,
        mock_calc_bbox: Mock,
        mock_window_show: Mock,
        visualizer: TractographyVisualizer,
        mock_tract_file: Path,
        mock_t1w_file: Path,
        mock_stateful_tractogram: Mock,
        mock_nibabel_image: Mock,
    ) -> None:
        """Test view_tract_interactive with custom window size."""
        mock_tract = Mock()
        mock_tract.streamlines = mock_stateful_tractogram.streamlines
        mock_load_trk.return_value = mock_tract
        mock_nib_load.return_value = mock_nibabel_image
        mock_transform.return_value = mock_stateful_tractogram.streamlines
        mock_downsample_streamlines.return_value = mock_stateful_tractogram.streamlines
        mock_calc_colors.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_calc_centroid.return_value = np.array([0, 0, 0])
        mock_calc_bbox.return_value = np.array([100, 100, 100])
        mock_actor = Mock()
        mock_create_actor.return_value = mock_actor
        mock_scene = Mock()
        mock_create_scene.return_value = (mock_scene, None)

        visualizer.view_tract_interactive(
            mock_tract_file,
            ref_img=mock_t1w_file,
            window_size=(1200, 1200),
        )

        # Verify window size was passed
        mock_window_show.assert_called_once()
        call_kwargs = mock_window_show.call_args[1]
        assert call_kwargs["size"] == (1200, 1200)
