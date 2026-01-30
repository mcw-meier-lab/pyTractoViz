"""Utility functions for tractography visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from dipy.tracking.streamline import Streamlines
    from fury import window

# Valid anatomical view names (used for iteration and validation; camera is set in set_anatomical_camera)
ANATOMICAL_VIEW_NAMES: tuple[str, ...] = ("coronal", "axial", "sagittal")


def calculate_centroid(streamlines: Streamlines) -> np.ndarray:
    """Calculate the centroid of streamlines.

    Parameters
    ----------
    streamlines : Streamlines
        The streamlines to calculate centroid for.

    Returns
    -------
    np.ndarray
        The centroid coordinates (3D).

    Raises
    ------
    ValueError
        If streamlines is empty (raises numpy's concatenation error).
    """
    # Let numpy raise the original error if empty - this matches test expectations
    all_points = np.vstack([np.array(sl) for sl in streamlines])
    return np.mean(all_points, axis=0)


def calculate_bbox_size(streamlines: Streamlines) -> np.ndarray:
    """Calculate bounding box size of streamlines.

    Parameters
    ----------
    streamlines : Streamlines
        The streamlines to calculate bbox for.

    Returns
    -------
    np.ndarray
        The bounding box size (3D).

    Raises
    ------
    ValueError
        If streamlines is empty (raises numpy's concatenation error).
    """
    # Let numpy raise the original error if empty - this matches test expectations
    all_points = np.vstack([np.array(sl) for sl in streamlines])
    return np.max(all_points, axis=0) - np.min(all_points, axis=0)


def calculate_direction_colors(streamlines: Streamlines) -> np.ndarray:
    """Calculate colors for streamlines based on their diffusion direction.

    Standard RGB mapping:
    - Red = X-axis (left/right)
    - Green = Y-axis (anterior/posterior)
    - Blue = Z-axis (superior/inferior)

    Parameters
    ----------
    streamlines : Streamlines
        The streamlines to calculate colors for.

    Returns
    -------
    np.ndarray
        Array of RGB colors, one per streamline (N x 3).
        Returns empty array (0, 3) if streamlines is empty.
    """
    # Return empty array if streamlines is empty (matches test expectations)
    if not streamlines or len(streamlines) == 0:
        return np.array([]).reshape(0, 3)
    streamline_colors = []
    max_range = 1e-10
    sl_len = 2
    for sl in streamlines:
        sl_array = np.array(sl)
        if len(sl_array) < sl_len:
            # Degenerate streamline, use default color
            streamline_colors.append([0.5, 0.5, 0.5])
            continue

        # Calculate direction vector (from start to end)
        direction = sl_array[-1] - sl_array[0]
        direction_norm = np.linalg.norm(direction)

        if direction_norm < max_range:
            # Degenerate direction, use default color
            streamline_colors.append([0.5, 0.5, 0.5])
            continue

        # Normalize direction to unit vector
        direction = direction / direction_norm

        # Map direction components to RGB using absolute values
        # X -> Red (left/right), Y -> Green (anterior/posterior), Z -> Blue (superior/inferior)
        r = abs(direction[0])  # X component -> Red
        g = abs(direction[1])  # Y component -> Green
        b = abs(direction[2])  # Z component -> Blue

        # Normalize by the maximum component to ensure colors are in 0-1 range
        # This preserves the relative direction while ensuring valid RGB values
        max_component = max(r, g, b)
        if max_component > max_range:
            r = r / max_component
            g = g / max_component
            b = b / max_component
        else:
            # Fallback for edge case
            r, g, b = 0.5, 0.5, 0.5

        streamline_colors.append([r, g, b])

    return np.array(streamline_colors)


def calculate_combined_centroid(*streamlines_groups: Streamlines) -> np.ndarray:
    """Calculate the centroid of multiple groups of streamlines combined.

    Parameters
    ----------
    *streamlines_groups : Streamlines
        One or more Streamlines objects to combine.

    Returns
    -------
    np.ndarray
        The combined centroid coordinates (3D).

    Raises
    ------
    ValueError
        If any streamlines group is empty (raises numpy's concatenation error).
    """
    # Let numpy raise the original error if any group is empty - this matches test expectations
    all_points_list = []
    for streamlines in streamlines_groups:
        all_points_list.append(np.vstack([np.array(sl) for sl in streamlines]))
    all_points = np.vstack(all_points_list)
    return np.mean(all_points, axis=0)


def calculate_combined_bbox_size(*streamlines_groups: Streamlines) -> np.ndarray:
    """Calculate bounding box size of multiple groups of streamlines combined.

    Parameters
    ----------
    *streamlines_groups : Streamlines
        One or more Streamlines objects to combine.

    Returns
    -------
    np.ndarray
        The combined bounding box size (3D).

    Raises
    ------
    ValueError
        If any streamlines group is empty (raises numpy's concatenation error).
    """
    # Let numpy raise the original error if any group is empty - this matches test expectations
    all_points_list = []
    for streamlines in streamlines_groups:
        all_points_list.append(np.vstack([np.array(sl) for sl in streamlines]))
    all_points = np.vstack(all_points_list)
    return np.max(all_points, axis=0) - np.min(all_points, axis=0)


def set_anatomical_camera(
    scene: window.Scene,
    centroid: np.ndarray,
    view_name: str,
    *,
    camera_distance: float | None = None,
    bbox_size: np.ndarray | None = None,
    template_space: Literal["ras", "mni"] = "ras",
) -> None:
    """Set camera position for standard anatomical views.

    This function positions the camera for coronal, axial, or sagittal views
    without rotating the streamlines, ensuring colors stay aligned.

    Parameters
    ----------
    scene : window.Scene
        The FURY scene to set the camera on.
    centroid : np.ndarray
        The centroid of the streamlines (3D coordinates).
    view_name : str
        Name of the view: "coronal", "axial", or "sagittal".
    camera_distance : float | None, optional
        Distance of camera from centroid. If None, calculated from bbox_size.
    bbox_size : np.ndarray | None, optional
        Bounding box size of streamlines. Used to calculate camera_distance if not provided.
    template_space : {"ras", "mni"}, optional
        Coordinate convention for the template. Use "ras" for standard RAS (e.g. native T1).
        Use "mni" for MNI-space templates, which uses opposite camera sides so left/right
        and superior/inferior match radiological convention. Default is "ras".

    Raises
    ------
    ValueError
        If view_name is not one of the standard anatomical views.
    """
    if view_name not in ANATOMICAL_VIEW_NAMES:
        raise ValueError(
            f"Invalid view name: {view_name}. Must be one of: {list(ANATOMICAL_VIEW_NAMES)}",
        )

    # Calculate camera distance if not provided
    if camera_distance is None:
        max_dim = np.max(bbox_size) if bbox_size is not None else 100.0
        camera_distance = max_dim * 2.5

    use_mni = template_space == "mni"

    # Define camera positions and view_up vectors for each anatomical view.
    # RAS: camera from posterior (-Y) coronal, from +Z axial, from -X sagittal.
    # MNI: flip camera side so anatomy matches radiological convention (anterior +Y, etc.).
    if view_name == "coronal":
        if use_mni:
            camera_position = centroid + np.array([0, camera_distance, 0])  # From anterior (+Y)
        else:
            camera_position = centroid + np.array([0, -camera_distance, 0])  # From posterior (-Y)
        view_up = np.array([0, 0, 1])  # Superior is up
    elif view_name == "axial":
        if use_mni:
            camera_position = centroid + np.array([0, 0, camera_distance])  # From inferior (+Z)
            view_up = np.array([0, 1, 0])  # Anterior as up
        else:
            camera_position = centroid + np.array([0, 0, camera_distance])  # From superior (+Z)
            view_up = np.array([0, -1, 0])  # Posterior as up (flips left/right)
    else:  # sagittal
        if use_mni:
            camera_position = centroid + np.array([camera_distance, 0, 0])  # From right (+X)
        else:
            camera_position = centroid + np.array([-camera_distance, 0, 0])  # From left (-X)
        view_up = np.array([0, 0, 1])  # Superior is up

    # Set camera
    scene.set_camera(
        position=camera_position,
        focal_point=centroid,
        view_up=view_up,
    )
