"""Automated tractography QC pipeline: views, CCI, AFQ profiles, and HTML reports."""

from __future__ import annotations

import csv
import gc
import json
import logging
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.typing as npt
from dipy.core.histeq import histeq
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.tracking.distances import approx_polygon_track
from dipy.tracking.streamline import (
    Streamlines,
    cluster_confidence,
    orient_by_streamline,
    set_number_of_points,
    transform_streamlines,
)
from dipy.tracking.utils import length
from fury import actor, window
from fury.colormap import create_colormap

from pytractoviz.qc_config import QCPathConfig, load_qc_path_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def _load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI volume and return a typed image handle."""
    return cast("nib.Nifti1Image", nib.load(str(path)))


def _optional_float(value: Any) -> float | None:
    """Parse a scalar for QC metadata, returning None when missing or invalid."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(parsed) else parsed


def _ref_cache(cache: RefCache | None) -> RefCache:
    return {} if cache is None else cache


def _streamline_cache(cache: StreamlineCache | None) -> StreamlineCache:
    return {} if cache is None else cache


def _t1_cache(cache: T1Cache | None) -> T1Cache:
    return {} if cache is None else cache


def _metric_cache(cache: MetricCache | None) -> MetricCache:
    return {} if cache is None else cache


def _atlas_standard_cache(cache: AtlasStandardCache | None) -> AtlasStandardCache:
    return {} if cache is None else cache


ViewName = Literal["coronal", "axial", "sagittal"]
TemplateSpace = Literal["mni", "ras"]
RenderBackend = Literal["osmesa", "egl"]

NDArrayFloat = npt.NDArray[np.floating]

RefCacheEntry = dict[str, Any]
T1CacheEntry = dict[str, Any]
RefCache = dict[str, RefCacheEntry]
T1Cache = dict[str, T1CacheEntry]
StreamlineCache = dict[str, Streamlines]
MetricCache = dict[str, tuple[nib.Nifti1Image, NDArrayFloat]]
AtlasStandardCache = dict[str, Any]

CCIStats = dict[str, int | float]
CciViewPair = dict[str, str]
CciViewDict = dict[str, Any]
ViewPathMap = dict[str, str]
AfqViewMap = dict[str, str]
ProfileRow = tuple[str, str, NDArrayFloat]
SpeedConfig = dict[str, int | None]
SubjectTractTask = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str,
    str,
    ViewPathMap,
    list[str],
    list[str],
    SpeedConfig,
]
TractResultExtra = dict[str, Any]
TractResult = list[ViewPathMap | CciViewDict | TractResultExtra]
SubjectResults = dict[str, TractResult | ViewPathMap | dict[str, Any]]
GroupQcData = dict[str, Any]
PairwiseShapeTask = tuple[str, list[str | None], int]

_MIN_STREAMLINES_FOR_GAUSSIAN_WEIGHTS = 2
_MIN_PAIRWISE_SHAPE_SUBJECTS = 2
_SUSPICIOUS_SHAPE_THRESHOLD = 0.2
_SUSPICIOUS_CCI_RETENTION_PCT = 20.0
_TRACT_RESULT_MIN_LEN = 3


def _configure_qc_logging() -> None:
    """Configure simple message-only logging for CLI and worker processes."""
    if logging.root.handlers:
        return
    level_name = os.environ.get("PYTRACTOVIZ_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


SUBJECT_TO_ATLAS: dict[str, str] = {
    "AFL": "AF_L",
    "AFR": "AF_R",
    "CCForcepsMajor": "CC_ForcepsMajor",
    "CCForcepsMinor": "CC_ForcepsMinor",
    "CCMid": "CCMid",
    "CL": "C_L",
    "CR": "C_R",
    "CSTL": "CST_L",
    "CSTR": "CST_R",
    "FL": "F_L",
    "FR": "F_R",
    "IFOFL": "IFOF_L",
    "IFOFR": "IFOF_R",
    "MCP": "MCP",
    "UFL": "UF_L",
    "UFR": "UF_R",
}

views: list[ViewName] = ["coronal", "axial", "sagittal"]


def _get_ref_data(ref_img: str | Path, cache: RefCache | None = None) -> RefCacheEntry:
    local_cache = _ref_cache(cache)
    key = str(ref_img)
    if key not in local_cache:
        img = _load_nifti(key)
        local_cache[key] = {
            "img": img,
            "affine": img.affine,
            "inv_affine": np.linalg.inv(img.affine),
            "mask_data": np.asarray(img.get_fdata(), dtype=float),
        }
    return local_cache[key]


def _get_t1_data(t1_img: str | Path, cache: T1Cache | None = None) -> T1CacheEntry:
    local_cache = _t1_cache(cache)
    key = str(t1_img)
    if key not in local_cache:
        img = _load_nifti(key)
        data = np.asarray(img.get_fdata(), dtype=float)
        local_cache[key] = {
            "img": img,
            "data": data,
            "shape": data.shape,
        }
    return local_cache[key]


def _resolve_t1_path(anat_dir: str | Path, subject: str) -> str | None:
    """Return the first existing ACPC T1 from common qsiprep naming patterns."""
    anat_dir = Path(anat_dir)
    candidates = [
        anat_dir / f"{subject}_space-ACPC_desc-preproc_T1w.nii.gz",
        anat_dir / f"{subject}_space-ACPC_T1w.nii.gz",
        anat_dir / f"{subject}_desc-preproc_T1w.nii.gz",
        anat_dir / f"{subject}_T1w.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _centroid_vox_indices(centroid: NDArrayFloat, shape: Sequence[int]) -> tuple[int, int, int]:
    """Clip centroid voxel indices to valid slice range."""
    shape_arr = np.asarray(shape, dtype=int)
    return (
        int(np.clip(centroid[0], 0, shape_arr[0] - 1)),
        int(np.clip(centroid[1], 0, shape_arr[1] - 1)),
        int(np.clip(centroid[2], 0, shape_arr[2] - 1)),
    )


def _t1_grid_matches_ref(t1_entry: T1CacheEntry, ref_data: RefCacheEntry) -> bool:
    return tuple(t1_entry["shape"]) == tuple(ref_data["mask_data"].shape)


def _subsample_streamlines(
    streamlines: Streamlines,
    max_streamlines: int | None = None,
    seed: int = 0,
) -> Streamlines:
    if max_streamlines is None or max_streamlines <= 0:
        return streamlines
    n = len(streamlines)
    if n <= max_streamlines:
        return streamlines
    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(n, size=max_streamlines, replace=False))
    return Streamlines([streamlines[ii] for ii in keep_idx])


def _normalize_streamline_cap(v: int | None) -> int | None:
    """Positive int = cap; None or <=0 = no cap (use all streamlines)."""
    if v is None:
        return None
    iv = int(v)
    return None if iv <= 0 else iv


def _cap_for_run_info(v: int | None) -> int | str:
    if v is None or (isinstance(v, int) and v <= 0):
        return "none"
    return int(v)


def _get_metric_data(metric_file: str | Path, cache: MetricCache | None = None) -> tuple[nib.Nifti1Image, NDArrayFloat]:
    local_cache = _metric_cache(cache)
    key = str(metric_file)
    if key not in local_cache:
        metric_img = _load_nifti(key)
        local_cache[key] = (metric_img, np.asarray(metric_img.get_fdata(), dtype=float))
    return local_cache[key]


def _get_rasmm_streamlines(tract_file: str | Path, cache: StreamlineCache | None = None) -> Streamlines:
    """Load tract streamlines in RASMM space (cached). Does not downsample."""
    local_cache = _streamline_cache(cache)
    key = str(tract_file)
    if key not in local_cache:
        t0 = time.perf_counter()
        tract = load_trk(key, "same", bbox_valid_check=False)
        tract.to_rasmm()
        local_cache[key] = tract.streamlines
        logger.debug(
            "    loaded %s: %d streamlines (%.1fs)",
            Path(tract_file).name,
            len(local_cache[key]),
            time.perf_counter() - t0,
        )
    return local_cache[key]


def _get_downsampled_rasmm_streamlines(
    tract_file: str | Path,
    cache: StreamlineCache | None = None,
    approx_pts: float = 0.25,
    max_streamlines: int | None = None,
    subsample_seed: int = 0,
) -> Streamlines:
    """Subsample (optional cap) then downsample — cap applies before expensive geometry work."""
    streamlines = _get_rasmm_streamlines(tract_file, cache)
    streamlines = _subsample_streamlines(
        streamlines,
        max_streamlines=max_streamlines,
        seed=subsample_seed,
    )
    return downsample_streamlines(streamlines, approx_pts=approx_pts)


def _get_atlas_standard_streamline(
    atlas_file: str | Path,
    cache: AtlasStandardCache | None,
    streamlines_cache: StreamlineCache | None,
) -> Any:
    local_cache = _atlas_standard_cache(cache)
    key = str(atlas_file)
    if key not in local_cache:
        atlas_streamlines = _get_downsampled_rasmm_streamlines(key, streamlines_cache)
        feature = ResampleFeature(nb_points=100)
        qb_metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=np.inf, metric=qb_metric)
        cluster_tract = qb.cluster(atlas_streamlines)
        local_cache[key] = cluster_tract.centroids[0]
    return local_cache[key]


def calc_centroid(streamlines: Streamlines | None) -> NDArrayFloat:
    """
    Safely compute centroid of a set of streamlines.

    Returns a 3-vector. If no points are available (e.g., empty bundle),
    returns the origin to avoid crashes in downstream camera logic.
    """
    if streamlines is None:
        return np.zeros(3, dtype=float)
    # len(Streamlines) is cheap and robust
    if len(streamlines) == 0:
        return np.zeros(3, dtype=float)
    all_pts = np.vstack([np.asarray(sl) for sl in streamlines if len(sl) > 0])
    if all_pts.size == 0:
        return np.zeros(3, dtype=float)
    return np.mean(all_pts, axis=0)


def calc_bbox_size(streamlines: Streamlines | None) -> NDArrayFloat | None:
    """
    Safely compute bounding-box size of a set of streamlines.

    Returns a 3-vector (max - min). If no points are available, returns None
    and callers should fall back to a default zoom.
    """
    if streamlines is None or len(streamlines) == 0:
        return None
    all_pts = np.vstack([np.asarray(sl) for sl in streamlines if len(sl) > 0])
    if all_pts.size == 0:
        return None
    return np.max(all_pts, axis=0) - np.min(all_pts, axis=0)


def init_scene(mask_data: NDArrayFloat | str | Path) -> Any:
    """Create a FURY scene with a faint brain-mask contour.

    Parameters
    ----------
    mask_data : array-like or str or Path
        Brain mask volume, or path to a NIfTI mask file.

    Returns
    -------
    fury.window.Scene
        Scene ready for streamline actors.
    """
    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    if isinstance(mask_data, (str, Path)):
        mask_data = np.asarray(_load_nifti(str(mask_data)).get_fdata(), dtype=float)
    brain_actor = actor.contour_from_roi(mask_data, color=[0, 0, 0], opacity=0.05)

    scene.add(brain_actor)
    del brain_actor
    gc.collect()

    return scene


def init_scene_with_t1_slice(t1_data: NDArrayFloat, view: ViewName, centroid_vox: NDArrayFloat) -> Any:
    """Orthogonal T1 slice through centroid; streamlines in the same voxel grid.

    Caller must apply ``set_anat_camera`` (same RAS convention as ``gen_views``).
    """
    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    t1_display = histeq(np.asarray(t1_data, dtype=float).copy(), num_bins=256)
    value_range = (float(t1_display.min()), float(t1_display.max()))
    vol_actor = actor.slicer(t1_display, value_range=value_range)
    ix, iy, iz = _centroid_vox_indices(centroid_vox, t1_data.shape)
    if view == "sagittal":
        vol_actor.display(x=ix, y=None, z=None)
    elif view == "coronal":
        vol_actor.display(x=None, y=iy, z=None)
    else:
        vol_actor.display(x=None, y=None, z=iz)
    scene.add(vol_actor)
    return scene


def set_anat_camera(
    scene: Any,
    centroid: NDArrayFloat,
    view_name: ViewName,
    bbox_size: NDArrayFloat | None,
    template_space: TemplateSpace,
) -> Any:
    """Position the scene camera for coronal, axial, or sagittal viewing.

    Parameters
    ----------
    scene : fury.window.Scene
        Scene to configure.
    centroid : array-like
        3D focal point in the same space as the displayed anatomy.
    view_name : str
        One of ``"coronal"``, ``"axial"``, or ``"sagittal"``.
    bbox_size : array-like or None
        Streamline bounding-box extent used to set camera distance.
    template_space : str
        ``"mni"`` or ``"ras"``; controls camera handedness conventions.

    Returns
    -------
    fury.window.Scene
        The same scene with an updated camera.
    """
    max_dim = np.max(bbox_size) if bbox_size is not None else 100.0
    camera_distance = max_dim * 2.5
    use_mni = template_space == "mni"

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

    return scene


def downsample_streamlines(streamlines: Streamlines, approx_pts: float = 0.25) -> Streamlines:
    """Downsample streamlines with DIPY polygon approximation.

    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines in voxel or world space.
    approx_pts : float, optional
        Approximation parameter passed to :func:`dipy.tracking.distances.approx_polygon_track`.

    Returns
    -------
    Streamlines
        Downsampled streamlines.
    """
    downsampled = [approx_polygon_track(s, approx_pts) for s in streamlines]
    return Streamlines(downsampled)


def load_tract(tract_file: str | Path, ref_img: str | Path) -> Any:
    """Load a tractogram and return a FURY line actor in reference voxel space.

    Parameters
    ----------
    tract_file : str or Path
        Path to a ``.trk`` tractography file.
    ref_img : str or Path
        Reference NIfTI used to transform streamlines into voxel coordinates.

    Returns
    -------
    fury.actor.line
        Line actor for the tract, transformed to reference space.
    """
    tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
    tract.to_rasmm()
    ref_img_obj = _load_nifti(ref_img)
    tract_to_ref = transform_streamlines(tract.streamlines, np.linalg.inv(ref_img_obj.affine))

    return actor.line(tract_to_ref)


def weighted_afq(
    tract_streamlines: Streamlines,
    standard_tract: Any,
    metric_file: str | Path,
    metric_cache: MetricCache,
    max_streamlines: int | None = None,
) -> NDArrayFloat:
    """Compute a weighted AFQ profile along a bundle for one scalar map.

    Parameters
    ----------
    tract_streamlines : Streamlines
        Subject bundle streamlines in RASMM space.
    standard_tract : Streamlines
        Atlas reference streamline used for orientation.
    metric_file : str or Path
        Scalar map NIfTI (for example FA or NODDI ICVF).
    metric_cache : dict
        Mutable cache of loaded metric volumes.
    max_streamlines : int, optional
        Optional cap on streamlines used for the profile.

    Returns
    -------
    numpy.ndarray
        Length-100 profile along the bundle, or NaNs when the bundle is empty.
    """
    metric_img, metric_data = _get_metric_data(metric_file, metric_cache)
    # Keep profile length consistent with AFQ defaults and downstream CSV logic.
    n_nodes = 100
    if tract_streamlines is None or len(tract_streamlines) == 0:
        return np.full(n_nodes, np.nan, dtype=float)
    tract_streamlines = _subsample_streamlines(tract_streamlines, max_streamlines=max_streamlines, seed=11)

    oriented_tract = orient_by_streamline(tract_streamlines, standard_tract)
    if oriented_tract is None or len(oriented_tract) == 0:
        return np.full(n_nodes, np.nan, dtype=float)

    # dipy's gaussian_weights can fail for very small/degenerate bundles (e.g., one
    # streamline after CCI filtering). In those cases, fall back to unweighted AFQ.
    w_tract = None
    if len(oriented_tract) >= _MIN_STREAMLINES_FOR_GAUSSIAN_WEIGHTS:
        try:
            w_tract = gaussian_weights(oriented_tract)
        except (ValueError, RuntimeError, TypeError) as exc:
            logger.debug("gaussian_weights failed, using unweighted AFQ: %s", exc)
            w_tract = None

    try:
        profile = afq_profile(
            metric_data,
            oriented_tract,
            affine=metric_img.affine,
            weights=w_tract,
        )
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        logger.debug("afq_profile failed for %s: %s", metric_file, exc)
        profile = np.full(n_nodes, np.nan, dtype=float)

    return profile


def _normalize_hist_bins(bins: int | str | NDArrayFloat, default: int = 100) -> int | str | NDArrayFloat:
    """Return bins arg safe for numpy.histogram (int, str, or edge array)."""
    if isinstance(bins, str):
        return bins
    arr = np.asarray(bins)
    if arr.ndim == 0:
        try:
            return max(1, int(arr.item()))
        except (TypeError, ValueError):
            return default
    if arr.ndim == 1:
        if arr.size == 1:
            try:
                return max(1, int(arr[0]))
            except (TypeError, ValueError):
                return default
        return arr.astype(float, copy=False)
    return default


def _save_cci_histogram(cci: NDArrayFloat, hist_path: str | Path, *, n_hist_bins: int = 100) -> None:
    cci_vals = np.asarray(cci, dtype=float).ravel()
    cci_vals = cci_vals[np.isfinite(cci_vals)]
    fig, ax = plt.subplots(1, figsize=(8, 6))
    if cci_vals.size == 0:
        ax.text(
            0.5,
            0.5,
            "No CCI values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("CCI Distribution")
    else:
        bins_arg = _normalize_hist_bins(n_hist_bins)
        if isinstance(bins_arg, (str, np.ndarray)):
            counts, edges = np.histogram(cci_vals, bins=bins_arg)
        else:
            counts, edges = np.histogram(cci_vals, bins=int(bins_arg))
        ax.stairs(counts, edges, fill=False)
        ax.set_xlabel("CCI")
        ax.set_ylabel("# streamlines")
        ax.set_title("CCI Distribution")
        ax.grid(visible=True, alpha=0.3)
    fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def calc_cci(
    tract_streamlines: Streamlines,
    ref_img_obj: nib.Nifti1Image,
    min_length_mm: float = 40,
    cci_threshold: float = 1.0,
    max_streamlines: int | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat, StatefulTractogram, Streamlines, CCIStats]:
    """
    Compute cluster confidence index (CCI) and keep streamlines above threshold.

    CCI (dipy) measures how much support each streamline has from similar neighbors;
    high CCI = many nearby streamlines (likely valid), low CCI = isolated (likely outlier).
    We keep streamlines with CCI >= cci_threshold for QA and downstream profiles.

    Short streamlines (< min_length_mm) are excluded before CCI (dipy recommends
    removing < 20 mm; 40 mm avoids short fragments that can distort the distribution).
    """
    tract_streamlines = _subsample_streamlines(tract_streamlines, max_streamlines=max_streamlines, seed=17)
    lengths = list(length(tract_streamlines))
    long_streamlines = Streamlines()
    for ii, sl in enumerate(tract_streamlines):
        if lengths[ii] > min_length_mm:
            long_streamlines.append(sl)

    n_long = len(long_streamlines)
    if n_long == 0:
        # No sufficiently long streamlines: return empty outputs but valid structures
        empty_arr = np.asarray([], dtype=float)
        keep_streamlines = Streamlines()
        keep_tract = StatefulTractogram(keep_streamlines, ref_img_obj, Space.RASMM)
        stats = {
            "n_long": 0,
            "n_keep": 0,
            "pct_retained": 0.0,
            "mean_cci": np.nan,
            "mean_cci_keep": np.nan,
        }
        return empty_arr, empty_arr, keep_tract, long_streamlines, stats

    cci = np.asarray(cluster_confidence(long_streamlines), dtype=float).ravel()
    keep_mask = cci >= cci_threshold

    keep_streamlines = Streamlines()
    for ii, sl in enumerate(long_streamlines):
        if keep_mask[ii]:
            keep_streamlines.append(sl)

    keep_cci = cci[keep_mask]
    keep_tract = StatefulTractogram(keep_streamlines, ref_img_obj, Space.RASMM)

    n_keep = int(np.sum(keep_mask))
    pct_retained = (100.0 * n_keep / n_long) if n_long else 0.0
    mean_cci = float(np.mean(cci)) if n_long else np.nan
    mean_cci_keep = float(np.mean(keep_cci)) if n_keep else np.nan

    return (
        cci,
        keep_cci,
        keep_tract,
        long_streamlines,
        {
            "n_long": n_long,
            "n_keep": n_keep,
            "pct_retained": pct_retained,
            "mean_cci": mean_cci,
            "mean_cci_keep": mean_cci_keep,
        },
    )


def plot_before_after_cci(
    tract_file: str | Path,
    ref_img: str | Path,
    output_dir: str | Path,
    *,
    fig_size: tuple[int, int] = (800, 800),
    n_hist_bins: int = 100,
    overwrite: bool = False,
    cci_threshold: float = 1.0,
    streamlines_cache: StreamlineCache | None = None,
    ref_cache: RefCache | None = None,
    max_streamlines_cci: int | None = None,
) -> CciViewDict:
    """Render CCI before/after views, histogram, and retained bundle assets.

    Computes cluster confidence indices, writes ``*_cci.csv`` and
    ``*_cci_keep.trk``, and saves coronal/axial/sagittal before/after PNGs.

    Parameters
    ----------
    tract_file : str or Path
        Subject bundle ``.trk`` file.
    ref_img : str or Path
        Reference brain-mask NIfTI.
    output_dir : str or Path
        Directory for CCI PNGs, CSV, and retained tractogram.
    fig_size : tuple of int, optional
        Output image size in pixels.
    n_hist_bins : int, optional
        Number of histogram bins for the CCI plot.
    overwrite : bool, optional
        Recompute outputs even when files already exist.
    cci_threshold : float, optional
        Minimum CCI for streamlines retained in the ``*_cci_keep`` tract.
    streamlines_cache : dict, optional
        Shared streamline cache for repeated loads.
    ref_cache : dict, optional
        Shared reference-image cache.
    max_streamlines_cci : int, optional
        Cap on streamlines used for CCI.

    Returns
    -------
    dict
        Paths to generated assets plus ``cci_stats`` when available.
    """
    tract_name = Path(tract_file).stem
    hist_path = f"{output_dir}/{tract_name}_cci_histogram.png"
    cci_csv_path = Path(output_dir) / f"{tract_name}_cci.csv"
    generated_views: CciViewDict = {}

    all_files_exist = True
    if not Path(hist_path).exists() or overwrite:
        all_files_exist = False
    else:
        generated_views["histogram"] = hist_path

    for v in views:
        out_before = f"{output_dir}/cci_before_{tract_name}_{v}.png"
        out_after = f"{output_dir}/cci_after_{tract_name}_{v}.png"
        if Path(out_before).exists() and Path(out_after).exists() and not overwrite:
            generated_views[v] = {"before": out_before, "after": out_after}
        else:
            all_files_exist = False

    keep_tract_path = Path(output_dir) / f"{tract_name}_cci_keep.trk"
    if all_files_exist:
        if cci_csv_path.exists():
            try:
                cci_vals = []
                with open(cci_csv_path, newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        try:
                            cci_vals.append(float(row.get("cci", "")))
                        except (TypeError, ValueError):
                            continue
                cci_arr = np.asarray(cci_vals, dtype=float)
                if cci_arr.size > 0:
                    keep_mask = cci_arr >= float(cci_threshold)
                    keep_vals = cci_arr[keep_mask]
                    n_long = int(cci_arr.size)
                    n_keep = int(np.sum(keep_mask))
                    generated_views["cci_stats"] = {
                        "n_long": n_long,
                        "n_keep": n_keep,
                        "pct_retained": (100.0 * n_keep / n_long) if n_long else 0.0,
                        "mean_cci": float(np.mean(cci_arr)) if n_long else np.nan,
                        "mean_cci_keep": float(np.mean(keep_vals)) if n_keep else np.nan,
                    }
                    generated_views["cci_data"] = str(cci_csv_path)
            except (OSError, csv.Error, ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to reconstruct CCI stats from %s; using image-only outputs: %s",
                    cci_csv_path,
                    exc,
                )
        if keep_tract_path.exists():
            generated_views["keep_tract"] = str(keep_tract_path)
        return generated_views

    ref_data = _get_ref_data(ref_img, ref_cache)
    tract_streamlines = _get_downsampled_rasmm_streamlines(
        tract_file,
        streamlines_cache,
        max_streamlines=max_streamlines_cci,
        subsample_seed=17,
    )

    t0 = time.perf_counter()
    cci, _keep_cci, keep_tract, long_streamlines, cci_stats = calc_cci(
        tract_streamlines,
        ref_img_obj=cast("nib.Nifti1Image", ref_data["img"]),
        cci_threshold=cci_threshold,
        max_streamlines=None,
    )
    logger.debug("    calc_cci: %.2fs", time.perf_counter() - t0)
    generated_views["cci_stats"] = cci_stats

    cci_csv_path_written = write_cci_csv(tract_name, cci, output_dir)
    generated_views["cci_data"] = cci_csv_path_written

    keep_tract_path = Path(output_dir) / f"{tract_name}_cci_keep.trk"
    save_trk(keep_tract, str(keep_tract_path), bbox_valid_check=False)
    generated_views["keep_tract"] = str(keep_tract_path)

    inv_affine = ref_data["inv_affine"]

    if not Path(hist_path).exists() or overwrite:
        t0 = time.perf_counter()
        _save_cci_histogram(cci, hist_path, n_hist_bins=n_hist_bins)
        logger.debug("    CCI histogram: %.2fs", time.perf_counter() - t0)

    generated_views["histogram"] = hist_path

    # If there are no long streamlines or no retained streamlines, skip
    # 3D before/after visualizations to avoid centroid/colormap errors.
    if len(long_streamlines) == 0 or len(keep_tract.streamlines) == 0 or len(cci) == 0:
        logger.warning("    No valid streamlines for CCI before/after views; skipping 3D plots.")
        return generated_views

    before_streamlines = transform_streamlines(long_streamlines, inv_affine)
    after_streamlines = transform_streamlines(keep_tract.streamlines, inv_affine)
    cci_before = np.asarray(cci, dtype=np.float64)
    cci_min = float(np.min(cci))
    cci_max = float(np.max(cci))
    hue = [0.5, 1]
    saturation = [0.0, 1.0]
    lut_cmap = actor.colormap_lookup_table(
        scale_range=(cci_min, cci_max / 4),
        hue_range=hue,
        saturation_range=saturation,
    )
    bar = actor.scalar_bar(lookup_table=lut_cmap)
    before_centroid = calc_centroid(before_streamlines)
    after_centroid = calc_centroid(after_streamlines)

    for v in views:
        out_before = f"{output_dir}/cci_before_{tract_name}_{v}.png"
        out_after = f"{output_dir}/cci_after_{tract_name}_{v}.png"

        t0 = time.perf_counter()
        scene_before = init_scene(ref_data["mask_data"])
        scene_before.add(bar)
        before_actor = actor.line(before_streamlines, colors=cci_before, lookup_colormap=lut_cmap)
        scene_before.add(before_actor)
        bbox_size_before = calc_bbox_size(before_streamlines)
        scene_before = set_anat_camera(
            scene_before,
            before_centroid,
            v,
            bbox_size=bbox_size_before,
            template_space="ras",
        )
        window.record(scene_before, out_path=str(out_before), size=fig_size)
        scene_before.clear()
        del before_actor, scene_before
        gc.collect()
        logger.debug("    CCI before %s: %.2fs", v, time.perf_counter() - t0)

        t0 = time.perf_counter()
        scene_after = init_scene(ref_data["mask_data"])
        after_actor = actor.line(after_streamlines)
        scene_after.add(after_actor)
        bbox_size_after = calc_bbox_size(after_streamlines)
        scene_after = set_anat_camera(scene_after, after_centroid, v, bbox_size=bbox_size_after, template_space="ras")
        window.record(scene_after, out_path=str(out_after), size=fig_size)
        scene_after.clear()
        del scene_after, after_actor
        gc.collect()
        logger.debug("    CCI after %s: %.2fs", v, time.perf_counter() - t0)

        generated_views[v] = {"before": out_before, "after": out_after}

    return generated_views


def gen_views(
    tract_file: str | Path,
    ref_img: str | Path,
    output_dir: str | Path,
    *,
    atlas: bool = False,
    overwrite: bool = False,
    fig_size: tuple[int, int] = (800, 800),
    streamlines_cache: StreamlineCache | None = None,
    ref_cache: RefCache | None = None,
    max_streamlines_view: int | None = None,
) -> ViewPathMap:
    """Render coronal, axial, and sagittal tract PNGs over a brain mask.

    Parameters
    ----------
    tract_file : str or Path
        Tractography ``.trk`` file to render.
    ref_img : str or Path
        Reference brain-mask NIfTI.
    output_dir : str or Path
        Directory for output PNG files.
    atlas : bool, optional
        If True, use MNI camera conventions and ``*_atlas_*`` filenames.
    overwrite : bool, optional
        Recompute views even when PNGs already exist.
    fig_size : tuple of int, optional
        Output image size in pixels.
    streamlines_cache : dict, optional
        Shared streamline cache.
    ref_cache : dict, optional
        Shared reference-image cache.
    max_streamlines_view : int, optional
        Cap on streamlines rendered per view.

    Returns
    -------
    dict
        Mapping of view name to output PNG path.
    """
    template_space: TemplateSpace = "mni" if atlas or "MNI" in str(ref_img) else "ras"
    tract_name = Path(tract_file).stem
    generated_views: ViewPathMap = {}

    all_files_exist = True
    for v in views:
        out_img = f"{output_dir}/{tract_name}_atlas_{v}.png" if atlas else f"{output_dir}/{tract_name}_{v}.png"

        if Path(out_img).exists() and not overwrite:
            generated_views[v] = out_img
        else:
            all_files_exist = False

    if all_files_exist:
        return generated_views

    t0 = time.perf_counter()
    ref_data = _get_ref_data(ref_img, ref_cache)
    tract_streamlines = _get_downsampled_rasmm_streamlines(
        tract_file,
        streamlines_cache,
        max_streamlines=max_streamlines_view,
        subsample_seed=23,
    )
    tract_streamlines = transform_streamlines(tract_streamlines, ref_data["inv_affine"])
    centroid = calc_centroid(tract_streamlines)
    logger.debug("    load+transform+downsample: %.2fs", time.perf_counter() - t0)

    for v in views:
        out_img = f"{output_dir}/{tract_name}_atlas_{v}.png" if atlas else f"{output_dir}/{tract_name}_{v}.png"
        t0 = time.perf_counter()
        scene = init_scene(ref_data["mask_data"])
        tract_actor = actor.line(tract_streamlines)
        scene.add(tract_actor)
        bbox_size = calc_bbox_size(tract_streamlines)
        scene = set_anat_camera(scene, centroid, v, bbox_size, template_space)
        window.record(scene=scene, out_path=str(out_img), size=fig_size)
        generated_views[v] = out_img
        scene.clear()
        if tract_actor is not None:
            del tract_actor
        del scene
        gc.collect()
        logger.debug("    view %s: %.2fs", v, time.perf_counter() - t0)

    del tract_streamlines
    gc.collect()

    return generated_views


def gen_t1_views(
    tract_file: str | Path,
    ref_img: str | Path,
    t1_img: str | Path | None,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    fig_size: tuple[int, int] = (800, 800),
    streamlines_cache: StreamlineCache | None = None,
    ref_cache: RefCache | None = None,
    t1_cache: T1Cache | None = None,
    max_streamlines_view: int | None = None,
) -> ViewPathMap:
    """Render tract streamlines overlaid on orthogonal T1 slices through the bundle centroid."""
    if not t1_img or not Path(t1_img).exists():
        return {}

    tract_name = Path(tract_file).stem
    generated_views: ViewPathMap = {}
    all_files_exist = True
    for v in views:
        out_img = f"{output_dir}/{tract_name}_t1_{v}.png"
        if Path(out_img).exists() and not overwrite:
            generated_views[v] = out_img
        else:
            all_files_exist = False

    if all_files_exist:
        return generated_views

    ref_data = _get_ref_data(ref_img, ref_cache)
    t1_entry = _get_t1_data(t1_img, t1_cache)
    if not _t1_grid_matches_ref(t1_entry, ref_data):
        logger.warning(
            "    T1 grid %s does not match ref %s; skipping T1 views for %s",
            t1_entry["shape"],
            ref_data["mask_data"].shape,
            tract_name,
        )
        return {}

    t0 = time.perf_counter()
    tract_streamlines = _get_downsampled_rasmm_streamlines(
        tract_file,
        streamlines_cache,
        max_streamlines=max_streamlines_view,
        subsample_seed=23,
    )
    tract_streamlines = transform_streamlines(tract_streamlines, ref_data["inv_affine"])
    centroid = calc_centroid(tract_streamlines)
    logger.debug("    T1 load+transform: %.2fs", time.perf_counter() - t0)

    bbox_size = calc_bbox_size(tract_streamlines)
    for v in views:
        out_img = f"{output_dir}/{tract_name}_t1_{v}.png"
        t0 = time.perf_counter()
        scene = init_scene_with_t1_slice(t1_entry["data"], v, centroid)
        tract_actor = actor.line(tract_streamlines)
        scene.add(tract_actor)
        scene = set_anat_camera(
            scene,
            centroid,
            v,
            bbox_size=bbox_size,
            template_space="ras",
        )
        window.record(scene=scene, out_path=str(out_img), size=fig_size)
        generated_views[v] = out_img
        scene.clear()
        del tract_actor, scene
        gc.collect()
        logger.debug("    T1 view %s: %.2fs", v, time.perf_counter() - t0)

    del tract_streamlines
    gc.collect()
    return generated_views


def _bundle_colors(n_bundles: int) -> list[tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors: list[tuple[float, float, float]] = []
    for ii in range(n_bundles):
        rgba = cmap(ii % 20)[:3]
        colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2])))
    return colors


def gen_subject_anatomy_views(
    subject_files: Sequence[str | Path],
    ref_img: str | Path,
    t1_img: str | Path | None,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    fig_size: tuple[int, int] = (800, 800),
    streamlines_cache: StreamlineCache | None = None,
    ref_cache: RefCache | None = None,
    t1_cache: T1Cache | None = None,
    max_streamlines_per_bundle: int = 400,
) -> ViewPathMap:
    """Once-per-subject T1 slices through brain center with all bundles overlaid."""
    if not t1_img or not Path(t1_img).exists():
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_views: ViewPathMap = {}
    all_files_exist = True
    for v in views:
        out_img = output_dir / f"subject_anatomy_{v}.png"
        if out_img.exists() and not overwrite:
            generated_views[v] = str(out_img)
        else:
            all_files_exist = False

    if all_files_exist:
        return generated_views

    ref_data = _get_ref_data(ref_img, ref_cache)
    t1_entry = _get_t1_data(t1_img, t1_cache)
    if not _t1_grid_matches_ref(t1_entry, ref_data):
        logger.warning(
            "    T1 grid %s does not match ref %s; skipping subject anatomy views",
            t1_entry["shape"],
            ref_data["mask_data"].shape,
        )
        return {}

    mask_coords = np.argwhere(ref_data["mask_data"] > 0)
    if mask_coords.size == 0:
        centroid = np.asarray(ref_data["mask_data"].shape, dtype=float) / 2.0
    else:
        centroid = np.mean(mask_coords, axis=0)

    bundle_streamlines = []
    colors = _bundle_colors(len(subject_files))
    t0 = time.perf_counter()
    for ii, sf in enumerate(subject_files):
        sls = _get_downsampled_rasmm_streamlines(
            str(sf),
            streamlines_cache,
            max_streamlines=max_streamlines_per_bundle,
            subsample_seed=31 + ii,
        )
        sls = transform_streamlines(sls, ref_data["inv_affine"])
        if len(sls) > 0:
            bundle_streamlines.append((sls, colors[ii]))
    logger.debug("    subject anatomy load+transform: %.2fs", time.perf_counter() - t0)

    combined = Streamlines()
    for sls, _color in bundle_streamlines:
        for sl in sls:
            combined.append(sl)
    bbox_size = calc_bbox_size(combined)

    for v in views:
        out_img = output_dir / f"subject_anatomy_{v}.png"
        t0 = time.perf_counter()
        scene = init_scene_with_t1_slice(t1_entry["data"], v, centroid)
        for sls, color in bundle_streamlines:
            scene.add(actor.line(sls, colors=color))
        scene = set_anat_camera(
            scene,
            centroid,
            v,
            bbox_size=bbox_size,
            template_space="ras",
        )
        window.record(scene=scene, out_path=str(out_img), size=fig_size)
        generated_views[v] = str(out_img)
        scene.clear()
        del scene
        gc.collect()
        logger.debug("    subject anatomy %s: %.2fs", v, time.perf_counter() - t0)

    return generated_views


def plot_afq(
    profile: NDArrayFloat,
    _metric_file: str | Path,
    metric_str: str,
    tract_file: str | Path,
    _atlas_file: str | Path,
    ref_img: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    colormap: str = "Spectral",
) -> AfqViewMap | tuple[AfqViewMap, NDArrayFloat]:
    """Render AFQ-colored tract views and a 1D profile plot for one metric.

    Parameters
    ----------
    profile : array-like
        Length-100 AFQ profile used to color streamlines and plot the ribbon.
    metric_file : str or Path
        Scalar map NIfTI path (used for metadata only after profile computation).
    metric_str : str
        Short metric label used in filenames and plot titles.
    tract_file : str or Path
        Subject bundle ``.trk`` file.
    atlas_file : str or Path
        Atlas bundle used for orientation (metadata).
    ref_img : str or Path
        Reference brain-mask NIfTI.
    output_dir : str or Path
        Directory for profile plot and view PNGs.
    overwrite : bool, optional
        Recompute outputs even when files already exist.
    colormap : str, optional
        Matplotlib/FURY colormap name for streamline coloring.

    Returns
    -------
    dict
        Mapping of view names and ``profile_plot`` to output paths.
    """
    t0 = time.perf_counter()
    template_space: TemplateSpace = "mni" if "MNI" in str(ref_img) else "ras"
    tract_name = Path(tract_file).stem
    generated_views: AfqViewMap = {}
    profile_plot_path = f"{output_dir}/{tract_name}_{metric_str}_profile.png"
    generated_views["profile_plot"] = profile_plot_path

    all_view_files_exist = True
    for v in views:
        out_img = f"{output_dir}/{tract_name}_{metric_str}_{v}.png"
        if Path(out_img).exists() and not overwrite:
            generated_views[v] = out_img
        else:
            all_view_files_exist = False

    if all_view_files_exist and Path(profile_plot_path).exists() and not overwrite:
        return generated_views, profile

    t0 = time.perf_counter()
    tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
    tract.to_rasmm()
    ref_img_obj = _load_nifti(ref_img)
    tract_streamlines = transform_streamlines(tract.streamlines, np.linalg.inv(ref_img_obj.affine))
    tract_streamlines = downsample_streamlines(tract_streamlines)
    streamline_point_colors = []
    for sl in tract_streamlines:
        sl_array = np.array(sl)
        interpolated_vals = np.interp(np.linspace(0, 1, len(sl_array)), np.linspace(0, 1, len(profile)), profile)
        point_colors = create_colormap(interpolated_vals, name=colormap)
        streamline_point_colors.append(point_colors)
    centroid = calc_centroid(tract_streamlines)
    logger.debug("      load+colormap: %.2fs", time.perf_counter() - t0)

    for v in views:
        out_img = f"{output_dir}/{tract_name}_{metric_str}_{v}.png"
        t0 = time.perf_counter()
        scene = init_scene(ref_img)
        for _ii, (sl, point_colors) in enumerate(zip(tract_streamlines, streamline_point_colors)):
            line_actor = actor.line([sl], colors=point_colors)
            scene.add(line_actor)
        bbox_size = calc_bbox_size(tract_streamlines)
        scene = set_anat_camera(scene, centroid, v, bbox_size=bbox_size, template_space=template_space)
        window.record(scene=scene, out_path=str(out_img), size=(800, 800))
        generated_views[v] = out_img
        scene.clear()
        del scene
        gc.collect()
        logger.debug("      %s view %s: %.2fs", metric_str, v, time.perf_counter() - t0)

    if not Path(profile_plot_path).exists() or overwrite:
        t0 = time.perf_counter()
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.plot(profile)
        ax.set_xlabel("Node along tract")
        ax.set_ylabel(metric_str)
        ax.set_title(f"AFQ Profile: {metric_str}")
        ax.grid(visible=True, alpha=0.3)
        fig.savefig(str(profile_plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("      profile plot: %.2fs", time.perf_counter() - t0)

    return generated_views


def write_profile_csv(profiles_rows: Sequence[ProfileRow], csv_path: str | Path) -> None:
    """Write profile data to CSV.

    profiles_rows: list of (tract, metric, profile_array).
    CSV columns: tract, metric, node, value.
    """
    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tract", "metric", "node", "value"])
        for tract, metric, profile_values in profiles_rows:
            profile_arr = np.asarray(profile_values)
            for node, val in enumerate(profile_arr):
                w.writerow([tract, metric, node, float(val)])


def write_cci_csv(tract_name: str, cci: NDArrayFloat, output_dir: str | Path) -> str:
    """Save CCI values (one per long streamline) to CSV for later use.

    CSV columns: tract, streamline_index, cci.
    """
    output_dir = Path(output_dir)
    csv_path = output_dir / f"{tract_name}_cci.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tract", "streamline_index", "cci"])
        for idx, val in enumerate(np.asarray(cci)):
            w.writerow([tract_name, idx, float(val)])
    return str(csv_path)


def write_tract_stats(
    results: Mapping[str, Any],
    profile_rows: Sequence[ProfileRow],
    output_path: str | Path,
) -> None:
    """Write one row per tract: shape_score, CCI QA stats, plus mean profile for each metric.

    results: dict tract_name -> [..., extra] with extra["shape_score"], extra["n_long"], etc.
    profile_rows: list of (tract, metric, profile_array).
    CSV columns: tract, shape_score, cci_n_long, cci_n_keep, cci_pct_retained, cci_mean, mean_<metric>, ...
    """
    output_path = Path(output_path)
    means = {}
    for tract, metric, profile in profile_rows:
        key = (tract, metric)
        means[key] = float(np.mean(np.asarray(profile)))
    metrics_sorted = sorted({m for (_, m) in means})
    cci_cols = ["cci_n_long", "cci_n_keep", "cci_pct_retained", "cci_mean"]
    header = ["tract", "shape_score"] + cci_cols + [f"mean_{m}" for m in metrics_sorted]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for tract_name in results:
            if str(tract_name).startswith("_"):
                continue
            tract_data = results[tract_name]
            extra = tract_data[-1] if isinstance(tract_data, (list, tuple)) and len(tract_data) > 0 else {}
            if not isinstance(extra, dict):
                extra = {}
            shape_score = extra.get("shape_score", "")
            if shape_score != "" and shape_score is not None:
                shape_score = float(shape_score)
            row = [tract_name, shape_score]
            for col in cci_cols:
                val = extra.get(col, "")
                if val != "" and val is not None and not (isinstance(val, float) and np.isnan(val)):
                    if col in {"cci_pct_retained", "cci_mean"}:
                        val = float(val)
                    elif col in ("cci_n_long", "cci_n_keep"):
                        val = int(val)
                row.append(val)
            for m in metrics_sorted:
                row.append(means.get((tract_name, m), ""))
            w.writerow(row)


def generate_subject_qc_html(
    qc_data_path: str | Path,
    output_html_path: str | Path | None = None,
) -> str:
    """
    Generate a single HTML report for one subject from qc_data.json.

    qc_data_path: path to qc_data.json (e.g. output_dir/subject/qc_data.json)
    output_html_path: where to write the HTML file (default: same dir as qc_data.json, name qc_report.html)
    """
    qc_data_path = Path(qc_data_path)
    if not qc_data_path.exists():
        raise FileNotFoundError(f"QC data not found: {qc_data_path}")
    base_dir = qc_data_path.parent
    if output_html_path is None:
        output_html_path = base_dir / "qc_report.html"
    output_html_path = Path(output_html_path)

    with open(qc_data_path) as f:
        data = json.load(f)

    # Expect one subject in the file
    subjects = list(data.keys())
    if not subjects:
        raise ValueError("qc_data.json has no subject keys")
    subject_id = subjects[0]
    results = data[subject_id]

    def rel(path: str | Path | None) -> str:
        """Path relative to report base (subject output dir) for correct img src (tract subfolders or shared atlas_views)."""
        if not path:
            return ""
        p = Path(path)
        try:
            return str(p.relative_to(base_dir))
        except ValueError:
            # Path is outside base_dir (e.g. shared atlas_views); use relpath so ../atlas_views/... works
            return os.path.relpath(str(p), str(base_dir))

    # Stable tract ordering across subjects/reports.
    preferred_order = [
        "CCForcepsMajor",
        "CCForcepsMinor",
        "CCMid",
        "MCP",
        "CSTL",
        "CSTR",
        "CL",
        "CR",
        "AFL",
        "AFR",
        "IFOFL",
        "IFOFR",
        "UFL",
        "UFR",
        "FL",
        "FR",
    ]
    order_index = {name: ii for ii, name in enumerate(preferred_order)}

    def tract_sort_key(name: str) -> tuple[int, str]:
        # Unknown names go after known names, then alphabetical.
        return (order_index.get(name, len(preferred_order)), name)

    def suspicious_flags(extra: Any) -> tuple[str, str]:
        """Return suspicious-bundle flag type and reason.

        flag_type is one of:
        - "" (not flagged)
        - "low_shape"
        - "low_cci_retention".
        """
        if not isinstance(extra, dict):
            return "", ""

        shape = _optional_float(extra.get("shape_score"))
        pct = _optional_float(extra.get("pct_retained", extra.get("cci_pct_retained")))

        if shape is not None and shape < _SUSPICIOUS_SHAPE_THRESHOLD:
            return "low_shape", f"low shape similarity ({shape:.3f} < 0.2)"
        if pct is not None and pct < _SUSPICIOUS_CCI_RETENTION_PCT:
            return "low_cci_retention", f"low CCI retention ({pct:.1f}% < 20%)"
        return "", ""

    chunks = []
    chunks.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QC Report — """
        + subject_id
        + """</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #f4f4f5; color: #27272a; }
    h1 { color: #18181b; border-bottom: 1px solid #d4d4d8; padding-bottom: 0.5rem; }
    h2 { color: #1e40af; margin-top: 2rem; }
    h3 { color: #166534; margin-top: 1.5rem; }
    .tract { margin-bottom: 2.5rem; padding: 1rem; border: 1px solid #d4d4d8; border-radius: 8px; background: #ffffff; }
    .tract.suspicious { border-color: #ca8a04; box-shadow: 0 0 0 1px rgba(202, 138, 4, 0.25) inset; }
    .section { margin: 1rem 0; }
    .section h4 { color: #a16207; margin: 0.5rem 0; }
    .grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
    .grid img { max-width: 320px; height: auto; border: 1px solid #d4d4d8; border-radius: 4px; }
    .grid .cell { flex: 0 0 auto; }
    .score { font-size: 1.1rem; color: #15803d; }
    .flag { display: inline-block; margin-left: 0.5rem; padding: 0.15rem 0.45rem; border-radius: 999px; background: #fef3c7; color: #92400e; font-size: 0.8rem; vertical-align: middle; }
    .flag-note { color: #a16207; margin-top: 0.25rem; }
    .note { margin: 0.75rem 0 1rem; padding: 0.75rem 1rem; border: 1px solid #d4d4d8; border-radius: 8px; background: #ffffff; color: #3f3f46; }
    .note strong { color: #18181b; }
    .note table { border-collapse: collapse; margin-top: 0.5rem; }
    .note th, .note td { padding: 0.35rem 0.65rem; text-align: left; border: 1px solid #d4d4d8; }
    .note th { background: #f4f4f5; color: #1e40af; }
  </style>
</head>
<body>
  <h1>QC Report: """
        + subject_id
        + """</h1>
""",
    )

    run_info = results.get("_run_info") if isinstance(results, dict) else None
    if isinstance(run_info, dict) and run_info:
        chunks.append('  <div class="note">')
        chunks.append("    <p><strong>Run settings</strong></p>")
        chunks.append("    <table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>")
        for k, v in run_info.items():
            chunks.append(f"      <tr><td>{k}</td><td>{v}</td></tr>")
        chunks.append("    </tbody></table>")
        chunks.append("  </div>")

    # Summary of flagged tracts for this subject
    flagged_tracts = []
    if isinstance(results, dict):
        for tract_name in results:
            tract_data = results.get(tract_name)
            if not isinstance(tract_data, (list, tuple)) or len(tract_data) < _TRACT_RESULT_MIN_LEN:
                continue
            extra = tract_data[2]
            flag_type, _reason = suspicious_flags(extra)
            if flag_type:
                flagged_tracts.append((tract_name, extra))

    def _flagged_tract_summary(name: str, extra: Any) -> str:
        score_bits = []
        shape = _optional_float(extra.get("shape_score"))
        if shape is not None:
            score_bits.append(f"shape {shape:.3f}")
        pct = _optional_float(extra.get("pct_retained", extra.get("cci_pct_retained")))
        if pct is not None:
            score_bits.append(f"retention {pct:.1f}%")
        if score_bits:
            return f"{name} ({', '.join(score_bits)})"
        return name

    if flagged_tracts:
        flagged_tracts.sort(key=lambda kv: tract_sort_key(kv[0]))
        tract_names = ", ".join(_flagged_tract_summary(name, extra) for name, extra in flagged_tracts)
        chunks.append('  <div class="note">')
        chunks.append(
            f"    <p><strong>Flagged tracts:</strong> {len(flagged_tracts)} "
            f"(shape similarity &lt; 0.2 and/or CCI retention &lt; 20%) — {tract_names}</p>",
        )
        chunks.append("  </div>")

    subject_anatomy = results.get("_subject_anatomy")
    if isinstance(subject_anatomy, dict) and subject_anatomy:
        chunks.append('  <div class="section"><h2>Subject anatomy</h2>')
        chunks.append("    <p>All bundles overlaid on orthogonal T1 slices through brain center.</p>")
        chunks.append('    <div class="grid">')
        for view in ["coronal", "axial", "sagittal"]:
            p = subject_anatomy.get(view)
            if p:
                chunks.append(
                    f'      <div class="cell"><img src="{rel(p)}" alt="anatomy {view}" title="Subject anatomy {view}"/></div>',
                )
        chunks.append("    </div></div>")

    for tract_name in sorted(results.keys(), key=tract_sort_key):
        tract_data = results.get(tract_name)
        if not isinstance(tract_data, (list, tuple)) or len(tract_data) < _TRACT_RESULT_MIN_LEN:
            continue
        # subject_views, cci_plots, dti_res, noddi_res, extra = tract_data[0], tract_data[1], tract_data[2], tract_data[3], tract_data[4]
        subject_views, cci_plots, extra = tract_data[0], tract_data[1], tract_data[2]
        shape_score = extra.get("shape_score", "")
        flag_type, suspicious_reason = suspicious_flags(extra)
        is_suspicious = bool(flag_type)

        tract_class = "tract suspicious" if is_suspicious else "tract"
        chunks.append(f'  <div class="{tract_class}">')
        if is_suspicious:
            chunks.append(f'    <h2>{tract_name}<span class="flag">Suspicious</span></h2>')
            chunks.append(f'    <p class="flag-note">Flag reason: {suspicious_reason}</p>')
        else:
            chunks.append(f"    <h2>{tract_name}</h2>")
        if shape_score != "":
            chunks.append(f'    <p class="score">Shape similarity score: {shape_score}</p>')

        # Tract views (coronal, axial, sagittal)
        chunks.append('    <div class="section"><h4>Tract views</h4><div class="grid">')
        if isinstance(subject_views, dict):
            for view in ["coronal", "axial", "sagittal"]:
                p = subject_views.get(view)
                if p:
                    chunks.append(f'      <div class="cell"><img src="{rel(p)}" alt="{view}" title="{view}"/></div>')
        chunks.append("    </div></div>")

        t1_views = extra.get("t1_views") if isinstance(extra, dict) else None
        if isinstance(t1_views, dict) and t1_views:
            chunks.append('    <div class="section"><h4>T1 alignment</h4><div class="grid">')
            for view in ["coronal", "axial", "sagittal"]:
                p = t1_views.get(view)
                if p:
                    chunks.append(
                        f'      <div class="cell"><img src="{rel(p)}" alt="T1 {view}" title="T1 alignment {view}"/></div>',
                    )
            chunks.append("    </div></div>")

        # Atlas views (same views for the matching atlas bundle, stored in tract folder)
        atlas_views = extra.get("atlas_views") if isinstance(extra, dict) else None
        if isinstance(atlas_views, dict):
            chunks.append('    <div class="section"><h4>Atlas views</h4><div class="grid">')
            for view in ["coronal", "axial", "sagittal"]:
                p = atlas_views.get(view)
                if p:
                    chunks.append(
                        f'      <div class="cell"><img src="{rel(p)}" alt="atlas {view}" title="Atlas {view}"/></div>',
                    )
            chunks.append("    </div></div>")

        # CCI: histogram + before/after views
        if isinstance(cci_plots, dict):
            chunks.append('    <div class="section"><h4>CCI</h4>')
            cci_stats = cci_plots.get("cci_stats") or {}
            if (
                cci_stats
                and isinstance(cci_stats.get("n_long"), (int, float))
                and isinstance(cci_stats.get("pct_retained"), (int, float))
            ):
                n_long = cci_stats["n_long"]
                n_keep = cci_stats.get("n_keep", "")
                pct = cci_stats["pct_retained"]
                mean_cci = cci_stats.get("mean_cci", "")
                try:
                    mean_s = (
                        f"; mean CCI: {float(mean_cci):.3f}"
                        if mean_cci is not None and not np.isnan(float(mean_cci))
                        else ""
                    )
                except (TypeError, ValueError):
                    mean_s = ""
                chunks.append(
                    f'    <p class="score">Long streamlines: {n_long} → retained (CCI threshold): {n_keep} ({pct:.1f}%){mean_s}</p>',
                )
            chunks.append('<div class="grid">')
            hist = cci_plots.get("histogram")
            if hist:
                chunks.append(
                    f'      <div class="cell"><img src="{rel(hist)}" alt="CCI histogram" title="CCI histogram"/></div>',
                )
            for view in ["coronal", "axial", "sagittal"]:
                pair = cci_plots.get(view)
                if isinstance(pair, dict):
                    for label, p in [("before", pair.get("before")), ("after", pair.get("after"))]:
                        if p:
                            chunks.append(
                                f'      <div class="cell"><img src="{rel(p)}" alt="CCI {label} {view}" title="CCI {label} {view}"/></div>',
                            )
            chunks.append("    </div></div>")

        # DKI (profile + views)
        # if isinstance(dki_res, dict):
        #    chunks.append('    <div class="section"><h4>DKI</h4><div class="grid">')
        #    pp = dki_res.get("profile_plot")
        #    if pp:
        #        chunks.append(f'      <div class="cell"><img src="{rel(pp)}" alt="DKI profile" title="DKI profile"/></div>')
        #    for view in ["coronal", "axial", "sagittal"]:
        #        p = dki_res.get(view)
        #        if p:
        #            chunks.append(f'      <div class="cell"><img src="{rel(p)}" alt="DKI {view}" title="DKI {view}"/></div>')
        #    chunks.append("    </div></div>")

        # DTI (profile + views)
        # if isinstance(dti_res, dict):
        #    chunks.append('    <div class="section"><h4>DTI</h4><div class="grid">')
        #    pp = dti_res.get("profile_plot")
        #    if pp:
        #        chunks.append(f'      <div class="cell"><img src="{rel(pp)}" alt="DTI profile" title="DTI profile"/></div>')
        #    for view in ["coronal", "axial", "sagittal"]:
        #        p = dti_res.get(view)
        #        if p:
        #            chunks.append(f'      <div class="cell"><img src="{rel(p)}" alt="DTI {view}" title="DTI {view}"/></div>')
        #    chunks.append("    </div></div>")

        # NODDI (profile + views)
        # if isinstance(noddi_res, dict):
        #    chunks.append('    <div class="section"><h4>NODDI</h4><div class="grid">')
        #    pp = noddi_res.get("profile_plot")
        #    if pp:
        #        chunks.append(f'      <div class="cell"><img src="{rel(pp)}" alt="NODDI profile" title="NODDI profile"/></div>')
        #    for view in ["coronal", "axial", "sagittal"]:
        #        p = noddi_res.get(view)
        #        if p:
        #            chunks.append(f'      <div class="cell"><img src="{rel(p)}" alt="NODDI {view}" title="NODDI {view}"/></div>')
        #    chunks.append("    </div></div>")

        chunks.append("  </div>")

    chunks.append("</body>\n</html>")

    html = "\n".join(chunks)
    output_html_path.write_text(html, encoding="utf-8")
    return str(output_html_path)


def _suspicious_flag_type(extra: Any) -> str:
    """Group-level mirror of subject HTML suspicious heuristic.

    Returns a flag type string, or "" if not flagged.
    """
    if not isinstance(extra, dict):
        return ""

    shape = _optional_float(extra.get("shape_score"))
    pct = _optional_float(extra.get("pct_retained", extra.get("cci_pct_retained")))

    if shape is not None and shape < _SUSPICIOUS_SHAPE_THRESHOLD:
        return "low_shape"
    if pct is not None and pct < _SUSPICIOUS_CCI_RETENTION_PCT:
        return "low_cci_retention"
    return ""


def collect_group_qc_data(group_qc_dir: str | Path) -> GroupQcData:
    """
    Scan a group QC directory and aggregate per-subject outputs.

    Returns a dict with subject_rows, suspicious_by_subject, tract_stats_rows,
    profile_values, tract_scores, tract_flag_stats, total_seen, total_flagged.
    """
    group_qc_dir = Path(group_qc_dir)
    if not group_qc_dir.is_dir():
        raise FileNotFoundError(f"Group QC directory not found: {group_qc_dir}")

    subject_rows = []
    suspicious_by_subject = {}
    tract_stats_rows = []
    profile_values: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    tract_scores: dict[str, dict[str, float]] = {}
    tract_flag_stats: dict[str, dict[str, int]] = {}
    total_seen = 0
    total_flagged = 0
    for subdir in sorted(group_qc_dir.iterdir()):
        if not subdir.is_dir():
            continue
        qc_json = subdir / "qc_data.json"
        qc_html = subdir / "qc_report.html"
        tract_stats_csv = subdir / "tract_stats.csv"
        profile_csv = subdir / "profile_data.csv"
        report_path = None
        if qc_html.exists() or qc_json.exists():
            report_path = subdir.name + "/qc_report.html"
        if report_path is None:
            continue
        subject_id = subdir.name
        n_tracts = ""
        n_suspicious = 0
        if qc_json.exists():
            try:
                with open(qc_json) as f:
                    data = json.load(f)
                subj_key = next(iter(data.keys())) if data else subject_id
                results = data.get(subj_key, {})
                if isinstance(results, dict):
                    # Count tracts with full data
                    valid_tracts = [
                        k
                        for k, v in results.items()
                        if isinstance(v, (list, tuple)) and len(v) >= _TRACT_RESULT_MIN_LEN
                    ]
                    n_tracts = str(len(valid_tracts))
                    # Extract shape similarity scores for matrix
                    for tract_name, tract_data in results.items():
                        if not (isinstance(tract_data, (list, tuple)) and len(tract_data) >= _TRACT_RESULT_MIN_LEN):
                            continue
                        extra = tract_data[2]
                        if not isinstance(extra, dict):
                            continue
                        # Suspicious bundle accounting
                        flag_type = _suspicious_flag_type(extra)
                        flagged = bool(flag_type)
                        stats = tract_flag_stats.setdefault(tract_name, {"flagged": 0, "seen": 0})
                        stats["seen"] += 1
                        total_seen += 1
                        if flagged:
                            stats["flagged"] += 1
                            n_suspicious += 1
                            total_flagged += 1
                        shape_score = extra.get("shape_score", None)
                        parsed_shape = _optional_float(shape_score)
                        tract_scores.setdefault(tract_name, {})[subject_id] = (
                            np.nan if parsed_shape is None else parsed_shape
                        )
            except (json.JSONDecodeError, IndexError, TypeError, OSError):
                pass
        suspicious_by_subject[subject_id] = int(n_suspicious)
        subject_rows.append((subject_id, n_tracts, n_suspicious, report_path))

        # Parse tract_stats.csv if present (this is the primary source for group stats)
        if tract_stats_csv.exists():
            try:
                with open(tract_stats_csv, newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if not isinstance(row, dict):
                            continue
                        tract = row.get("tract", "")
                        if not tract:
                            continue
                        tract_stats_rows.append({"subject": subject_id, **row})
            except OSError:
                pass

        # Parse profile_data.csv if present (node-wise values)
        if profile_csv.exists():
            try:
                with open(profile_csv, newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if not isinstance(row, dict):
                            continue
                        tract = row.get("tract", "")
                        metric = row.get("metric", "")
                        if not tract or not metric:
                            continue
                        try:
                            node = int(float(row.get("node", "")))
                            val = float(row.get("value", ""))
                        except (TypeError, ValueError):
                            continue
                        key = (tract, metric)
                        subj_map = profile_values.setdefault(key, {})
                        node_map = subj_map.setdefault(subject_id, {})
                        # if duplicates exist, last one wins
                        node_map[node] = val
            except OSError:
                pass

    return {
        "subject_rows": subject_rows,
        "suspicious_by_subject": suspicious_by_subject,
        "tract_stats_rows": tract_stats_rows,
        "profile_values": profile_values,
        "tract_scores": tract_scores,
        "tract_flag_stats": tract_flag_stats,
        "total_seen": total_seen,
        "total_flagged": total_flagged,
    }


def write_combined_group_csvs(
    group_qc_dir: str | Path,
    tract_stats_rows: list[dict[str, Any]],
    profile_values: dict[tuple[str, str], dict[str, dict[int, float]]],
) -> None:
    """Write combined_tract_stats.csv and combined_profile_data.csv under group_qc_dir."""
    group_qc_dir = Path(group_qc_dir)
    if tract_stats_rows:
        try:
            combined_path = group_qc_dir / "combined_tract_stats.csv"
            all_keys: set[str] = set()
            for row in tract_stats_rows:
                all_keys.update(row.keys())
            base_first = ["subject", "tract"]
            other = sorted([k for k in all_keys if k not in base_first])
            header = base_first + other
            with open(combined_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                for row in tract_stats_rows:
                    out_row = {k: row.get(k, "") for k in header}
                    w.writerow(out_row)
        except OSError:
            pass

    if profile_values:
        try:
            combined_prof_path = group_qc_dir / "combined_profile_data.csv"
            with open(combined_prof_path, "w", newline="") as f:
                prof_writer = csv.writer(f)
                prof_writer.writerow(["subject", "tract", "metric", "node", "value"])
                for tract, metric in sorted(profile_values.keys(), key=lambda tm: (tm[0], tm[1])):
                    subj_map = profile_values[(tract, metric)]
                    for subject_id in sorted(subj_map.keys()):
                        node_map = subj_map[subject_id]
                        for node in sorted(node_map.keys()):
                            val = node_map[node]
                            try:
                                prof_writer.writerow([subject_id, tract, metric, int(node), float(val)])
                            except (TypeError, ValueError):
                                continue
        except OSError:
            pass


def generate_group_summary_html(
    group_qc_dir: str | Path,
    output_html_path: str | Path | None = None,
    *,
    pairwise_shape: bool = True,
    pairwise_shape_max_streamlines: int = 4000,
    pairwise_shape_jobs: int = 1,
) -> str:
    """
    Generate a group summary HTML page with links to each subject's QC report.

    Scans group_qc_dir for subdirectories that contain qc_data.json (or qc_report.html),
    then builds a QC triage page: flagged-bundle summaries, shape-similarity diagnostics,
    and a subject index. Also writes combined CSVs and group_interactive.html for
    interactive tract/profile exploration (replaces static boxplots and profile ribbons).

    group_qc_dir: path to the parent QC directory (e.g. .../derivatives/QC) containing
                  one subdir per subject, each with qc_data.json and qc_report.html.
    output_html_path: where to write the summary HTML (default: group_qc_dir/group_summary.html)
    pairwise_shape: if True, compute subjectxsubject shape similarity from bundle .trk files
                    under each subject's tract subfolders: one matrix per tract (BUAN-style)
                    plus a summary matrix averaged across tracts.
    pairwise_shape_max_streamlines: cap passed to pairwise shape (0 or None = no cap).
    pairwise_shape_jobs: parallel tract workers for pairwise shape (1 = serial). Each worker
                          loads one tract's bundles once per subject then scores all pairs.
    """
    group_qc_dir = Path(group_qc_dir)
    if output_html_path is None:
        output_html_path = group_qc_dir / "group_summary.html"
    output_html_path = Path(output_html_path)

    collected = collect_group_qc_data(group_qc_dir)
    subject_rows = collected["subject_rows"]
    suspicious_by_subject = collected["suspicious_by_subject"]
    tract_stats_rows = collected["tract_stats_rows"]
    profile_values = collected["profile_values"]
    tract_scores = collected["tract_scores"]
    tract_flag_stats = collected["tract_flag_stats"]
    total_seen = collected["total_seen"]
    total_flagged = collected["total_flagged"]

    write_combined_group_csvs(group_qc_dir, tract_stats_rows, profile_values)

    interactive_rel = "group_interactive.html"
    interactive_written = False
    try:
        from pytractoviz.group_interactive_report import generate_group_interactive_html  # noqa: PLC0415

        generate_group_interactive_html(
            [(group_qc_dir.name, group_qc_dir)],
            group_qc_dir / interactive_rel,
        )
        interactive_written = True
    except (ImportError, OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Could not generate group interactive report: %s", exc)

    chunks = []
    chunks.append("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QC Group Summary</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #f4f4f5; color: #27272a; }
    h1 { color: #18181b; border-bottom: 1px solid #d4d4d8; padding-bottom: 0.5rem; }
    h2 { color: #1e40af; }
    table { border-collapse: collapse; margin-top: 1rem; }
    th, td { padding: 0.5rem 1rem; text-align: left; border: 1px solid #d4d4d8; }
    th { background: #f4f4f5; color: #1e40af; }
    tr:nth-child(even) { background: #fafafa; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .count { color: #52525b; }
  </style>
</head>
<body>
  <h1>QC Group Summary</h1>
  <p>Subject-level QC reports. Click a subject to open their full report.</p>
""")
    if interactive_written:
        chunks.append(
            f"""  <p><a href="{interactive_rel}">Open interactive stats explorer</a> """
            """— tract metrics, profiles, and group comparisons from combined CSV data.</p>
""",
        )

    chunks.append(
        """
  <p><strong>Flagged bundles:</strong> """
        + str(total_flagged)
        + """ / """
        + str(total_seen)
        + """ evaluated bundles (shape similarity &lt; 0.2 and/or CCI retention &lt; 20%)</p>
""",
    )

    # Bar chart: suspicious bundle counts per subject
    if suspicious_by_subject:
        try:
            subj_ids = sorted(suspicious_by_subject.keys())
            counts = [int(suspicious_by_subject[sid]) for sid in subj_ids]
            fig, ax = plt.subplots(1, figsize=(max(8, 0.35 * len(subj_ids)), 4))
            ax.bar(np.arange(len(subj_ids)), counts, color="#a36b3a", alpha=0.85)
            ax.set_xticks(np.arange(len(subj_ids)))
            ax.set_xticklabels(subj_ids, rotation=90)
            ax.set_ylabel("Suspicious bundles (count)")
            ax.set_title("Suspicious bundle counts per subject")
            ax.grid(axis="y", color="#333333", linestyle="--", linewidth=0.6, alpha=0.7)
            fig.tight_layout()
            suspicious_path = group_qc_dir / "suspicious_counts_per_subject.png"
            fig.savefig(suspicious_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            chunks.append(
                """  <h2>Suspicious bundle counts</h2>
  <p>Number of tracts flagged as suspicious per subject.</p>
  <p><img src="suspicious_counts_per_subject.png" alt="Suspicious counts per subject" style="max-width: 100%; height: auto; border: 1px solid #d4d4d8; border-radius: 4px;" /></p>""",
            )
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.warning("Failed to write suspicious bundle counts chart: %s", exc)

    if tract_flag_stats:
        tract_rows = sorted(
            tract_flag_stats.items(),
            key=lambda kv: (
                -(kv[1]["flagged"] / kv[1]["seen"] if kv[1]["seen"] else 0.0),
                -kv[1]["flagged"],
                kv[0],
            ),
        )
        chunks.append("""
  <h2>Suspicious frequency by tract</h2>
  <p>How often each tract was flagged as suspicious across available subjects.</p>
  <table>
    <thead><tr><th>Tract</th><th>Flagged</th><th>Seen</th><th>Flag rate</th></tr></thead>
    <tbody>
""")
        for tract_name, stats in tract_rows:
            seen = stats["seen"]
            flagged = stats["flagged"]
            rate = (100.0 * flagged / seen) if seen else 0.0
            chunks.append(
                f'      <tr><td>{tract_name}</td><td class="count">{flagged}</td><td class="count">{seen}</td><td class="count">{rate:.1f}%</td></tr>',
            )
        chunks.append("""    </tbody>
  </table>
""")

    # Shape similarity heatmap (subject vs atlas); distributions live in interactive report.
    if tract_scores:
        # Stable ordering for axes
        all_tracts = sorted(tract_scores.keys())
        all_subjects = sorted({sid for tdict in tract_scores.values() for sid in tdict})
        if all_tracts and all_subjects:
            mat = np.full((len(all_subjects), len(all_tracts)), np.nan, dtype=float)
            for j, tract_name in enumerate(all_tracts):
                scores_by_subj = tract_scores.get(tract_name, {})
                for i, sid in enumerate(all_subjects):
                    val = scores_by_subj.get(sid, np.nan)
                    try:
                        mat[i, j] = float(val)
                    except (TypeError, ValueError):
                        mat[i, j] = np.nan
            # Plot heatmap; NaNs will appear as a separate color
            fig, ax = plt.subplots(1, figsize=(max(6, 0.4 * len(all_tracts)), max(4, 0.3 * len(all_subjects))))
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="#444444")  # dark gray for missing
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_xticks(np.arange(len(all_tracts)))
            ax.set_xticklabels(all_tracts, rotation=90)
            ax.set_yticks(np.arange(len(all_subjects)))
            ax.set_yticklabels(all_subjects)
            ax.set_xlabel("Tract")
            ax.set_ylabel("Subject")
            ax.set_title("Shape similarity (subject vs atlas)")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Shape similarity score")
            fig.tight_layout()
            matrix_path = group_qc_dir / "shape_similarity_matrix.png"
            fig.savefig(matrix_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Add section with the generated matrix image
            chunks.append("""  <h2>Shape similarity across subjects and tracts</h2>
  <p>Heatmap of shape similarity scores (subject bundles vs. atlas bundles) for all subjects and tracts with available data.</p>
  <p><img src="shape_similarity_matrix.png" alt="Shape similarity matrix" style="max-width: 100%; height: auto; border: 1px solid #d4d4d8; border-radius: 4px;" /></p>""")

            # Subject-subject bundle shape similarity: per-tract (BUAN-style) + mean across tracts
            if pairwise_shape and len(all_subjects) >= _MIN_PAIRWISE_SHAPE_SUBJECTS:
                try:
                    n_sub = len(all_subjects)
                    pair_sum = np.zeros((n_sub, n_sub), dtype=float)
                    pair_count = np.zeros((n_sub, n_sub), dtype=int)
                    by_tract_dir = group_qc_dir / "shape_similarity_subject_by_tract"
                    by_tract_dir.mkdir(parents=True, exist_ok=True)
                    per_tract_sections = []

                    tract_tasks = []
                    for tract_name in all_tracts:
                        paths = [find_qc_bundle_trk(group_qc_dir / sid, tract_name) for sid in all_subjects]
                        path_strs = [str(p.resolve()) if p is not None and p.exists() else None for p in paths]
                        tract_tasks.append(
                            (tract_name, path_strs, pairwise_shape_max_streamlines),
                        )

                    nj = max(1, int(pairwise_shape_jobs))
                    tract_results = []
                    if nj <= 1:
                        for tract_name, path_strs, msl in tract_tasks:
                            path_list = [Path(p) if p else None for p in path_strs]
                            cap = _normalize_streamline_cap(msl)
                            mat_t, had_off = _compute_pairwise_shape_one_tract(
                                path_list,
                                n_sub,
                                cap,
                            )
                            tract_results.append((tract_name, mat_t, had_off))
                    else:
                        mp_ctx = mp.get_context("spawn")
                        with ProcessPoolExecutor(max_workers=nj, mp_context=mp_ctx) as ex:
                            futures = [ex.submit(_pairwise_shape_tract_worker, t) for t in tract_tasks]
                            for fut in as_completed(futures):
                                tract_results.append(fut.result())
                    tract_results.sort(key=lambda r: r[0])

                    for tract_name, mat_t, had_off_diagonal in tract_results:
                        if not had_off_diagonal:
                            continue
                        for ii in range(n_sub):
                            for jj in range(ii + 1, n_sub):
                                v = mat_t[ii, jj]
                                if np.isfinite(v):
                                    pair_sum[ii, jj] += float(v)
                                    pair_sum[jj, ii] += float(v)
                                    pair_count[ii, jj] += 1
                                    pair_count[jj, ii] += 1

                        safe = _safe_tract_filename(tract_name)
                        stem = by_tract_dir / safe
                        save_subject_subject_shape_matrix_assets(
                            stem,
                            mat_t,
                            all_subjects,
                            title=f"Shape similarity (pairwise): {tract_name}",
                            cbar_label="Shape similarity",
                        )
                        rel_png = f"shape_similarity_subject_by_tract/{safe}.png"
                        rel_csv = f"shape_similarity_subject_by_tract/{safe}.csv"
                        per_tract_sections.append(
                            (
                                tract_name,
                                safe,
                                f"""  <h3 id="pairwise-tract-{safe}">{tract_name}</h3>
  <p><img src="{rel_png}" alt="Pairwise shape similarity {tract_name}" style="max-width: 100%; height: auto; border: 1px solid #d4d4d8; border-radius: 4px;" /></p>
  <p><code>{rel_csv}</code></p>""",
                            ),
                        )

                    subj_mat = np.full((n_sub, n_sub), np.nan, dtype=float)
                    np.fill_diagonal(subj_mat, 1.0)
                    for ii in range(n_sub):
                        for jj in range(n_sub):
                            if ii == jj:
                                continue
                            c = pair_count[ii, jj]
                            if c > 0:
                                subj_mat[ii, jj] = pair_sum[ii, jj] / float(c)

                    save_subject_subject_shape_matrix_assets(
                        group_qc_dir / "shape_similarity_subject_matrix",
                        subj_mat,
                        all_subjects,
                        title="Shape similarity: subject vs subject (mean across tracts)",
                        cbar_label="Mean shape similarity",
                    )

                    intro = """  <h2>Subject-subject shape similarity</h2>
  <p>Pairwise bundle shape similarity using DIPY <code>bundle_shape_similarity</code> (same family of scores as BUAN). Bundles are read from <code>&lt;subject&gt;/&lt;tract&gt;/</code> (preferring <code>*_cci_keep.trk</code>). The diagonal is 1. Off-diagonal cells are gray when that pair could not be compared for any tract.</p>
  <h3>Mean across tracts</h3>
  <p>Each off-diagonal entry averages the pairwise score over tracts where both subjects had a <code>.trk</code> file.</p>
  <p><img src="shape_similarity_subject_matrix.png" alt="Mean subject-subject shape similarity" style="max-width: 100%; height: auto; border: 1px solid #d4d4d8; border-radius: 4px;" /></p>
  <p><code>shape_similarity_subject_matrix.csv</code></p>"""

                    if per_tract_sections:
                        toc_items = "".join(
                            f'    <li><a href="#pairwise-tract-{safe}">{tract_name}</a></li>\n'
                            for tract_name, safe, _sec in per_tract_sections
                        )
                        per_block = (
                            """  <h3>Per-tract pairwise matrices (BUAN-style)</h3>
  <p>One subjectxsubject matrix per bundle—only the tract shown is used for that heatmap.</p>
  <ul>
"""
                            + toc_items
                            + """  </ul>
"""
                            + "\n".join(sec for _tn, _sf, sec in per_tract_sections)
                        )
                    else:
                        per_block = (
                            "  <p><em>No per-tract pairwise matrices were written "
                            "(need bundle <code>.trk</code> files under each subject's tract folders).</em></p>"
                        )

                    chunks.append(intro + "\n" + per_block)
                except (OSError, ValueError, RuntimeError, TypeError) as exc:
                    logger.warning("Failed to generate pairwise shape similarity matrices: %s", exc)

    chunks.append("""
  <h2>Subjects</h2>
  <table>
    <thead><tr><th>Subject</th><th>Tracts</th><th>Flagged</th><th>Report</th></tr></thead>
    <tbody>
""")
    for subject_id, n_tracts, n_flagged, report_path in subject_rows:
        link = (
            f'      <tr><td>{subject_id}</td><td class="count">{n_tracts}</td>'
            f'<td class="count">{n_flagged}</td>'
            f'<td><a href="{report_path}">Open report</a></td></tr>'
        )
        chunks.append(link)

    chunks.append("""    </tbody>
  </table>
""")

    chunks.append("""
</body>
</html>""")

    html = "\n".join(chunks)
    output_html_path.write_text(html, encoding="utf-8")
    return str(output_html_path)


def _streamlines_for_shape_metric(
    tract_file: str | Path,
    max_streamlines: int | None = None,
    subsample_seed: int = 0,
    nb_points: int = 20,
    min_streamlines: int = 10,
) -> Streamlines | None:
    """Load fixed-point streamlines for bundle_shape_similarity.

    Returns None if the bundle is too small.
    """
    tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
    tract.to_rasmm()
    tract_streamlines = _subsample_streamlines(
        tract.streamlines,
        max_streamlines=max_streamlines,
        seed=subsample_seed,
    )
    tract_streamlines = downsample_streamlines(tract_streamlines)
    if len(tract_streamlines) < min_streamlines:
        return None
    return set_number_of_points(tract_streamlines, nb_points=nb_points)


def calc_shape_similarity(
    tract_file: str | Path,
    atlas_file: str | Path,
    clust_thr: tuple[float, float, float] = (5, 3, 1.5),
    threshold: float = 6,
    nb_points: int = 20,
    min_streamlines: int = 10,
    max_streamlines: int | None = None,
) -> float:
    """
    Bundle shape similarity (DIPY BUAN/BA metric).

    Notes
    -----
    - Resampling to a fixed number of points improves stability of streamline-space
      comparisons/registration. Many DIPY bundle registration workflows assume
      fixed-point streamlines (often 20).
    - Very small bundles can yield unstable/degenerate scores; we return NaN to
      flag "insufficient data" rather than reporting an apparently meaningful 0.
    """
    tract_streamlines = _streamlines_for_shape_metric(
        tract_file,
        max_streamlines=max_streamlines,
        subsample_seed=31,
        nb_points=nb_points,
        min_streamlines=min_streamlines,
    )
    atlas_streamlines = _streamlines_for_shape_metric(
        atlas_file,
        max_streamlines=max_streamlines,
        subsample_seed=37,
        nb_points=nb_points,
        min_streamlines=min_streamlines,
    )
    if tract_streamlines is None or atlas_streamlines is None:
        return float("nan")

    similarity_score = bundle_shape_similarity(
        tract_streamlines,
        atlas_streamlines,
        rng=np.random.default_rng(),
        clust_thr=clust_thr,
        threshold=threshold,
    )

    return float(similarity_score)


def calc_shape_similarity_pair(
    tract_file_a: str | Path,
    tract_file_b: str | Path,
    clust_thr: tuple[float, float, float] = (5, 3, 1.5),
    threshold: float = 6,
    nb_points: int = 20,
    min_streamlines: int = 10,
    max_streamlines: int | None = None,
) -> float:
    """
    Shape similarity between two subject bundles (same DIPY metric as vs-atlas).

    Uses the same subsample seed for both bundles so pairwise comparisons are
    comparable; only unordered pairs need to be computed (metric is symmetric in
    practice for a fixed RNG).
    """
    a = _streamlines_for_shape_metric(
        tract_file_a,
        max_streamlines=max_streamlines,
        subsample_seed=41,
        nb_points=nb_points,
        min_streamlines=min_streamlines,
    )
    b = _streamlines_for_shape_metric(
        tract_file_b,
        max_streamlines=max_streamlines,
        subsample_seed=41,
        nb_points=nb_points,
        min_streamlines=min_streamlines,
    )
    if a is None or b is None:
        return float("nan")
    similarity_score = bundle_shape_similarity(
        a,
        b,
        rng=np.random.default_rng(42),
        clust_thr=clust_thr,
        threshold=threshold,
    )
    return float(similarity_score)


def _compute_pairwise_shape_one_tract(
    path_list: Sequence[Path | None],
    n_sub: int,
    cap_pair: int | None,
    clust_thr: tuple[float, float, float] = (5, 3, 1.5),
    threshold: float = 6,
    nb_points: int = 20,
    min_streamlines: int = 10,
) -> tuple[NDArrayFloat, bool]:
    """
    One tract's subjectxsubject shape similarity matrix.

    Loads each subject's bundle once, then runs ``bundle_shape_similarity`` on
    every pair (avoids re-reading .trk for each pair).
    Returns (matrix, had_off_diagonal).
    """
    mat_t = np.full((n_sub, n_sub), np.nan, dtype=float)
    np.fill_diagonal(mat_t, 1.0)
    sls: list[Streamlines | None] = [None] * n_sub
    for i in range(n_sub):
        p = path_list[i]
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        try:
            sls[i] = _streamlines_for_shape_metric(
                p,
                max_streamlines=cap_pair,
                subsample_seed=41,
                nb_points=nb_points,
                min_streamlines=min_streamlines,
            )
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.debug("Failed to load streamlines for shape metric from %s: %s", p, exc)
            sls[i] = None

    had_off_diagonal = False
    for ii in range(n_sub):
        for jj in range(ii + 1, n_sub):
            a, b = sls[ii], sls[jj]
            if a is None or b is None:
                continue
            try:
                s = bundle_shape_similarity(
                    a,
                    b,
                    rng=np.random.default_rng(42),
                    clust_thr=clust_thr,
                    threshold=threshold,
                )
            except (ValueError, RuntimeError, TypeError) as exc:
                logger.debug(
                    "bundle_shape_similarity failed for subjects %d and %d: %s",
                    ii,
                    jj,
                    exc,
                )
                continue
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            fs = float(s)
            mat_t[ii, jj] = fs
            mat_t[jj, ii] = fs
            had_off_diagonal = True
    return mat_t, had_off_diagonal


def _pairwise_shape_tract_worker(task: PairwiseShapeTask) -> tuple[str, NDArrayFloat, bool]:
    """Picklable worker: (tract_name, path_strs, max_streamlines_raw) -> (tract_name, mat, had)."""
    tract_name, path_strs, max_streamlines_raw = task
    path_list = [Path(p) if p else None for p in path_strs]
    n_sub = len(path_list)
    cap = _normalize_streamline_cap(max_streamlines_raw)
    mat_t, had = _compute_pairwise_shape_one_tract(path_list, n_sub, cap)
    return tract_name, mat_t, had


def find_qc_bundle_trk(subject_dir: str | Path, tract_name: str) -> Path | None:
    """
    Locate a bundle .trk under ``subject_dir/tract_name/``.

    Prefers ``*_cci_keep.trk`` (written by the QC pipeline) so paths match completed
    runs; otherwise uses any ``*.trk`` in that folder.
    """
    subject_dir = Path(subject_dir)
    td = subject_dir / tract_name
    if not td.is_dir():
        return None
    keep = sorted(td.glob("*_cci_keep.trk"))
    if keep:
        return keep[0]
    trks = sorted(td.glob("*.trk"))
    return trks[0] if trks else None


def _safe_tract_filename(tract_name: str) -> str:
    """Filesystem-safe slug from a tract / bundle name."""
    s = re.sub(r"[^\w.\-]+", "_", str(tract_name))
    s = s.strip("._")
    return s or "tract"


def save_subject_subject_shape_matrix_assets(
    base_path: str | Path,
    matrix: NDArrayFloat,
    subject_labels: Sequence[str],
    title: str,
    cbar_label: str = "Shape similarity",
) -> None:
    """
    Write a subjectxsubject shape similarity heatmap (PNG) and numeric table (CSV).

    ``base_path`` is without extension (e.g. .../shape_similarity_subject_matrix).
    """
    base_path = Path(base_path)
    png_path = base_path.with_suffix(".png")
    csv_path = base_path.with_suffix(".csv")
    n_sub = len(subject_labels)
    fig, ax = plt.subplots(
        1,
        figsize=(max(5, 0.35 * n_sub), max(4, 0.35 * n_sub)),
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#444444")
    im = ax.imshow(
        matrix,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(n_sub))
    ax.set_yticks(np.arange(n_sub))
    ax.set_xticklabels(subject_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(subject_labels, fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *list(subject_labels)])
        for i, row in enumerate(np.asarray(matrix).tolist()):
            w.writerow(
                [subject_labels[i]]
                + [f"{v:.6g}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in row],
            )


def subject_bundle_key(stem: str) -> str:
    """Extract the short bundle name from a PYAFQ or BIDS-like tract filename stem."""
    s = str(stem)
    if "desc-" not in s:
        return s.split("_")[-1] if "_" in s else s
    after_desc = s.split("desc-", 1)[1]
    if "_tractography" in after_desc:
        after_desc = after_desc.split("_tractography")[0].rstrip("_")
    else:
        after_desc = after_desc.split("_")[0]
    return after_desc


def _process_subject_tract(task: SubjectTractTask) -> tuple[str, TractResult, list[ProfileRow]]:
    _configure_qc_logging()
    (
        sf,
        subject,
        output_dir,
        subj_ref_img,
        subj_t1_img,
        atlas_file,
        _atlas_ref_img,
        atlas_view_paths,
        dti_file_list,
        noddi_file_list,
        speed,
    ) = task
    tract_name = subject_bundle_key(Path(sf).stem)
    logger.info(">> Processing %s...", tract_name)
    tract_out_dir = f"{output_dir}/{subject}/{tract_name}"
    Path(tract_out_dir).mkdir(parents=True, exist_ok=True)

    streamlines_cache: StreamlineCache = {}
    ref_cache: RefCache = {}
    t1_cache: T1Cache = {}
    metric_cache: MetricCache = {}
    atlas_standard_cache: AtlasStandardCache = {}

    t0 = time.perf_counter()
    logger.info(">> Generating views...")
    subject_views = gen_views(
        str(sf),
        subj_ref_img,
        output_dir=tract_out_dir,
        streamlines_cache=streamlines_cache,
        ref_cache=ref_cache,
        max_streamlines_view=speed.get("max_streamlines_view"),
    )
    logger.info("   [total subject views: %.2fs]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info(">> Generating T1 alignment views...")
    t1_views = gen_t1_views(
        str(sf),
        subj_ref_img,
        subj_t1_img,
        output_dir=tract_out_dir,
        streamlines_cache=streamlines_cache,
        ref_cache=ref_cache,
        t1_cache=t1_cache,
        max_streamlines_view=speed.get("max_streamlines_view"),
    )
    logger.info("   [total T1 views: %.2fs]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    cci_plots = plot_before_after_cci(
        str(sf),
        subj_ref_img,
        output_dir=tract_out_dir,
        streamlines_cache=streamlines_cache,
        ref_cache=ref_cache,
        max_streamlines_cci=speed.get("max_streamlines_cci"),
    )
    logger.info("   [total CCI: %.2fs]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    shape_score = calc_shape_similarity(
        str(sf),
        str(atlas_file),
        max_streamlines=speed.get("max_streamlines_shape"),
    )
    logger.info("   [shape_similarity: %.2fs]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info(">> Generating metric AFQs...")
    profile_rows_local = []
    subj_streamlines = _get_downsampled_rasmm_streamlines(
        str(sf),
        streamlines_cache,
        max_streamlines=speed.get("max_streamlines_afq"),
        subsample_seed=11,
    )
    atlas_standard = _get_atlas_standard_streamline(str(atlas_file), atlas_standard_cache, streamlines_cache)
    for dti_file in dti_file_list:
        metric_str = dti_file.split("-")[-1].split("_")[0]
        profile = weighted_afq(
            subj_streamlines,
            atlas_standard,
            dti_file,
            metric_cache,
            max_streamlines=speed.get("max_streamlines_afq"),
        )
        logger.debug("      weighted_afq: %.2fs", time.perf_counter() - t0)
        profile_rows_local.append((tract_name, metric_str, profile))
    logger.info("    [total DTI AFQ: %.2fs]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for noddi_file in noddi_file_list:
        if "modulated" in noddi_file:
            metric_str = f"{noddi_file.split('-')[-2].split('_')[0]}_mod"
        else:
            metric_str = noddi_file.split("-")[-1].split("_")[0]
        profile = weighted_afq(
            subj_streamlines,
            atlas_standard,
            noddi_file,
            metric_cache,
            max_streamlines=speed.get("max_streamlines_afq"),
        )
        logger.debug("      weighted_afq: %.2fs", time.perf_counter() - t0)
        profile_rows_local.append((tract_name, metric_str, profile))
    logger.info("   [total NODDI AFQ: %.2fs]", time.perf_counter() - t0)

    keep_tract_path = cci_plots.get("keep_tract") if isinstance(cci_plots, dict) else None
    if keep_tract_path and Path(keep_tract_path).exists():
        keep_tract_name = f"{tract_name}_cci_keep"
        t0 = time.perf_counter()
        keep_streamlines = _get_downsampled_rasmm_streamlines(
            keep_tract_path,
            streamlines_cache,
            max_streamlines=speed.get("max_streamlines_afq"),
            subsample_seed=11,
        )
        for dti_file in dti_file_list:
            metric_str = dti_file.split("-")[-1].split("_")[0]
            profile = weighted_afq(
                keep_streamlines,
                atlas_standard,
                dti_file,
                metric_cache,
                max_streamlines=speed.get("max_streamlines_afq"),
            )
            profile_rows_local.append((keep_tract_name, metric_str, profile))
        for noddi_file in noddi_file_list:
            if "modulated" in noddi_file:
                metric_str = f"{noddi_file.split('-')[-2].split('_')[0]}_mod"
            else:
                metric_str = noddi_file.split("-")[-1].split("_")[0]
            profile = weighted_afq(
                keep_streamlines,
                atlas_standard,
                noddi_file,
                metric_cache,
                max_streamlines=speed.get("max_streamlines_afq"),
            )
            profile_rows_local.append((keep_tract_name, metric_str, profile))
        logger.info("   [CCI-keep tract profiles: %.2fs]", time.perf_counter() - t0)

    cci_stats = cci_plots.get("cci_stats", {}) if isinstance(cci_plots, dict) else {}
    extra = {
        "shape_score": shape_score,
        "atlas_views": atlas_view_paths,
        "t1_views": t1_views,
        **cci_stats,
    }
    if cci_stats:
        extra.setdefault("cci_n_long", cci_stats.get("n_long"))
        extra.setdefault("cci_n_keep", cci_stats.get("n_keep"))
        extra.setdefault("cci_pct_retained", cci_stats.get("pct_retained"))
        extra.setdefault("cci_mean", cci_stats.get("mean_cci"))

    return tract_name, [subject_views, cci_plots, extra], profile_rows_local


def main(
    subject: str,
    project: str,
    *,
    path_config: QCPathConfig | None = None,
    n_jobs: int = 1,
    render_backend: RenderBackend = "osmesa",
    max_streamlines_view: int = 5000,
    max_streamlines_cci: int = 5000,
    max_streamlines_afq: int = 6000,
    max_streamlines_shape: int = 4000,
) -> None:
    """Run the full per-subject QC pipeline and write CSV, JSON, and HTML outputs.

    Parameters
    ----------
    subject : str
        Subject identifier (for example ``sub-0000``).
    project : str
        Project or cohort name (for example ``UNC``).
    path_config : QCPathConfig, optional
        Filesystem layout. Defaults to :func:`pytractoviz.qc_config.load_qc_path_config`.
    n_jobs : int, optional
        Number of parallel tract workers.
    render_backend : str, optional
        ``"osmesa"`` or ``"egl"`` for offscreen VTK/FURY rendering.
    max_streamlines_view : int, optional
        Cap on streamlines used for view rendering (0 = no cap).
    max_streamlines_cci : int, optional
        Cap on streamlines used for CCI (0 = no cap).
    max_streamlines_afq : int, optional
        Cap on streamlines used for AFQ profiles (0 = no cap).
    max_streamlines_shape : int, optional
        Cap on streamlines used for shape similarity (0 = no cap).
    """
    _configure_qc_logging()
    cap_view = _normalize_streamline_cap(max_streamlines_view)
    cap_cci = _normalize_streamline_cap(max_streamlines_cci)
    cap_afq = _normalize_streamline_cap(max_streamlines_afq)
    cap_shape = _normalize_streamline_cap(max_streamlines_shape)

    os.environ.setdefault("VTK_STREAM_READER", "1")
    os.environ.setdefault("VTK_STREAM_WRITER", "1")
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
    if str(render_backend).lower() == "osmesa":
        os.environ.setdefault("VTK_USE_OSMESA", "1")
    else:
        os.environ.pop("VTK_USE_OSMESA", None)

    paths = path_config or load_qc_path_config()
    atlas_dir = paths.resolve_atlas_dir()
    atlas_ref_img = str(paths.resolve_atlas_ref_img())
    output_dir = str(paths.output_dir_for_project(project))
    atlas_views_dir = f"{output_dir}/atlas_views"
    Path(atlas_views_dir).mkdir(parents=True, exist_ok=True)

    anat_dir = paths.anat_dir(project, subject)
    dki_dir = paths.dki_dir(project, subject)
    noddi_dir = paths.noddi_dir(project, subject)

    subj_ref_img = str(anat_dir / f"{subject}_space-ACPC_desc-brain_mask.nii.gz")
    subj_t1_img = _resolve_t1_path(anat_dir, subject)
    if subj_t1_img:
        logger.info(">> Using T1 for anatomy overlays: %s", subj_t1_img)
    else:
        logger.warning(">> No T1 found under %s; T1 alignment views will be skipped", anat_dir)

    atlas_files = list(atlas_dir.glob("*trk"))
    subject_files = list(paths.pyafq_bundles_dir(project, subject).glob("*trk"))
    # dki_file = "acq-multiband_dir-AP_space-ACPC_model-dki_param"
    # dki_file_list = [
    #    f"{dki_dir}/{subject}_{dki_file}-ad_dwimap.nii.gz",
    #    f"{dki_dir}/{subject}_{dki_file}-rd_dwimap.nii.gz",
    #    f"{dki_dir}/{subject}_{dki_file}-kfa_dwimap.nii.gz",
    #    f"{dki_dir}/{subject}_{dki_file}-md_dwimap.nii.gz",
    #    f"{dki_dir}/{subject}_{dki_file}-mk_dwimap.nii.gz"
    # ]
    dti_file = "acq-multiband_dir-AP_space-ACPC_model"
    dti_file_list = [str(dki_dir / f"{subject}_{dti_file}-tensor_param-fa_dwimap.nii.gz")]
    noddi_file = "acq-multiband_dir-AP_space-ACPC_model-noddi_param"
    noddi_file_list = [
        str(noddi_dir / f"{subject}_{noddi_file}-icvf_dwimap.nii.gz"),
        str(noddi_dir / f"{subject}_{noddi_file}-isovf_dwimap.nii.gz"),
        str(noddi_dir / f"{subject}_{noddi_file}-od_dwimap.nii.gz"),
        str(noddi_dir / f"{subject}_{noddi_file}-icvf_desc-modulated_dwimap.nii.gz"),
        str(noddi_dir / f"{subject}_{noddi_file}-od_desc-modulated_dwimap.nii.gz"),
    ]

    results: SubjectResults = {}
    subject_results: dict[str, SubjectResults] = {}
    profile_rows = []

    atlas_by_stem = {af.stem: af for af in atlas_files}
    # Pre-generate atlas views once in main process to avoid write races.
    atlas_views_by_stem: dict[str, ViewPathMap] = {}
    atlas_stream_cache: StreamlineCache = {}
    atlas_ref_cache: RefCache = {}
    for atlas_stem_name, atlas_bundle_file in atlas_by_stem.items():
        logger.info(">> Generating atlas views for %s...", atlas_stem_name)
        atlas_views_by_stem[atlas_stem_name] = gen_views(
            str(atlas_bundle_file),
            atlas_ref_img,
            atlas=True,
            output_dir=atlas_views_dir,
            streamlines_cache=atlas_stream_cache,
            ref_cache=atlas_ref_cache,
            max_streamlines_view=cap_view,
        )

    speed = {
        "max_streamlines_view": cap_view,
        "max_streamlines_cci": cap_cci,
        "max_streamlines_afq": cap_afq,
        "max_streamlines_shape": cap_shape,
    }

    tasks = []
    for sf in subject_files:
        tract_name = subject_bundle_key(sf.stem)
        mapped_atlas_stem = SUBJECT_TO_ATLAS.get(tract_name)
        if mapped_atlas_stem is None:
            raise ValueError(
                f"No atlas mapping defined for subject bundle '{tract_name}'. "
                f"Expected one of: {sorted(SUBJECT_TO_ATLAS.keys())}",
            )
        atlas_bundle_path = atlas_by_stem.get(mapped_atlas_stem)
        if atlas_bundle_path is None:
            raise ValueError(
                f"Atlas file with stem '{mapped_atlas_stem}' not found. Available atlas stems: {sorted(atlas_by_stem.keys())}",
            )
        tasks.append(
            (
                str(sf),
                subject,
                output_dir,
                subj_ref_img,
                subj_t1_img,
                str(atlas_bundle_path),
                atlas_ref_img,
                atlas_views_by_stem.get(mapped_atlas_stem, {}),
                dti_file_list,
                noddi_file_list,
                speed,
            ),
        )

    if n_jobs <= 1:
        for task in tasks:
            tract_name, tract_result, tract_profiles = _process_subject_tract(task)
            results[tract_name] = tract_result
            profile_rows.extend(tract_profiles)
    else:
        logger.info(">> Parallel mode enabled with %d workers (spawn)", n_jobs)
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_ctx) as ex:
            futures = [ex.submit(_process_subject_tract, task) for task in tasks]
            for fut in as_completed(futures):
                tract_name, tract_result, tract_profiles = fut.result()
                results[tract_name] = tract_result
                profile_rows.extend(tract_profiles)

    subj_out_dir = f"{output_dir}/{subject}"
    if subj_t1_img:
        t0 = time.perf_counter()
        logger.info(">> Generating subject-level anatomy views...")
        anatomy_cache: T1Cache = {}
        ref_cache: RefCache = {}
        streamlines_cache: StreamlineCache = {}
        results["_subject_anatomy"] = gen_subject_anatomy_views(
            subject_files,
            subj_ref_img,
            subj_t1_img,
            subj_out_dir,
            streamlines_cache=streamlines_cache,
            ref_cache=ref_cache,
            t1_cache=anatomy_cache,
        )
        logger.info("   [subject anatomy: %.2fs]", time.perf_counter() - t0)

    # Record run settings in output (visible in HTML if present).
    results["_run_info"] = {
        "jobs": int(n_jobs),
        "render_backend": str(render_backend),
        "max_streamlines_view": _cap_for_run_info(cap_view),
        "max_streamlines_cci": _cap_for_run_info(cap_cci),
        "max_streamlines_afq": _cap_for_run_info(cap_afq),
        "max_streamlines_shape": _cap_for_run_info(cap_shape),
        "t1_img": subj_t1_img or "",
    }

    subject_results[subject] = results
    logger.info(">> Saving results...")
    Path(subj_out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    profile_csv_path = f"{subj_out_dir}/profile_data.csv"
    write_profile_csv(profile_rows, profile_csv_path)
    stats_csv_path = f"{subj_out_dir}/tract_stats.csv"
    write_tract_stats(results, profile_rows, stats_csv_path)
    qc_data_path = f"{subj_out_dir}/qc_data.json"
    with open(qc_data_path, "w") as f:
        json.dump(subject_results, f, indent=4)
    logger.info("   CSV + JSON: %.2fs", time.perf_counter() - t0)
    logger.info("   Profile CSV written to %s", profile_csv_path)
    logger.info("   Tract stats written to %s", stats_csv_path)
    t0 = time.perf_counter()
    html_path = generate_subject_qc_html(qc_data_path)
    logger.info("   HTML report: %.2fs", time.perf_counter() - t0)
    logger.info("   Report written to %s", html_path)
