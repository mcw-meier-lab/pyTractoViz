"""Visualization module for diffusion tractography."""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import multiprocessing
import os
import resource
import sys
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

try:
    import psutil
except ImportError:
    psutil = None

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import vtk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.stats.analysis import afq_profile, assignment_map, gaussian_weights
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
from scipy.spatial.transform import Rotation
from xvfbwrapper import Xvfb

from pytractoviz.html import create_quality_check_html, create_summary_csv
from pytractoviz.utils import (
    ANATOMICAL_VIEW_NAMES,
    calculate_bbox_size,
    calculate_centroid,
    calculate_combined_bbox_size,
    calculate_combined_centroid,
    calculate_direction_colors,
    set_anatomical_camera,
)

logger = logging.getLogger(__name__)

# Constants for memory estimation and warnings
MAX_POINTS_PER_STREAMLINE_FRAGMENTATION_THRESHOLD = 10000  # Streamlines with >10k points risk fragmentation
DENSE_TRACT_POINT_THRESHOLD = 1000000  # >1M total points triggers dense tract warning
VERY_LONG_STREAMLINE_POINT_THRESHOLD = 50000  # Streamlines with >50k points trigger warning
LARGE_TRACT_POINT_THRESHOLD = 100000  # >100k total points triggers large tract info message
MIN_MEMORY_CHECK_MB = 100  # Minimum estimated memory (MB) before checking availability


def _log_memory_usage(
    label: str = "",
    *,
    enable_tracemalloc: bool = False,
    log_level: int = logging.DEBUG,
) -> dict[str, float | int] | None:
    """Log current memory usage for debugging memory issues.

    This function provides comprehensive memory monitoring using:
    - psutil (if available): Process-level memory (RSS, VMS)
    - tracemalloc (built-in): Python-level memory tracking

    Parameters
    ----------
    label : str, optional
        Label to identify this memory checkpoint (e.g., "after loading tract").
    enable_tracemalloc : bool, default=False
        If True, also log top memory allocations from tracemalloc.
        Note: tracemalloc must be started with tracemalloc.start() first.
    log_level : int, default=logging.DEBUG
        Logging level to use for memory information.

    Returns
    -------
    dict[str, float | int] | None
        Dictionary with memory statistics, or None if psutil is not available.
        Keys: 'rss_mb', 'vms_mb', 'percent', 'available_mb' (if psutil available).

    Examples
    --------
    >>> # Basic usage
    >>> _log_memory_usage("Before processing")
    >>> # Process data...
    >>> _log_memory_usage("After processing")

    >>> # With tracemalloc for detailed tracking
    >>> import tracemalloc
    >>> tracemalloc.start()
    >>> _log_memory_usage("Checkpoint 1", enable_tracemalloc=True)
    """
    memory_info: dict[str, float | int] = {}

    # Method 1: resource module (built-in, Unix/macOS/Linux, no dependencies)
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on macOS, MB on Linux
        if sys.platform == "darwin":  # macOS
            resource_rss_mb = usage.ru_maxrss / 1024
        else:  # Linux
            resource_rss_mb = usage.ru_maxrss
        memory_info["resource_rss_mb"] = round(resource_rss_mb, 2)
        logger.log(
            log_level,
            "Memory usage%s (resource): RSS=%.2f MB",
            f" [{label}]" if label else "",
            resource_rss_mb,
        )
    except (OSError, AttributeError):
        # resource module not available on this platform
        pass

    # Method 2: Process-level memory using psutil (if available, more detailed)
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()

        # Get system memory info
        try:
            sys_mem = psutil.virtual_memory()
            available_mb = sys_mem.available / (1024**2)
        except (OSError, RuntimeError, AttributeError):
            available_mb = 0.0

        rss_mb = mem_info.rss / (1024**2)  # Resident Set Size
        vms_mb = mem_info.vms / (1024**2)  # Virtual Memory Size

        memory_info = {
            "rss_mb": round(rss_mb, 2),
            "vms_mb": round(vms_mb, 2),
            "percent": round(mem_percent, 2),
            "available_mb": round(available_mb, 2),
        }

        logger.log(
            log_level,
            "Memory usage%s: RSS=%.2f MB, VMS=%.2f MB, Process=%.2f%%, System Available=%.2f MB",
            f" [{label}]" if label else "",
            rss_mb,
            vms_mb,
            mem_percent,
            available_mb,
        )
    else:
        logger.log(
            log_level,
            "Memory monitoring: psutil not available. Install with: pip install psutil",
        )

    # Python-level memory using tracemalloc (if enabled)
    if enable_tracemalloc and tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        logger.log(log_level, "Top 5 memory allocations%s:", f" [{label}]" if label else "")
        for index, stat in enumerate(top_stats[:5], 1):
            logger.log(
                log_level,
                "  #%d: %s: %.1f MB",
                index,
                stat.traceback[0],
                stat.size / (1024**2),
            )

        # Get current and peak memory
        current, peak = tracemalloc.get_traced_memory()
        logger.log(
            log_level,
            "Tracemalloc: Current=%.2f MB, Peak=%.2f MB",
            current / (1024**2),
            peak / (1024**2),
        )

        memory_info["tracemalloc_current_mb"] = round(current / (1024**2), 2)
        memory_info["tracemalloc_peak_mb"] = round(peak / (1024**2), 2)

    return memory_info if memory_info else None


def _set_memory_limit(memory_limit_mb: float | None = None) -> None:
    """Set a hard memory limit for the current process.

    This uses resource.setrlimit() to set a maximum virtual memory (address space)
    limit. If the process exceeds this limit, it will be killed by the OS.

    Parameters
    ----------
    memory_limit_mb : float | None, optional
        Maximum memory limit in MB. If None, no limit is set.
        If set, the process will be killed if it exceeds this limit.

    Examples
    --------
    >>> # Limit to 8 GB
    >>> _set_memory_limit(8192)

    >>> # Limit to 4 GB
    >>> _set_memory_limit(4096)

    Notes
    -----
    - This sets RLIMIT_AS (virtual memory/address space limit)
    - On macOS, this is enforced by the kernel
    - The limit applies to the entire process, including all threads
    - Once set, the limit cannot be increased (only decreased)
    """
    if memory_limit_mb is None:
        return

    try:
        # Convert MB to bytes
        memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)

        # Set virtual memory limit (RLIMIT_AS)
        # This limits the total address space the process can use
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, resource.RLIM_INFINITY))

        logger.info("Memory limit set to %.2f MB (%.2f GB)", memory_limit_mb, memory_limit_mb / 1024)
    except (OSError, ValueError) as e:
        logger.warning("Failed to set memory limit: %s", e)


def _estimate_actor_memory_mb(streamlines: Streamlines, figure_size: tuple[int, int] = (800, 800)) -> float:
    """Estimate memory needed for creating VTK actors and rendering.

    Parameters
    ----------
    streamlines : Streamlines
        The streamlines to visualize.
    figure_size : tuple[int, int], default=(800, 800)
        Size of output image in pixels.

    Returns
    -------
    float
        Estimated memory in MB needed for actor creation and rendering.
    """
    # Count total points
    total_points = sum(len(sl) for sl in streamlines)
    num_streamlines = len(streamlines)

    # Calculate average points per streamline
    avg_points_per_sl = total_points / num_streamlines if num_streamlines > 0 else 0
    max_points_per_sl = max(len(sl) for sl in streamlines) if streamlines else 0

    # Estimate memory:
    # - VTK actor data: ~200 bytes per point (coordinates, colors, normals, etc.)
    #   (increased from 100 to account for VTK's internal structures)
    # - Scene rendering buffer: image_size * 4 bytes (RGBA) * 3 (triple buffering for safety)
    # - Additional overhead: ~30% for VTK internal structures and fragmentation
    # - Large contiguous allocation penalty: if max streamline is very long, VTK may need
    #   a large contiguous buffer, which can fail even with available memory due to fragmentation

    actor_memory_mb = (total_points * 200) / (1024**2)  # Actor data (increased estimate)
    render_memory_mb = (figure_size[0] * figure_size[1] * 4 * 3) / (1024**2)  # Render buffer (triple buffering)

    # Add penalty for very long streamlines (fragmentation risk)
    # If any streamline has >10k points, VTK needs a large contiguous buffer
    fragmentation_penalty_mb = 0.0
    if max_points_per_sl > MAX_POINTS_PER_STREAMLINE_FRAGMENTATION_THRESHOLD:
        # Large contiguous allocation needed - add 50% overhead for fragmentation
        fragmentation_penalty_mb = actor_memory_mb * 0.5
        logger.warning(
            "Very long streamline detected (%d points). VTK may need large contiguous memory allocation.",
            max_points_per_sl,
        )

    overhead_mb = (actor_memory_mb + render_memory_mb) * 0.3  # 30% overhead
    total_mb = actor_memory_mb + render_memory_mb + overhead_mb + fragmentation_penalty_mb

    logger.info(
        "Memory estimate for actor: %.2f MB (total_points=%d, streamlines=%d, "
        "avg_points/sl=%.1f, max_points/sl=%d, image=%dx%d)",
        total_mb,
        total_points,
        num_streamlines,
        avg_points_per_sl,
        max_points_per_sl,
        figure_size[0],
        figure_size[1],
    )

    return total_mb


def _check_memory_available(required_mb: float, safety_margin: float = 0.2) -> bool:
    """Check if enough memory is available before loading data.

    Parameters
    ----------
    required_mb : float
        Estimated memory required in MB.
    safety_margin : float, default=0.2
        Safety margin as fraction (0.2 = 20% extra buffer).

    Returns
    -------
    bool
        True if enough memory is available, False otherwise.
    """
    if psutil is None:
        # If psutil not available, assume we have enough
        logger.debug("psutil not available, skipping memory check")
        return True

    try:
        # Get system memory
        sys_mem = psutil.virtual_memory()
        available_mb = sys_mem.available / (1024**2)

        # Get current process memory
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / (1024**2)

        # Calculate required with safety margin
        required_with_margin = required_mb * (1 + safety_margin)

        # Check if we have enough available memory
        if available_mb < required_with_margin:
            logger.warning(
                "Insufficient memory: Need %.2f MB (with %.0f%% margin), "
                "but only %.2f MB available. Current process: %.2f MB",
                required_with_margin,
                safety_margin * 100,
                available_mb,
                current_mb,
            )
            return False

        logger.debug(
            "Memory check: Need %.2f MB, Available: %.2f MB, Current: %.2f MB",
            required_with_margin,
            available_mb,
            current_mb,
        )
    except (OSError, RuntimeError, AttributeError) as e:
        logger.warning("Memory check failed: %s. Proceeding anyway.", e)
        return True
    else:
        return True


def _get_n_jobs_with_memory_limit(
    base_n_jobs: int,
    estimated_memory_per_job_mb: float = 2000.0,
    safety_margin: float = 0.2,
) -> int:
    """Calculate n_jobs considering available memory.

    Reduces n_jobs if there isn't enough memory to run all jobs in parallel.

    Parameters
    ----------
    base_n_jobs : int
        Base number of jobs to use (from CPU/SLURM settings).
    estimated_memory_per_job_mb : float, default=2000.0
        Estimated memory per job in MB. Default assumes ~2GB per worker.
    safety_margin : float, default=0.2
        Safety margin as fraction (0.2 = 20% extra buffer).

    Returns
    -------
    int
        Adjusted number of jobs considering memory constraints.
    """
    if psutil is None:
        # If psutil not available, return base_n_jobs
        return base_n_jobs

    try:
        # Get available system memory
        sys_mem = psutil.virtual_memory()
        available_mb = sys_mem.available / (1024**2)

        # Get current process memory
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / (1024**2)

        # Calculate how much memory we can use for workers
        # Reserve some for the main process
        usable_mb = available_mb - (current_mb * 0.5)  # Reserve 50% of current for main process

        # Calculate required memory with safety margin
        required_per_job = estimated_memory_per_job_mb * (1 + safety_margin)

        # Calculate max jobs based on memory
        max_jobs_by_memory = max(1, int(usable_mb / required_per_job))

        # Use the minimum of base_n_jobs and memory-limited jobs
        optimal_jobs = min(base_n_jobs, max_jobs_by_memory)

        if optimal_jobs < base_n_jobs:
            logger.info(
                "Reduced n_jobs from %d to %d due to memory constraints "
                "(Available: %.2f MB, Estimated per job: %.2f MB)",
                base_n_jobs,
                optimal_jobs,
                usable_mb,
                required_per_job,
            )
    except (OSError, RuntimeError, AttributeError) as e:
        logger.warning("Memory-based n_jobs calculation failed: %s. Using base_n_jobs=%d", e, base_n_jobs)
        return base_n_jobs
    else:
        return optimal_jobs


def _get_optimal_n_jobs() -> int:
    """Calculate optimal number of jobs considering SLURM and OpenMP settings.

    This function respects SLURM CPU allocations and OpenMP thread settings
    to prevent oversubscription. It checks:
    1. SLURM_CPUS_PER_TASK (if running under SLURM)
    2. SLURM_JOB_CPUS_PER_NODE (if running under SLURM)
    3. OMP_NUM_THREADS (to avoid conflicts with OpenMP)
    4. Falls back to multiprocessing.cpu_count() if not in SLURM

    Returns
    -------
    int
        Optimal number of parallel jobs to use.
    """
    # Check if running under SLURM
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_job_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")

    if slurm_cpus_per_task:
        # Use SLURM allocation
        allocated_cpus = int(slurm_cpus_per_task)
        logger.debug("Using SLURM_CPUS_PER_TASK=%d", allocated_cpus)

        # Check OMP_NUM_THREADS to avoid oversubscription
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        if omp_threads > 1:
            # If OpenMP uses multiple threads, reduce n_jobs accordingly
            # Formula: n_jobs = allocated_cpus / omp_threads
            optimal_jobs = max(1, allocated_cpus // omp_threads)
            logger.debug(
                "OMP_NUM_THREADS=%d detected. Using n_jobs=%d (allocated_cpus=%d / omp_threads=%d)",
                omp_threads,
                optimal_jobs,
                allocated_cpus,
                omp_threads,
            )
            return optimal_jobs
        return allocated_cpus
    if slurm_job_cpus:
        # Use SLURM job CPUs (may be a range like "8-16", take the first value)
        allocated_cpus = int(slurm_job_cpus.split("-")[0].split(",")[0])
        logger.debug("Using SLURM_JOB_CPUS_PER_NODE=%d", allocated_cpus)

        # Check OMP_NUM_THREADS
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        if omp_threads > 1:
            optimal_jobs = max(1, allocated_cpus // omp_threads)
            logger.debug(
                "OMP_NUM_THREADS=%d detected. Using n_jobs=%d",
                omp_threads,
                optimal_jobs,
            )
            return optimal_jobs
        return allocated_cpus

    # Not in SLURM, check OMP_NUM_THREADS and fall back to cpu_count()
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if omp_threads > 0:
        # If OMP_NUM_THREADS is set, use it to calculate n_jobs
        total_cpus = multiprocessing.cpu_count()
        optimal_jobs = max(1, total_cpus // omp_threads)
        logger.debug(
            "OMP_NUM_THREADS=%d detected. Using n_jobs=%d (cpu_count=%d / omp_threads=%d)",
            omp_threads,
            optimal_jobs,
            total_cpus,
            omp_threads,
        )
        return optimal_jobs

    # Default: use all available CPUs
    total_cpus = multiprocessing.cpu_count()
    logger.debug("Using all available CPUs: n_jobs=%d", total_cpus)
    return total_cpus


def _process_tract_worker(
    subject_id: str,
    tract_name: str,
    tract_file: str | Path,
    subject_ref_img: str | Path,
    tract_output_dir: str | Path,
    subjects_mni_space: dict[str, dict[str, str | Path]] | None,
    atlas_files: dict[str, str | Path] | None,
    metric_files: dict[str, dict[str, str | Path]] | None,
    atlas_ref_img: str | Path | None,
    *,
    flip_lr: bool,
    skip_checks: list[str],
    visualizer_params: dict[str, Any],
    subject_kwargs: dict[str, Any] | None = None,
    atlas_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, str, dict[str, str | Path]]:
    """Worker function for parallel processing of tracts.

    This function creates a new visualizer instance and processes a single tract.
    Returns (subject_id, tract_name, results_dict).
    """
    visualizer = None
    results: dict[str, str | Path] = {}

    try:
        # Initialize VTK offscreen rendering in worker process
        # This is critical for headless cluster environments
        # Set environment variables before any VTK operations
        os.environ.setdefault("VTK_STREAM_READER", "1")
        os.environ.setdefault("VTK_STREAM_WRITER", "1")
        # Force offscreen rendering to prevent segfaults
        os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
        # Disable OpenGL to prevent segfaults in headless environments
        os.environ.setdefault("VTK_USE_OSMESA", "0")

        # Disable VTK warnings and errors that might cause issues
        with contextlib.suppress(AttributeError, RuntimeError, ImportError):
            vtk.vtkObject.GlobalWarningDisplayOff()
            # Try to set error callback to prevent crashes
            with contextlib.suppress(AttributeError, RuntimeError):
                vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)

        # Set matplotlib to non-interactive backend for headless environments
        with contextlib.suppress(ImportError, ValueError, RuntimeError):
            # Use Agg backend (no display needed) for headless environments
            plt.switch_backend("Agg")

        # Create a new visualizer instance in this worker process
        # Wrap in try-except to catch initialization errors that might cause segfaults
        try:
            visualizer = TractographyVisualizer(**visualizer_params)
        except (OSError, RuntimeError, MemoryError) as e:
            logger.exception(
                "Failed to initialize visualizer in worker for %s/%s: %s",
                subject_id,
                tract_name,
                type(e).__name__,
            )
            return (subject_id, tract_name, {})

        # Process the tract with additional error handling
        try:
            results = visualizer._process_single_tract(
                subject_id=subject_id,
                tract_name=tract_name,
                tract_file=tract_file,
                subject_ref_img=Path(subject_ref_img),
                tract_output_dir=Path(tract_output_dir),
                subjects_mni_space=subjects_mni_space,
                atlas_files=atlas_files,
                metric_files=metric_files,
                atlas_ref_img=atlas_ref_img,
                flip_lr=flip_lr,
                skip_checks=skip_checks,
                subject_kwargs=subject_kwargs,
                atlas_kwargs=atlas_kwargs,
                **kwargs,
            )
        except MemoryError:
            # Memory errors are critical - log and return empty results
            logger.exception(
                "Memory error in worker process for %s/%s. Consider reducing n_jobs or increasing memory allocation.",
                subject_id,
                tract_name,
            )
            results = {}
        except (OSError, ValueError, RuntimeError) as e:
            # Catch specific exceptions that might cause process crashes
            # Log the error but don't re-raise to prevent process pool from breaking
            logger.exception(
                "Error in worker process for %s/%s (%s)",
                subject_id,
                tract_name,
                type(e).__name__,
            )
            # Return empty results dict on error
            results = {}
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.exception(
                "Unexpected error in worker process for %s/%s (%s)",
                subject_id,
                tract_name,
                type(e).__name__,
            )
            # Return empty results dict on error
            results = {}
    except (OSError, ValueError, RuntimeError, MemoryError) as e:
        # Catch errors during initialization or setup
        logger.exception(
            "Error during worker initialization for %s/%s (%s)",
            subject_id,
            tract_name,
            type(e).__name__,
        )
        results = {}
    except Exception as e:
        # Catch any other unexpected exceptions during setup
        logger.exception(
            "Unexpected error during worker setup for %s/%s (%s)",
            subject_id,
            tract_name,
            type(e).__name__,
        )
        results = {}
    finally:
        # Clean up the visualizer instance and force garbage collection
        # This is critical to prevent memory leaks that could cause OOM kills
        if visualizer is not None:
            # Try to clean up any VTK objects
            with contextlib.suppress(AttributeError, RuntimeError, TypeError):
                del visualizer
        # Clear results dict reference if it's large
        if results:
            # Keep only the minimal results (file paths, not data)
            pass  # Results dict should only contain paths, not large data

        # Force garbage collection multiple times to handle circular references
        # VTK objects can have complex reference cycles
        for _ in range(3):
            gc.collect()

    return (subject_id, tract_name, results)


class TractographyVisualizationError(Exception):
    """Base exception for tractography visualization errors."""


class InvalidInputError(TractographyVisualizationError):
    """Raised when input data is invalid."""


class TractographyVisualizer:
    """A class for visualizing diffusion tractography data.

    This class provides methods for loading, processing, and visualizing
    tractography data, including generating static images, animations, and
    quality metrics.

    Parameters
    ----------
    reference_image : str | Path, optional
        Path to the reference T1-weighted image. Can be set later via
        `set_reference_image()`.
    output_directory : str | Path, optional
        Default output directory for generated files. Can be set later via
        `set_output_directory()`.
    gif_size : tuple[int, int], optional
        Size of generated GIFs in pixels. Default is (608, 608).
    gif_duration : float, optional
        Duration per frame in seconds. Default is 0.2.
    gif_palette_size : int, optional
        Color palette size for GIF optimization. Default is 64.
    gif_frames : int, optional
        Number of frames in rotation animation. Default is 60.
    min_streamline_length : float, optional
        Minimum streamline length for CCI calculation. Default is 40.0.
    cci_threshold : float, optional
        Minimum CCI value to keep streamlines. Default is 1.0.
    afq_resample_points : int, optional
        Number of points for AFQ resampling. Default is 100.
    n_jobs : int, optional
        Number of parallel jobs to run for processing multiple subjects/tracts.
        Default is 1 (sequential processing). Use -1 to automatically determine
        optimal number based on available resources (respects SLURM allocations
        and OpenMP thread settings to prevent oversubscription).
        Only used in `run_quality_check_workflow()`.

        Note: When running under SLURM, this will automatically use
        SLURM_CPUS_PER_TASK or SLURM_JOB_CPUS_PER_NODE. If OMP_NUM_THREADS is set,
        it will divide the available CPUs by the number of OpenMP threads to
        prevent resource contention.
    max_memory_mb : float | None, optional
        Maximum memory limit in MB for the process. If set, the process will be
        killed by the OS if it exceeds this limit. This helps prevent OOM kills
        by setting a hard limit. Default is None (no limit).

        Note: This uses resource.setrlimit() which sets a virtual memory limit.
        Once set, the limit cannot be increased (only decreased).
    figure_size : tuple[int, int], optional
        Default size for generated static images in pixels. Default is (800, 800).
        Can be overridden per method call. Smaller sizes use less memory and may
        help avoid VTK segfaults with large tractograms.

    Examples
    --------
    Single subject usage:
    >>> visualizer = TractographyVisualizer(
    ...     reference_image="path/to/t1w.nii.gz", output_directory="output/"
    ... )
    >>> visualizer.generate_videos(
    ...     tract_files=["tract1.trk", "tract2.trk"], ref_file="t1w.nii.gz"
    ... )

    Multiple subjects usage (initialize once, set data per subject):
    >>> visualizer = TractographyVisualizer(output_directory="output/")
    >>> # Process subject 1
    >>> visualizer.generate_videos(
    ...     tract_files=["subj1_tract1.trk"], ref_file="subj1_t1w.nii.gz"
    ... )
    >>> # Process subject 2
    >>> visualizer.generate_videos(
    ...     tract_files=["subj2_tract1.trk"], ref_file="subj2_t1w.nii.gz"
    ... )
    """

    def __init__(
        self,
        reference_image: str | Path | None = None,
        output_directory: str | Path | None = None,
        *,
        gif_size: tuple[int, int] = (608, 608),
        gif_duration: float = 0.2,
        gif_palette_size: int = 64,
        gif_frames: int = 60,
        min_streamline_length: float = 40.0,
        cci_threshold: float = 1.0,
        afq_resample_points: int = 100,
        n_jobs: int = 1,
        max_memory_mb: float | None = None,
        figure_size: tuple[int, int] = (800, 800),
    ) -> None:
        """Initialize the TractographyVisualizer."""
        # Initialize VTK for headless rendering BEFORE any VTK operations
        # This is critical for cluster jobs that don't have a display
        # Set environment variables before any VTK imports are used
        if "DISPLAY" not in os.environ:
            # No display available - force offscreen rendering
            os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
            os.environ.setdefault("VTK_USE_OSMESA", "0")
            logger.debug("No DISPLAY detected, enabling VTK offscreen rendering")
        else:
            # Display available, but still set offscreen as fallback
            os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
            logger.debug("DISPLAY detected, but VTK offscreen rendering enabled as fallback")

        # Additional VTK settings for stability
        os.environ.setdefault("VTK_STREAM_READER", "1")
        os.environ.setdefault("VTK_STREAM_WRITER", "1")

        # Disable VTK warnings (can cause issues in some environments)
        with contextlib.suppress(AttributeError, RuntimeError, ImportError):
            vtk.vtkObject.GlobalWarningDisplayOff()
            with contextlib.suppress(AttributeError, RuntimeError):
                vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)

        # Set matplotlib to non-interactive backend for headless environments
        with contextlib.suppress(ImportError, ValueError, RuntimeError):
            plt.switch_backend("Agg")

        self._reference_image: Path | None = None
        self._output_directory: Path | None = None
        self.max_memory_mb: float | None = None

        if reference_image is not None:
            self.set_reference_image(reference_image)
        if output_directory is not None:
            self.set_output_directory(output_directory)

        self.gif_size = gif_size
        self.gif_duration = gif_duration
        self.gif_palette_size = gif_palette_size
        self.gif_frames = gif_frames
        self.min_streamline_length = min_streamline_length
        self.cci_threshold = cci_threshold
        self.afq_resample_points = afq_resample_points
        self.figure_size = figure_size
        # Handle n_jobs: -1 means use all CPUs, otherwise use specified value
        if n_jobs == -1:
            self.n_jobs = _get_optimal_n_jobs()
        else:
            self.n_jobs = n_jobs

        # Set memory limit if specified
        if max_memory_mb is not None:
            _set_memory_limit(max_memory_mb)
            self.max_memory_mb = max_memory_mb
        else:
            self.max_memory_mb = None

    def set_reference_image(self, reference_image: str | Path) -> None:
        """Set the reference T1-weighted image.

        Parameters
        ----------
        reference_image : str | Path
            Path to the reference image file.

        Raises
        ------
        FileNotFoundError
            If the reference image file does not exist.
        """
        ref_path = Path(reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image}")
        self._reference_image = ref_path

    def set_output_directory(self, output_directory: str | Path) -> None:
        """Set the output directory for generated files.

        Parameters
        ----------
        output_directory : str | Path
            Path to the output directory. Will be created if it doesn't exist.

        Raises
        ------
        OSError
            If the directory cannot be created.
        """
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._output_directory = out_dir

    @property
    def reference_image(self) -> Path | None:
        """Get the current reference image path."""
        return self._reference_image

    @property
    def output_directory(self) -> Path | None:
        """Get the current output directory path."""
        return self._output_directory

    def get_glass_brain(self, t1w_img: str | Path | None = None) -> actor.Actor:
        """Get the glass brain actor for the T1-weighted image.

        Parameters
        ----------
        t1w_img : str | Path | None, optional
            Path to the T1-weighted image. If None, uses the reference image
            set during initialization or via `set_reference_image()`.

        Returns
        -------
        actor.Actor
            The glass brain actor.

        Raises
        ------
        FileNotFoundError
            If the image file is not found.
        InvalidInputError
            If no reference image is provided and none was set.
        """
        if t1w_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            t1w_img = self._reference_image

        try:
            mask_img = nib.load(str(t1w_img))
            mask_data = mask_img.get_fdata()  # type: ignore[attr-defined]
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to load glass brain from {t1w_img}: {e}",
            ) from e
        else:
            return actor.contour_from_roi(mask_data, color=[0, 0, 0], opacity=0.05)

    def _set_anatomical_camera(
        self,
        scene: window.Scene,
        centroid: np.ndarray,
        view_name: str,
        *,
        camera_distance: float | None = None,
        bbox_size: np.ndarray | None = None,
        template_space: Literal["ras", "mni"] = "ras",
    ) -> None:
        """Set camera position for standard anatomical views.

        Wrapper around utils.set_anatomical_camera for class method compatibility.

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
        template_space : str, optional
            "ras" for standard templates, "mni" for MNI-space (different camera sides).
            Default is "ras".

        Raises
        ------
        InvalidInputError
            If view_name is not one of the standard anatomical views.
        """
        try:
            set_anatomical_camera(
                scene,
                centroid,
                view_name,
                camera_distance=camera_distance,
                bbox_size=bbox_size,
                template_space=template_space,
            )
        except ValueError as e:
            raise InvalidInputError(str(e)) from e

    def _create_scene(
        self,
        *,
        ref_img: str | Path | None = None,
        show_glass_brain: bool = True,
    ) -> tuple[window.Scene, actor.Actor | None]:
        """Create a new scene with optional glass brain.

        Parameters
        ----------
        show_glass_brain : bool, optional
            Whether to add glass brain to the scene. Default is True.
        ref_img : str | Path | None, optional
            Reference image for glass brain. If None, uses default.

        Returns
        -------
        tuple[window.Scene, actor.Actor | None]
            The scene and brain actor (if added, otherwise None).
        """
        scene = window.Scene()
        scene.SetBackground(1, 1, 1)

        brain_actor = None
        if show_glass_brain:
            try:
                brain_actor = self.get_glass_brain(ref_img)
                scene.add(brain_actor)
            except (RuntimeError, OSError, MemoryError) as e:
                # Glass brain creation can fail with VTK errors
                logger.warning(
                    "Failed to create glass brain: %s. Continuing without glass brain.",
                    e,
                )
                # Continue without glass brain rather than failing completely
                brain_actor = None

        return scene, brain_actor

    def _create_streamline_actor(
        self,
        streamlines: Streamlines,
        colors: np.ndarray | None = None,
    ) -> actor.Actor:
        """Create a streamline actor with optional colors, handling length mismatches.

        Parameters
        ----------
        streamlines : Streamlines
            The streamlines to visualize.
        colors : np.ndarray | None, optional
            Colors array. If provided, must match streamlines length or will be adjusted.

        Returns
        -------
        actor.Actor
            The streamline actor.

        Raises
        ------
        RuntimeError
            If VTK memory allocation fails (std::bad_alloc). The error message will
            contain information about the memory failure.
        """
        # Pre-emptive memory check to avoid C++ std::bad_alloc that might bypass Python exception handling
        # Estimate memory needed (conservative estimate for actor creation)
        total_points = sum(len(sl) for sl in streamlines)
        num_streamlines = len(streamlines)

        # Conservative estimate: ~500 bytes per point for VTK actor (coordinates, colors, normals, etc.)
        # Plus overhead for VTK internal structures
        estimated_memory_mb = (total_points * 500) / (1024**2) * 1.5  # 1.5x safety factor

        # Check if we have enough memory before attempting allocation
        # This prevents C++ std::bad_alloc that might terminate the process
        if (
            psutil is not None
            and estimated_memory_mb > MIN_MEMORY_CHECK_MB
            and not _check_memory_available(estimated_memory_mb, safety_margin=0.5)
        ):
            total_points = sum(len(sl) for sl in streamlines)
            error_msg = (
                f"Insufficient memory to create actor. "
                f"Tract has {num_streamlines} streamlines with {total_points} total points. "
                f"Estimated memory needed: {estimated_memory_mb:.1f} MB. "
                f"Try filtering streamlines, reducing image resolution, or using n_jobs=1."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            if colors is not None:
                if len(colors) == len(streamlines):
                    return actor.line(streamlines, colors=colors)
                # Handle length mismatch
                if len(colors) < len(streamlines):
                    repeat_factor = len(streamlines) // len(colors) + 1
                    extended_colors = np.tile(colors, (repeat_factor, 1))[: len(streamlines)]
                    return actor.line(streamlines, colors=extended_colors)
                return actor.line(streamlines, colors=colors[: len(streamlines)])
            return actor.line(streamlines)
        except RuntimeError as e:
            # Catch std::bad_alloc and other VTK memory errors
            error_msg = str(e).lower()
            if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                total_points = sum(len(sl) for sl in streamlines)
                logger.exception(
                    "VTK memory allocation failed in _create_streamline_actor (likely std::bad_alloc). "
                    "Tract has %d streamlines with %d total points. "
                    "Try filtering streamlines, reducing image resolution, or using n_jobs=1.",
                    len(streamlines),
                    total_points,
                )
            raise

    def _downsample_streamlines(
        self,
        streamlines: Streamlines,
        *,
        set_points: int | None = None,
        approx_points: float | None = 0.25,
    ) -> Streamlines:
        """Downsample streamlines to reduce memory usage.

        Parameters
        ----------
        streamlines : Streamlines
            The streamlines to filter.
        set_points: int
            A set number of points per streamline. (Uses DIPY's set_number_of_points).
        approx_points: float, optional
            Reduce the number of points so that there are more points
            in curvy regions and less points in less curvy regions.
            (Uses DIPY's approx_polygon_track).

        Returns
        -------
        Streamlines
            Downsampled streamlines.
        """
        n_pts = sum([len(streamline) for streamline in streamlines])

        # Downsample using set_number_of points
        if set_points is not None:
            downsampled = set_number_of_points(streamlines, nb_points=set_points)
            n_pts_ds = sum([len(s) for s in downsampled])
        elif approx_points is not None:
            downsampled = [approx_polygon_track(s, approx_points) for s in streamlines]
            n_pts_ds = sum([len(s) for s in downsampled])

        logger.info(
            "Downsampled streamlines: %d points -> %d points (reduction: %.1f%%)",
            n_pts,
            n_pts_ds,
            (1 - n_pts_ds / n_pts) * 100 if n_pts > 0 else 0,
        )

        return Streamlines(downsampled)

    def load_tract(
        self,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
    ) -> actor.Actor:
        """Load the tractography file and create an actor.

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file (.trk).
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization or via `set_reference_image()`.

        Returns
        -------
        actor.Actor
            The tractography actor.

        Raises
        ------
        FileNotFoundError
            If the tract or reference image file is not found.
        InvalidInputError
            If no reference image is provided and none was set.
        TractographyVisualizationError
            If loading or transformation fails.
        """
        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image

        try:
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()
            ref_img_obj = nib.load(str(ref_img))
            tract_to_ref = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]  # type: ignore[attr-defined]
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to load tract from {tract_file}: {e}",
            ) from e
        else:
            try:
                return actor.line(tract_to_ref)
            except RuntimeError as e:
                # Catch std::bad_alloc and other VTK errors
                error_msg = str(e).lower()
                if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                    total_points = sum(len(sl) for sl in tract_to_ref)
                    logger.exception(
                        "VTK memory allocation failed in load_tract (likely std::bad_alloc). "
                        "Tract has %d streamlines with %d total points. "
                        "Try filtering streamlines or using n_jobs=1.",
                        len(tract_to_ref),
                        total_points,
                    )
                raise TractographyVisualizationError(
                    f"Failed to create actor from tract: {e}. Try filtering streamlines or processing with n_jobs=1.",
                ) from e

    def weighted_afq(
        self,
        tract_file: str | Path,
        atlas_file: str | Path,
        metric_file: str | Path,
    ) -> np.ndarray:
        """Calculate weighted AFQ profile for tractography.

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file.
        atlas_file : str | Path
            Path to the atlas tractography file.
        metric_file : str | Path
            Path to the metric image file.

        Returns
        -------
        np.ndarray
            The AFQ profile array.

        Raises
        ------
        FileNotFoundError
            If any required file is not found.
        InvalidInputError
            If clustering fails or no centroids are found.
        TractographyVisualizationError
            If processing fails.
        """
        try:
            atlas_tract = load_trk(str(atlas_file), "same", bbox_valid_check=False)
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)

            # Check if tract is empty before processing
            if not tract.streamlines or len(tract.streamlines) == 0:
                raise InvalidInputError(
                    f"Tract file {tract_file} is empty (0 streamlines). Cannot calculate AFQ profile.",
                )

            # Check if atlas tract is empty
            if not atlas_tract.streamlines or len(atlas_tract.streamlines) == 0:
                raise InvalidInputError(
                    f"Atlas tract file {atlas_file} is empty (0 streamlines). Cannot calculate AFQ profile.",
                )

            metric_img = nib.load(str(metric_file))
            metric = metric_img.get_fdata()  # type: ignore[attr-defined]

            feature = ResampleFeature(nb_points=self.afq_resample_points)
            qb_metric = AveragePointwiseEuclideanMetric(feature)
            qb = QuickBundles(threshold=np.inf, metric=qb_metric)
            cluster_tract = qb.cluster(atlas_tract.streamlines)

            if len(cluster_tract.centroids) == 0:
                raise InvalidInputError(
                    "No centroids found in atlas tractography clustering.",
                )

            standard_tract = cluster_tract.centroids[0]
            oriented_tract = orient_by_streamline(tract.streamlines, standard_tract)
            w_tract = gaussian_weights(oriented_tract)
            profile = afq_profile(
                metric,
                oriented_tract,
                affine=metric_img.affine,  # type: ignore[attr-defined]
                weights=w_tract,
            )

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError, IndexError) as e:
            raise TractographyVisualizationError(
                f"Failed to calculate weighted AFQ profile: {e}",
            ) from e
        else:
            return profile

    def calc_cci(
        self,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, StatefulTractogram, Streamlines]:
        """Calculate Cluster Confidence Index (CCI) for tractography.

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization or via `set_reference_image()`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, StatefulTractogram, Streamlines]
            A tuple containing:
            - CCI values for all long streamlines (one per streamline).
            - CCI values above threshold (aligned with kept streamlines).
            - Filtered tractogram with streamlines above threshold.
            - Long streamlines used for CCI (length > min_streamline_length).

        Raises
        ------
        InvalidInputError
            If the tractogram is empty or processing fails, or if no reference
            image is provided and none was set.
        TractographyVisualizationError
            If calculation fails.
        """
        tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
        tract.to_rasmm()

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image

        try:
            lengths = list(length(tract.streamlines))
            long_streamlines = Streamlines()
            for i, sl in enumerate(tract.streamlines):
                if lengths[i] > self.min_streamline_length:
                    long_streamlines.append(sl)

            if len(long_streamlines) == 0:
                raise InvalidInputError(
                    f"No streamlines longer than {self.min_streamline_length}mm found.",
                )

            cci = cluster_confidence(long_streamlines)

            # Create boolean mask for streamlines above threshold
            keep_mask = cci >= self.cci_threshold

            # Filter streamlines and CCI values to match
            keep_streamlines = Streamlines()
            for i, sl in enumerate(long_streamlines):
                if keep_mask[i]:
                    keep_streamlines.append(sl)

            # Filter CCI array to match kept streamlines
            keep_cci = cci[keep_mask]
            keep_tract = StatefulTractogram(keep_streamlines, nib.load(str(ref_img)), Space.RASMM)

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError, IndexError) as e:
            raise TractographyVisualizationError(
                f"Failed to calculate CCI: {e}",
            ) from e
        else:
            return cci, keep_cci, keep_tract, long_streamlines

    def generate_anatomical_views(
        self,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] | None = None,
        show_glass_brain: bool = True,
        flip_lr: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Generate snapshots from standard anatomical views.

        Creates static images from coronal, axial, and sagittal views of the tract.

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization or via `set_reference_image()`.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int] | None, optional
            Size of the output images in pixels. If None, uses the default
            set during initialization (default: (800, 800)). Default is None.
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        flip_lr : bool, optional
            Whether to flip left-right (X-axis) when transforming tract.
            This may be needed for some coordinate conventions or file formats
            where the left-right orientation differs (e.g., when working with
            MNI space). Default is False.
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping view names to their output file paths.
            Keys: "coronal", "axial", "sagittal".

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available or invalid view name.
        TractographyVisualizationError
            If image generation fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(
        ...     reference_image="t1w.nii.gz", output_directory="output/"
        ... )
        >>> # Generate all views
        >>> views = visualizer.generate_anatomical_views("tract.trk")
        >>> # Generate specific views
        >>> views = visualizer.generate_anatomical_views(
        ...     "tract.trk", views=["coronal", "axial"]
        ... )
        """
        # Use instance default if figure_size not provided
        if figure_size is None:
            figure_size = self.figure_size

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        # Determine which views to generate and template space for camera (MNI vs RAS)
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )
        template_space: Literal["ras", "mni"] = "mni" if "MNI" in str(ref_img) else "ras"

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        tract_name = Path(tract_file).stem
        generated_views: dict[str, Path] = {}

        # Check if all output files already exist BEFORE loading tract
        # This prevents unnecessary memory usage when files are already generated
        all_files_exist = True
        for view_name in views_to_generate:
            output_image = output_dir / f"{tract_name}_{view_name}.png"
            if output_image.exists() and not overwrite:
                generated_views[view_name] = output_image
                logger.debug("Skipping generation of %s (file already exists)", output_image)
            else:
                all_files_exist = False

        # If all files exist, return early without loading tract
        if all_files_exist:
            return generated_views

        try:
            # Load tract only if we need to generate at least one view
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()
            ref_img_obj = nib.load(str(ref_img))
            tract_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Apply left-right flip if needed (common when tract is in MNI space)
            if flip_lr:
                # Flip X-axis (left-right) by negating X coordinates
                # This is needed when MNI and native space have different L/R conventions
                if not tract_streamlines or len(tract_streamlines) == 0:
                    logger.warning(
                        "Tract has 0 streamlines before flip. Skipping visualization for %s",
                        tract_file,
                    )
                    return {}
                tract_streamlines = Streamlines(
                    [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in tract_streamlines],
                )

            # Filter streamlines if requested (to avoid VTK segfaults)
            tract_streamlines = self._downsample_streamlines(tract_streamlines)

            # Check if streamlines are empty after filtering
            if not tract_streamlines or len(tract_streamlines) == 0:
                logger.warning(
                    "Tract has 0 streamlines after filtering. Skipping visualization for %s",
                    tract_file,
                )
                return {}

            # Calculate colors based on streamline directions using utility function
            streamline_colors = calculate_direction_colors(tract_streamlines)

            # Get centroid using utility function
            centroid = calculate_centroid(tract_streamlines)

            # Log tract statistics for debugging
            total_points = sum(len(sl) for sl in tract_streamlines)
            avg_points = total_points / len(tract_streamlines) if tract_streamlines else 0
            max_points = max(len(sl) for sl in tract_streamlines) if tract_streamlines else 0
            logger.info(
                "Tract statistics: %d streamlines, %d total points, "
                "%.1f avg points/streamline, %d max points/streamline",
                len(tract_streamlines),
                total_points,
                avg_points,
                max_points,
            )

            # Warn if tract is very dense (high risk of std::bad_alloc or segfault)
            if total_points > DENSE_TRACT_POINT_THRESHOLD:
                logger.warning(
                    "Very dense tract detected (%d total points). "
                    "This may cause std::bad_alloc errors or segfaults due to memory fragmentation. "
                    "Consider: (1) resampling streamlines to reduce points, "
                    "(2) filtering with higher cci_threshold, (3) reducing image resolution, "
                    "(4) disabling glass brain (show_glass_brain=False).",
                    total_points,
                )
            if max_points > VERY_LONG_STREAMLINE_POINT_THRESHOLD:
                logger.warning(
                    "Very long streamline detected (%d points). "
                    "VTK may need large contiguous memory allocation which can fail due to fragmentation. "
                    "Consider disabling glass brain (show_glass_brain=False) or reducing figure_size.",
                    max_points,
                )
            # Warn if tract is moderately large and might cause issues
            if total_points > LARGE_TRACT_POINT_THRESHOLD:
                logger.info(
                    "Large tract detected (%d total points). "
                    "If you experience segfaults during rendering, try: "
                    "(1) reducing figure_size to (400, 400) or smaller, "
                    "(2) disabling glass brain (show_glass_brain=False), "
                    "(3) filtering streamlines with higher cci_threshold.",
                    total_points,
                )

            # Clean up loaded objects that are no longer needed
            del tract, ref_img_obj

            # Generate each requested view
            for view_name in views_to_generate:
                output_image = output_dir / f"{tract_name}_{view_name}.png"

                # Skip if file already exists (already added to generated_views above)
                if output_image.exists() and not overwrite:
                    continue

                # Check memory before creating VTK actor
                estimated_memory = _estimate_actor_memory_mb(tract_streamlines, figure_size)
                if not _check_memory_available(estimated_memory, safety_margin=0.3):
                    logger.warning(
                        "Insufficient memory to create actor for view %s. Skipping.",
                        view_name,
                    )
                    continue

                # Create scene using helper method (can segfault in VTK)
                logger.debug("Creating scene for view %s", view_name)
                try:
                    scene, _ = self._create_scene(ref_img=ref_img, show_glass_brain=show_glass_brain)
                except (RuntimeError, OSError, MemoryError):
                    # Catch VTK errors during scene creation
                    logger.exception(
                        "Failed to create scene for view %s. "
                        "This may indicate a VTK segfault or memory issue. "
                        "Try reducing image resolution or disabling glass brain.",
                        view_name,
                    )
                    continue
                except Exception:
                    # Catch any other unexpected errors
                    logger.exception(
                        "Unexpected error creating scene for view %s",
                        view_name,
                    )
                    continue

                logger.debug("Scene created successfully for view %s", view_name)

                # Create actor with original streamlines using helper method
                logger.debug("Creating streamline actor for view %s", view_name)
                tract_actor = None
                try:
                    tract_actor = self._create_streamline_actor(tract_streamlines, streamline_colors)
                    logger.debug("Streamline actor created, adding to scene")
                    scene.add(tract_actor)
                    logger.debug("Streamline actor added to scene successfully")
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed for view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution or filtering streamlines.",
                            view_name,
                            len(tract_streamlines),
                            sum(len(sl) for sl in tract_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()
                        if tract_actor is not None:
                            del tract_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Set camera position for anatomical view (no streamline rotation needed)
                # Calculate bbox for camera distance using utility function
                logger.debug("Calculating bbox and setting camera for view %s", view_name)
                bbox_size = calculate_bbox_size(tract_streamlines)

                # Use helper method to set camera
                try:
                    self._set_anatomical_camera(
                        scene,
                        centroid,
                        view_name,
                        bbox_size=bbox_size,
                        template_space=template_space,
                    )
                    logger.debug("Camera set successfully for view %s", view_name)
                except (RuntimeError, OSError, MemoryError):
                    logger.exception(
                        "Failed to set camera for view %s. Skipping view.",
                        view_name,
                    )
                    # Clean up and skip this view
                    scene.clear()
                    if tract_actor is not None:
                        del tract_actor
                    del scene
                    gc.collect()
                    continue

                # Record the scene (this can also fail with std::bad_alloc or segfault)
                # Force aggressive memory cleanup before recording to reduce fragmentation
                logger.debug("Recording scene for view %s", view_name)
                logger.debug(
                    "Before recording: tract has %d streamlines, %d total points, image size=%dx%d",
                    len(tract_streamlines),
                    sum(len(sl) for sl in tract_streamlines),
                    figure_size[0],
                    figure_size[1],
                )
                # Aggressive cleanup before recording to reduce memory pressure
                gc.collect()
                gc.collect()  # Multiple passes for circular references

                try:
                    window.record(
                        scene=scene,  # noqa: F821
                        out_path=str(output_image),
                        size=figure_size,
                    )
                    logger.debug("Scene recorded successfully for view %s", view_name)
                    generated_views[view_name] = output_image
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during rendering
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(tract_streamlines),
                            sum(len(sl) for sl in tract_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()  # noqa: F821
                        if tract_actor is not None:
                            del tract_actor
                        del scene  # noqa: F821
                        gc.collect()
                        continue
                    raise

                # Explicitly clean up scene and actors to free memory
                # VTK/FURY objects can hold circular references, so explicit cleanup is critical
                scene.clear()  # noqa: F821
                if tract_actor is not None:
                    del tract_actor
                del scene  # noqa: F821

                # Force garbage collection between views to free memory
                # This is critical for large tractograms to prevent OOM kills
                gc.collect()

            # Clean up remaining large objects
            del tract_streamlines, streamline_colors
            gc.collect()

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to generate anatomical views: {e}",
            ) from e
        else:
            return generated_views

    def view_tract_interactive(
        self,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        show_glass_brain: bool = True,
        flip_lr: bool = False,
        window_size: tuple[int, int] = (800, 800),
        max_streamlines: int | None = None,  # noqa: ARG002
        subsample_factor: float | None = None,  # noqa: ARG002
        max_points_per_streamline: int | None = None,  # noqa: ARG002
        resample_streamlines: bool = False,  # noqa: ARG002
    ) -> None:
        """Display a tract in an interactive 3D viewer.

        Opens a FURY window that allows the user to rotate, zoom, and interact
        with the tract visualization. The window blocks until closed.

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization or via `set_reference_image()`.
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        flip_lr : bool, optional
            Whether to flip left-right (X-axis) when transforming tract.
            This may be needed for some coordinate conventions or file formats
            where the left-right orientation differs (e.g., when working with
            MNI space). Default is False.
        window_size : tuple[int, int], optional
            Size of the interactive window in pixels. Default is (800, 800).

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no reference image is provided and none was set.
        TractographyVisualizationError
            If loading or visualization fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(reference_image="t1w.nii.gz")
        >>> # View a single tract interactively
        >>> visualizer.view_tract_interactive("tract.trk")
        """
        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        tract_file = Path(tract_file)
        if not tract_file.exists():
            raise FileNotFoundError(f"Tract file not found: {tract_file}")

        try:
            # Load tract
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()
            ref_img_obj = nib.load(str(ref_img))
            tract_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Apply left-right flip if needed
            if flip_lr:
                if not tract_streamlines or len(tract_streamlines) == 0:
                    logger.warning("Tract has 0 streamlines before flip. Skipping visualization.")
                    return
                tract_streamlines = Streamlines(
                    [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in tract_streamlines],
                )

            # Filter streamlines if requested
            tract_streamlines = self._downsample_streamlines(tract_streamlines)

            # Check if streamlines are empty after filtering
            if not tract_streamlines or len(tract_streamlines) == 0:
                logger.warning("Tract has 0 streamlines after filtering. Skipping visualization.")
                return

            # Calculate colors based on streamline directions
            streamline_colors = calculate_direction_colors(tract_streamlines)

            # Get centroid for camera positioning
            centroid = calculate_centroid(tract_streamlines)

            # Clean up loaded objects
            del tract, ref_img_obj

            # Create scene
            scene, _ = self._create_scene(ref_img=ref_img, show_glass_brain=show_glass_brain)

            # Create and add streamline actor
            tract_actor = self._create_streamline_actor(tract_streamlines, streamline_colors)
            scene.add(tract_actor)

            # Set camera to show the tract nicely
            bbox_size = calculate_bbox_size(tract_streamlines)
            # Use a default view (coronal) for initial camera position
            self._set_anatomical_camera(
                scene,
                centroid,
                "coronal",
                bbox_size=bbox_size,
            )

            # Show interactive window (blocks until closed)
            # Use window.show() which is blocking and simpler
            window.show(
                scene,
                size=window_size,
                title=str(tract_file.name),
            )

        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to display tract interactively: {e}",
            ) from e

    def generate_atlas_views(
        self,
        atlas_file: str | Path,
        *,
        atlas_ref_img: str | Path | None = None,
        ref_img: str | Path | None = None,  # Alias for atlas_ref_img
        flip_lr: bool = False,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] = (800, 800),
        show_glass_brain: bool = True,
        atlas_name: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Generate anatomical views for an atlas tract.

        Creates static images from coronal, axial, and sagittal views of the atlas
        tract. This is useful for comparing subject tracts to atlas tracts using
        the same viewing angles.

        Parameters
        ----------
        atlas_file : str | Path
            Path to the atlas tractography file.
        atlas_ref_img : str | Path | None, optional
            Path to the reference image that matches the atlas coordinate space
            (e.g., MNI template if atlas is in MNI space).
            This is important if the atlas is in a different space (e.g., MNI)
            than the subject reference image.
        flip_lr: bool, optional
            Whether to flip left-right (X-axis) when transforming atlas.
            This may be needed for some coordinate conventions or file formats
            where the left-right orientation differs. Try this if the atlas
            appears on the wrong side compared to the subject. Default is False.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int], optional
            Size of the output images in pixels. Default is (800, 800).
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        atlas_name : str | None, optional
            Name prefix for output files. If None, uses the stem of atlas_file.
            Default is None.
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping view names to their output file paths.
            Keys: "coronal", "axial", "sagittal".

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available or invalid view name.
        TractographyVisualizationError
            If image generation fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(
        ...     reference_image="t1w.nii.gz", output_directory="output/"
        ... )
        >>> # Generate all atlas views
        >>> atlas_views = visualizer.generate_atlas_views("atlas_tract.trk")
        >>> # Generate specific views
        >>> atlas_views = visualizer.generate_atlas_views(
        ...     "atlas_tract.trk", views=["coronal", "axial"]
        ... )
        """
        # Determine reference image for atlas coordinate transformation
        # If atlas is in different space (e.g., MNI), use atlas_ref_img
        # Support both atlas_ref_img and ref_img (alias) for backward compatibility
        if atlas_ref_img is None and ref_img is not None:
            atlas_ref_img = ref_img
        atlas_ref_img = self._reference_image if atlas_ref_img is None else Path(atlas_ref_img)

        # Determine which views to generate
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided atlas_name or derive from file
        atlas_name = Path(atlas_file).stem if atlas_name is None else str(atlas_name)

        generated_views: dict[str, Path] = {}

        # Check if all output files already exist BEFORE loading atlas tract
        # This prevents unnecessary memory usage when files are already generated
        all_files_exist = True
        for view_name in views_to_generate:
            output_image = output_dir / f"{atlas_name}_atlas_{view_name}.png"
            if output_image.exists() and not overwrite:
                generated_views[view_name] = output_image
                logger.debug("Skipping generation of %s (file already exists)", output_image)
            else:
                all_files_exist = False

        # If all files exist, return early without loading atlas tract
        if all_files_exist:
            return generated_views

        try:
            # Load atlas tract only if we need to generate at least one view
            atlas_tract = load_trk(str(atlas_file), "same", bbox_valid_check=False)
            atlas_tract.to_rasmm()

            # Load reference image
            atlas_ref_img_obj = nib.load(str(atlas_ref_img))

            # Transform atlas streamlines to visualization reference space
            # The atlas tract is already in RASMM after to_rasmm(), so we transform
            # directly to the visualization reference space
            atlas_streamlines = transform_streamlines(
                atlas_tract.streamlines,
                np.linalg.inv(atlas_ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Apply left-right flip if needed (common when atlas is in MNI space)
            if flip_lr:
                # Flip X-axis (left-right) by negating X coordinates
                # This is needed when MNI and native space have different L/R conventions
                if not atlas_streamlines or len(atlas_streamlines) == 0:
                    logger.warning(
                        "Atlas tract has 0 streamlines before flip. Skipping visualization for %s",
                        atlas_file,
                    )
                    return {}
                atlas_streamlines = Streamlines(
                    [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in atlas_streamlines],
                )

                # Also update centroid after flip
                all_points = np.vstack([np.array(sl) for sl in atlas_streamlines])
                centroid = np.mean(all_points, axis=0)

            # Filter streamlines if requested (to avoid VTK segfaults)
            atlas_streamlines = self._downsample_streamlines(
                atlas_streamlines,
                approx_points=0.25,
            )

            # Check if streamlines are empty after filtering
            if not atlas_streamlines or len(atlas_streamlines) == 0:
                logger.warning(
                    "Atlas tract has 0 streamlines after filtering. Skipping visualization for %s",
                    atlas_file,
                )
                return {}

            # Calculate colors based on streamline directions using utility function
            streamline_colors = calculate_direction_colors(atlas_streamlines)

            # Get centroid using utility function (recalculate after any transformations)
            centroid = calculate_centroid(atlas_streamlines)

            # Generate each requested view
            for view_name in views_to_generate:
                output_image = output_dir / f"{atlas_name}_atlas_{view_name}.png"

                # Skip if file already exists (already added to generated_views above)
                if output_image.exists() and not overwrite:
                    continue

                # Create scene using helper method
                scene, _ = self._create_scene(ref_img=atlas_ref_img, show_glass_brain=show_glass_brain)

                # Create actor with original streamlines using helper method
                try:
                    tract_actor = self._create_streamline_actor(atlas_streamlines, streamline_colors)
                    scene.add(tract_actor)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        total_points = sum(len(sl) for sl in atlas_streamlines)
                        logger.exception(
                            "VTK memory allocation failed for atlas comparison view %s (likely std::bad_alloc). "
                            "Atlas has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(atlas_streamlines),
                            total_points,
                        )
                        # Clean up and skip this view
                        scene.clear()
                        del scene
                        gc.collect()
                        continue
                    raise

                # Set camera position for anatomical view using utility function
                bbox_size = calculate_bbox_size(atlas_streamlines)

                # Use helper method to set camera
                self._set_anatomical_camera(
                    scene,
                    centroid,
                    view_name,
                    bbox_size=bbox_size,
                )

                # Record the scene (this can also fail with std::bad_alloc)
                try:
                    window.record(
                        scene=scene,
                        out_path=str(output_image),
                        size=figure_size,
                    )
                    generated_views[view_name] = output_image
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during rendering
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(atlas_streamlines),
                            sum(len(sl) for sl in atlas_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()
                        del tract_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Explicitly clean up scene and actors to free memory
                # VTK/FURY objects can hold circular references, so explicit cleanup is critical
                scene.clear()
                del tract_actor
                del scene

                # Force garbage collection between views to free memory
                # This is critical for large tractograms to prevent OOM kills
                gc.collect()

            # Clean up large objects after all views are generated
            del atlas_tract, atlas_ref_img_obj, atlas_streamlines, streamline_colors
            gc.collect()

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to generate atlas views: {e}",
            ) from e
        else:
            return generated_views

    def plot_afq(
        self,
        metric_file: str | Path,
        metric_str: str,
        tract_file: str | Path,
        atlas_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] = (800, 800),
        show_glass_brain: bool = True,
        colormap: str = "Spectral",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Plot AFQ profile with anatomical views.

        Generates anatomical views (coronal, axial, sagittal) with streamlines
        colored by AFQ profile values.

        Parameters
        ----------
        metric_file : str | Path
            Path to the metric image file.
        metric_str : str
            Name of the metric (e.g., "FA", "MD") for labeling.
        tract_file : str | Path
            Path to the tractography file.
        atlas_file : str | Path
            Path to the atlas tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int], optional
            Size of the output images in pixels. Default is (800, 800).
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        colormap : str, optional
            Name of the colormap to use for AFQ profile values.
            Default is "Spectral".
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping view names to their output file paths.
            Also includes "profile_plot" key for the profile line plot.

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available.
        TractographyVisualizationError
            If image generation fails.
        """
        # Calculate AFQ profile (this will raise InvalidInputError if tract is empty)
        try:
            profile = self.weighted_afq(tract_file, atlas_file, metric_file)
        except InvalidInputError as e:
            # If tract is empty, return empty dict instead of raising
            if "empty" in str(e).lower() or "0 streamlines" in str(e).lower():
                logger.warning(
                    "Skipping AFQ profile for %s: %s",
                    tract_file,
                    e,
                )
                return {}
            raise

        # Determine which views to generate
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        tract_name = Path(tract_file).stem
        generated_views: dict[str, Path] = {}

        # Check profile plot path
        profile_plot_path = output_dir / f"{tract_name}_{metric_str}_profile.png"
        if profile_plot_path.exists() and not overwrite:
            logger.debug("Skipping generation of %s (file already exists)", profile_plot_path)
        generated_views["profile_plot"] = profile_plot_path

        # Check if all view files already exist BEFORE loading tract
        # This prevents unnecessary memory usage when files are already generated
        all_view_files_exist = True
        for view_name in views_to_generate:
            output_image = output_dir / f"{tract_name}_{metric_str}_{view_name}.png"
            if output_image.exists() and not overwrite:
                generated_views[view_name] = output_image
                logger.debug("Skipping generation of %s (file already exists)", output_image)
            else:
                all_view_files_exist = False

        # If all view files exist and profile plot exists, return early without loading tract
        if all_view_files_exist and profile_plot_path.exists() and not overwrite:
            return generated_views

        try:
            # Load tract first to check if it's empty before calculating AFQ profile
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)

            # Check if tract is empty before any processing
            if not tract.streamlines or len(tract.streamlines) == 0:
                logger.warning(
                    "Tract has 0 streamlines. Skipping AFQ profile calculation and visualization for %s",
                    tract_file,
                )
                return {}

            tract.to_rasmm()
            ref_img_obj = nib.load(str(ref_img))
            tract_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Downsample points (to avoid VTK segfaults)
            tract_streamlines = self._downsample_streamlines(tract_streamlines)

            # Check if streamlines are empty after filtering
            if not tract_streamlines or len(tract_streamlines) == 0:
                logger.warning(
                    "Tract has 0 streamlines after filtering. Skipping visualization for %s",
                    tract_file,
                )
                return {}

            # Calculate AFQ profile colors for each streamline
            # Store per-point colors for each streamline
            streamline_point_colors = []
            for sl in tract_streamlines:
                sl_array = np.array(sl)
                # Interpolate profile values to match streamline points
                interpolated_values = np.interp(
                    np.linspace(0, 1, len(sl_array)),
                    np.linspace(0, 1, len(profile)),
                    profile,
                )
                # Create colormap colors for each point
                point_colors = create_colormap(interpolated_values, name=colormap)
                streamline_point_colors.append(point_colors)

            # Get centroid using utility function
            centroid = calculate_centroid(tract_streamlines)

            # Generate each requested view
            for view_name in views_to_generate:
                output_image = output_dir / f"{tract_name}_{metric_str}_{view_name}.png"

                # Skip if file already exists (already added to generated_views above)
                if output_image.exists() and not overwrite:
                    continue

                # Create scene using helper method
                scene, _ = self._create_scene(ref_img=ref_img, show_glass_brain=show_glass_brain)

                # Create actors with AFQ profile colors using original streamlines
                # Colors are already calculated per point, so we can use them directly
                for _i, (sl, point_colors) in enumerate(
                    zip(tract_streamlines, streamline_point_colors),
                ):
                    try:
                        line_actor = actor.line([sl], colors=point_colors)
                        scene.add(line_actor)
                    except RuntimeError as e:
                        # Catch std::bad_alloc and other VTK errors
                        error_msg = str(e).lower()
                        if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                            logger.warning(
                                "VTK memory allocation failed for AFQ profile streamline %d/%d (likely std::bad_alloc). "
                                "Skipping this streamline. Try reducing image resolution or filtering streamlines.",
                                _i + 1,
                                len(tract_streamlines),
                            )
                            continue
                        raise

                # Set camera position for anatomical view using utility function
                bbox_size = calculate_bbox_size(tract_streamlines)

                # Use helper method to set camera
                self._set_anatomical_camera(
                    scene,
                    centroid,
                    view_name,
                    bbox_size=bbox_size,
                )

                # Record the scene (this can also fail with std::bad_alloc)
                try:
                    window.record(
                        scene=scene,
                        out_path=str(output_image),
                        size=figure_size,
                    )
                    generated_views[view_name] = output_image
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during rendering
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(tract_streamlines),
                            sum(len(sl) for sl in tract_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()
                        del scene
                        gc.collect()
                        continue
                    raise

                # Explicitly clean up scene to free memory
                # VTK/FURY objects can hold circular references, so explicit cleanup is critical
                scene.clear()
                del scene

                # Force garbage collection between views to free memory
                # This is critical for large tractograms to prevent OOM kills
                gc.collect()

            # Also create the profile line plot (if it doesn't already exist or overwrite)
            if not profile_plot_path.exists() or overwrite:
                fig, ax = plt.subplots(1, figsize=(8, 6))
                ax.plot(profile)
                ax.set_xlabel("Node along tract")
                ax.set_ylabel(metric_str)
                ax.set_title(f"AFQ Profile: {metric_str}")
                ax.grid(visible=True, alpha=0.3)
                fig.savefig(str(profile_plot_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                logger.debug("Skipping generation of %s (file already exists)", profile_plot_path)

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to plot AFQ profile: {e}",
            ) from e
        else:
            return generated_views

    def calculate_shape_similarity(
        self,
        tract_file: str | Path,
        atlas_file: str | Path,
        *,
        atlas_ref_img: str | Path | None = None,
        flip_lr: bool = False,
        clust_thr: tuple[float, float, float] = (5, 3, 1.5),
        threshold: float = 6,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Calculate shape similarity score between tract and atlas.

        Uses DIPY's bundle_shape_similarity function with the Bundle Adjacency (BA) metric
        to compute how closely the shapes of two bundles align.

        Parameters
        ----------
        tract_file : str | Path
            Path to the subject tractography file.
        atlas_file : str | Path
            Path to the atlas tractography file.
        atlas_ref_img : str | Path | None, optional
            Path to the reference image that matches the atlas coordinate space
            (e.g., MNI template if atlas is in MNI space). If None, assumes atlas
            is in the same space as the subject tract. This is important if the
            atlas is in a different space (e.g., MNI) than the subject.
        flip_lr : bool, optional
            Whether to flip left-right (X-axis) when transforming atlas.
            This may be needed for some coordinate conventions or file formats
            where the left-right orientation differs. Default is False.
        clust_thr : tuple[float, float, float], optional
            Clustering thresholds for QuickBundlesX used internally.
            Default is (5, 3, 1.5).
        threshold : float, optional
            Threshold controlling the strictness of the shape similarity assessment.
            A smaller threshold requires the bundles to be more similar to achieve
            a higher score. Default is 6.
        rng : np.random.Generator | None, optional
            Random number generator. If None, creates a new one. Default is None.

        Returns
        -------
        float
            Bundle similarity score (BA value). Higher values indicate greater
            similarity between the bundles.

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If tracts are empty or invalid.
        TractographyVisualizationError
            If calculation fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer()
        >>> score = visualizer.calculate_shape_similarity(
        ...     "subject_tract.trk", "atlas_tract.trk"
        ... )
        >>> print(f"Shape similarity score: {score}")
        """
        try:
            # Load both tracts
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()

            atlas_tract = load_trk(str(atlas_file), "same", bbox_valid_check=False)
            atlas_tract.to_rasmm()

            # Check if tracts are empty
            if not tract.streamlines or len(tract.streamlines) == 0:
                raise InvalidInputError("Subject tract is empty.")
            if not atlas_tract.streamlines or len(atlas_tract.streamlines) == 0:
                raise InvalidInputError("Atlas tract is empty.")

            # Transform atlas to subject space if needed
            if atlas_ref_img is not None:
                atlas_ref_img_obj = nib.load(str(atlas_ref_img))

                # Transform atlas streamlines to subject reference space
                atlas_streamlines = transform_streamlines(
                    atlas_tract.streamlines,
                    np.linalg.inv(atlas_ref_img_obj.affine),  # type: ignore[attr-defined]
                )

                # Apply left-right flip if needed
                if flip_lr:
                    atlas_streamlines = Streamlines(
                        [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in atlas_streamlines],
                    )
            else:
                # No transformation needed, use streamlines directly
                atlas_streamlines = atlas_tract.streamlines
                if flip_lr:
                    atlas_streamlines = Streamlines(
                        [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in atlas_streamlines],
                    )

            # Use subject tract streamlines directly (already in RASMM)
            subject_streamlines = tract.streamlines

            # Create random number generator if not provided
            if rng is None:
                rng = np.random.default_rng()

            # Calculate shape similarity using DIPY's function
            similarity_score = bundle_shape_similarity(
                subject_streamlines,
                atlas_streamlines,
                rng,
                clust_thr=clust_thr,
                threshold=threshold,
            )

            return float(similarity_score)
        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError, IndexError) as e:
            raise TractographyVisualizationError(
                f"Failed to calculate shape similarity: {e}",
            ) from e

    def visualize_shape_similarity(
        self,
        tract_file: str | Path,
        atlas_file: str | Path,
        *,
        atlas_ref_img: str | Path | None = None,
        flip_lr: bool = False,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] = (800, 800),
        show_glass_brain: bool = True,
        subject_color: tuple[float, float, float] = (1.0, 0.0, 0.0),  # Red
        atlas_color: tuple[float, float, float] = (0.0, 0.0, 1.0),  # Blue
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Visualize shape similarity by overlaying subject and atlas tracts.

        Generates anatomical views (coronal, axial, sagittal) showing both tracts
        overlaid with different colors to visualize their shape similarity.

        Parameters
        ----------
        tract_file : str | Path
            Path to the subject tractography file.
        atlas_file : str | Path
            Path to the atlas tractography file.
        atlas_ref_img : str | Path | None, optional
            Path to the reference image that matches the atlas coordinate space
            (e.g., MNI template if atlas is in MNI space).
        flip_lr : bool, optional
            Whether to flip left-right (X-axis) when transforming atlas.
            Default is False.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int], optional
            Size of the output images in pixels. Default is (800, 800).
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        subject_color : tuple[float, float, float], optional
            RGB color for subject tract (0-1 range). Default is (1.0, 0.0, 0.0) (red).
        atlas_color : tuple[float, float, float], optional
            RGB color for atlas tract (0-1 range). Default is (0.0, 0.0, 1.0) (blue).
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping view names to their output file paths.
            Keys: "coronal", "axial", "sagittal".

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available or invalid view name.
        TractographyVisualizationError
            If visualization fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(
        ...     reference_image="t1w.nii.gz", output_directory="output/"
        ... )
        >>> views = visualizer.visualize_shape_similarity(
        ...     "subject_tract.trk", "atlas_tract.trk"
        ... )
        """
        # Determine which views to generate
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        tract_name = Path(tract_file).stem
        atlas_name = Path(atlas_file).stem
        generated_views: dict[str, Path] = {}

        # Check if all output files already exist BEFORE loading tracts
        # This prevents unnecessary memory usage when files are already generated
        all_files_exist = True
        for view_name in views_to_generate:
            output_image = output_dir / f"similarity_{tract_name}_vs_{atlas_name}_{view_name}.png"
            if output_image.exists() and not overwrite:
                generated_views[view_name] = output_image
                logger.debug("Skipping generation of %s (file already exists)", output_image)
            else:
                all_files_exist = False

        # If all files exist, return early without loading tracts
        if all_files_exist:
            return generated_views

        try:
            # Load both tracts only if we need to generate at least one view
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()

            atlas_tract = load_trk(str(atlas_file), "same", bbox_valid_check=False)
            atlas_tract.to_rasmm()

            # Check if tracts are empty
            if not tract.streamlines or len(tract.streamlines) == 0:
                raise InvalidInputError("Subject tract is empty.")
            if not atlas_tract.streamlines or len(atlas_tract.streamlines) == 0:
                raise InvalidInputError("Atlas tract is empty.")

            # Load reference images and transform tracts to same space
            atlas_ref_img_obj = nib.load(str(atlas_ref_img))

            # Transform subject tract to reference space
            subject_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(atlas_ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # No transformation needed, use streamlines directly
            atlas_streamlines = transform_streamlines(
                atlas_tract.streamlines,
                np.linalg.inv(atlas_ref_img_obj.affine),  # type: ignore[attr-defined]
            )
            if flip_lr:
                atlas_streamlines = Streamlines(
                    [np.column_stack([-sl[:, 0], sl[:, 1], sl[:, 2]]) for sl in atlas_streamlines],
                )

            # Downsample points (to avoid VTK segfaults)
            subject_streamlines = self._downsample_streamlines(subject_streamlines)
            atlas_streamlines = self._downsample_streamlines(atlas_streamlines)

            # Check if streamlines are empty after filtering
            if (not subject_streamlines or len(subject_streamlines) == 0) and (
                not atlas_streamlines or len(atlas_streamlines) == 0
            ):
                logger.warning(
                    "Both subject and atlas tracts have 0 streamlines after filtering. Skipping visualization.",
                )
                return {}
            if not subject_streamlines or len(subject_streamlines) == 0:
                logger.warning(
                    "Subject tract has 0 streamlines after filtering. Skipping visualization.",
                )
                return {}
            if not atlas_streamlines or len(atlas_streamlines) == 0:
                logger.warning(
                    "Atlas tract has 0 streamlines after filtering. Skipping visualization.",
                )
                return {}

            # Calculate combined centroid for rotation (from both tracts)
            # Calculate combined centroid using utility function
            centroid = calculate_combined_centroid(subject_streamlines, atlas_streamlines)

            # Generate each requested view
            for view_name in views_to_generate:
                output_image = output_dir / f"similarity_{tract_name}_vs_{atlas_name}_{view_name}.png"

                # Skip if file already exists (already added to generated_views above)
                if output_image.exists() and not overwrite:
                    continue

                # Create scene using helper method
                scene, brain_actor = self._create_scene(ref_img=atlas_ref_img, show_glass_brain=show_glass_brain)

                # Add subject tract with subject color (single color for all streamlines)
                # Use original streamlines - camera handles view
                subject_colors = np.tile(subject_color, (len(subject_streamlines), 1))
                try:
                    subject_actor = actor.line(subject_streamlines, colors=subject_colors, opacity=0.05)
                    scene.add(subject_actor)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        total_points = sum(len(sl) for sl in subject_streamlines)
                        logger.exception(
                            "VTK memory allocation failed for shape similarity subject actor (likely std::bad_alloc). "
                            "Subject has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            len(subject_streamlines),
                            total_points,
                        )
                        # Clean up and skip this view
                        scene.clear()
                        if brain_actor is not None:
                            del brain_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Add atlas tract with atlas color (single color for all streamlines)
                # Use original streamlines - camera handles view
                atlas_colors = np.tile(atlas_color, (len(atlas_streamlines), 1))
                try:
                    atlas_actor = actor.line(atlas_streamlines, colors=atlas_colors, opacity=0.05)
                    scene.add(atlas_actor)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        total_points = sum(len(sl) for sl in atlas_streamlines)
                        logger.exception(
                            "VTK memory allocation failed for shape similarity atlas actor (likely std::bad_alloc). "
                            "Atlas has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            len(atlas_streamlines),
                            total_points,
                        )
                        # Clean up and skip this view
                        scene.clear()
                        if subject_actor is not None:
                            del subject_actor
                        if brain_actor is not None:
                            del brain_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Set camera position for anatomical view
                # Use combined bbox of both tracts for camera distance
                # Calculate combined bbox using utility function
                bbox_size = calculate_combined_bbox_size(subject_streamlines, atlas_streamlines)

                # Use helper method to set camera
                self._set_anatomical_camera(
                    scene,
                    centroid,
                    view_name,
                    bbox_size=bbox_size,
                )

                # Record the scene (this can also fail with std::bad_alloc)
                try:
                    window.record(
                        scene=scene,
                        out_path=str(output_image),
                        size=figure_size,
                    )
                    generated_views[view_name] = output_image
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during rendering
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Subject has %d streamlines, atlas has %d streamlines. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(subject_streamlines),
                            len(atlas_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()
                        del subject_actor, atlas_actor
                        if show_glass_brain:
                            del brain_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Explicitly clean up scene and actors to free memory
                # VTK/FURY objects can hold circular references, so explicit cleanup is critical
                scene.clear()
                del subject_actor, atlas_actor
                if show_glass_brain:
                    del brain_actor
                del scene

                # Force garbage collection between views to free memory
                # This is critical for large tractograms to prevent OOM kills
                gc.collect()

                generated_views[view_name] = output_image

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to visualize shape similarity: {e}",
            ) from e
        else:
            return generated_views

    def compare_before_after_cci(
        self,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] = (800, 800),
        show_glass_brain: bool = True,
        bins: int = 100,
        overwrite: bool = False,
        cci: np.ndarray | None = None,
        keep_cci: np.ndarray | None = None,
        keep_tract: StatefulTractogram | None = None,
        long_streamlines: Streamlines | None = None,
    ) -> dict[str, Path] | dict[str, Path | dict[str, Path]]:
        """Generate CCI histogram and before/after CCI views (original vs filtered).

        Calls :meth:`calc_cci` (or uses optional precomputed outputs) and produces:
        - A histogram of CCI values (`<tract>_cci_histogram.png`).
        - For each anatomical view: original tract colored by CCI (`cci_before_<tract>_<view>.png`)
          and filtered tract colored by filtered CCI (`cci_after_<tract>_<view>.png`).

        Parameters
        ----------
        tract_file : str | Path
            Path to the tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int], optional
            Size of each image in pixels. Default is (800, 800).
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        bins : int, optional
            Number of bins for the CCI histogram. Default is 100.
        max_streamlines : int | None, optional
            Maximum number of streamlines to keep for visualization. If None, no limit.
        subsample_factor : float | None, optional
            Fraction of streamlines to keep (0-1). If None, no subsampling.
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.
        cci : np.ndarray | None, optional
            Precomputed CCI for all long streamlines. If provided with keep_cci,
            keep_tract, and long_streamlines, :meth:`calc_cci` is not called.
        keep_cci : np.ndarray | None, optional
            Precomputed CCI above threshold (aligned with keep_tract).
        keep_tract : StatefulTractogram | None, optional
            Precomputed filtered tractogram.
        long_streamlines : Streamlines | None, optional
            Precomputed long streamlines used for CCI (length > min_streamline_length).

        Returns
        -------
        dict[str, Path] | dict[str, Path | dict[str, Path]]
            - Key "histogram": Path to the CCI histogram image.
            - For each view (coronal, axial, sagittal): a dict with keys
              before and after, each mapping to the Path of that image.

        Raises
        ------
        InvalidInputError
            If no output directory or reference image, or invalid view name.
        TractographyVisualizationError
            If comparison fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(
        ...     reference_image="t1w.nii.gz", output_directory="output/"
        ... )
        >>> result = visualizer.compare_before_after_cci("tract.trk")
        >>> result["histogram"]
        >>> list(result["coronal"].keys())
        ['before', 'after']
        """
        # Determine which views to generate
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        tract_name = Path(tract_file).stem
        hist_path = output_dir / f"{tract_name}_cci_histogram.png"
        generated_views: dict[str, Path | dict[str, Path]] = {}

        # Check if histogram and all view files already exist (skip loading tract)
        all_files_exist = True
        if not hist_path.exists() or overwrite:
            all_files_exist = False
        else:
            generated_views["histogram"] = hist_path
        for view_name in views_to_generate:
            output_before = output_dir / f"cci_before_{tract_name}_{view_name}.png"
            output_after = output_dir / f"cci_after_{tract_name}_{view_name}.png"
            if output_before.exists() and output_after.exists() and not overwrite:
                generated_views[view_name] = {"before": output_before, "after": output_after}
                logger.debug(
                    "Skipping generation of before/after CCI for %s (%s, %s already exist)",
                    view_name,
                    output_before,
                    output_after,
                )
            else:
                all_files_exist = False

        if all_files_exist:
            return generated_views

        try:
            # Get CCI and tracts: use precomputed or call calc_cci
            if cci is not None and keep_cci is not None and keep_tract is not None and long_streamlines is not None:
                pass  # use provided arrays/tracts
            else:
                cci, keep_cci, keep_tract, long_streamlines = self.calc_cci(tract_file, ref_img=ref_img)

            if len(cci) == 0:
                logger.warning("No CCI values; skipping CCI visualization.")
                return generated_views

            ref_img_obj = nib.load(str(ref_img))
            inv_affine = np.linalg.inv(ref_img_obj.affine)  # type: ignore[attr-defined]

            # Histogram
            if not hist_path.exists() or overwrite:
                fig, ax = plt.subplots(1, figsize=(8, 6))
                ax.hist(cci, bins=bins, histtype="step")
                ax.set_xlabel("CCI")
                ax.set_ylabel("# streamlines")
                ax.set_title("CCI Distribution")
                ax.grid(visible=True, alpha=0.3)
                fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
            generated_views["histogram"] = hist_path

            # Transform to reference space and optionally subsample
            before_streamlines = transform_streamlines(long_streamlines, inv_affine)
            after_streamlines = transform_streamlines(
                keep_tract.streamlines,
                inv_affine,
            )
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
            centroid = calculate_combined_centroid(before_streamlines, after_streamlines)

            # Generate each requested view
            for view_name in views_to_generate:
                output_before = output_dir / f"cci_before_{tract_name}_{view_name}.png"
                output_after = output_dir / f"cci_after_{tract_name}_{view_name}.png"

                if output_before.exists() and output_after.exists() and not overwrite:
                    continue

                # Before: original tract colored by CCI
                scene_before, brain_actor_before = self._create_scene(
                    ref_img=ref_img,
                    show_glass_brain=show_glass_brain,
                )
                scene_before.add(bar)
                try:
                    before_actor = actor.line(
                        before_streamlines,
                        colors=cci_before,
                        linewidth=0.1,
                        lookup_colormap=lut_cmap,
                    )
                    scene_before.add(before_actor)
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed for before CCI actor (likely std::bad_alloc). "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                        )
                        scene_before.clear()
                        del scene_before
                        if brain_actor_before is not None:
                            del brain_actor_before
                        gc.collect()
                        continue
                    raise

                bbox_size_before = calculate_bbox_size(before_streamlines)
                self._set_anatomical_camera(
                    scene_before,
                    centroid,
                    view_name,
                    bbox_size=bbox_size_before,
                )

                # After: filtered tract colored by filtered CCI
                scene_after, brain_actor_after = self._create_scene(
                    ref_img=ref_img,
                    show_glass_brain=show_glass_brain,
                )
                try:
                    after_actor = actor.line(
                        after_streamlines,
                        linewidth=0.1,
                    )
                    scene_after.add(after_actor)
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed for after CCI actor (likely std::bad_alloc). "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                        )
                        scene_before.clear()
                        scene_after.clear()
                        del before_actor
                        if brain_actor_before is not None:
                            del brain_actor_before
                        if brain_actor_after is not None:
                            del brain_actor_after
                        del scene_before, scene_after
                        gc.collect()
                        continue
                    raise

                bbox_size_after = calculate_bbox_size(after_streamlines)
                self._set_anatomical_camera(
                    scene_after,
                    centroid,
                    view_name,
                    bbox_size=bbox_size_after,
                )

                try:
                    window.record(
                        scene=scene_before,
                        out_path=str(output_before),
                        size=figure_size,
                    )
                    window.record(
                        scene=scene_after,
                        out_path=str(output_after),
                        size=figure_size,
                    )
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                        )
                        scene_before.clear()
                        scene_after.clear()
                        del before_actor, after_actor
                        if brain_actor_before is not None:
                            del brain_actor_before
                        if brain_actor_after is not None:
                            del brain_actor_after
                        del scene_before, scene_after
                        gc.collect()
                        continue
                    raise

                scene_before.clear()
                scene_after.clear()
                del before_actor, after_actor
                if brain_actor_before is not None:
                    del brain_actor_before
                if brain_actor_after is not None:
                    del brain_actor_after
                del scene_before, scene_after
                gc.collect()

                generated_views[view_name] = {"before": output_before, "after": output_after}

        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to compare before/after CCI: {e}",
            ) from e
        else:
            return generated_views

    def visualize_bundle_assignment(
        self,
        tract_file: str | Path,
        atlas_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        n_segments: int = 100,
        views: list[str] | None = None,
        output_dir: str | Path | None = None,
        figure_size: tuple[int, int] = (800, 800),
        show_glass_brain: bool = True,
        colormap: str = "random",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Visualize bundle assignment map using DIPY's assignment_map.

        Assigns each streamline in the target tract to a segment of the model
        bundle using DIPY's assignment_map function. Color-codes streamlines
        by their assigned segment. Generates anatomical views showing which
        streamlines belong to which bundle segment.

        Parameters
        ----------
        tract_file : str | Path
            Path to the target tractography file (streamlines to assign).
        atlas_file : str | Path
            Path to the atlas tractography file (reference bundle for assignment).
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization.
        n_segments : int, optional
            Number of segments to divide the model bundle into. Each streamline
            in the target tract will be assigned to the closest segment.
            Default is 100.
        views : list[str] | None, optional
            List of views to generate. Options: "coronal", "axial", "sagittal".
            If None, generates all three views. Default is None.
        output_dir : str | Path | None, optional
            Output directory for generated images. If None, uses the output
            directory set during initialization.
        figure_size : tuple[int, int], optional
            Size of the output images in pixels. Default is (800, 800).
        show_glass_brain : bool, optional
            Whether to show the glass brain outline. Default is True.
        colormap : str, optional
            Name of the colormap to use for segment colors. Should be a
            discrete colormap (e.g., "tab20", "Set3", "Paired").
            Default is "tab20".
        overwrite : bool, optional
            If True, regenerate images even when output files already exist.
            Default is False.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping view names to their output file paths.
            Keys: "coronal", "axial", "sagittal".

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available or invalid view name.
        TractographyVisualizationError
            If visualization fails.

        Examples
        --------
        >>> visualizer = TractographyVisualizer(
        ...     reference_image="t1w.nii.gz", output_directory="output/"
        ... )
        >>> views = visualizer.visualize_bundle_assignment(
        ...     "target_tract.trk", "model_tract.trk", n_segments=100
        ... )
        """
        rgb = (2, 3)

        # Determine which views to generate
        if views is None:
            views_to_generate = list(ANATOMICAL_VIEW_NAMES)
        else:
            views_to_generate = views
            invalid_views = [v for v in views_to_generate if v not in ANATOMICAL_VIEW_NAMES]
            if invalid_views:
                raise InvalidInputError(
                    f"Invalid view names: {invalid_views}. Valid options: {list(ANATOMICAL_VIEW_NAMES)}",
                )

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        tract_name = Path(tract_file).stem
        generated_views: dict[str, Path] = {}

        # Check if all output files already exist BEFORE loading tracts
        # This prevents unnecessary memory usage when files are already generated
        all_files_exist = True
        for view_name in views_to_generate:
            output_image = output_dir / f"bundle_assignment_{tract_name}_{view_name}.png"
            if output_image.exists() and not overwrite:
                generated_views[view_name] = output_image
                logger.debug("Skipping generation of %s (file already exists)", output_image)
            else:
                all_files_exist = False

        # If all files exist, return early without loading tracts
        if all_files_exist:
            return generated_views

        try:
            # Load both tracts only if we need to generate at least one view
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()

            atlas_tract = load_trk(str(atlas_file), "same", bbox_valid_check=False)
            atlas_tract.to_rasmm()

            if not tract.streamlines or len(tract.streamlines) == 0:
                raise InvalidInputError("Target tractogram is empty.")
            if not atlas_tract.streamlines or len(atlas_tract.streamlines) == 0:
                raise InvalidInputError("Atlas tractogram is empty.")

            # Transform both tracts to reference space for assignment
            # assignment_map requires both tracts to be in the same coordinate space
            ref_img_obj = nib.load(str(ref_img))
            tract_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )
            atlas_streamlines = transform_streamlines(
                atlas_tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Downsample points (to avoid VTK segfaults)
            tract_streamlines = self._downsample_streamlines(tract_streamlines)
            atlas_streamlines = self._downsample_streamlines(atlas_streamlines)

            # Calculate assignments on transformed streamlines (both in same space)
            # This ensures consistent colors across all rotations
            # Use assignment_map to assign target streamlines to model bundle segments
            # assignment_map returns per-point assignments (one assignment per point)
            # Assignments are in the order points appear when iterating through streamlines sequentially
            assignment_indices = assignment_map(tract_streamlines, atlas_streamlines, n_segments)
            assignment_indices = np.array(assignment_indices)

            # Validate that the number of assignments matches the number of points
            total_points = sum(len(sl) for sl in tract_streamlines)
            if len(assignment_indices) != total_points:
                msg = (
                    f"Mismatch between number of assignments ({len(assignment_indices)}) "
                    f"and number of points ({total_points}). This may indicate an issue "
                    "with streamline filtering or assignment_map."
                )
                raise ValueError(msg)  # noqa: TRY301

            # Generate colors for each segment
            # Use random colors like DIPY example (produces wide spectrum of distinct colors)
            # or use a colormap if specified (e.g., "Spectral", "hsv", "rainbow" for wide spectrum)
            if colormap == "random" or colormap is None:
                # Match DIPY example: use random colors for maximum color differentiation
                # This produces a wide spectrum of distinct colors like the example image
                rng = np.random.default_rng()
                segment_colors = [tuple(rng.random(3)) for si in range(n_segments)]
            else:
                # Use specified colormap
                # For wide spectrum, consider: "Spectral", "hsv", "rainbow", "turbo"
                segment_colors_array = create_colormap(
                    np.linspace(0, 1, n_segments),
                    name=colormap,
                )
                # Convert to RGB (0-1 range), remove alpha if present
                segment_colors_array = (
                    segment_colors_array[:, : rgb[1]]
                    if segment_colors_array.shape[1] > rgb[1]
                    else segment_colors_array
                )
                # Convert to list of tuples for compatibility
                segment_colors = [tuple(segment_colors_array[i]) for i in range(n_segments)]

            # Create per-point colors based on assignment (matching DIPY example pattern)
            # Convert to list of tuples as in DIPY example to ensure proper color application
            # This is calculated once and reused for all rotations
            # Each point gets the color corresponding to its segment assignment
            # This creates the banding effect as points along a streamline are assigned to different segments
            # IMPORTANT: assignment_indices are in the order points appear when iterating through
            # streamlines sequentially (streamline 0 all points, then streamline 1 all points, etc.)
            # Construct colors by explicitly iterating through streamlines to ensure order matches actor.line()
            point_colors = []
            assignment_idx = 0
            for sl in tract_streamlines:
                for _ in range(len(sl)):
                    # Ensure we don't go out of bounds
                    if assignment_idx >= len(assignment_indices):
                        msg = (
                            f"Assignment index {assignment_idx} exceeds number of assignments "
                            f"({len(assignment_indices)}). This indicates a mismatch between "
                            "streamlines and assignments."
                        )
                        raise ValueError(msg)  # noqa: TRY301
                    # Get the color for this point's assigned segment
                    seg_idx = assignment_indices[assignment_idx]
                    # Ensure index is within bounds
                    if seg_idx < 0 or seg_idx >= len(segment_colors):
                        # Fallback to first color if index is out of bounds
                        point_colors.append(segment_colors[0])
                    else:
                        point_colors.append(tuple(segment_colors[seg_idx]))
                    assignment_idx += 1

            # Final validation: ensure we processed all assignments
            if assignment_idx != len(assignment_indices):
                msg = (
                    f"Processed {assignment_idx} points but have {len(assignment_indices)} assignments. "
                    "This indicates a mismatch between streamlines and assignments."
                )
                raise ValueError(msg)  # noqa: TRY301

            # Debug: Show sample of assignments and colors for first streamline (after point_colors is created)
            point_idx = 0
            first_sl = tract_streamlines[0]
            first_sl_assignments = assignment_indices[point_idx : point_idx + len(first_sl)]
            first_sl_colors = point_colors[point_idx : point_idx + len(first_sl)]
            logger.debug(
                "First streamline: %d points, %d unique segments",
                len(first_sl),
                len(np.unique(first_sl_assignments)),
            )
            logger.debug("Sample assignments (first 10): %s", first_sl_assignments[:10])
            logger.debug("Sample colors (first 3): %s", first_sl_colors[:3])

            # Get centroid using utility function
            centroid = calculate_centroid(tract_streamlines)
            gc.collect()

            # Generate each requested view
            for view_name in views_to_generate:
                output_image = output_dir / f"bundle_assignment_{tract_name}_{view_name}.png"

                # Skip if file already exists (already added to generated_views above)
                if output_image.exists() and not overwrite:
                    continue

                # Create scene using helper method
                scene, brain_actor = self._create_scene(ref_img=ref_img, show_glass_brain=show_glass_brain)

                # Convert colors to numpy array format (N x 3) for actor.line
                point_colors_array = np.array(point_colors, dtype=np.float32)

                # Ensure the array has the correct shape (N points x 3 RGB values)
                if point_colors_array.ndim != rgb[0] or point_colors_array.shape[1] != rgb[1]:

                    def _create_shape_error(error_msg: str) -> ValueError:
                        return ValueError(error_msg)

                    msg = (
                        f"point_colors_array has incorrect shape: {point_colors_array.shape}. "
                        f"Expected (N, 3) where N is the number of points."
                    )
                    raise _create_shape_error(msg)

                # Use original streamlines with original colors - no rotation needed
                try:
                    tract_actor = actor.line(tract_streamlines, colors=point_colors_array, fake_tube=True, linewidth=6)
                    scene.add(tract_actor)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        total_points = sum(len(sl) for sl in tract_streamlines)
                        logger.exception(
                            "VTK memory allocation failed for bundle assignment view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(tract_streamlines),
                            total_points,
                        )
                        # Clean up and skip this view
                        scene.clear()
                        if brain_actor is not None:
                            del brain_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Set camera position for anatomical view (no streamline rotation needed)
                # Calculate bbox for camera distance using utility function
                bbox_size = calculate_bbox_size(tract_streamlines)

                # Use helper method to set camera
                self._set_anatomical_camera(
                    scene,
                    centroid,
                    view_name,
                    bbox_size=bbox_size,
                )

                # Record the scene (this can also fail with std::bad_alloc)
                try:
                    window.record(
                        scene=scene,
                        out_path=str(output_image),
                        size=figure_size,
                    )
                    generated_views[view_name] = output_image
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during rendering
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during rendering for view %s (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing image resolution (figure_size) or processing with n_jobs=1.",
                            view_name,
                            len(tract_streamlines),
                            sum(len(sl) for sl in tract_streamlines),
                        )
                        # Clean up and skip this view
                        scene.clear()
                        del tract_actor
                        if show_glass_brain:
                            del brain_actor
                        del scene
                        gc.collect()
                        continue
                    raise

                # Explicitly clean up scene and actors to free memory
                # VTK/FURY objects can hold circular references, so explicit cleanup is critical
                scene.clear()
                del tract_actor
                if show_glass_brain:
                    del brain_actor
                del scene

                # Force garbage collection between views to free memory
                # This is critical for large tractograms to prevent OOM kills
                gc.collect()

                generated_views[view_name] = output_image

        except (OSError, ValueError, RuntimeError, IndexError) as e:
            raise TractographyVisualizationError(
                f"Failed to visualize bundle assignment: {e}",
            ) from e
        else:
            return generated_views

    def generate_gif(
        self,
        name: str,
        tract_file: str | Path,
        ref_img: str | Path | None = None,
        *,
        ref_file: str | Path | None = None,  # Alias for ref_img
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        max_streamlines: int | None = None,  # noqa: ARG002
        subsample_factor: float | None = None,  # noqa: ARG002
        max_points_per_streamline: int | None = None,  # noqa: ARG002
        resample_streamlines: bool = False,  # noqa: ARG002
    ) -> Path:
        """Generate a GIF animation of rotating tractography.

        Parameters
        ----------
        name : str
            Base name for the output GIF file (without extension).
        tract_file : str | Path
            Path to the tractography file.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization.
        output_dir : str | Path | None, optional
            Output directory. If None, uses the output directory set during
            initialization or creates a default directory.
        overwrite : bool, optional
            If True, regenerate GIF even when output file already exists.
            Default is False.

        Returns
        -------
        Path
            Path to the generated GIF file.

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If no output directory is available.
        TractographyVisualizationError
            If GIF generation fails.
        """
        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        gif_filename = output_dir / f"{name}.gif"

        # Skip if file already exists
        if gif_filename.exists() and not overwrite:
            logger.debug("Skipping generation of %s (file already exists)", gif_filename)
            return gif_filename

        # Support both ref_img and ref_file (alias) for backward compatibility
        if ref_img is None and ref_file is not None:
            ref_img = ref_file

        try:
            # Load tract and get streamlines for rotation
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()
            ref_img_obj = nib.load(str(ref_img))
            tract_streamlines = transform_streamlines(
                tract.streamlines,
                np.linalg.inv(ref_img_obj.affine),  # type: ignore[attr-defined]
            )

            # Downsample points (to avoid VTK segfaults)
            tract_streamlines = self._downsample_streamlines(tract_streamlines)

            # Create initial actors with error handling for std::bad_alloc
            try:
                tract_actor = actor.line(tract_streamlines)
            except RuntimeError as e:
                # Catch std::bad_alloc and other VTK errors
                error_msg = str(e).lower()
                if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                    total_points = sum(len(sl) for sl in tract_streamlines)
                    logger.exception(
                        "VTK memory allocation failed when creating initial actor for GIF (likely std::bad_alloc). "
                        "Tract has %d streamlines with %d total points. "
                        "Try filtering streamlines, reducing gif_size, or using n_jobs=1.",
                        len(tract_streamlines),
                        total_points,
                    )
                    raise TractographyVisualizationError(
                        f"Failed to create actor for GIF: {e}. "
                        "Try reducing tract size, gif_size, or processing with n_jobs=1.",
                    ) from e
                raise
            brain_actor = self.get_glass_brain(ref_img)
            scene = window.Scene()
            scene.add(brain_actor)
            scene.add(tract_actor)
            scene.setBackground(color=(1, 1, 1))

            angles = np.linspace(0, 360, self.gif_frames, endpoint=False)
            rotation_axis = np.array([0, 0, 1])  # Rotate around Z-axis
            rotation_center = np.array([0, 0, 0])
            gif_frames = []

            for angle in angles:
                rot_matrix = Rotation.from_rotvec(
                    angle * np.pi / 180 * rotation_axis,
                ).as_matrix()

                # Rotate streamlines around the origin
                rotated_streamlines = [
                    np.dot(s - rotation_center, rot_matrix.T) + rotation_center for s in tract_streamlines
                ]
                # Create actor for rotated streamlines with error handling
                try:
                    stream_actor = actor.line(rotated_streamlines)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.warning(
                            "VTK memory allocation failed when creating rotated actor for GIF frame at angle %.1f° (likely std::bad_alloc). "
                            "Skipping this frame. Try filtering streamlines, reducing gif_size, or using n_jobs=1.",
                            angle,
                        )
                        # Clean up and skip this frame
                        del rotated_streamlines
                        gc.collect()
                        continue
                    raise

                # Convert to 4x4 transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = rotation_center - np.dot(
                    rot_matrix,
                    rotation_center,
                )

                # Rotate glass brain using VTK transform
                transform = vtk.vtkTransform()
                transform.Concatenate(transform_matrix.flatten())

                brain_actor.SetUserTransform(transform)

                # Clear scene and re-add actors
                scene.clear()
                scene.add(stream_actor)
                scene.add(brain_actor)

                # Snapshot can also fail with std::bad_alloc
                try:
                    frame = window.snapshot(scene, size=self.gif_size)
                    gif_frames.append(frame)
                except RuntimeError as e:
                    # Catch std::bad_alloc and other VTK errors during snapshot
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "VTK memory allocation failed during GIF frame snapshot (likely std::bad_alloc). "
                            "Tract has %d streamlines with %d total points. "
                            "Try reducing gif_size or processing with n_jobs=1.",
                            len(tract_streamlines),
                            sum(len(sl) for sl in tract_streamlines),
                        )
                        # Clean up and break out of loop
                        scene.clear()
                        del stream_actor, rotated_streamlines
                        gc.collect()
                        # If we have some frames, save what we have
                        if gif_frames:
                            logger.warning("Saving partial GIF with %d frames", len(gif_frames))
                        else:
                            # No frames captured, skip this view
                            del gif_frames, tract_actor, brain_actor, scene
                            gc.collect()
                            continue
                    raise

                # Force memory cleanup
                del stream_actor, rotated_streamlines
                gc.collect()

            # Explicitly clean up scene and actors after all frames are captured
            # VTK/FURY objects can hold circular references, so explicit cleanup is critical
            scene.clear()
            del tract_actor, brain_actor, scene
            del tract, tract_streamlines, ref_img_obj

            # Force garbage collection to free memory
            # This is critical for large tractograms to prevent OOM kills
            gc.collect()

            # Save as optimized GIF
            imageio.mimsave(
                str(gif_filename),
                gif_frames,
                duration=self.gif_duration,
                palettesize=self.gif_palette_size,
            )

            # Clean up frames after saving
            del gif_frames
            gc.collect()
        except TractographyVisualizationError:
            raise
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to generate GIF: {e}",
            ) from e
        else:
            return gif_filename

    def convert_gif_to_mp4(
        self,
        gif_path: str | Path,
        mp4_path: str | Path | None = None,
        *,
        fps: int = 10,
        overwrite: bool = False,
    ) -> Path:
        """Convert a GIF file to MP4 format.

        Parameters
        ----------
        gif_path : str | Path
            Path to the input GIF file.
        mp4_path : str | Path | None, optional
            Path to the output MP4 file. If None, uses the same name as GIF
            with .mp4 extension.
        fps : int, optional
            Frames per second for the video. Default is 10.
        overwrite : bool, optional
            If True, overwrite MP4 even when output file already exists.
            Default is False.

        Returns
        -------
        Path
            Path to the generated MP4 file.

        Raises
        ------
        FileNotFoundError
            If the GIF file is not found.
        TractographyVisualizationError
            If conversion fails.
        """
        gif_path_obj = Path(gif_path)
        if mp4_path is None:
            mp4_path = gif_path_obj.with_suffix(".mp4")
        else:
            mp4_path = Path(mp4_path)
            mp4_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists
        if mp4_path.exists() and not overwrite:
            logger.debug("Skipping conversion of %s to %s (file already exists)", gif_path, mp4_path)
            return mp4_path

        try:
            reader = imageio.get_reader(str(gif_path_obj))
            writer = imageio.get_writer(
                str(mp4_path),
                format="FFMPEG",  # type: ignore[arg-type]
                fps=fps,
                codec="libx264",
            )

            for frame in reader:  # type: ignore[attr-defined]
                writer.append_data(frame)

            writer.close()
            reader.close()
        except (OSError, ValueError, RuntimeError) as e:
            raise TractographyVisualizationError(
                f"Failed to convert GIF to MP4: {e}",
            ) from e
        else:
            return mp4_path

    def generate_videos(
        self,
        tract_files: list[str | Path],
        ref_img: str | Path | None = None,
        *,
        ref_file: str | Path | None = None,  # Alias for ref_img
        output_dir: str | Path | None = None,
        remove_gifs: bool = True,
        overwrite: bool = False,
        max_streamlines: int | None = None,
        subsample_factor: float | None = None,
        max_points_per_streamline: int | None = None,
        resample_streamlines: bool = False,
    ) -> dict[str, Path]:
        """Generate MP4 videos from multiple tractography files.

        Parameters
        ----------
        tract_files : list[str | Path]
            List of paths to tractography files.
        ref_img : str | Path | None, optional
            Path to the reference image. If None, uses the reference image
            set during initialization.
        output_dir : str | Path | None, optional
            Output directory. If None, uses the output directory set during
            initialization.
        remove_gifs : bool, optional
            Whether to remove intermediate GIF files. Default is True.
        overwrite : bool, optional
            If True, regenerate videos even when output files already exist.
            Default is False.
        max_streamlines : int | None, optional
            Passed through to :meth:`generate_gif`. Maximum streamlines per tract.
        subsample_factor : float | None, optional
            Passed through to :meth:`generate_gif`. Fraction of streamlines to keep.
        max_points_per_streamline : int | None, optional
            Passed through to :meth:`generate_gif`. Max points per streamline.
        resample_streamlines : bool, optional
            Passed through to :meth:`generate_gif`. Whether to resample streamlines.

        Returns
        -------
        dict[str, Path]
            Dictionary mapping tract names to their MP4 file paths.

        Raises
        ------
        FileNotFoundError
            If required files are not found.
        InvalidInputError
            If the tract_files list is empty.
        TractographyVisualizationError
            If video generation fails.
        """
        if not tract_files:
            raise InvalidInputError("No tract files provided.")

        # Support both ref_img and ref_file (alias) for backward compatibility
        if ref_img is None and ref_file is not None:
            ref_img = ref_file

        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            ref_img = self._reference_image
        else:
            ref_img = Path(ref_img)

        tract_videos: dict[str, Path] = {}

        for tract_file in tract_files:
            try:
                tract_path = Path(tract_file)
                tract_name = tract_path.stem
                tract_mp4 = output_dir / f"{tract_name}.mp4"

                # Skip if MP4 already exists
                if tract_mp4.exists() and not overwrite:
                    logger.debug("Skipping generation of %s (file already exists)", tract_mp4)
                    tract_videos[tract_name] = tract_mp4
                    continue

                tract_gif = self.generate_gif(
                    name=tract_name,
                    tract_file=tract_path,
                    ref_img=ref_img,
                    output_dir=output_dir,
                    max_streamlines=max_streamlines,
                    subsample_factor=subsample_factor,
                    max_points_per_streamline=max_points_per_streamline,
                    resample_streamlines=resample_streamlines,
                    overwrite=overwrite,
                )
                self.convert_gif_to_mp4(tract_gif, tract_mp4, overwrite=overwrite)

                if remove_gifs:
                    tract_gif.unlink()

                tract_videos[tract_name] = tract_mp4
            except (OSError, ValueError, RuntimeError) as e:
                raise TractographyVisualizationError(
                    f"Failed to generate video for {tract_file}: {e}",
                ) from e

        return tract_videos

    def _process_single_tract(
        self,
        subject_id: str,
        tract_name: str,
        tract_file: str | Path,
        subject_ref_img: Path,
        tract_output_dir: Path,
        subjects_mni_space: dict[str, dict[str, str | Path]] | None,
        atlas_files: dict[str, str | Path] | None,
        metric_files: dict[str, dict[str, str | Path]] | None,
        atlas_ref_img: str | Path | None,
        *,
        flip_lr: bool,
        skip_checks: list[str],
        subject_kwargs: dict[str, Any] | None = None,
        atlas_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str | Path]:
        """Process a single subject/tract combination.

        This is a helper method used for parallel processing.
        Returns a dictionary of results for this tract.
        """
        tract_results: dict[str, str | Path] = {}
        # Track metrics and errors for summary table
        metrics: dict[str, Any] = {}
        errors: list[str] = []
        missing_data: list[str] = []

        # Merge kwargs: subject_kwargs override general kwargs for subject methods
        subject_merged_kwargs = {**kwargs, **(subject_kwargs or {})}
        # Merge kwargs: atlas_kwargs override general kwargs for atlas methods
        atlas_merged_kwargs = {**kwargs, **(atlas_kwargs or {})}

        # Load tract to get initial metrics
        try:
            tract = load_trk(str(tract_file), "same", bbox_valid_check=False)
            tract.to_rasmm()
            initial_streamline_count = len(tract.streamlines) if tract.streamlines else 0
            metrics["initial_streamline_count"] = initial_streamline_count
            del tract
            gc.collect()
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Failed to load tract for metrics: %s", e)
            metrics["initial_streamline_count"] = None
            errors.append(f"Failed to load tract: {e!s}")

        try:
            # 1. Standard anatomical views
            if "anatomical_views" not in skip_checks:
                try:
                    anatomical_views = self.generate_anatomical_views(
                        tract_file,
                        output_dir=tract_output_dir,
                        ref_img=subject_ref_img,
                        **subject_merged_kwargs,
                    )
                    # Add anatomical views to results
                    for view_name, view_path in anatomical_views.items():
                        tract_results[f"anatomical_{view_name}"] = view_path
                except (OSError, ValueError, RuntimeError, MemoryError, SystemExit, KeyboardInterrupt) as e:
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "Memory allocation failed (likely std::bad_alloc) when generating anatomical views for %s/%s. "
                            "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                            subject_id,
                            tract_name,
                        )
                    else:
                        error_msg_short = str(e)[:100]
                        logger.warning(
                            "Failed to generate anatomical views for %s/%s: %s",
                            subject_id,
                            tract_name,
                            e,
                        )
                        errors.append(f"Anatomical views: {error_msg_short}")
                except Exception as e:
                    # Catch-all for any other exceptions (including C++ exceptions that might not be properly caught)
                    error_msg = str(e).lower()
                    if (
                        "bad_alloc" in error_msg
                        or "memory" in error_msg
                        or "allocation" in error_msg
                        or "terminate" in error_msg
                    ):
                        logger.exception(
                            "Memory allocation failed (likely std::bad_alloc) when generating anatomical views for %s/%s. "
                            "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                            subject_id,
                            tract_name,
                        )
                    else:
                        error_msg_short = str(e)[:100]
                        logger.exception(
                            "Unexpected error generating anatomical views for %s/%s",
                            subject_id,
                            tract_name,
                        )
                        errors.append(f"Anatomical views: {error_msg_short}")

            # 2. CCI calculation and visualization (histogram + before/after CCI views)
            if "cci" not in skip_checks:
                try:
                    cci_values = None
                    keep_cci = None
                    keep_tract = None
                    long_streamlines = None
                    try:
                        cci_values, keep_cci, keep_tract, long_streamlines = self.calc_cci(
                            tract_file,
                            ref_img=subject_ref_img,
                        )
                        if len(cci_values) > 0:
                            metrics["cci_mean"] = float(np.mean(cci_values))
                            metrics["cci_median"] = float(np.median(cci_values))
                            metrics["cci_min"] = float(np.min(cci_values))
                            metrics["cci_max"] = float(np.max(cci_values))
                            metrics["cci_after_filter_count"] = len(keep_cci)
                            metrics["cci_removed_count"] = (
                                metrics.get("initial_streamline_count", 0) - len(keep_cci)
                                if metrics.get("initial_streamline_count") is not None
                                else None
                            )
                        else:
                            metrics["cci_after_filter_count"] = 0
                            missing_data.append("CCI: No streamlines after filtering")
                    except (OSError, ValueError, RuntimeError, InvalidInputError, TractographyVisualizationError) as e:
                        logger.warning("Failed to calculate CCI metrics: %s", e)
                        errors.append(f"CCI metrics calculation failed: {e!s}")

                    # Histogram + before/after CCI views (reuse precomputed CCI/tracts)
                    if cci_values is not None and keep_tract is not None and len(cci_values) > 0:
                        cci_result = self.compare_before_after_cci(
                            tract_file,
                            ref_img=subject_ref_img,
                            output_dir=tract_output_dir,
                            cci=cci_values,
                            keep_cci=keep_cci,
                            keep_tract=keep_tract,
                            long_streamlines=long_streamlines,
                            **subject_merged_kwargs,
                        )
                        hist_path = cci_result["histogram"]
                        if isinstance(hist_path, Path):
                            tract_results["cci_histogram"] = hist_path
                        for view_name in ("coronal", "axial", "sagittal"):
                            view_data = cci_result.get(view_name)
                            if isinstance(view_data, dict):
                                tract_results[f"before_cci_{view_name}"] = view_data["before"]
                                tract_results[f"after_cci_{view_name}"] = view_data["after"]
                                tract_results[f"cci_{view_name}"] = view_data["after"]
                        del cci_values, keep_cci, keep_tract, long_streamlines
                        gc.collect()
                except (OSError, ValueError, RuntimeError, MemoryError) as e:
                    error_msg = str(e).lower()
                    if "bad_alloc" in error_msg or "memory" in error_msg or "allocation" in error_msg:
                        logger.exception(
                            "Memory allocation failed (likely std::bad_alloc) when generating CCI plots for %s/%s. "
                            "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                            subject_id,
                            tract_name,
                        )
                    else:
                        error_msg_short = str(e)[:100]  # Truncate long error messages
                        logger.warning(
                            "Failed to generate CCI plots for %s/%s: %s",
                            subject_id,
                            tract_name,
                            e,
                        )
                        errors.append(f"CCI visualization: {error_msg_short}")
                except Exception as e:
                    error_msg = str(e).lower()
                    error_msg_short = str(e)[:100]
                    if (
                        "bad_alloc" in error_msg
                        or "memory" in error_msg
                        or "allocation" in error_msg
                        or "terminate" in error_msg
                    ):
                        logger.exception(
                            "Memory allocation failed (likely std::bad_alloc) when generating CCI plots for %s/%s. "
                            "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                            subject_id,
                            tract_name,
                        )
                        errors.append(f"CCI visualization: Memory error ({error_msg_short})")
                    else:
                        logger.exception(
                            "Unexpected error generating CCI plots for %s/%s",
                            subject_id,
                            tract_name,
                        )
                        errors.append(f"CCI visualization: {error_msg_short}")

            # 4. Atlas comparison (uses MNI space tracts for subject views)
            if "atlas_comparison" not in skip_checks and atlas_files is not None and tract_name in atlas_files:
                try:
                    atlas_file = atlas_files[tract_name]
                    # Generate subject views in MNI space if available, otherwise skip
                    if (
                        subjects_mni_space is not None
                        and subject_id in subjects_mni_space
                        and tract_name in subjects_mni_space[subject_id]
                    ):
                        tract_file_mni = subjects_mni_space[subject_id][tract_name]
                        # Generate subject views in MNI space
                        # Note: Don't apply flip_lr here - MNI views should match original anatomical views
                        subject_mni_views = self.generate_anatomical_views(
                            tract_file_mni,
                            ref_img=atlas_ref_img,  # Use atlas ref image for MNI space
                            output_dir=tract_output_dir,
                            **subject_merged_kwargs,  # Subject files use subject_kwargs
                        )
                        # Add subject MNI views to results
                        for view_name, view_path in subject_mni_views.items():
                            tract_results[f"subject_mni_{view_name}"] = view_path

                    # Generate atlas views
                    # Note: Don't apply flip_lr here - atlas views should match MNI views
                    atlas_views = self.generate_atlas_views(
                        atlas_file,
                        atlas_ref_img=atlas_ref_img,
                        flip_lr=False,  # Set to False to match MNI views
                        output_dir=tract_output_dir,
                        **atlas_merged_kwargs,  # Atlas files use atlas_kwargs
                    )
                    # Add atlas views to results
                    for view_name, view_path in atlas_views.items():
                        tract_results[f"atlas_{view_name}"] = view_path
                except (OSError, ValueError, RuntimeError) as e:
                    error_msg_short = str(e)[:100]
                    logger.warning(
                        "Failed to generate atlas comparison views for %s/%s: %s",
                        subject_id,
                        tract_name,
                        e,
                    )
                    errors.append(f"Atlas comparison: {error_msg_short}")
            elif "atlas_comparison" not in skip_checks:
                if atlas_files is None or tract_name not in atlas_files:
                    missing_data.append("Atlas comparison: Missing atlas file")
                elif (
                    subjects_mni_space is None
                    or subject_id not in subjects_mni_space
                    or tract_name not in subjects_mni_space[subject_id]
                ):
                    missing_data.append("Atlas comparison: Missing MNI space tract")

            # 5. Shape similarity (uses MNI space tracts)
            if "shape_similarity" not in skip_checks:
                if (
                    atlas_files is not None
                    and subjects_mni_space is not None
                    and subject_id in subjects_mni_space
                    and tract_name in subjects_mni_space[subject_id]
                    and tract_name in atlas_files
                ):
                    try:
                        # Use MNI space tract for shape similarity
                        tract_file_mni = subjects_mni_space[subject_id][tract_name]
                        atlas_file = atlas_files[tract_name]
                        # Calculate shape similarity score
                        similarity_score = self.calculate_shape_similarity(
                            tract_file_mni,
                            atlas_file,
                            atlas_ref_img=atlas_ref_img,
                            flip_lr=flip_lr,
                            **kwargs,
                        )
                        tract_results["shape_similarity_score"] = str(similarity_score)
                        metrics["shape_similarity_score"] = float(similarity_score)

                        # Visualize shape similarity (uses subject tract, so use subject_kwargs)
                        similarity_views = self.visualize_shape_similarity(
                            tract_file_mni,
                            atlas_file,
                            atlas_ref_img=atlas_ref_img,
                            flip_lr=flip_lr,
                            output_dir=tract_output_dir,
                            **subject_merged_kwargs,
                        )
                        # Add similarity views to results
                        for view_name, view_path in similarity_views.items():
                            tract_results[f"similarity_{view_name}"] = view_path
                    except (OSError, ValueError, RuntimeError, IndexError) as e:
                        error_msg_short = str(e)[:100]
                        logger.warning(
                            "Failed to calculate/visualize shape similarity for %s/%s: %s",
                            subject_id,
                            tract_name,
                            e,
                        )
                        errors.append(f"Shape similarity: {error_msg_short}")
                elif atlas_files is None or tract_name not in atlas_files:
                    missing_data.append("Shape similarity: Missing atlas file")
                elif (
                    subjects_mni_space is None
                    or subject_id not in subjects_mni_space
                    or tract_name not in subjects_mni_space[subject_id]
                ):
                    missing_data.append("Shape similarity: Missing MNI space tract")

            # 6. AFQ profile (requires metric files per subject and atlas files as model files)
            if ("afq_profile" not in skip_checks and metric_files is not None and atlas_files is not None) and (
                subject_id in metric_files and tract_name in atlas_files
            ):
                model_file = atlas_files[tract_name]  # Use atlas file as model file
                for metric_name, metric_file in metric_files[subject_id].items():
                    try:
                        afq_plots = self.plot_afq(
                            metric_file,
                            metric_name,
                            tract_file,
                            model_file,
                            ref_img=subject_ref_img,
                            output_dir=tract_output_dir,
                            **subject_merged_kwargs,
                        )
                        # Add AFQ plots to results (skip if empty dict returned for empty tracts)
                        if afq_plots:
                            for plot_type, plot_path in afq_plots.items():
                                tract_results[f"afq_{metric_name}_{plot_type}"] = plot_path
                    except (OSError, ValueError, RuntimeError, IndexError, InvalidInputError) as e:
                        # InvalidInputError is raised for empty tracts
                        if isinstance(e, InvalidInputError) and (
                            "empty" in str(e).lower() or "0 streamlines" in str(e).lower()
                        ):
                            logger.warning(
                                "Skipping AFQ profile for %s/%s/%s: tract is empty",
                                subject_id,
                                tract_name,
                                metric_name,
                            )
                        else:
                            error_msg_short = str(e)[:100]
                            logger.warning(
                                "Failed to generate AFQ profile for %s/%s/%s: %s",
                                subject_id,
                                tract_name,
                                metric_name,
                                e,
                            )
                            errors.append(f"AFQ profile ({metric_name}): {error_msg_short}")
            elif "afq_profile" not in skip_checks:
                if metric_files is None or subject_id not in metric_files:
                    missing_data.append("AFQ profile: Missing metric files")
                elif atlas_files is None or tract_name not in atlas_files:
                    missing_data.append("AFQ profile: Missing atlas file")

            # 7. Bundle assignment (uses MNI space tracts and atlas files as model files)
            if (
                "bundle_assignment" not in skip_checks and atlas_files is not None and subjects_mni_space is not None
            ) and (
                subject_id in subjects_mni_space
                and tract_name in subjects_mni_space[subject_id]
                and tract_name in atlas_files
            ):
                try:
                    # Use MNI space tract for bundle assignment
                    tract_file_mni = subjects_mni_space[subject_id][tract_name]
                    model_file = atlas_files[tract_name]  # Use atlas file as model file
                    assignment_views = self.visualize_bundle_assignment(
                        tract_file_mni,
                        model_file,
                        output_dir=tract_output_dir,
                        ref_img=subject_ref_img,
                        **subject_merged_kwargs,
                    )
                    # Add assignment views to results
                    for view_name, view_path in assignment_views.items():
                        tract_results[f"assignment_{view_name}"] = view_path
                except (OSError, ValueError, RuntimeError, IndexError) as e:
                    error_msg_short = str(e)[:100]
                    logger.warning(
                        "Failed to generate bundle assignment for %s/%s: %s",
                        subject_id,
                        tract_name,
                        e,
                    )
                    errors.append(f"Bundle assignment: {error_msg_short}")
            elif "bundle_assignment" not in skip_checks:
                if atlas_files is None or tract_name not in atlas_files:
                    missing_data.append("Bundle assignment: Missing atlas file")
                elif (
                    subjects_mni_space is None
                    or subject_id not in subjects_mni_space
                    or tract_name not in subjects_mni_space[subject_id]
                ):
                    missing_data.append("Bundle assignment: Missing MNI space tract")

        except (OSError, ValueError, RuntimeError, MemoryError, SystemExit, KeyboardInterrupt) as e:
            error_msg = str(e).lower()
            if (
                "bad_alloc" in error_msg
                or "memory" in error_msg
                or "allocation" in error_msg
                or "terminate" in error_msg
            ):
                logger.exception(
                    "Memory allocation failed (likely std::bad_alloc) when processing %s/%s. "
                    "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                    subject_id,
                    tract_name,
                )
            else:
                error_msg_short = str(e)[:100]
                logger.exception("Error processing %s/%s", subject_id, tract_name)
                errors.append(f"General processing error: {error_msg_short}")
        except Exception as e:
            # Catch-all for any other exceptions (including C++ exceptions that might not be properly caught)
            error_msg = str(e).lower()
            error_msg_short = str(e)[:100]
            if (
                "bad_alloc" in error_msg
                or "memory" in error_msg
                or "allocation" in error_msg
                or "terminate" in error_msg
            ):
                logger.exception(
                    "Memory allocation failed (likely std::bad_alloc) when processing %s/%s. "
                    "Try reducing image resolution, filtering streamlines, or using n_jobs=1.",
                    subject_id,
                    tract_name,
                )
                errors.append(f"General processing error: Memory error ({error_msg_short})")
            else:
                logger.exception(
                    "Unexpected error processing %s/%s (type: %s)",
                    subject_id,
                    tract_name,
                    type(e).__name__,
                )
                errors.append(f"General processing error: {error_msg_short} ({type(e).__name__})")
        finally:
            # Store metrics and errors in results
            if metrics:
                tract_results["_metrics"] = json.dumps(metrics)
            if errors:
                tract_results["_errors"] = json.dumps(errors)
            if missing_data:
                tract_results["_missing_data"] = json.dumps(missing_data)
            # Clean up memory after processing this tract
            gc.collect()

        return tract_results

    def run_quality_check_workflow(
        self,
        subjects_original_space: dict[str, dict[str, str | Path]],
        ref_img: str | Path | dict[str, str | Path] | None = None,
        *,
        subjects_mni_space: dict[str, dict[str, str | Path]] | None = None,
        atlas_files: dict[str, str | Path] | None = None,
        metric_files: dict[str, dict[str, str | Path]] | None = None,
        atlas_ref_img: str | Path | None = None,
        flip_lr: bool = False,
        output_dir: str | Path | None = None,
        html_output: str | Path | None = None,
        skip_checks: list[str] | None = None,
        n_jobs: int | None = None,
        subject_kwargs: dict[str, Any] | None = None,
        atlas_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> dict[str, dict[str, dict[str, str | Path]]]:
        """Run comprehensive quality checks for multiple subjects and tracts.

        This workflow function orchestrates all available quality check methods
        for each subject/tract combination and generates an HTML report.

        Different quality checks require tracts in different coordinate spaces:
        - **Original space** (subjects_original_space): Used for anatomical views,
          CCI calculations, before/after CCI comparison, and AFQ profiles
          (which need to align with subject-specific metric files).
        - **MNI/Atlas space** (subjects_mni_space): Used for shape similarity
          calculations and bundle assignment (which compare with atlas files).

        Parameters
        ----------
        subjects_original_space : dict[str, dict[str, str | Path]]
            Dictionary mapping subject IDs to their tract files in original/native space.
            Format: {subject_id: {tract_name: tract_file_path}}
            Example:
            {
                "sub-001": {
                    "AF_L": "path/to/sub-001_AF_L_original.trk",
                    "AF_R": "path/to/sub-001_AF_R_original.trk"
                },
                "sub-002": {
                    "AF_L": "path/to/sub-002_AF_L_original.trk"
                }
            }
            Used for: anatomical views, CCI, before/after CCI, AFQ profiles.
        ref_img : str | Path | dict[str, str | Path] | None, optional
            Reference image(s) for subjects. Can be:
            - A single path (str | Path): Used for all subjects
            - A dictionary mapping subject IDs to reference images:
              {subject_id: ref_image_path}
            - None: Uses the reference image set during initialization or via
              `set_reference_image()` for all subjects
            Example for per-subject reference images:
            {"sub-001": "path/to/sub-001_t1w.nii.gz", "sub-002": "path/to/sub-002_t1w.nii.gz"}
        subjects_mni_space : dict[str, dict[str, str | Path]] | None, optional
            Dictionary mapping subject IDs to their tract files in MNI/atlas space.
            Format: {subject_id: {tract_name: tract_file_path}}
            Example:
            {
                "sub-001": {
                    "AF_L": "path/to/sub-001_AF_L_mni.trk",
                    "AF_R": "path/to/sub-001_AF_R_mni.trk"
                },
                "sub-002": {
                    "AF_L": "path/to/sub-002_AF_L_mni.trk"
                }
            }
            Used for: shape similarity, bundle assignment.
            If None, these checks will be skipped.
        atlas_files : dict[str, str | Path] | None, optional
            Dictionary mapping tract names to their corresponding atlas/model files.
            These files are shared across all subjects and used for:
            - Atlas comparison visualizations
            - Shape similarity calculations
            - AFQ profile calculations (as model files)
            - Bundle assignment visualizations (as model files)
            Format: {tract_name: atlas_file_path}
            Example: {"AF_L": "path/to/atlas_AF_L.trk", "AF_R": "path/to/atlas_AF_R.trk"}
        metric_files : dict[str, dict[str, str | Path]] | None, optional
            Dictionary mapping subject IDs to their metric files.
            All tracts within a subject will use the same metric files.
            Format: {subject_id: {metric_name: metric_file_path}}
            Example:
            {
                "sub-001": {"FA": "path/to/sub-001_FA.nii.gz", "MD": "path/to/sub-001_MD.nii.gz"},
                "sub-002": {"FA": "path/to/sub-002_FA.nii.gz"}
            }
            If provided, AFQ profile calculations will be run for all tracts in each subject.
        atlas_ref_img : str | Path | None, optional
            Path to the reference image matching the atlas coordinate space
            (e.g., MNI template). Required if atlas files are in a different
            coordinate space than subject tracts.
        flip_lr : bool, optional
            Whether to flip left-right (X-axis) when transforming atlas.
            Default is False.
        output_dir : str | Path | None, optional
            Output directory for generated files. If None, uses the output
            directory set during initialization.
        html_output : str | Path | None, optional
            Path for the HTML report file(s). Individual reports are generated per subject.
            Options:
            - None: Creates "{subject_id}_quality_check_report.html" for each subject
              in the output directory.
            - Directory path: Creates "{subject_id}_quality_check_report.html" for each
              subject in the specified directory.
            - File path: If the path contains "{subject}" or "{SUBJECT}" placeholder,
              it will be replaced with the subject ID. Otherwise, the subject ID will
              be prepended to the filename (e.g., "report.html" becomes "sub-001_report.html").
            Example: "reports/{subject}_qc.html" will create "reports/sub-001_qc.html",
            "reports/sub-002_qc.html", etc.
        skip_checks : list[str] | None, optional
            List of quality checks to skip. Valid options:
            - "anatomical_views": Skip standard anatomical views
            - "atlas_comparison": Skip atlas comparison views
            - "cci": Skip CCI calculation and visualization
            - "before_after_cci": Skip before/after CCI comparison
            - "afq_profile": Skip AFQ profile visualization
            - "bundle_assignment": Skip bundle assignment visualization
            - "shape_similarity": Skip shape similarity calculation and visualization
        n_jobs : int | None, optional
            Number of parallel jobs to run for processing multiple subjects/tracts.
            If None, uses the value set during initialization (default: 1).
            Use -1 to automatically determine optimal number based on available
            resources (respects SLURM allocations and OpenMP thread settings to
            prevent oversubscription). Only effective when processing multiple
            subjects/tracts. Default is None.

            Note: When running under SLURM, -1 will automatically use
            SLURM_CPUS_PER_TASK or SLURM_JOB_CPUS_PER_NODE. If OMP_NUM_THREADS is set,
            it will divide the available CPUs by the number of OpenMP threads to
            prevent resource contention.
        subject_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments passed only to subject tract visualization
            methods (e.g., `generate_anatomical_views` for subject files).
            These kwargs are NOT passed to atlas visualization methods.
            Example: `{"max_streamlines": 500, "max_points_per_streamline": 100}`
        atlas_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments passed only to atlas visualization methods
            (e.g., `generate_atlas_views`). These kwargs are NOT passed to subject
            visualization methods. If None, atlas methods use default parameters.
        overwrite : bool, optional
            If True, regenerate all images even when output files already exist.
            Passed to all quality check methods. Default is False.
        **kwargs
            Additional keyword arguments passed to ALL quality check methods
            (both subject and atlas). Use `subject_kwargs` or `atlas_kwargs` to
            pass different parameters to subject vs atlas methods.

        Returns
        -------
        dict[str, dict[str, dict[str, str | Path]]]
            Nested dictionary structure: {subject_id: {tract_name: {media_type: file_path}}}
            This structure is compatible with `create_quality_check_html()`.

        Raises
        ------
        InvalidInputError
            If required files are missing or invalid.
        TractographyVisualizationError
            If quality check workflow fails.

        Examples
        --------
        Single subject or all subjects share same reference image:
        >>> visualizer = TractographyVisualizer(output_directory="output/")
        >>> subjects_original = {
        ...     "sub-001": {
        ...         "AF_L": "sub-001_AF_L_original.trk",
        ...         "AF_R": "sub-001_AF_R_original.trk",
        ...     }
        ... }
        >>> results = visualizer.run_quality_check_workflow(
        ...     subjects_original_space=subjects_original,
        ...     ref_img="shared_t1w.nii.gz",  # Single image for all subjects
        ...     html_output="quality_report.html",
        ... )

        Multiple subjects with different reference images:
        >>> visualizer = TractographyVisualizer(output_directory="output/")
        >>> subjects_original = {
        ...     "sub-001": {"AF_L": "sub-001_AF_L_original.trk"},
        ...     "sub-002": {"AF_L": "sub-002_AF_L_original.trk"},
        ... }
        >>> # Each subject has its own reference image
        >>> ref_images = {
        ...     "sub-001": "sub-001_t1w.nii.gz",
        ...     "sub-002": "sub-002_t1w.nii.gz",
        ... }
        >>> subjects_mni = {
        ...     "sub-001": {"AF_L": "sub-001_AF_L_mni.trk"},
        ...     "sub-002": {"AF_L": "sub-002_AF_L_mni.trk"},
        ... }
        >>> atlas_files = {"AF_L": "atlas_AF_L.trk"}
        >>> metric_files = {
        ...     "sub-001": {"FA": "sub-001_FA.nii.gz"},
        ...     "sub-002": {"FA": "sub-002_FA.nii.gz"},
        ... }
        >>> results = visualizer.run_quality_check_workflow(
        ...     subjects_original_space=subjects_original,
        ...     ref_img=ref_images,  # Dictionary mapping subject_id -> ref_image
        ...     subjects_mni_space=subjects_mni,
        ...     atlas_files=atlas_files,
        ...     metric_files=metric_files,
        ...     html_output="quality_report.html",
        ... )
        """
        # Merge overwrite into kwargs so all quality check methods receive it
        kwargs = dict(overwrite=overwrite, **kwargs)

        # Initialize XVFB (X Virtual Framebuffer) if requested for headless environments
        vdisplay = None
        if os.environ.get("XVFB", "").lower() in ("1", "true", "yes"):
            logger.info("Initializing XVFB for headless rendering")
            vdisplay = Xvfb()
            vdisplay.start()

        # Get output directory
        if output_dir is None:
            if self._output_directory is None:
                raise InvalidInputError(
                    "No output directory provided. Set it via constructor or "
                    "set_output_directory() method, or pass it as an argument.",
                )
            output_dir = self._output_directory
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Handle reference image(s) - can be single path or dict mapping subject_id -> path
        if ref_img is None:
            if self._reference_image is None:
                raise InvalidInputError(
                    "No reference image provided. Set it via constructor or "
                    "set_reference_image() method, or pass it as an argument.",
                )
            # Use instance reference image for all subjects
            subject_ref_imgs: dict[str, Path] = dict.fromkeys(subjects_original_space.keys(), self._reference_image)
        elif isinstance(ref_img, dict):
            # Dictionary mapping subject IDs to reference images
            subject_ref_imgs = {subject_id: Path(path) for subject_id, path in ref_img.items()}
            # Validate all subjects have reference images
            missing = set(subjects_original_space.keys()) - set(subject_ref_imgs.keys())
            if missing:
                raise InvalidInputError(
                    f"Missing reference images for subjects: {', '.join(sorted(missing))}",
                )
        else:
            # Single reference image for all subjects
            single_ref_img = Path(ref_img)
            subject_ref_imgs = dict.fromkeys(subjects_original_space.keys(), single_ref_img)

        # Set default skip_checks
        if skip_checks is None:
            skip_checks = []

        # Determine number of jobs to use
        if n_jobs is None:
            n_jobs = self.n_jobs
        elif n_jobs == -1:
            # Use optimal n_jobs considering SLURM and OpenMP settings
            base_n_jobs = _get_optimal_n_jobs()
            # Further reduce if memory is limited
            n_jobs = _get_n_jobs_with_memory_limit(
                base_n_jobs,
                estimated_memory_per_job_mb=2000.0,  # ~2GB per worker
                safety_margin=0.2,
            )
        else:
            n_jobs = max(1, n_jobs)
            # Still check memory even if n_jobs is explicitly set
            if psutil is not None:
                n_jobs = _get_n_jobs_with_memory_limit(
                    n_jobs,
                    estimated_memory_per_job_mb=2000.0,
                    safety_margin=0.2,
                )

        # Log the final n_jobs value for debugging
        logger.debug("Using n_jobs=%d for parallel processing", n_jobs)

        # Initialize results dictionary
        results: dict[str, dict[str, dict[str, str | Path]]] = {}

        # Prepare all tasks (subject_id, tract_name, tract_file combinations)
        tasks: list[tuple[str, str, str | Path, Path, Path]] = []
        for subject_id, tracts in subjects_original_space.items():
            subject_ref_img = subject_ref_imgs[subject_id]
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            for tract_name, tract_file in tracts.items():
                tract_path = Path(tract_file)
                if not tract_path.exists():
                    raise FileNotFoundError(f"Tract file not found: {tract_file}")

                tract_output_dir = subject_output_dir / tract_name
                tract_output_dir.mkdir(parents=True, exist_ok=True)

                tasks.append((subject_id, tract_name, tract_file, subject_ref_img, tract_output_dir))

        # Prepare visualizer parameters for worker processes
        visualizer_params = {
            "gif_size": self.gif_size,
            "gif_duration": self.gif_duration,
            "gif_palette_size": self.gif_palette_size,
            "gif_frames": self.gif_frames,
            "min_streamline_length": self.min_streamline_length,
            "cci_threshold": self.cci_threshold,
            "afq_resample_points": self.afq_resample_points,
            "n_jobs": 1,  # Workers don't need parallelization
        }

        # Process tasks in parallel or sequentially
        if n_jobs > 1 and len(tasks) > 1:
            logger.info("Processing %d tracts using %d workers", len(tasks), n_jobs)
            try:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = {
                        executor.submit(
                            _process_tract_worker,
                            subject_id=subject_id,
                            tract_name=tract_name,
                            tract_file=tract_file,
                            subject_ref_img=subject_ref_img,
                            tract_output_dir=tract_output_dir,
                            subjects_mni_space=subjects_mni_space,
                            atlas_files=atlas_files,
                            metric_files=metric_files,
                            atlas_ref_img=atlas_ref_img,
                            flip_lr=flip_lr,
                            skip_checks=skip_checks,
                            visualizer_params=visualizer_params,
                            subject_kwargs=subject_kwargs,
                            atlas_kwargs=atlas_kwargs,
                            **kwargs,
                        ): (subject_id, tract_name)
                        for subject_id, tract_name, tract_file, subject_ref_img, tract_output_dir in tasks
                    }

                    # Collect results as they complete
                    broken_pool_detected = False
                    completed_count = 0
                    try:
                        for future in as_completed(futures):
                            subject_id, tract_name = futures[future]
                            try:
                                # Get result with timeout - this is where BrokenProcessPool is raised
                                result_subject_id, result_tract_name, tract_results = future.result(timeout=None)
                                if result_subject_id not in results:
                                    results[result_subject_id] = {}
                                results[result_subject_id][result_tract_name] = tract_results
                                # Clean up the future reference
                                del tract_results
                            except RuntimeError as e:
                                # RuntimeError catches BrokenProcessPool (which is a subclass of RuntimeError)
                                # Check if this is a BrokenProcessPool by checking the error message and type name
                                error_type = type(e).__name__
                                error_msg = str(e)
                                if (
                                    "BrokenProcessPool" in error_type
                                    or "BrokenProcessPool" in error_msg
                                    or "process pool" in error_msg.lower()
                                    or "was terminated abruptly" in error_msg
                                ):
                                    broken_pool_detected = True
                                    logger.exception(
                                        "BrokenProcessPool detected for %s/%s. Worker process may have crashed",
                                        subject_id,
                                        tract_name,
                                    )
                                    # Mark this task as failed
                                    if subject_id not in results:
                                        results[subject_id] = {}
                                    if tract_name not in results[subject_id]:
                                        results[subject_id][tract_name] = {}
                                    # Break out of the loop immediately - pool is broken, can't process more
                                    break
                                logger.exception("RuntimeError processing %s/%s", subject_id, tract_name)
                                if subject_id not in results:
                                    results[subject_id] = {}
                                if tract_name not in results[subject_id]:
                                    results[subject_id][tract_name] = {}
                            except (OSError, ValueError) as e:
                                # OSError catches system-level errors
                                # ValueError catches other processing errors
                                logger.exception("Error processing %s/%s: %s", subject_id, tract_name, type(e).__name__)
                                if subject_id not in results:
                                    results[subject_id] = {}
                                if tract_name not in results[subject_id]:
                                    results[subject_id][tract_name] = {}
                            except Exception as e:
                                # Catch any other unexpected exceptions, including BrokenProcessPool
                                error_type = type(e).__name__
                                error_msg = str(e)
                                if (
                                    "BrokenProcessPool" in error_type
                                    or "BrokenProcessPool" in error_msg
                                    or "process pool" in error_msg.lower()
                                    or "was terminated abruptly" in error_msg
                                ):
                                    broken_pool_detected = True
                                    logger.exception(
                                        "BrokenProcessPool detected for %s/%s (caught as Exception). Worker process may have crashed",
                                        subject_id,
                                        tract_name,
                                    )
                                    # Mark this task as failed
                                    if subject_id not in results:
                                        results[subject_id] = {}
                                    if tract_name not in results[subject_id]:
                                        results[subject_id][tract_name] = {}
                                    # Break out of the loop immediately - pool is broken, can't process more
                                    break
                                logger.exception(
                                    "Unexpected error processing %s/%s: %s",
                                    subject_id,
                                    tract_name,
                                    type(e).__name__,
                                )
                                if subject_id not in results:
                                    results[subject_id] = {}
                                if tract_name not in results[subject_id]:
                                    results[subject_id][tract_name] = {}
                            finally:
                                # Increment counter regardless of success/failure
                                completed_count += 1

                                # Clean up future reference and remove from futures dict
                                # This is critical to prevent memory accumulation
                                futures.pop(future, None)
                                del future

                                # Periodic garbage collection every 10 completed tasks
                                # This helps prevent memory buildup during long-running jobs
                                if completed_count > 0 and completed_count % 10 == 0:
                                    gc.collect()
                    except RuntimeError as e:
                        # Catch BrokenProcessPool that might break out of the loop
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if (
                            "BrokenProcessPool" in error_type
                            or "BrokenProcessPool" in error_msg
                            or "process pool" in error_msg.lower()
                            or "was terminated abruptly" in error_msg
                        ):
                            broken_pool_detected = True
                            logger.exception(
                                "BrokenProcessPool detected during result collection. Falling back to sequential processing",
                            )
                        else:
                            # Re-raise if it's a different RuntimeError
                            raise
                    except Exception as e:
                        # Catch any other exceptions that might break the loop
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if (
                            "BrokenProcessPool" in error_type
                            or "BrokenProcessPool" in error_msg
                            or "process pool" in error_msg.lower()
                            or "was terminated abruptly" in error_msg
                        ):
                            broken_pool_detected = True
                            logger.exception(
                                "BrokenProcessPool detected during result collection (caught as Exception). Falling back to sequential processing",
                            )
                        else:
                            # Log but don't re-raise - try to continue with what we have
                            logger.exception("Unexpected error during result collection: %s", type(e).__name__)

                    # If we detected a broken process pool, break out and fall back to sequential
                    if broken_pool_detected:
                        # Raise error to trigger fallback to sequential processing
                        # The exception will be caught by the outer try-except block
                        raise RuntimeError("BrokenProcessPool detected, falling back to sequential processing")  # noqa: TRY301

                    # Clear futures dictionary to free memory
                    futures.clear()

                    # Force garbage collection after all parallel tasks complete
                    # Run multiple times to handle circular references
                    for _ in range(3):
                        gc.collect()
            except RuntimeError as e:
                # RuntimeError catches BrokenProcessPool (which is a subclass of RuntimeError)
                # and other runtime errors that might occur with process pools
                error_msg = str(e)
                if "BrokenProcessPool" in error_msg or "falling back" in error_msg.lower():
                    logger.warning(
                        "Process pool error occurred (worker process may have crashed with core dump). "
                        "Falling back to sequential processing. "
                        "If this persists, try: (1) reducing n_jobs, (2) increasing memory allocation, "
                        "or (3) using n_jobs=1 to force sequential processing.",
                    )
                else:
                    logger.exception("RuntimeError in process pool. Falling back to sequential processing")

                # Fall back to sequential processing if process pool fails
                # Only process tasks that haven't been completed yet
                remaining_tasks = [
                    (s_id, t_name, t_file, s_ref, t_out)
                    for s_id, t_name, t_file, s_ref, t_out in tasks
                    if s_id not in results or t_name not in results.get(s_id, {})
                ]

                if remaining_tasks:
                    logger.info("Processing %d remaining tracts sequentially (fallback)", len(remaining_tasks))
                    for subject_id, tract_name, tract_file, subject_ref_img, tract_output_dir in remaining_tasks:
                        if subject_id not in results:
                            results[subject_id] = {}
                        # Skip if already processed
                        if tract_name in results[subject_id]:
                            continue
                        try:
                            tract_result = self._process_single_tract(
                                subject_id=subject_id,
                                tract_name=tract_name,
                                tract_file=tract_file,
                                subject_ref_img=subject_ref_img,
                                tract_output_dir=tract_output_dir,
                                subjects_mni_space=subjects_mni_space,
                                atlas_files=atlas_files,
                                metric_files=metric_files,
                                atlas_ref_img=atlas_ref_img,
                                flip_lr=flip_lr,
                                skip_checks=skip_checks,
                                subject_kwargs=subject_kwargs,
                                atlas_kwargs=atlas_kwargs,
                                **kwargs,
                            )
                            results[subject_id][tract_name] = tract_result
                            # Clean up result reference
                            del tract_result
                        except (OSError, ValueError, RuntimeError, MemoryError) as process_error:
                            logger.exception(
                                "Error processing %s/%s in fallback mode (%s)",
                                subject_id,
                                tract_name,
                                type(process_error).__name__,
                            )
                            results[subject_id][tract_name] = {}
                        except Exception as process_error:
                            logger.exception(
                                "Unexpected error processing %s/%s in fallback mode (%s)",
                                subject_id,
                                tract_name,
                                type(process_error).__name__,
                            )
                            results[subject_id][tract_name] = {}
                        finally:
                            # Force garbage collection after each tract in fallback mode
                            # Run multiple times to handle circular references in VTK objects
                            for _ in range(2):
                                gc.collect()
                else:
                    logger.info("All tasks already completed, skipping fallback processing")
        else:
            # Sequential processing
            logger.info("Processing %d tracts sequentially", len(tasks))
            for subject_id, tract_name, tract_file, subject_ref_img, tract_output_dir in tasks:
                if subject_id not in results:
                    results[subject_id] = {}
                try:
                    tract_result = self._process_single_tract(
                        subject_id=subject_id,
                        tract_name=tract_name,
                        tract_file=tract_file,
                        subject_ref_img=subject_ref_img,
                        tract_output_dir=tract_output_dir,
                        subjects_mni_space=subjects_mni_space,
                        atlas_files=atlas_files,
                        metric_files=metric_files,
                        atlas_ref_img=atlas_ref_img,
                        flip_lr=flip_lr,
                        skip_checks=skip_checks,
                        subject_kwargs=subject_kwargs,
                        atlas_kwargs=atlas_kwargs,
                        **kwargs,
                    )
                    results[subject_id][tract_name] = tract_result
                    # Clean up result reference
                    del tract_result
                finally:
                    # Clean up after each tract in sequential processing
                    # Run multiple times to handle circular references in VTK objects
                    for _ in range(2):
                        gc.collect()

        # Generate HTML reports (one per subject) - after all processing is complete
        if html_output is None:
            html_output_dir = output_dir
            html_output_template = None
        else:
            html_output_path = Path(html_output)
            # If html_output is a directory, use it as the output directory
            if html_output_path.is_dir() or (not html_output_path.exists() and html_output_path.suffix == ""):
                html_output_dir = html_output_path
                html_output_template = None
            else:
                # If html_output is a file path, use its parent as directory and name as template
                html_output_dir = html_output_path.parent
                html_output_template = html_output_path.stem

        # Generate individual HTML reports per subject
        html_files_generated = []
        for subject_id, subject_tracts in results.items():
            # Convert Path objects to strings for HTML function for this subject only
            subject_results_for_html: dict[str, dict[str, dict[str, str]]] = {
                subject_id: {},
            }
            if isinstance(subject_tracts, dict):
                for tract_name, media_dict in subject_tracts.items():
                    subject_results_for_html[subject_id][tract_name] = {}
                    if isinstance(media_dict, dict):
                        for media_type, file_path in media_dict.items():
                            # Convert Path to string, or keep as string/number
                            if isinstance(file_path, Path):
                                subject_results_for_html[subject_id][tract_name][media_type] = str(file_path)
                            else:
                                subject_results_for_html[subject_id][tract_name][media_type] = str(file_path)

            # Determine output file path for this subject
            # Ensure output directory exists
            html_output_dir.mkdir(parents=True, exist_ok=True)

            if html_output_template:
                # Use template: replace {subject} placeholder or prepend subject_id
                if "{subject}" in html_output_template or "{SUBJECT}" in html_output_template:
                    # Replace placeholder with subject_id
                    template_with_subject = html_output_template.replace("{subject}", subject_id).replace(
                        "{SUBJECT}",
                        subject_id,
                    )
                    subject_html_file = html_output_dir / f"{template_with_subject}.html"
                else:
                    # No placeholder found, prepend subject_id to filename
                    subject_html_file = html_output_dir / f"{subject_id}_{html_output_template}.html"
            else:
                # Default: subject_id_quality_check_report.html
                subject_html_file = html_output_dir / f"{subject_id}_quality_check_report.html"

            # Generate HTML report
            try:
                create_quality_check_html(
                    subject_results_for_html,
                    str(subject_html_file),
                    title=f"Tractography Quality Check Report - {subject_id}",
                )
                html_files_generated.append(subject_html_file)
                logger.info("Quality check report generated for %s: %s", subject_id, subject_html_file)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to generate HTML report for %s: %s", subject_id, e)

            # Generate JSON results file for this subject
            # Determine JSON file path (same location and naming pattern as HTML, but with .json extension)
            subject_json_file = subject_html_file.with_suffix(".json")
            try:
                # Convert Path objects to strings for JSON serialization
                # Use Any for values since they can be strings, numbers, or Path objects
                subject_results_for_json: dict[str, dict[str, dict[str, Any]]] = {
                    subject_id: {},
                }
                if isinstance(subject_tracts, dict):
                    for tract_name, media_dict in subject_tracts.items():
                        subject_results_for_json[subject_id][tract_name] = {}
                        if isinstance(media_dict, dict):
                            for media_type, file_path in media_dict.items():
                                # Convert Path to string, or keep as string/number
                                if isinstance(file_path, Path):
                                    subject_results_for_json[subject_id][tract_name][media_type] = str(file_path)
                                elif isinstance(file_path, (int, float)):
                                    # Keep numeric values as-is (JSON can handle them)
                                    subject_results_for_json[subject_id][tract_name][media_type] = file_path
                                else:
                                    subject_results_for_json[subject_id][tract_name][media_type] = str(file_path)

                # Write JSON file
                with open(subject_json_file, "w", encoding="utf-8") as f:
                    json.dump(subject_results_for_json, f, indent=2, ensure_ascii=False)
                logger.info("Results JSON generated for %s: %s", subject_id, subject_json_file)
            except (OSError, ValueError, TypeError, RuntimeError) as e:
                logger.warning("Failed to generate JSON results for %s: %s", subject_id, e)
            finally:
                # Clean up subject HTML conversion dictionary after use
                del subject_results_for_html
                gc.collect()

        if html_files_generated:
            logger.info("Generated %d individual subject report(s)", len(html_files_generated))

        # Generate CSV summary file with all subjects' data
        csv_output_file = html_output_dir / "quality_check_summary.csv"
        try:
            # Convert all results to format expected by CSV function
            all_results_for_csv: dict[str, dict[str, dict[str, str]]] = {}
            for subject_id, subject_tracts in results.items():
                all_results_for_csv[subject_id] = {}
                if isinstance(subject_tracts, dict):
                    for tract_name, media_dict in subject_tracts.items():
                        all_results_for_csv[subject_id][tract_name] = {}
                        if isinstance(media_dict, dict):
                            for media_type, file_path in media_dict.items():
                                # Convert Path to string, or keep as string/number
                                if isinstance(file_path, Path):
                                    all_results_for_csv[subject_id][tract_name][media_type] = str(file_path)
                                else:
                                    all_results_for_csv[subject_id][tract_name][media_type] = str(file_path)

            create_summary_csv(all_results_for_csv, str(csv_output_file))
            logger.info("Summary CSV generated: %s", csv_output_file)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Failed to generate summary CSV: %s", e)

        # Stop XVFB if it was started
        if vdisplay is not None:
            logger.info("Stopping XVFB")
            vdisplay.stop()
        return results
