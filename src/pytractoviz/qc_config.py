"""Path configuration for the automated QC pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_TRACTOGRAPHY_ROOT = ("PYTRACTOVIZ_TRACTOGRAPHY_ROOT", "TRACTOGRAPHY_ROOT")
_ENV_ATLAS_DIR = ("PYTRACTOVIZ_ATLAS_DIR",)
_ENV_ATLAS_REF_IMG = ("PYTRACTOVIZ_ATLAS_REF_IMG",)
_ENV_QSIPREP_ROOT = ("PYTRACTOVIZ_QSIPREP_ROOT",)
_ENV_QSIRECON_ROOT = ("PYTRACTOVIZ_QSIRECON_ROOT",)
_ENV_QC_OUTPUT_DIR = ("PYTRACTOVIZ_QC_OUTPUT_DIR",)

_DEFAULT_ATLAS_DIR = Path("hcp_atlas_16_bundles/Atlas_in_MNI_Space_16_bundles/bundles")
_DEFAULT_ATLAS_REF_IMG = Path(
    "templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


@dataclass(frozen=True)
class QCPathConfig:
    """Filesystem layout for ``pytractoviz run`` inputs and outputs.

    Paths are resolved relative to ``tractography_root`` unless overridden.
    Use :func:`load_qc_path_config` to build an instance from CLI flags or
    environment variables.
    """

    tractography_root: Path
    atlas_dir: Path | None = None
    atlas_ref_img: Path | None = None
    qsiprep_root: Path | None = None
    qsirecon_root: Path | None = None
    qc_output_dir: Path | None = None

    def _qsiprep_root(self) -> Path:
        return self.qsiprep_root or self.tractography_root

    def _qsirecon_root(self) -> Path:
        return self.qsirecon_root or self.tractography_root

    def resolve_atlas_dir(self) -> Path:
        """Return the directory containing atlas bundle ``.trk`` files."""
        if self.atlas_dir is not None:
            return self.atlas_dir
        return self.tractography_root / _DEFAULT_ATLAS_DIR

    def resolve_atlas_ref_img(self) -> Path:
        """Return the MNI brain-mask NIfTI used for atlas rendering."""
        if self.atlas_ref_img is not None:
            return self.atlas_ref_img
        return self.tractography_root / _DEFAULT_ATLAS_REF_IMG

    def output_dir_for_project(self, project: str) -> Path:
        """Return the group QC output directory for ``project``."""
        if self.qc_output_dir is not None:
            return self.qc_output_dir
        return self._qsirecon_root() / "qsirecon" / project / "derivatives" / "QC"

    def anat_dir(self, project: str, subject: str) -> Path:
        """Return the qsiprep anatomical directory for one subject."""
        return self._qsiprep_root() / "qsiprep" / project / subject / "anat"

    def pyafq_bundles_dir(self, project: str, subject: str) -> Path:
        """Return the PYAFQ bundle ``.trk`` directory for one subject."""
        return self._qsirecon_root() / "qsirecon" / project / "derivatives" / "PYAFQ" / subject / "bundles"

    def dki_dir(self, project: str, subject: str) -> Path:
        """Return the qsirecon-DKI ``dwi`` directory for one subject."""
        return self._qsirecon_root() / "qsirecon" / project / "derivatives" / "qsirecon-DKI" / subject / "dwi"

    def noddi_dir(self, project: str, subject: str) -> Path:
        """Return the qsirecon-NODDI ``dwi`` directory for one subject."""
        return self._qsirecon_root() / "qsirecon" / project / "derivatives" / "qsirecon-NODDI" / subject / "dwi"


def load_qc_path_config(
    *,
    tractography_root: str | Path | None = None,
    atlas_dir: str | Path | None = None,
    atlas_ref_img: str | Path | None = None,
    qsiprep_root: str | Path | None = None,
    qsirecon_root: str | Path | None = None,
    qc_output_dir: str | Path | None = None,
) -> QCPathConfig:
    """Build :class:`QCPathConfig` from explicit arguments, then environment variables.

    Precedence: function arguments > environment variables.

    Parameters
    ----------
    tractography_root : str or Path, optional
        Top-level tractography directory. Required unless
        ``PYTRACTOVIZ_TRACTOGRAPHY_ROOT`` or ``TRACTOGRAPHY_ROOT`` is set.
    atlas_dir : str or Path, optional
        Override atlas bundle directory. Env: ``PYTRACTOVIZ_ATLAS_DIR``.
    atlas_ref_img : str or Path, optional
        Override MNI reference mask. Env: ``PYTRACTOVIZ_ATLAS_REF_IMG``.
    qsiprep_root : str or Path, optional
        Root containing ``qsiprep/<project>/<subject>/anat``.
        Env: ``PYTRACTOVIZ_QSIPREP_ROOT``.
    qsirecon_root : str or Path, optional
        Root containing ``qsirecon/<project>/derivatives``. Env: ``PYTRACTOVIZ_QSIRECON_ROOT``.
    qc_output_dir : str or Path, optional
        Full QC output directory for a project run. Env: ``PYTRACTOVIZ_QC_OUTPUT_DIR``.

    Returns
    -------
    QCPathConfig
        Resolved path configuration.

    Raises
    ------
    ValueError
        If no tractography root is provided via argument or environment.
    """
    root = _resolve_optional_path(tractography_root) or _resolve_optional_path(_first_env(_ENV_TRACTOGRAPHY_ROOT))
    if root is None:
        env_names = " or ".join(f"${name}" for name in _ENV_TRACTOGRAPHY_ROOT)
        raise ValueError(
            "Tractography root is required. Pass tractography_root=... or set "
            f"{env_names}, or use --tractography-root on `pytractoviz run`.",
        )

    return QCPathConfig(
        tractography_root=root,
        atlas_dir=_resolve_optional_path(atlas_dir) or _resolve_optional_path(_first_env(_ENV_ATLAS_DIR)),
        atlas_ref_img=_resolve_optional_path(atlas_ref_img) or _resolve_optional_path(_first_env(_ENV_ATLAS_REF_IMG)),
        qsiprep_root=_resolve_optional_path(qsiprep_root) or _resolve_optional_path(_first_env(_ENV_QSIPREP_ROOT)),
        qsirecon_root=_resolve_optional_path(qsirecon_root) or _resolve_optional_path(_first_env(_ENV_QSIRECON_ROOT)),
        qc_output_dir=_resolve_optional_path(qc_output_dir) or _resolve_optional_path(_first_env(_ENV_QC_OUTPUT_DIR)),
    )
