"""Tests for QC path configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from pytractoviz.qc_config import QCPathConfig, load_qc_path_config


class TestQCPathConfig:
    """Tests for :class:`pytractoviz.qc_config.QCPathConfig` and :func:`load_qc_path_config`."""

    def test_default_layout(self, tmp_path: Path) -> None:
        """Default paths resolve under ``tractography_root``."""
        root = tmp_path / "tractography"
        root.mkdir()
        cfg = QCPathConfig(tractography_root=root)

        assert cfg.resolve_atlas_dir() == root / "hcp_atlas_16_bundles/Atlas_in_MNI_Space_16_bundles/bundles"
        assert cfg.resolve_atlas_ref_img() == root / (
            "templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
        )
        assert cfg.output_dir_for_project("UNC") == root / "qsirecon/UNC/derivatives/QC"
        assert cfg.anat_dir("UNC", "sub-0001") == root / "qsiprep/UNC/sub-0001/anat"
        assert cfg.pyafq_bundles_dir("UNC", "sub-0001") == (root / "qsirecon/UNC/derivatives/PYAFQ/sub-0001/bundles")
        assert cfg.dki_dir("UNC", "sub-0001") == root / "qsirecon/UNC/derivatives/qsirecon-DKI/sub-0001/dwi"
        assert cfg.noddi_dir("UNC", "sub-0001") == root / "qsirecon/UNC/derivatives/qsirecon-NODDI/sub-0001/dwi"

    def test_overrides(self, tmp_path: Path) -> None:
        """Explicit overrides replace default atlas and derivative roots."""
        root = tmp_path / "tractography"
        root.mkdir()
        cfg = QCPathConfig(
            tractography_root=root,
            atlas_dir=tmp_path / "custom_atlas",
            qc_output_dir=tmp_path / "qc_out",
            qsiprep_root=tmp_path / "prep",
            qsirecon_root=tmp_path / "recon",
        )

        assert cfg.resolve_atlas_dir() == tmp_path / "custom_atlas"
        assert cfg.output_dir_for_project("UNC") == tmp_path / "qc_out"
        assert cfg.anat_dir("UNC", "sub-0001") == tmp_path / "prep/qsiprep/UNC/sub-0001/anat"
        assert cfg.pyafq_bundles_dir("UNC", "sub-0001") == (
            tmp_path / "recon/qsirecon/UNC/derivatives/PYAFQ/sub-0001/bundles"
        )

    def test_load_from_argument(self, tmp_path: Path) -> None:
        """``load_qc_path_config`` accepts an explicit tractography root."""
        cfg = load_qc_path_config(tractography_root=tmp_path)
        assert cfg.tractography_root == tmp_path

    def test_load_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``TRACTOGRAPHY_ROOT`` is used when no argument is passed."""
        monkeypatch.delenv("PYTRACTOVIZ_TRACTOGRAPHY_ROOT", raising=False)
        monkeypatch.setenv("TRACTOGRAPHY_ROOT", str(tmp_path))
        cfg = load_qc_path_config()
        assert cfg.tractography_root == tmp_path

    def test_load_prefers_pytractoviz_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``PYTRACTOVIZ_TRACTOGRAPHY_ROOT`` takes precedence over ``TRACTOGRAPHY_ROOT``."""
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TRACTOGRAPHY_ROOT", str(other))
        monkeypatch.setenv("PYTRACTOVIZ_TRACTOGRAPHY_ROOT", str(tmp_path))
        cfg = load_qc_path_config()
        assert cfg.tractography_root == tmp_path

    def test_load_requires_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing root via argument and environment raises ``ValueError``."""
        monkeypatch.delenv("PYTRACTOVIZ_TRACTOGRAPHY_ROOT", raising=False)
        monkeypatch.delenv("TRACTOGRAPHY_ROOT", raising=False)
        with pytest.raises(ValueError, match="Tractography root is required"):
            load_qc_path_config()

    def test_cli_arg_beats_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit ``tractography_root`` overrides environment variables."""
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.delenv("PYTRACTOVIZ_TRACTOGRAPHY_ROOT", raising=False)
        monkeypatch.setenv("TRACTOGRAPHY_ROOT", str(other))
        cfg = load_qc_path_config(tractography_root=tmp_path)
        assert cfg.tractography_root == tmp_path
