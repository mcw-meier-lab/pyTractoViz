# pyTractoViz

[![ci](https://github.com/mcw-meier-lab/pytractoviz/workflows/ci/badge.svg)](https://github.com/mcw-meier-lab/pytractoviz/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mcw-meier-lab.github.io/pytractoviz/)
[![pypi version](https://img.shields.io/pypi/v/pytractoviz.svg)](https://pypi.org/project/pytractoviz/)

Python tools for diffusion tractography visualization and QC reporting.

## Installation

```bash
pip install pytractoviz
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install pytractoviz
```

## Quick start

**Interactive 3D review** of tract files:

```bash
pytractoviz interactive /path/to/bundles --ref /path/to/T1w.nii.gz
```

**Batch QC pipeline** for a subject (PYAFQ + qsirecon layout):

```bash
export PYTRACTOVIZ_TRACTOGRAPHY_ROOT=/path/to/tractography
pytractoviz run sub-0000 UNC -j 4
```

**Regenerate reports** without re-running the pipeline:

```bash
pytractoviz html /path/to/qc_data.json
pytractoviz group-html /path/to/QC
pytractoviz group-interactive UNC=/path/to/QC
```

## Documentation

Full usage guide, path configuration, and API reference:

**https://mcw-meier-lab.github.io/pytractoviz/**
