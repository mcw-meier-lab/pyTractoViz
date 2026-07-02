# pyTractoViz

Python tools for diffusion tractography visualization and QC reporting.

## Installation

```
pip install pytractoviz
```

With [`uv`](https://docs.astral.sh/uv/):

```
uv tool install pytractoviz
```

## Quick start

**Interactive 3D review** of tract files:

```
pytractoviz interactive /path/to/bundles --ref /path/to/T1w.nii.gz
```

**Batch QC pipeline** for a subject (PYAFQ + qsirecon layout):

```
export PYTRACTOVIZ_TRACTOGRAPHY_ROOT=/path/to/tractography
pytractoviz run sub-0000 UNC -j 4
```

**Regenerate reports** without re-running the pipeline:

```
pytractoviz html /path/to/qc_data.json
pytractoviz group-html /path/to/QC
pytractoviz group-interactive UNC=/path/to/QC
```

## Documentation

Full usage guide, path configuration, and API reference:

**<https://mcw-meier-lab.github.io/pytractoviz/>**
