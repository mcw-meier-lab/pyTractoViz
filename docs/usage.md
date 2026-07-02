---
title: Usage
---

# Usage

pyTractoViz provides a single CLI, `pytractoviz`, with subcommands for interactive tract review, automated batch QC, and HTML report generation.

## Command overview

| Command | Purpose |
|---------|---------|
| [`interactive`](#interactive-3d-viewer) | Review `.trk` bundles one at a time in a 3D viewer |
| [`run`](#batch-qc-pipeline) | Full subject QC pipeline (views, CCI, AFQ, shape, CSV/JSON/HTML) |
| [`html`](#regenerate-a-subject-report) | Regenerate `qc_report.html` from `qc_data.json` |
| [`group-html`](#group-summary-report) | Group triage page + combined CSVs |
| [`group-interactive`](#interactive-group-comparison) | Plotly-based group stats comparison (no extra Python deps) |

```bash
pytractoviz --help
pytractoviz run --help
```

## Interactive 3D viewer

Walk through tract files interactively. Close each window to advance to the next bundle.

```bash
pytractoviz interactive /path/to/bundles/*.trk --ref /path/to/T1w.nii.gz
```

You can pass individual `.trk` files or directories containing `.trk` files.

Common options:

- `--max-streamlines` — cap how many streamlines are rendered
- `--subsample-factor` — keep a fraction of streamlines (0.0–1.0)
- `--resample-streamlines` — reduce points per streamline
- `--skip N` — skip the first *N* tracts

!!! note "Breaking change (0.6.0)"
    The 3D viewer command was renamed from `qc-interactive` to `interactive`.

## Batch QC pipeline

`run` executes the full per-subject QC workflow for PYAFQ bundles against a 16-bundle HCP atlas. It writes:

- Per-tract PNG views (subject, atlas, CCI, AFQ)
- `profile_data.csv`, `tract_stats.csv`, `qc_data.json`
- `qc_report.html`

### Path configuration

`run` needs a **tractography root** — the top-level directory that contains your `qsiprep/`, `qsirecon/`, `templates/`, and atlas folders.

Set it once per session (recommended on HPC):

```bash
export PYTRACTOVIZ_TRACTOGRAPHY_ROOT=/scratch/group/project/tractography
```

`TRACTOGRAPHY_ROOT` is also accepted; `PYTRACTOVIZ_TRACTOGRAPHY_ROOT` takes precedence if both are set.

Or pass it on the command line:

```bash
pytractoviz run sub-0000 UNC \
  --tractography-root /scratch/group/project/tractography \
  -j 4
```

#### Default layout

Given `tractography_root` and `project` / `subject`, paths resolve as:

| Input | Default path |
|-------|----------------|
| Atlas bundles | `{root}/hcp_atlas_16_bundles/Atlas_in_MNI_Space_16_bundles/bundles` |
| Atlas reference mask | `{root}/templates/tpl-MNI152NLin2009cAsym/..._brain_mask.nii.gz` |
| Subject anatomy | `{root}/qsiprep/{project}/{subject}/anat` |
| PYAFQ bundles | `{root}/qsirecon/{project}/derivatives/PYAFQ/{subject}/bundles` |
| DKI / NODDI maps | `{root}/qsirecon/{project}/derivatives/qsirecon-DKI|NODDI/{subject}/dwi` |
| QC output | `{root}/qsirecon/{project}/derivatives/QC/{subject}/` |

Override individual paths with CLI flags or environment variables:

| CLI flag | Environment variable |
|----------|---------------------|
| `--tractography-root` | `PYTRACTOVIZ_TRACTOGRAPHY_ROOT`, `TRACTOGRAPHY_ROOT` |
| `--atlas-dir` | `PYTRACTOVIZ_ATLAS_DIR` |
| `--atlas-ref-img` | `PYTRACTOVIZ_ATLAS_REF_IMG` |
| `--qsiprep-root` | `PYTRACTOVIZ_QSIPREP_ROOT` |
| `--qsirecon-root` | `PYTRACTOVIZ_QSIRECON_ROOT` |
| `--output-dir` | `PYTRACTOVIZ_QC_OUTPUT_DIR` |

CLI arguments override environment variables.

### Example: cluster job

```bash
export PYTRACTOVIZ_TRACTOGRAPHY_ROOT=/scratch/g/bbrett/K-Award/NFLLONG/tractography
export VTK_USE_OFFSCREEN=1
export VTK_USE_OSMESA=1

pytractoviz run sub-0000 UNC -j 4 --render-backend osmesa
```

Set `PYTRACTOVIZ_LOG_LEVEL=DEBUG` to include per-step timing messages (default: `INFO`).

### Performance options

| Flag | Default | Description |
|------|---------|-------------|
| `-j`, `--jobs` | 1 | Parallel tract workers |
| `--render-backend` | `osmesa` | `osmesa` or `egl` (GPU nodes) |
| `--max-streamlines-view` | 5000 | Cap for view rendering (0 = no cap) |
| `--max-streamlines-cci` | 5000 | Cap for CCI |
| `--max-streamlines-afq` | 6000 | Cap for AFQ profiles |
| `--max-streamlines-shape` | 4000 | Cap for shape similarity |
| `--no-streamline-caps` | off | Disable all caps |

## Regenerate a subject report

Rebuild HTML from an existing `qc_data.json` without re-running the pipeline:

```bash
pytractoviz html /path/to/subject/qc_data.json
```

Output defaults to `qc_report.html` in the same directory.

## Group summary report

Scan a group QC directory (one subdirectory per subject) and build a triage page with links to each subject report:

```bash
pytractoviz group-html /path/to/derivatives/QC
```

This also writes `combined_tract_stats.csv`, `combined_profile_data.csv`, and `group_interactive.html` when possible.

Options:

- `-o`, `--output` — custom output path (default: `group_dir/group_summary.html`)
- `--skip-pairwise-shape` — skip subject×subject shape similarity matrices
- `--pairwise-shape-max-streamlines` — streamline cap for pairwise shape (default: 4000)
- `--pairwise-shape-jobs` — parallel workers per tract (0 = all CPUs)

## Interactive group comparison

Build a self-contained HTML file with Plotly charts for tract stats and AFQ-style profile ribbons. Works from combined CSVs or by scanning subject subdirectories.

Single group:

```bash
pytractoviz group-interactive /path/to/derivatives/QC
```

Multiple groups (side-by-side comparison):

```bash
pytractoviz group-interactive \
  UNC=/data/QC/UNC \
  ADNI=/data/QC/ADNI \
  -o /data/QC/comparison.html
```

Use `--refresh-csv` to rescan subject folders and rewrite `combined_*.csv` before building the report.

## Python API

```python
from pytractoviz import (
    load_qc_path_config,
    generate_subject_qc_html,
    generate_group_summary_html,
    generate_group_interactive_html,
    parse_group_spec,
)
from pytractoviz.qc_viz import main as run_subject_qc

cfg = load_qc_path_config(tractography_root="/data/tractography")
run_subject_qc("sub-0000", "UNC", path_config=cfg, n_jobs=4)

generate_subject_qc_html("/data/QC/sub-0000/qc_data.json")
generate_group_summary_html("/data/QC")
generate_group_interactive_html([parse_group_spec("UNC=/data/QC")])
```

See the [API reference](reference/api.md) for full signatures.

## Installing a development version

To test unreleased changes on a cluster, install from a git branch:

```bash
pip install --upgrade "git+https://github.com/mcw-meier-lab/pyTractoViz.git@your-branch"
```

For iterative development, clone the repo and use an editable install:

```bash
git clone https://github.com/mcw-meier-lab/pyTractoViz.git
cd pyTractoViz
pip install -e .
```
