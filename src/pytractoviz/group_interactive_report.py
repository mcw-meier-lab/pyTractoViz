"""
Interactive group QC report from combined_tract_stats.csv and combined_profile_data.csv.

Supports one or more groups for side-by-side comparison. Uses Plotly.js (CDN) for
interactive plots in a self-contained HTML file — no extra Python dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def _qc_viz_collect_write() -> tuple[Callable[..., Any], Callable[..., None]]:
    from pytractoviz.qc_viz import collect_group_qc_data, write_combined_group_csvs  # noqa: PLC0415

    return collect_group_qc_data, write_combined_group_csvs


_BASE_NUMERIC_COLS = (
    "shape_score",
    "cci_n_long",
    "cci_n_keep",
    "cci_pct_retained",
    "cci_mean",
)

# CCI-keep bundle profiles are written alongside full-tract profiles; exclude from ribbon plots only.
_PROFILE_TRACT_SUFFIX_EXCLUDE = "_cci_keep"


def _is_profile_tract_excluded(tract: str) -> bool:
    return tract.endswith(_PROFILE_TRACT_SUFFIX_EXCLUDE)


def _filter_profile_rows(profile_rows: list[dict]) -> list[dict]:
    return [r for r in profile_rows if not _is_profile_tract_excluded(r.get("tract", ""))]


def parse_group_spec(spec: str) -> tuple[str, Path]:
    """Parse a group label and QC directory from a CLI ``group-interactive`` argument.

    Parameters
    ----------
    spec : str
        Either ``label=/path/to/qc`` or ``/path/to/qc``. When no label is given,
        the directory name is used as the label.

    Returns
    -------
    tuple[str, Path]
        ``(label, group_dir)`` pair.
    """
    spec = spec.strip()
    if "=" in spec:
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
    else:
        path = Path(spec)
        label = path.name
    if not label:
        label = path.name
    return label, path


def _coerce_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v


def _numeric_columns_from_rows(rows: list[dict]) -> list[str]:
    if not rows:
        return list(_BASE_NUMERIC_COLS)
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    cols = [c for c in _BASE_NUMERIC_COLS if c in keys]
    mean_cols = sorted(k for k in keys if isinstance(k, str) and k.startswith("mean_"))
    cols.extend(mean_cols)
    return cols


def _load_tract_stats_csv(path: Path, group: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            tract = row.get("tract", "")
            if not tract:
                continue
            out = {"group": group, "subject": row.get("subject", ""), "tract": tract}
            for k, v in row.items():
                if k in ("subject", "tract"):
                    continue
                fv = _coerce_float(v)
                out[k] = fv if fv is not None else v
            rows.append(out)
    return rows


def _load_profile_csv(path: Path, group: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            rows.append(
                {
                    "group": group,
                    "subject": row.get("subject", ""),
                    "tract": tract,
                    "metric": metric,
                    "node": node,
                    "value": val,
                },
            )
    return rows


def _profile_values_to_rows(profile_values: dict, group: str) -> list[dict]:
    rows: list[dict] = []
    for (tract, metric), subj_map in profile_values.items():
        for subject_id, node_map in subj_map.items():
            for node, val in node_map.items():
                try:
                    rows.append(
                        {
                            "group": group,
                            "subject": subject_id,
                            "tract": tract,
                            "metric": metric,
                            "node": int(node),
                            "value": float(val),
                        },
                    )
                except (TypeError, ValueError):
                    continue
    return rows


def _summarize_profiles(profile_rows: list[dict]) -> list[dict]:
    """Pre-aggregate node-wise median and IQR per (group, tract, metric)."""
    buckets: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    for row in profile_rows:
        key = (row["group"], row["tract"], row["metric"])
        node = int(row["node"])
        val = float(row["value"])
        buckets.setdefault(key, {}).setdefault(node, []).append(val)

    summaries = []
    for (group, tract, metric), node_map in sorted(buckets.items()):
        nodes = sorted(node_map.keys())
        medians, q1s, q3s = [], [], []
        for n in nodes:
            vals = np.asarray(node_map[n], dtype=float)
            medians.append(float(np.nanmedian(vals)))
            q1s.append(float(np.nanpercentile(vals, 25)))
            q3s.append(float(np.nanpercentile(vals, 75)))
        n_subjects = len(
            {
                r["subject"]
                for r in profile_rows
                if r["group"] == group and r["tract"] == tract and r["metric"] == metric
            },
        )
        summaries.append(
            {
                "group": group,
                "tract": tract,
                "metric": metric,
                "n_subjects": n_subjects,
                "nodes": nodes,
                "median": medians,
                "q1": q1s,
                "q3": q3s,
            },
        )
    return summaries


def load_group_combined_data(
    group_dir: Path,
    group_label: str | None = None,
    *,
    refresh_csv: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Load combined tract stats and profile rows for one group.

    Reads ``combined_tract_stats.csv`` and ``combined_profile_data.csv`` when
    present; otherwise scans subject subdirectories and writes those CSVs.

    Parameters
    ----------
    group_dir : Path
        Group QC directory containing one subdirectory per subject.
    group_label : str, optional
        Label used in output rows. Defaults to ``group_dir.name``.
    refresh_csv : bool, optional
        If True, rescan subject subdirs and rewrite combined CSVs before loading.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(tract_rows, profile_rows)`` with a ``group`` column on each row.
    """
    group_dir = Path(group_dir)
    label = group_label or group_dir.name
    tract_csv = group_dir / "combined_tract_stats.csv"
    profile_csv = group_dir / "combined_profile_data.csv"

    if refresh_csv or not tract_csv.exists() or not profile_csv.exists():
        collect_group_qc_data, write_combined_group_csvs = _qc_viz_collect_write()
        collected = collect_group_qc_data(group_dir)
        write_combined_group_csvs(
            group_dir,
            collected["tract_stats_rows"],
            collected["profile_values"],
        )
        tract_rows = [{"group": label, **row} for row in collected["tract_stats_rows"]]
        profile_rows = _profile_values_to_rows(collected["profile_values"], label)
        return tract_rows, profile_rows

    tract_rows = _load_tract_stats_csv(tract_csv, label) if tract_csv.exists() else []
    profile_rows = _load_profile_csv(profile_csv, label) if profile_csv.exists() else []

    if not tract_rows and not profile_rows:
        collect_group_qc_data, write_combined_group_csvs = _qc_viz_collect_write()
        collected = collect_group_qc_data(group_dir)
        write_combined_group_csvs(
            group_dir,
            collected["tract_stats_rows"],
            collected["profile_values"],
        )
        tract_rows = [{"group": label, **row} for row in collected["tract_stats_rows"]]
        profile_rows = _profile_values_to_rows(collected["profile_values"], label)

    return tract_rows, profile_rows


def _group_overview(tract_rows: list[dict], profile_rows: list[dict]) -> list[dict]:
    groups = sorted({r["group"] for r in tract_rows} | {r["group"] for r in profile_rows})
    overview = []
    for g in groups:
        tr = [r for r in tract_rows if r["group"] == g]
        pr = [r for r in profile_rows if r["group"] == g]
        overview.append(
            {
                "group": g,
                "n_subjects": len({r["subject"] for r in tr if r.get("subject")}),
                "n_tract_stat_rows": len(tr),
                "n_profile_rows": len(pr),
                "tracts": sorted({r["tract"] for r in tr}),
                "profile_metrics": sorted({r["metric"] for r in pr}),
            },
        )
    return overview


_INTERACTIVE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QC Interactive Group Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f4f4f5;
      --panel: #ffffff;
      --border: #d4d4d8;
      --text: #27272a;
      --muted: #71717a;
      --accent: #2563eb;
      --heading: #1e40af;
    }
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
    }
    header h1 { margin: 0 0 0.35rem; color: #18181b; font-size: 1.5rem; }
    header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
    main { padding: 1.25rem 2rem 2.5rem; max-width: 1400px; }
    section {
      margin-bottom: 2rem;
      padding: 1rem 1.25rem;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    section h2 {
      margin: 0 0 0.75rem;
      color: var(--heading);
      font-size: 1.15rem;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1.25rem;
      align-items: flex-end;
      margin-bottom: 0.75rem;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      font-size: 0.85rem;
      color: var(--muted);
    }
    select {
      min-width: 10rem;
      padding: 0.35rem 0.5rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
    }
    .plot { width: 100%; min-height: 420px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th, td {
      padding: 0.45rem 0.75rem;
      text-align: left;
      border: 1px solid var(--border);
    }
    th { background: #f4f4f5; color: var(--heading); }
    tr:nth-child(even) { background: #fafafa; }
    .group-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
    .pill {
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.8rem;
      border: 1px solid var(--border);
      background: #f4f4f5;
    }
    .hint { color: var(--muted); font-size: 0.85rem; margin: 0.25rem 0 0.75rem; }
  </style>
</head>
<body>
  <header>
    <h1>QC Interactive Group Report</h1>
    <p>Explore combined tract stats and profile data. Select metrics and tracts; compare groups when multiple cohorts are loaded.</p>
    <div class="group-pills" id="group-pills"></div>
  </header>
  <main>
    <section id="overview-section">
      <h2>Group overview</h2>
      <table id="overview-table">
        <thead>
          <tr><th>Group</th><th>Subjects</th><th>Tract-stat rows</th><th>Profile rows</th><th>Tracts</th><th>Profile metrics</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>

    <section>
      <h2>Tract stats</h2>
      <p class="hint">Box plots of numeric tract-level metrics. When multiple groups are loaded, each box is colored by group.</p>
      <div class="controls">
        <label>Metric
          <select id="metric-select"></select>
        </label>
        <label>Tracts (hold Ctrl/Cmd to multi-select)
          <select id="tract-select" multiple size="8"></select>
        </label>
        <label>Plot type
          <select id="plot-type-select">
            <option value="box">Box plot</option>
            <option value="violin">Violin plot</option>
          </select>
        </label>
      </div>
      <div id="tract-stats-plot" class="plot"></div>
    </section>

    <section>
      <h2>Group comparison heatmap</h2>
      <p class="hint">Mean metric value per tract and group (subjects averaged within each group).</p>
      <div class="controls">
        <label>Metric
          <select id="heatmap-metric-select"></select>
        </label>
      </div>
      <div id="heatmap-plot" class="plot"></div>
    </section>

    <section>
      <h2>Profile along tract</h2>
      <p class="hint">Median profile with interquartile ribbon (25th-75th percentile) across subjects in each group.</p>
      <div class="controls">
        <label>Tract
          <select id="profile-tract-select"></select>
        </label>
        <label>Metric
          <select id="profile-metric-select"></select>
        </label>
      </div>
      <div id="profile-plot" class="plot"></div>
    </section>
  </main>
  <script>
    const DATA = __DATA_JSON__;
    const GROUP_COLORS = ["#6eb5ff", "#90ee90", "#ffb347", "#dda0dd", "#f08080", "#87ceeb", "#ffd700", "#98fb98"];

    const plotLayout = {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#fafafa",
      font: { color: "#333333" },
      margin: { t: 48, r: 24, b: 80, l: 56 },
      xaxis: { gridcolor: "#e4e4e7", zerolinecolor: "#d4d4d8", color: "#333333" },
      yaxis: { gridcolor: "#e4e4e7", zerolinecolor: "#d4d4d8", color: "#333333" },
      legend: { orientation: "h", y: 1.12 },
    };

    function groupColorMap() {
      const map = {};
      DATA.groups.forEach((g, i) => { map[g] = GROUP_COLORS[i % GROUP_COLORS.length]; });
      return map;
    }

    function renderOverview() {
      const pills = document.getElementById("group-pills");
      const tbody = document.querySelector("#overview-table tbody");
      pills.innerHTML = "";
      tbody.innerHTML = "";
      const colors = groupColorMap();
      for (const row of DATA.overview) {
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.style.borderColor = colors[row.group];
        pill.textContent = row.group;
        pills.appendChild(pill);

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.group}</td>
          <td>${row.n_subjects}</td>
          <td>${row.n_tract_stat_rows}</td>
          <td>${row.n_profile_rows}</td>
          <td>${row.tracts.length}</td>
          <td>${row.profile_metrics.join(", ") || "—"}</td>`;
        tbody.appendChild(tr);
      }
    }

    function fillSelect(id, options, selectedValues) {
      const sel = document.getElementById(id);
      sel.innerHTML = "";
      for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        if (selectedValues && selectedValues.includes(opt)) o.selected = true;
        sel.appendChild(o);
      }
    }

    function selectedTracts() {
      const sel = document.getElementById("tract-select");
      return Array.from(sel.selectedOptions).map(o => o.value);
    }

    function buildDistTrace(plotType, name, matched, metric, color) {
      const vals = matched.map(r => Number(r[metric]));
      const subjects = matched.map(r => r.subject || "unknown");
      const tract = matched[0].tract;
      const trace = {
        type: plotType,
        name,
        x: Array(vals.length).fill(tract),
        y: vals,
        text: subjects,
        hovertemplate:
          "Subject: %{text}<br>Tract: %{x}<br>Value: %{y}<extra>" + name + "</extra>",
        marker: { color },
        line: { color },
      };
      if (plotType === "box") {
        trace.boxmean = "sd";
        trace.boxpoints = "outliers";
      } else {
        trace.points = "outliers";
        trace.meanline = { visible: true };
      }
      return trace;
    }

    function drawTractStats() {
      const metric = document.getElementById("metric-select").value;
      const tracts = selectedTracts();
      const plotType = document.getElementById("plot-type-select").value;
      const colors = groupColorMap();
      const rows = DATA.tract_stats.filter(r =>
        tracts.includes(r.tract) && r[metric] !== null && r[metric] !== "" && !Number.isNaN(Number(r[metric]))
      );
      const traces = [];
      const multiGroup = DATA.groups.length > 1;
      if (multiGroup) {
        for (const group of DATA.groups) {
          for (const tract of tracts) {
            const matched = rows.filter(r => r.group === group && r.tract === tract);
            if (!matched.length) continue;
            traces.push(buildDistTrace(plotType, `${group} — ${tract}`, matched, metric, colors[group]));
          }
        }
      } else {
        for (const tract of tracts) {
          const matched = rows.filter(r => r.tract === tract);
          if (!matched.length) continue;
          traces.push(buildDistTrace(plotType, tract, matched, metric, colors[DATA.groups[0]]));
        }
      }
      const groupMode = multiGroup && tracts.length > 0 ? "group" : undefined;
      Plotly.react("tract-stats-plot", traces, {
        ...plotLayout,
        title: metric.replace(/_/g, " "),
        xaxis: { ...plotLayout.xaxis, title: "Tract", tickangle: -45 },
        yaxis: { ...plotLayout.yaxis, title: metric.replace(/_/g, " ") },
        boxmode: plotType === "box" ? groupMode : undefined,
        violinmode: plotType === "violin" ? groupMode : undefined,
      }, { responsive: true, displayModeBar: true });
    }

    function drawHeatmap() {
      const metric = document.getElementById("heatmap-metric-select").value;
      const tracts = DATA.tracts;
      const groups = DATA.groups;
      const z = [];
      const text = [];
      for (const group of groups) {
        const row = [];
        const textRow = [];
        for (const tract of tracts) {
          const vals = DATA.tract_stats
            .filter(r => r.group === group && r.tract === tract && r[metric] !== null && r[metric] !== "" && !Number.isNaN(Number(r[metric])))
            .map(r => Number(r[metric]));
          const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
          row.push(mean);
          textRow.push(mean === null ? "n/a" : mean.toFixed(4));
        }
        z.push(row);
        text.push(textRow);
      }
      Plotly.react("heatmap-plot", [{
        type: "heatmap",
        x: tracts,
        y: groups,
        z: z,
        text: text,
        texttemplate: "%{text}",
        colorscale: "Viridis",
        colorbar: { title: metric.replace(/_/g, " ") },
        hoverongaps: false,
      }], {
        ...plotLayout,
        title: `Mean ${metric.replace(/_/g, " ")} by tract and group`,
        xaxis: { ...plotLayout.xaxis, title: "Tract", tickangle: -45 },
        yaxis: { ...plotLayout.yaxis, title: "Group", autorange: "reversed" },
        margin: { ...plotLayout.margin, b: 120 },
      }, { responsive: true, displayModeBar: true });
    }

    function drawProfile() {
      const tract = document.getElementById("profile-tract-select").value;
      const metric = document.getElementById("profile-metric-select").value;
      const colors = groupColorMap();
      const traces = [];
      for (const group of DATA.groups) {
        const summary = DATA.profile_summaries.find(s => s.group === group && s.tract === tract && s.metric === metric);
        if (!summary) continue;
        const color = colors[group];
        const x = summary.nodes;
        traces.push({
          type: "scatter",
          mode: "lines",
          name: `${group} median (n=${summary.n_subjects})`,
          x: x,
          y: summary.median,
          line: { color: color, width: 2.5 },
        });
        traces.push({
          type: "scatter",
          mode: "lines",
          name: `${group} IQR`,
          x: x.concat([...x].reverse()),
          y: summary.q1.concat([...summary.q3].reverse()),
          fill: "toself",
          fillcolor: color.length === 7 ? color + "40" : color,
          line: { color: "transparent" },
          showlegend: false,
          hoverinfo: "skip",
        });
      }
      Plotly.react("profile-plot", traces, {
        ...plotLayout,
        title: `${tract} — ${metric}`,
        xaxis: { ...plotLayout.xaxis, title: "Node along tract" },
        yaxis: { ...plotLayout.yaxis, title: metric },
      }, { responsive: true, displayModeBar: true });
    }

    function init() {
      renderOverview();
      fillSelect("metric-select", DATA.numeric_columns, [DATA.numeric_columns[0]]);
      fillSelect("heatmap-metric-select", DATA.numeric_columns, [DATA.numeric_columns[0]]);
      fillSelect("tract-select", DATA.tracts, DATA.tracts.slice(0, Math.min(2, DATA.tracts.length)));
      fillSelect("profile-tract-select", DATA.profile_tracts, [DATA.profile_tracts[0]]);
      fillSelect("profile-metric-select", DATA.profile_metrics, [DATA.profile_metrics[0]]);

      ["metric-select", "tract-select", "plot-type-select"].forEach(id => {
        document.getElementById(id).addEventListener("change", drawTractStats);
      });
      document.getElementById("heatmap-metric-select").addEventListener("change", drawHeatmap);
      ["profile-tract-select", "profile-metric-select"].forEach(id => {
        document.getElementById(id).addEventListener("change", drawProfile);
      });

      drawTractStats();
      drawHeatmap();
      if (DATA.profile_summaries.length) drawProfile();
      else document.getElementById("profile-plot").innerHTML = "<p class='hint'>No profile data available.</p>";
    }

    init();
  </script>
</body>
</html>
"""


def generate_group_interactive_html(
    group_specs: list[tuple[str, Path]],
    output_html_path: Path | None = None,
    *,
    refresh_csv: bool = False,
) -> str:
    """
    Build an interactive HTML report from one or more group QC directories.

    The output is a self-contained HTML file using Plotly.js (CDN) for
    interactive tract stats, heatmaps, and profile ribbon plots.

    Parameters
    ----------
    group_specs : list of (str, Path)
        ``(label, group_dir)`` pairs. Labels must be unique.
    output_html_path : Path, optional
        Destination HTML path. Defaults to ``group_dir/group_interactive.html``
        for a single group, or ``group_interactive_comparison.html`` in the
        first group directory when multiple groups are provided.
    refresh_csv : bool, optional
        If True, rescan subject subdirs and rewrite combined CSVs before loading.

    Returns
    -------
    str
        Path to the written HTML file.

    Raises
    ------
    ValueError
        If no groups are provided, labels are duplicated, or no data is found.
    """
    if not group_specs:
        raise ValueError("At least one group must be provided")

    labels = [label for label, _ in group_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Group labels must be unique")

    all_tract_rows: list[dict] = []
    all_profile_rows: list[dict] = []
    for label, group_dir_raw in group_specs:
        group_dir = Path(group_dir_raw)
        tract_rows, profile_rows = load_group_combined_data(
            group_dir,
            label,
            refresh_csv=refresh_csv,
        )
        all_tract_rows.extend(tract_rows)
        all_profile_rows.extend(profile_rows)

    if not all_tract_rows and not all_profile_rows:
        raise ValueError("No combined tract stats or profile data found for any group")

    profile_rows_for_ribbon = _filter_profile_rows(all_profile_rows)

    groups = [label for label, _ in group_specs]
    tracts = sorted({r["tract"] for r in all_tract_rows})
    profile_tracts = sorted({r["tract"] for r in profile_rows_for_ribbon})
    profile_metrics = sorted({r["metric"] for r in profile_rows_for_ribbon})
    numeric_columns = _numeric_columns_from_rows(all_tract_rows)
    overview = _group_overview(all_tract_rows, all_profile_rows)
    profile_summaries = _summarize_profiles(profile_rows_for_ribbon)

    payload = {
        "groups": groups,
        "overview": overview,
        "tract_stats": all_tract_rows,
        "numeric_columns": numeric_columns,
        "tracts": tracts,
        "profile_tracts": profile_tracts,
        "profile_metrics": profile_metrics,
        "profile_summaries": profile_summaries,
    }

    if output_html_path is None:
        if len(group_specs) == 1:
            output_html_path = Path(group_specs[0][1]) / "group_interactive.html"
        else:
            output_html_path = Path(group_specs[0][1]) / "group_interactive_comparison.html"
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    data_json = json.dumps(payload, separators=(",", ":"))
    html = _INTERACTIVE_HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    output_html_path.write_text(html, encoding="utf-8")
    return str(output_html_path)
