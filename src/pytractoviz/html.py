"""HTML report generation for quality checking."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any


def create_quality_check_html(
    data: dict[str, dict[str, dict[str, str]]],
    output_file: str,
    title: str = "Tractography Quality Check",
    items_per_page: int = 50,
) -> None:
    """Create an interactive HTML quality check report.

    This function generates an HTML file with filtering, search, and pagination
    capabilities for efficiently reviewing large datasets of visualizations.

    Parameters
    ----------
    data : dict
        Nested dictionary structure: {subject_id: {tract_name: {media_type: file_path}}}
        Example:
        {
            "sub-001": {
                "AF_L": {
                    "image": "path/to/image.png",
                    "plot": "path/to/plot.png",
                    "gif": "path/to/animation.gif",
                    "video": "path/to/video.mp4"
                }
            }
        }
    output_file : str
        Path where the HTML file will be saved.
    title : str, optional
        Title for the HTML report. Default is "Tractography Quality Check".
    items_per_page : int, optional
        Number of items to display per page. Default is 50.

    Notes
    -----
    Supported media types in the data dictionary:
    - "image": Static images (PNG, JPG, etc.)
    - "plot": Matplotlib plots or other static plots
    - "gif": Animated GIF files
    - "video": Video files (MP4, WebM, etc.)

    The HTML report includes:
    - Filtering by subject and tract
    - Search functionality
    - Pagination for efficient loading
    - Thumbnail grid view with expandable detail views
    - Support for images, plots, GIFs, and videos
    """
    # Convert file paths to relative paths for HTML
    html_dir = Path(output_file).parent
    data_processed: dict[str, dict[str, dict[str, str]]] = {}
    summary_data: list[dict[str, Any]] = []  # For summary table

    for subject_id, tracts in data.items():
        data_processed[subject_id] = {}
        for tract_name, media in tracts.items():
            data_processed[subject_id][tract_name] = {}
            metrics_dict = {}
            errors_list = []
            missing_data_list = []

            for media_type, file_path in media.items():
                # Extract metrics, errors, and missing_data
                if media_type == "_metrics":
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        metrics_dict = json.loads(file_path)
                elif media_type == "_errors":
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        errors_list = json.loads(file_path)
                elif media_type == "_missing_data":
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        missing_data_list = json.loads(file_path)
                # Handle numeric scores (like shape_similarity_score)
                elif isinstance(file_path, (int, float)):
                    data_processed[subject_id][tract_name][media_type] = str(file_path)
                elif file_path and os.path.exists(file_path):
                    # Convert to relative path from HTML file location
                    try:
                        rel_path = os.path.relpath(file_path, html_dir)
                        data_processed[subject_id][tract_name][media_type] = rel_path
                    except ValueError:
                        # If paths are on different drives (Windows), use absolute
                        data_processed[subject_id][tract_name][media_type] = file_path
                elif file_path:
                    # Store as-is if it's a string but file doesn't exist (might be a score string)
                    data_processed[subject_id][tract_name][media_type] = file_path

            # Add to summary data
            summary_data.append(
                {
                    "subject": subject_id,
                    "tract": tract_name,
                    "metrics": metrics_dict,
                    "errors": errors_list,
                    "missing_data": missing_data_list,
                },
            )

    # Extract unique subjects and tracts for filters
    subjects = sorted(data_processed.keys())
    tract_names: list[str] = sorted({tract for tracts_dict in data_processed.values() for tract in tracts_dict})

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin-bottom: 0.5rem;
        }}

        .header .stats {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .controls {{
            background: white;
            padding: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
        }}

        .control-group label {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #555;
        }}

        .control-group select,
        .control-group input {{
            padding: 0.6rem;
            border: 2px solid #e0e0e0;
            border-radius: 4px;
            font-size: 0.9rem;
            transition: border-color 0.3s;
        }}

        .control-group select:focus,
        .control-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }}

        .pagination button {{
            padding: 0.5rem 1rem;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s;
        }}

        .pagination button:hover:not(:disabled) {{
            background: #667eea;
            color: white;
        }}

        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .pagination .page-info {{
            padding: 0 1rem;
            font-weight: 600;
        }}

        .grid-container {{
            padding: 2rem;
            max-width: 1800px;
            margin: 0 auto;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }}

        .item-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }}

        .item-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}

        .item-header {{
            padding: 1rem;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}

        .item-header h3 {{
            font-size: 1rem;
            margin-bottom: 0.25rem;
            color: #333;
        }}

        .item-header .tract-name {{
            font-size: 0.85rem;
            color: #667eea;
            font-weight: 600;
        }}

        .item-media {{
            position: relative;
            width: 100%;
            padding-top: 75%; /* 4:3 aspect ratio */
            background: #f0f0f0;
            overflow: hidden;
        }}

        .item-media img,
        .item-media video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: white;
        }}

        .item-media .media-placeholder {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #999;
            font-size: 0.9rem;
        }}

        .item-footer {{
            padding: 0.75rem 1rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .media-badge {{
            padding: 0.25rem 0.5rem;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .score-badge {{
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 700;
            text-align: center;
            display: inline-block;
            margin: 0.5rem 0;
        }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 1rem 0;
        }}

        .comparison-item {{
            background: #f8f9fa;
            border-radius: 4px;
            padding: 0.5rem;
            text-align: center;
        }}

        .comparison-item h5 {{
            margin-bottom: 0.5rem;
            color: #555;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .comparison-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            background: white;
        }}

        @media (max-width: 768px) {{
            .comparison-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            overflow: auto;
        }}

        .modal-content {{
            position: relative;
            margin: 2% auto;
            max-width: 90%;
            max-height: 90vh;
            background: white;
            border-radius: 8px;
            padding: 2rem;
            overflow-y: auto;
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e0e0e0;
        }}

        .modal-header h2 {{
            color: #333;
        }}

        .close {{
            color: #aaa;
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
            line-height: 1;
        }}

        .close:hover {{
            color: #000;
        }}

        .modal-media-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}

        .modal-media-item {{
            background: #f8f9fa;
            border-radius: 4px;
            padding: 1rem;
        }}

        .modal-media-item h4 {{
            margin-bottom: 0.5rem;
            color: #555;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}

        .modal-media-item img,
        .modal-media-item video {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            background: white;
        }}

        .no-results {{
            text-align: center;
            padding: 4rem 2rem;
            color: #999;
        }}

        .no-results h3 {{
            margin-bottom: 0.5rem;
            color: #666;
        }}

        .tabs {{
            background: white;
            padding: 0;
            margin: 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 99;
        }}

        .tab-buttons {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            background: white;
        }}

        .tab-button {{
            padding: 1rem 2rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}

        .tab-button:hover {{
            background: #f8f9fa;
            color: #333;
        }}

        .tab-button.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .summary-section {{
            background: white;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 1800px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .summary-section h2 {{
            margin-bottom: 1.5rem;
            color: #333;
            font-size: 1.5rem;
        }}

        .summary-table-container {{
            overflow-x: auto;
            width: 100%;
        }}

        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
            min-width: 1000px;
        }}

        .summary-table th,
        .summary-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}

        .summary-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .summary-table tr:hover {{
            background: #f8f9fa;
        }}

        .summary-table .metric-value {{
            font-weight: 600;
            color: #667eea;
        }}

        .summary-table .error-cell {{
            color: #d32f2f;
            font-size: 0.85rem;
            max-width: 300px;
            word-wrap: break-word;
        }}

        .summary-table .missing-cell {{
            color: #f57c00;
            font-size: 0.85rem;
            max-width: 300px;
            word-wrap: break-word;
        }}

        .summary-table .no-data {{
            color: #999;
            font-style: italic;
        }}

        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}

            .controls-grid {{
                grid-template-columns: 1fr;
            }}

            .modal-content {{
                max-width: 95%;
                padding: 1rem;
            }}

            .modal-media-container {{
                grid-template-columns: 1fr;
            }}

            .tab-button {{
                padding: 0.75rem 1rem;
                font-size: 0.9rem;
            }}

            .summary-table {{
                font-size: 0.8rem;
            }}

            .summary-table th,
            .summary-table td {{
                padding: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="stats">
            <span id="total-items">{len(subjects)} subjects x {len(tract_names)} tracts</span> |
            <span id="filtered-count">Showing all items</span>
        </div>
    </div>

    <div class="controls">
        <div class="controls-grid">
            <div class="control-group">
                <label for="subject-filter">Filter by Subject</label>
                <select id="subject-filter">
                    <option value="">All Subjects</option>
                    {"".join(f'<option value="{s}">{s}</option>' for s in subjects)}
                </select>
            </div>
            <div class="control-group">
                <label for="tract-filter">Filter by Tract</label>
                <select id="tract-filter">
                    <option value="">All Tracts</option>
                    {"".join(f'<option value="{t}">{t}</option>' for t in tract_names)}
                </select>
            </div>
            <div class="control-group">
                <label for="search">Search</label>
                <input type="text" id="search" placeholder="Search subject or tract...">
            </div>
        </div>
        <div class="pagination">
            <button id="prev-page" disabled>Previous</button>
            <span class="page-info">
                Page <span id="current-page">1</span> of <span id="total-pages">1</span>
            </span>
            <button id="next-page">Next</button>
        </div>
    </div>

    <div class="tabs">
        <div class="tab-buttons">
            <button class="tab-button active" data-tab="visualizations">Visualizations</button>
            <button class="tab-button" data-tab="summary">Summary</button>
        </div>
    </div>

    <div id="visualizations-tab" class="tab-content active">
        <div class="grid-container">
            <div class="grid" id="items-grid"></div>
            <div class="no-results" id="no-results" style="display: none;">
                <h3>No items found</h3>
                <p>Try adjusting your filters or search terms</p>
            </div>
        </div>
    </div>

    <div id="summary-tab" class="tab-content">
        <div class="summary-section">
            <h2>Summary Table: Metrics, Errors, and Missing Data</h2>
            <div class="summary-table-container">
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Subject</th>
                            <th>Tract</th>
                            <th>Initial Streamlines</th>
                            <th>After CCI Filter</th>
                            <th>CCI Mean</th>
                            <th>CCI Median</th>
                            <th>Shape Similarity</th>
                            <th>Errors</th>
                            <th>Missing Data</th>
                        </tr>
                    </thead>
                    <tbody id="summary-table-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div id="modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-title"></h2>
                <span class="close">&times;</span>
            </div>
            <div class="modal-media-container" id="modal-media"></div>
        </div>
    </div>

    <script>
        const data = {json.dumps(data_processed)};
        const summaryData = {json.dumps(summary_data)};
        const itemsPerPage = {items_per_page};

        let currentPage = 1;
        let filteredData = [];

        function getMediaType(filePath) {{
            if (!filePath) return null;
            // Check if it's a numeric score (for shape_similarity_score)
            if (typeof filePath === 'number' || (!isNaN(parseFloat(filePath)) && isFinite(filePath))) {{
                return 'score';
            }}
            const ext = filePath.split('.').pop().toLowerCase();
            if (['gif'].includes(ext)) return 'gif';
            if (['mp4', 'webm', 'ogg'].includes(ext)) return 'video';
            if (['png', 'jpg', 'jpeg', 'svg', 'webp'].includes(ext)) return 'image';
            return 'image';
        }}

        function isComparisonType(mediaType) {{
            return ['before_after_cci', 'atlas_comparison', 'shape_similarity_image'].includes(mediaType);
        }}

        function flattenData() {{
            const items = [];
            for (const [subject, tracts] of Object.entries(data)) {{
                for (const [tract, media] of Object.entries(tracts)) {{
                    items.push({{ subject, tract, media }});
                }}
            }}
            return items;
        }}

        function filterData() {{
            const subjectFilter = document.getElementById('subject-filter').value;
            const tractFilter = document.getElementById('tract-filter').value;
            const searchTerm = document.getElementById('search').value.toLowerCase();

            filteredData = flattenData().filter(item => {{
                const matchSubject = !subjectFilter || item.subject === subjectFilter;
                const matchTract = !tractFilter || item.tract === tractFilter;
                const matchSearch = !searchTerm ||
                    item.subject.toLowerCase().includes(searchTerm) ||
                    item.tract.toLowerCase().includes(searchTerm);
                return matchSubject && matchTract && matchSearch;
            }});

            currentPage = 1;
            renderItems();
            updatePagination();
        }}

        function renderItems() {{
            const grid = document.getElementById('items-grid');
            const noResults = document.getElementById('no-results');
            const filteredCount = document.getElementById('filtered-count');

            if (filteredData.length === 0) {{
                grid.style.display = 'none';
                noResults.style.display = 'block';
                filteredCount.textContent = 'No items found';
                return;
            }}

            grid.style.display = 'grid';
            noResults.style.display = 'none';
            filteredCount.textContent = `Showing ${{filteredData.length}} item${{filteredData.length !== 1 ? 's' : ''}}`;

            const start = (currentPage - 1) * itemsPerPage;
            const end = start + itemsPerPage;
            const pageItems = filteredData.slice(start, end);

            grid.innerHTML = pageItems.map(item => {{
                const mediaTypes = Object.keys(item.media);
                // Find primary media (skip scores)
                const primaryMedia = Object.entries(item.media).find(([type, path]) => {{
                    const mt = getMediaType(path);
                    return mt !== 'score' && mt !== null;
                }});
                const primaryMediaPath = primaryMedia ? primaryMedia[1] : null;
                const mediaType = primaryMediaPath ? getMediaType(primaryMediaPath) : null;

                // Find scores to display
                const scores = Object.entries(item.media)
                    .filter(([type, path]) => getMediaType(path) === 'score')
                    .map(([type, path]) => ({{
                        type: type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase()),
                        value: parseFloat(path)
                    }}));

                let mediaHtml = '<div class="media-placeholder">No media available</div>';
                if (primaryMediaPath) {{
                    if (mediaType === 'video') {{
                        mediaHtml = `<video src="${{primaryMediaPath}}" muted loop></video>`;
                    }} else if (mediaType === 'gif' || mediaType === 'image') {{
                        mediaHtml = `<img src="${{primaryMediaPath}}" alt="${{item.tract}}" loading="lazy">`;
                    }}
                }} else if (scores.length > 0) {{
                    // If only scores available, display them
                    mediaHtml = scores.map(score =>
                        `<div class="score-badge" style="margin: 0.5rem;">${{score.type}}: ${{score.value.toFixed(3)}}</div>`
                    ).join('');
                }}

                const badges = mediaTypes.map(type =>
                    `<span class="media-badge">${{type}}</span>`
                ).join('');

                return `
                    <div class="item-card" onclick="openModal('${{item.subject}}', '${{item.tract}}')">
                        <div class="item-header">
                            <h3>${{item.subject}}</h3>
                            <div class="tract-name">${{item.tract}}</div>
                        </div>
                        <div class="item-media">
                            ${{mediaHtml}}
                        </div>
                        <div class="item-footer">
                            ${{badges}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function updatePagination() {{
            const totalPages = Math.ceil(filteredData.length / itemsPerPage);
            document.getElementById('current-page').textContent = currentPage;
            document.getElementById('total-pages').textContent = totalPages || 1;
            document.getElementById('prev-page').disabled = currentPage === 1;
            document.getElementById('next-page').disabled = currentPage >= totalPages;
        }}

        function openModal(subject, tract) {{
            const item = filteredData.find(i => i.subject === subject && i.tract === tract);
            if (!item) return;

            document.getElementById('modal-title').textContent = `${{subject}} - ${{tract}}`;
            const modalMedia = document.getElementById('modal-media');

            // Separate scores and media files
            const scores = [];
            const mediaFiles = [];
            const comparisons = [];

            Object.entries(item.media).forEach(([type, path]) => {{
                if (!path) return;
                const mediaType = getMediaType(path);

                if (mediaType === 'score') {{
                    scores.push({{ type, value: parseFloat(path) }});
                }} else if (isComparisonType(type)) {{
                    comparisons.push({{ type, path }});
                }} else {{
                    mediaFiles.push({{ type, path }});
                }}
            }});

            let html = '';

            // Display scores as badges
            if (scores.length > 0) {{
                html += scores.map(score => `
                    <div class="modal-media-item">
                        <h4>${{score.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                        <div class="score-badge">${{score.value.toFixed(3)}}</div>
                    </div>
                `).join('');
            }}

            // Display comparisons side-by-side
            if (comparisons.length > 0) {{
                html += comparisons.map(comp => {{
                    const mediaType = getMediaType(comp.path);
                    let mediaHtml = '';

                    if (mediaType === 'video') {{
                        mediaHtml = `<video src="${{comp.path}}" controls autoplay></video>`;
                    }} else if (mediaType === 'gif' || mediaType === 'image') {{
                        mediaHtml = `<img src="${{comp.path}}" alt="${{comp.type}}">`;
                    }}

                    // For before_after_cci, the image is already side-by-side, just add labels
                    if (comp.type === 'before_after_cci') {{
                        return `
                            <div class="modal-media-item">
                                <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                ${{mediaHtml}}
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem; text-align: center;">
                                    <div style="font-size: 0.85rem; color: #555; font-weight: 600;">Before CCI</div>
                                    <div style="font-size: 0.85rem; color: #555; font-weight: 600;">After CCI</div>
                                </div>
                            </div>
                        `;
                    }} else if (comp.type === 'atlas_comparison') {{
                        // Check if we have separate subject and atlas images
                        const subjectImg = item.media.subject_image || item.media.atlas_comparison_subject;
                        const atlasImg = item.media.atlas_image || item.media.atlas_comparison_atlas;

                        if (subjectImg && atlasImg) {{
                            const subjectType = getMediaType(subjectImg);
                            const atlasType = getMediaType(atlasImg);
                            let subjectHtml = subjectType === 'image' || subjectType === 'gif'
                                ? `<img src="${{subjectImg}}" alt="Subject">`
                                : `<img src="${{comp.path}}" alt="Subject">`;
                            let atlasHtml = atlasType === 'image' || atlasType === 'gif'
                                ? `<img src="${{atlasImg}}" alt="Atlas">`
                                : `<img src="${{comp.path}}" alt="Atlas">`;

                            return `
                                <div class="modal-media-item">
                                    <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                    <div class="comparison-container">
                                        <div class="comparison-item">
                                            <h5>Subject</h5>
                                            ${{subjectHtml}}
                                        </div>
                                        <div class="comparison-item">
                                            <h5>Atlas</h5>
                                            ${{atlasHtml}}
                                        </div>
                                    </div>
                                </div>
                            `;
                        }} else {{
                            // Single comparison image
                            return `
                                <div class="modal-media-item">
                                    <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                    ${{mediaHtml}}
                                </div>
                            `;
                        }}
                    }} else if (comp.type === 'shape_similarity_image') {{
                        // Check if we have separate subject and atlas images
                        const subjectImg = item.media.shape_similarity_subject;
                        const atlasImg = item.media.shape_similarity_atlas;

                        if (subjectImg && atlasImg) {{
                            const subjectType = getMediaType(subjectImg);
                            const atlasType = getMediaType(atlasImg);
                            let subjectHtml = subjectType === 'image' || subjectType === 'gif'
                                ? `<img src="${{subjectImg}}" alt="Subject">`
                                : `<img src="${{comp.path}}" alt="Subject">`;
                            let atlasHtml = atlasType === 'image' || atlasType === 'gif'
                                ? `<img src="${{atlasImg}}" alt="Atlas">`
                                : `<img src="${{comp.path}}" alt="Atlas">`;

                            return `
                                <div class="modal-media-item">
                                    <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                    <div class="comparison-container">
                                        <div class="comparison-item">
                                            <h5>Subject</h5>
                                            ${{subjectHtml}}
                                        </div>
                                        <div class="comparison-item">
                                            <h5>Atlas</h5>
                                            ${{atlasHtml}}
                                        </div>
                                    </div>
                                </div>
                            `;
                        }} else {{
                            // Single overlay image
                            return `
                                <div class="modal-media-item">
                                    <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                    ${{mediaHtml}}
                                </div>
                            `;
                        }}
                    }} else {{
                        return `
                            <div class="modal-media-item">
                                <h4>${{comp.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                                ${{mediaHtml}}
                            </div>
                        `;
                    }}
                }}).join('');
            }}

            // Display other media files
            html += mediaFiles.map(media => {{
                const mediaType = getMediaType(media.path);
                let mediaHtml = '';

                if (mediaType === 'video') {{
                    mediaHtml = `<video src="${{media.path}}" controls autoplay></video>`;
                }} else if (mediaType === 'gif' || mediaType === 'image') {{
                    mediaHtml = `<img src="${{media.path}}" alt="${{media.type}}">`;
                }}

                return `
                    <div class="modal-media-item">
                        <h4>${{media.type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                        ${{mediaHtml}}
                    </div>
                `;
            }}).join('');

            modalMedia.innerHTML = html;

            document.getElementById('modal').style.display = 'block';
        }}

        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}

        // Event listeners
        document.getElementById('subject-filter').addEventListener('change', filterData);
        document.getElementById('tract-filter').addEventListener('change', filterData);
        document.getElementById('search').addEventListener('input', filterData);
        document.getElementById('prev-page').addEventListener('click', () => {{
            if (currentPage > 1) {{
                currentPage--;
                renderItems();
                updatePagination();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }});
        document.getElementById('next-page').addEventListener('click', () => {{
            const totalPages = Math.ceil(filteredData.length / itemsPerPage);
            if (currentPage < totalPages) {{
                currentPage++;
                renderItems();
                updatePagination();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }});
        document.querySelector('.close').addEventListener('click', closeModal);
        document.getElementById('modal').addEventListener('click', (e) => {{
            if (e.target.id === 'modal') closeModal();
        }});

        function renderSummaryTable() {{
            const tbody = document.getElementById('summary-table-body');
            tbody.innerHTML = summaryData.map(item => {{
                const metrics = item.metrics || {{}};
                const errors = item.errors || [];
                const missing = item.missing_data || [];

                const formatValue = (val) => {{
                    if (val === null || val === undefined) return '<span class="no-data">N/A</span>';
                    if (typeof val === 'number') {{
                        if (val % 1 === 0) return val.toString();
                        return val.toFixed(3);
                    }}
                    return val;
                }};

                const errorsHtml = errors.length > 0
                    ? errors.map(e => `<div class="error-cell">${{e}}</div>`).join('')
                    : '<span class="no-data">None</span>';

                const missingHtml = missing.length > 0
                    ? missing.map(m => `<div class="missing-cell">${{m}}</div>`).join('')
                    : '<span class="no-data">None</span>';

                return `
                    <tr>
                        <td>${{item.subject}}</td>
                        <td>${{item.tract}}</td>
                        <td class="metric-value">${{formatValue(metrics.initial_streamline_count)}}</td>
                        <td class="metric-value">${{formatValue(metrics.cci_after_filter_count)}}</td>
                        <td class="metric-value">${{formatValue(metrics.cci_mean)}}</td>
                        <td class="metric-value">${{formatValue(metrics.cci_median)}}</td>
                        <td class="metric-value">${{formatValue(metrics.shape_similarity_score)}}</td>
                        <td>${{errorsHtml}}</td>
                        <td>${{missingHtml}}</td>
                    </tr>
                `;
            }}).join('');
        }}

        function switchTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});

            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            // Activate the clicked button
            document.querySelector(`[data-tab="${{tabName}}"]`).classList.add('active');
        }}

        // Add click handlers to tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {{
            button.addEventListener('click', function() {{
                const tabName = this.getAttribute('data-tab');
                switchTab(tabName);
            }});
        }});

        // Initialize
        renderSummaryTable();
        filterData();
    </script>
</body>
</html>"""

    # Write HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
