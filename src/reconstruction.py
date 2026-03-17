from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh
from PIL import Image


@dataclass(frozen=True)
class ReconstructionConfig:
    max_points: int = 12000
    depth_strategy: str = "auto"
    resize_to: int = 384
    z_scale: float = 75.0
    xy_scale: float = 1.0


@dataclass(frozen=True)
class ReconstructionResult:
    preview_image: Image.Image
    depth_preview: Image.Image
    figure: go.Figure
    cotton_overlay: Image.Image
    cotton_figure: go.Figure
    object_preview: Image.Image
    object_figure: go.Figure
    object_model_file: str
    point_cloud_file: str
    mesh_file: str
    depth_npy_file: str
    output_dir: str
    num_points: int
    depth_strategy_used: str
    cotton_metrics: str


def reconstruct_image_to_assets(
    image_path: Path,
    output_root: Path,
    config: ReconstructionConfig,
) -> ReconstructionResult:
    image = Image.open(image_path).convert("RGB")
    image = _resize_preserving_aspect(image, config.resize_to)
    rgb = np.asarray(image).astype(np.float32) / 255.0

    depth, depth_strategy_used = estimate_depth(rgb, config.depth_strategy)
    points, colors = depth_to_point_cloud(rgb, depth, config.max_points, config.z_scale, config.xy_scale)
    figure = create_plotly_figure(points, colors)

    cotton_mask = detect_cotton_mask(rgb, depth)
    cotton_overlay = create_cotton_overlay(image, cotton_mask)
    cotton_figure, cotton_metrics = create_cotton_focus_figure(rgb, depth, cotton_mask, config.z_scale)
    object_figure = create_object_studio_figure(rgb, depth, cotton_mask, config.z_scale)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{image_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / "input_preview.png"
    depth_preview_path = output_dir / "depth_preview.png"
    point_cloud_path = output_dir / "point_cloud.ply"
    mesh_path = output_dir / "surface_mesh.obj"
    object_mesh_path = output_dir / "cotton_object.obj"
    depth_npy_path = output_dir / "depth.npy"
    object_preview_path = output_dir / "cotton_object_preview.png"

    image.save(preview_path)
    depth_preview = depth_to_image(depth)
    depth_preview.save(depth_preview_path)
    np.save(depth_npy_path, depth)
    object_preview = create_object_preview(image, cotton_mask)
    object_preview.save(object_preview_path)

    save_point_cloud(point_cloud_path, points, colors)
    save_surface_mesh(mesh_path, rgb, depth, config.z_scale, config.xy_scale)
    save_object_mesh(object_mesh_path, rgb, depth, cotton_mask, config.z_scale, config.xy_scale)

    return ReconstructionResult(
        preview_image=image,
        depth_preview=depth_preview,
        figure=figure,
        cotton_overlay=cotton_overlay,
        cotton_figure=cotton_figure,
        object_preview=object_preview,
        object_figure=object_figure,
        object_model_file=str(object_mesh_path),
        point_cloud_file=str(point_cloud_path),
        mesh_file=str(mesh_path),
        depth_npy_file=str(depth_npy_path),
        output_dir=str(output_dir),
        num_points=len(points),
        depth_strategy_used=depth_strategy_used,
        cotton_metrics=cotton_metrics,
    )


def estimate_depth(rgb: np.ndarray, strategy: str) -> tuple[np.ndarray, str]:
    if strategy in {"auto", "midas"}:
        try:
            depth = estimate_depth_with_midas(rgb)
            return depth, "midas"
        except Exception:
            if strategy == "midas":
                raise
    return estimate_depth_heuristic(rgb), "heuristic"


def estimate_depth_with_midas(rgb: np.ndarray) -> np.ndarray:
    import torch
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    model_id = "Intel/dpt-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = DPTFeatureExtractor.from_pretrained(model_id)
    model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    image = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = depth.detach().cpu().numpy()
    return normalize_depth(depth)


def estimate_depth_heuristic(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    gradient_energy = np.sqrt(gx**2 + gy**2)

    h, w = gray.shape
    y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
    vertical_prior = 1.0 - y_coords[:, None]

    depth = 0.65 * vertical_prior + 0.35 * (1.0 - normalize_depth(gradient_energy))
    return normalize_depth(depth)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    min_value = float(depth.min())
    max_value = float(depth.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - min_value) / (max_value - min_value)


def depth_to_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    max_points: int,
    z_scale: float,
    xy_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]

    x = (xx.astype(np.float32) - w / 2.0) * xy_scale
    y = ((h - yy).astype(np.float32) - h / 2.0) * xy_scale
    z = depth.astype(np.float32) * z_scale

    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    colors = rgb.reshape(-1, 3)

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[indices]
        colors = colors[indices]

    return points, colors


def create_plotly_figure(points: np.ndarray, colors: np.ndarray) -> go.Figure:
    color_strings = [
        f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for r, g, b in colors
    ]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2.0,
                    "color": color_strings,
                    "opacity": 0.9,
                },
            )
        ]
    )
    figure.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth",
            aspectmode="data",
        ),
        title="Monocular 3D reconstruction preview",
    )
    return figure


def create_cotton_focus_figure(
    rgb: np.ndarray,
    depth: np.ndarray,
    cotton_mask: np.ndarray,
    z_scale: float,
) -> tuple[go.Figure, str]:
    if not np.any(cotton_mask):
        empty = go.Figure()
        empty.update_layout(
            title="Cotton-focused 3D view unavailable",
            annotations=[
                dict(
                    text="No clear cotton candidates were isolated in this frame.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
            height=700,
        )
        return empty, "Cotton candidate coverage: 0.00%\nCotton objects detected: 0"

    rows, cols = np.where(cotton_mask)
    row_min, row_max = max(0, rows.min() - 6), min(depth.shape[0], rows.max() + 7)
    col_min, col_max = max(0, cols.min() - 6), min(depth.shape[1], cols.max() + 7)

    crop_depth = depth[row_min:row_max, col_min:col_max]
    crop_rgb = rgb[row_min:row_max, col_min:col_max]
    crop_mask = cotton_mask[row_min:row_max, col_min:col_max]

    surface_z = np.where(crop_mask, crop_depth * z_scale, np.nan)
    marker_rows, marker_cols = np.where(crop_mask)
    x_coords = marker_cols.astype(np.float32)
    y_coords = (crop_depth.shape[0] - marker_rows).astype(np.float32)
    z_coords = crop_depth[marker_rows, marker_cols].astype(np.float32) * z_scale
    marker_colors = crop_rgb[marker_rows, marker_cols]
    color_strings = [
        f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for r, g, b in marker_colors
    ]

    highlight_mask = crop_mask & (crop_rgb.mean(axis=2) > np.quantile(crop_rgb.mean(axis=2)[crop_mask], 0.75))
    hi_rows, hi_cols = np.where(highlight_mask)

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            z=surface_z,
            surfacecolor=surface_z,
            colorscale=[
                [0.0, "#7c5a46"],
                [0.35, "#a57b63"],
                [0.72, "#d1b39f"],
                [1.0, "#f7efe6"],
            ],
            opacity=0.92,
            showscale=False,
            name="Cotton surface",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker={"size": 3.0, "color": color_strings, "opacity": 0.7},
            name="Cotton structure",
        )
    )
    if len(hi_rows) > 0:
        figure.add_trace(
            go.Scatter3d(
                x=hi_cols.astype(np.float32),
                y=(crop_depth.shape[0] - hi_rows).astype(np.float32),
                z=crop_depth[hi_rows, hi_cols].astype(np.float32) * z_scale + 1.0,
                mode="markers",
                marker={"size": 4.8, "color": "#fff9ef", "opacity": 0.98},
                name="Bright cotton candidates",
            )
        )

    figure.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=46, b=0),
        title="Cotton-focused rotatable 3D reconstruction",
        scene=dict(
            xaxis_title="Image X",
            yaxis_title="Image Y",
            zaxis_title="Estimated depth",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.45, z=0.9)),
        ),
    )

    coverage = float(cotton_mask.mean() * 100.0)
    object_count, component_areas = connected_component_sizes(cotton_mask)
    cotton_depth = depth[cotton_mask]
    metrics = "\n".join(
        [
            f"Cotton candidate coverage: {coverage:.2f}%",
            f"Cotton objects detected: {object_count}",
            f"Mean cotton depth score: {float(cotton_depth.mean()):.3f}",
            f"Peak cotton depth score: {float(cotton_depth.max()):.3f}",
            f"Largest component size: {max(component_areas) if component_areas else 0} pixels",
            "Interaction: drag in the 3D view to rotate, zoom, and inspect cotton structure.",
        ]
    )
    return figure, metrics


def create_object_studio_figure(
    rgb: np.ndarray,
    depth: np.ndarray,
    cotton_mask: np.ndarray,
    z_scale: float,
) -> go.Figure:
    figure = go.Figure()
    if not np.any(cotton_mask):
        figure.update_layout(
            title="Cotton object studio unavailable",
            annotations=[
                dict(
                    text="No isolated cotton object was extracted from this frame.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="#e8eefc", size=16),
                )
            ],
            height=640,
            paper_bgcolor="#08111f",
            plot_bgcolor="#08111f",
        )
        return figure

    rows, cols = np.where(cotton_mask)
    row_min, row_max = max(0, rows.min() - 8), min(depth.shape[0], rows.max() + 9)
    col_min, col_max = max(0, cols.min() - 8), min(depth.shape[1], cols.max() + 9)

    crop_depth = depth[row_min:row_max, col_min:col_max]
    crop_rgb = rgb[row_min:row_max, col_min:col_max]
    crop_mask = cotton_mask[row_min:row_max, col_min:col_max]

    h, w = crop_depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx.astype(np.float32) - w / 2.0
    y = (h - yy).astype(np.float32) - h / 2.0
    z = crop_depth.astype(np.float32) * z_scale
    surface_z = np.where(crop_mask, z, np.nan)

    figure.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=surface_z,
            surfacecolor=np.where(crop_mask, crop_depth, np.nan),
            colorscale=[
                [0.0, "#6b4f3f"],
                [0.25, "#98715b"],
                [0.6, "#c8aa8f"],
                [0.86, "#f2eadf"],
                [1.0, "#ffffff"],
            ],
            showscale=False,
            opacity=0.98,
            lighting=dict(ambient=0.7, diffuse=0.95, specular=0.25, roughness=0.85),
            lightposition=dict(x=80, y=120, z=240),
        )
    )

    bright_mask = crop_mask & (crop_rgb.mean(axis=2) >= np.quantile(crop_rgb.mean(axis=2)[crop_mask], 0.84))
    if np.any(bright_mask):
        b_rows, b_cols = np.where(bright_mask)
        figure.add_trace(
            go.Scatter3d(
                x=b_cols.astype(np.float32) - w / 2.0,
                y=(h - b_rows).astype(np.float32) - h / 2.0,
                z=crop_depth[b_rows, b_cols].astype(np.float32) * z_scale + 1.2,
                mode="markers",
                marker={"size": 4.8, "color": "#fff8f0", "opacity": 0.95},
                name="Cotton bolls",
            )
        )

    figure.update_layout(
        title="Isolated cotton object studio",
        height=640,
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        scene=dict(
            bgcolor="#08111f",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.25, z=0.9)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        legend=dict(
            bgcolor="rgba(8, 17, 31, 0.65)",
            font=dict(color="#dfe9ff"),
        ),
        font=dict(color="#dfe9ff"),
    )
    return figure


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    color_bytes = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
    cloud = trimesh.points.PointCloud(vertices=points, colors=color_bytes)
    cloud.export(path)


def save_surface_mesh(path: Path, rgb: np.ndarray, depth: np.ndarray, z_scale: float, xy_scale: float) -> None:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    vertices = np.column_stack(
        [
            (xx.astype(np.float32) - w / 2.0) * xy_scale,
            ((h - yy).astype(np.float32) - h / 2.0) * xy_scale,
            depth.astype(np.float32) * z_scale,
        ]
    ).reshape(-1, 3)

    faces = []
    for row in range(h - 1):
        for col in range(w - 1):
            top_left = row * w + col
            top_right = top_left + 1
            bottom_left = top_left + w
            bottom_right = bottom_left + 1
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])

    vertex_colors = (rgb.reshape(-1, 3).clip(0.0, 1.0) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), vertex_colors=vertex_colors, process=False)
    mesh.export(path)


def depth_to_image(depth: np.ndarray) -> Image.Image:
    depth_uint8 = (normalize_depth(depth) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(depth_uint8, mode="L")


def _resize_preserving_aspect(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    scale = long_edge / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def detect_cotton_mask(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(rgb)
    value = hsv[..., 2]
    saturation = hsv[..., 1]

    brightness_threshold = float(np.quantile(value, 0.78))
    low_saturation_threshold = float(np.quantile(saturation, 0.55))
    depth_threshold = float(np.quantile(depth, 0.68))

    bright_whites = (value >= brightness_threshold) & (saturation <= low_saturation_threshold)
    elevated_texture = (value >= np.quantile(value, 0.63)) & (depth >= depth_threshold)
    mask = bright_whites | elevated_texture
    mask = smooth_binary_mask(mask, rounds=2)
    return keep_large_components(mask, min_size=max(24, int(mask.size * 0.00045)))


def create_cotton_overlay(image: Image.Image, cotton_mask: np.ndarray) -> Image.Image:
    base = np.asarray(image).astype(np.uint8)
    overlay = base.copy()
    tint = np.array([217, 84, 139], dtype=np.uint8)
    overlay[cotton_mask] = (
        0.55 * overlay[cotton_mask].astype(np.float32) + 0.45 * tint.astype(np.float32)
    ).astype(np.uint8)
    return Image.fromarray(overlay)


def create_object_preview(image: Image.Image, cotton_mask: np.ndarray) -> Image.Image:
    base = np.asarray(image).astype(np.uint8)
    rows, cols = np.where(cotton_mask)
    if len(rows) == 0:
        return image
    row_min, row_max = max(0, rows.min() - 12), min(base.shape[0], rows.max() + 13)
    col_min, col_max = max(0, cols.min() - 12), min(base.shape[1], cols.max() + 13)
    crop = base[row_min:row_max, col_min:col_max].copy()
    crop_mask = cotton_mask[row_min:row_max, col_min:col_max]
    background = np.full_like(crop, 18, dtype=np.uint8)
    preview = np.where(crop_mask[..., None], crop, background)
    border = dilate_mask(crop_mask, rounds=1) & ~crop_mask
    preview[border] = np.array([217, 84, 139], dtype=np.uint8)
    return Image.fromarray(preview)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc

    saturation = np.where(maxc == 0, 0, delta / np.clip(maxc, 1e-8, None))
    hue = np.zeros_like(maxc)

    nonzero = delta > 1e-8
    r_mask = nonzero & (maxc == r)
    g_mask = nonzero & (maxc == g)
    b_mask = nonzero & (maxc == b)

    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    hue /= 6.0

    return np.stack([hue, saturation, maxc], axis=2).astype(np.float32)


def smooth_binary_mask(mask: np.ndarray, rounds: int = 2) -> np.ndarray:
    mask = mask.astype(bool)
    for _ in range(rounds):
        neighbors = sum(
            np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        )
        mask = (mask & (neighbors >= 2)) | (neighbors >= 4)
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
    return mask


def dilate_mask(mask: np.ndarray, rounds: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    for _ in range(rounds):
        mask = np.logical_or.reduce(
            [
                np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
        )
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
    return mask


def keep_large_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)

    for row in range(h):
        for col in range(w):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            component: list[tuple[int, int]] = []
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if 0 <= next_row < h and 0 <= next_col < w and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            if len(component) >= min_size:
                for current_row, current_col in component:
                    kept[current_row, current_col] = True
    return kept


def connected_component_sizes(mask: np.ndarray) -> tuple[int, list[int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []

    for row in range(h):
        for col in range(w):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            size = 0
            while stack:
                current_row, current_col = stack.pop()
                size += 1
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if 0 <= next_row < h and 0 <= next_col < w and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            sizes.append(size)
    return len(sizes), sorted(sizes, reverse=True)


def save_object_mesh(path: Path, rgb: np.ndarray, depth: np.ndarray, cotton_mask: np.ndarray, z_scale: float, xy_scale: float) -> None:
    rows, cols = np.where(cotton_mask)
    if len(rows) == 0:
        save_surface_mesh(path, rgb, depth, z_scale, xy_scale)
        return

    row_min, row_max = max(0, rows.min() - 6), min(depth.shape[0], rows.max() + 7)
    col_min, col_max = max(0, cols.min() - 6), min(depth.shape[1], cols.max() + 7)

    crop_depth = depth[row_min:row_max, col_min:col_max]
    crop_rgb = rgb[row_min:row_max, col_min:col_max]
    crop_mask = cotton_mask[row_min:row_max, col_min:col_max]

    h, w = crop_depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    vertices = np.column_stack(
        [
            (xx.astype(np.float32) - w / 2.0) * xy_scale,
            ((h - yy).astype(np.float32) - h / 2.0) * xy_scale,
            crop_depth.astype(np.float32) * z_scale,
        ]
    ).reshape(-1, 3)
    vertex_colors = (crop_rgb.reshape(-1, 3).clip(0.0, 1.0) * 255).astype(np.uint8)

    faces = []
    for row in range(h - 1):
        for col in range(w - 1):
            quad_mask = [
                crop_mask[row, col],
                crop_mask[row + 1, col],
                crop_mask[row, col + 1],
                crop_mask[row + 1, col + 1],
            ]
            if sum(quad_mask) < 3:
                continue
            top_left = row * w + col
            top_right = top_left + 1
            bottom_left = top_left + w
            bottom_right = bottom_left + 1
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])

    if not faces:
        save_surface_mesh(path, crop_rgb, crop_depth, z_scale, xy_scale)
        return

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces),
        vertex_colors=vertex_colors,
        process=True,
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(path)
