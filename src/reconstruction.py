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
    point_cloud_file: str
    mesh_file: str
    depth_npy_file: str
    output_dir: str
    num_points: int
    depth_strategy_used: str


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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{image_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / "input_preview.png"
    depth_preview_path = output_dir / "depth_preview.png"
    point_cloud_path = output_dir / "point_cloud.ply"
    mesh_path = output_dir / "surface_mesh.obj"
    depth_npy_path = output_dir / "depth.npy"

    image.save(preview_path)
    depth_preview = depth_to_image(depth)
    depth_preview.save(depth_preview_path)
    np.save(depth_npy_path, depth)

    save_point_cloud(point_cloud_path, points, colors)
    save_surface_mesh(mesh_path, rgb, depth, config.z_scale, config.xy_scale)

    return ReconstructionResult(
        preview_image=image,
        depth_preview=depth_preview,
        figure=figure,
        point_cloud_file=str(point_cloud_path),
        mesh_file=str(mesh_path),
        depth_npy_file=str(depth_npy_path),
        output_dir=str(output_dir),
        num_points=len(points),
        depth_strategy_used=depth_strategy_used,
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
