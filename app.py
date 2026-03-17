from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional

import gradio as gr

from src.data import DATASET_ROOT, list_dataset_images, resolve_dataset_image
from src.reconstruction import ReconstructionConfig, reconstruct_image_to_assets


OUTPUT_ROOT = Path("outputs")
DEFAULT_CONFIG = ReconstructionConfig()


def find_free_port(start: int = 8860, end: int = 8999) -> int:
    requested = os.getenv("COTTON3D_PORT")
    if requested:
        return int(requested)

    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start} and {end}.")


def build_dataset_choices(phase: str) -> list[str]:
    images = list_dataset_images(phase=phase)
    return [item.display_name for item in images]


def update_image_choices(phase: str) -> tuple[gr.Dropdown, str]:
    choices = build_dataset_choices(phase)
    value = choices[0] if choices else None
    message = f"Loaded {len(choices)} images from {phase}."
    return gr.Dropdown(choices=choices, value=value), message


def run_reconstruction(
    uploaded_image,
    phase: str,
    dataset_image_name: Optional[str],
    max_points: int,
    depth_strategy: str,
):
    if uploaded_image is None and not dataset_image_name:
        raise gr.Error("Upload an image or select one from the dataset.")

    if uploaded_image is not None:
        image_path = Path(uploaded_image)
        source_label = f"Uploaded image: {image_path.name}"
    else:
        image_path = resolve_dataset_image(phase, dataset_image_name)
        source_label = f"Dataset image: {dataset_image_name}"

    config = ReconstructionConfig(
        max_points=max_points,
        depth_strategy=depth_strategy,
    )
    result = reconstruct_image_to_assets(
        image_path=image_path,
        output_root=OUTPUT_ROOT,
        config=config,
    )
    summary = "\n".join(
        [
            source_label,
            f"Depth strategy: {result.depth_strategy_used}",
            f"Point count: {result.num_points}",
            f"Output folder: {result.output_dir}",
        ]
    )
    return (
        result.preview_image,
        result.depth_preview,
        result.figure,
        result.point_cloud_file,
        result.mesh_file,
        result.depth_npy_file,
        summary,
    )


def create_app() -> gr.Blocks:
    initial_phase = "pre-deflation"
    initial_choices = build_dataset_choices(initial_phase)
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Blocks(title="Cotton 3D Reconstruction") as demo:
        gr.Markdown(
            """
            # Cotton 3D Reconstruction Demo
            Academic prototype for interactive 3D reconstruction from UAV imagery.

            This starter app supports:
            - pre-deflation and post-deflation cotton image browsing from `/Volumes/T9/ICML`
            - image upload for ad hoc experiments
            - monocular-depth-based 3D reconstruction as a first building block
            - Plotly visualization and exportable `.ply`, `.obj`, and `.npy` artifacts
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                phase = gr.Radio(
                    choices=["pre-deflation", "post-deflation"],
                    value=initial_phase,
                    label="Dataset split",
                )
                dataset_image = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_value,
                    label="Dataset image",
                    interactive=True,
                )
                dataset_status = gr.Textbox(
                    value=f"Dataset root: {DATASET_ROOT}",
                    label="Dataset status",
                    interactive=False,
                )
                uploaded_image = gr.Image(
                    type="filepath",
                    label="Upload UAV image",
                )
                max_points = gr.Slider(
                    minimum=2000,
                    maximum=40000,
                    value=12000,
                    step=1000,
                    label="Maximum 3D points",
                )
                depth_strategy = gr.Dropdown(
                    choices=["auto", "midas", "heuristic"],
                    value="auto",
                    label="Depth estimation strategy",
                )
                run_button = gr.Button("Reconstruct in 3D", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Preview"):
                    preview_image = gr.Image(label="Input preview")
                    depth_preview = gr.Image(label="Estimated depth map")
                with gr.Tab("3D View"):
                    plot = gr.Plot(label="Interactive 3D reconstruction")
                with gr.Tab("Exports"):
                    point_cloud_file = gr.File(label="Point cloud (.ply)")
                    mesh_file = gr.File(label="Mesh (.obj)")
                    depth_npy_file = gr.File(label="Depth array (.npy)")
                summary = gr.Textbox(label="Run summary", lines=5, interactive=False)

        phase.change(
            fn=update_image_choices,
            inputs=[phase],
            outputs=[dataset_image, dataset_status],
        )
        run_button.click(
            fn=run_reconstruction,
            inputs=[uploaded_image, phase, dataset_image, max_points, depth_strategy],
            outputs=[
                preview_image,
                depth_preview,
                plot,
                point_cloud_file,
                mesh_file,
                depth_npy_file,
                summary,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Cotton 3D app starting at {url}")
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
