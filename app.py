from __future__ import annotations

import html
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

import gradio as gr

from src.data import DATASET_ROOT, list_dataset_images, resolve_dataset_image
from src.reconstruction import ReconstructionConfig, reconstruct_image_to_assets


OUTPUT_ROOT = Path("outputs")
DEFAULT_PREFERRED_PORT = 8907
APP_CSS = """
:root {
  --bg-0: #06111f;
  --bg-1: #0b1830;
  --panel: rgba(11, 24, 48, 0.78);
  --panel-2: rgba(14, 28, 54, 0.9);
  --line: rgba(115, 161, 255, 0.18);
  --text: #e7eefc;
  --muted: #9aabc9;
  --accent: #53d1ff;
  --accent-2: #7b7cff;
  --accent-3: #ff8f5c;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(83, 209, 255, 0.12), transparent 26%),
    radial-gradient(circle at bottom right, rgba(123, 124, 255, 0.16), transparent 24%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
  color: var(--text);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container {
  max-width: 1620px !important;
}

.hero-panel {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(7, 18, 35, 0.94), rgba(13, 29, 56, 0.96));
  color: var(--text);
  padding: 28px 30px;
  border-radius: 28px;
  border: 1px solid rgba(110, 173, 255, 0.18);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
  margin-bottom: 18px;
}

.hero-panel::after {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(90deg, transparent 0%, rgba(83, 209, 255, 0.06) 48%, transparent 100%);
  pointer-events: none;
}

.hero-title {
  font-size: 2.3rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 10px;
}

.hero-build {
  display: inline-block;
  margin-left: 10px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(83, 209, 255, 0.08);
  border: 1px solid rgba(83, 209, 255, 0.22);
  color: #aee8ff;
  font-size: 0.82rem;
  vertical-align: middle;
}

.hero-subtitle {
  max-width: 1040px;
  line-height: 1.7;
  color: #b8c7e4;
  margin-bottom: 18px;
}

.hero-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.hero-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(115, 161, 255, 0.14);
  border-radius: 18px;
  padding: 14px 16px;
  backdrop-filter: blur(10px);
}

.hero-card strong {
  display: block;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #82b0ff;
  margin-bottom: 6px;
}

.hero-card span {
  font-size: 1rem;
  line-height: 1.5;
}

.section-note {
  background: rgba(12, 24, 46, 0.7);
  border: 1px solid rgba(115, 161, 255, 0.14);
  border-radius: 20px;
  padding: 16px 18px;
  margin-bottom: 16px;
  color: #bdd0f3;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
}

.panel-shell, .results-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 14px;
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(14px);
}

.studio-note {
  background: linear-gradient(135deg, rgba(83, 209, 255, 0.08), rgba(123, 124, 255, 0.08));
  border: 1px solid rgba(83, 209, 255, 0.16);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 12px;
  color: #d7e4ff;
}

button.primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: #06111f !important;
  border: none !important;
  box-shadow: 0 10px 24px rgba(83, 209, 255, 0.28);
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .tabitem {
  border-color: rgba(115, 161, 255, 0.14) !important;
}

.gradio-container label,
.gradio-container .prose,
.gradio-container .md,
.gradio-container .tabs,
.gradio-container .tab-nav button {
  color: var(--text) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .dropdown,
.gradio-container .input-container,
.gradio-container .scroll-hide {
  background: rgba(8, 18, 34, 0.82) !important;
  color: var(--text) !important;
}

@media (max-width: 1100px) {
  .hero-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
}
"""


def find_free_port(start: int = 8860, end: int = 8999) -> int:
    requested = os.getenv("COTTON3D_PORT")
    if requested:
        return int(requested)

    for preferred in [DEFAULT_PREFERRED_PORT]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass

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


def get_build_label() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except Exception:
        return "local"


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
    else:
        image_path = resolve_dataset_image(phase, dataset_image_name)

    config = ReconstructionConfig(
        max_points=max_points,
        depth_strategy=depth_strategy,
    )
    result = reconstruct_image_to_assets(
        image_path=image_path,
        output_root=OUTPUT_ROOT,
        config=config,
    )
    return (
        result.preview_image,
        result.depth_preview,
        result.figure,
        result.cotton_overlay,
        result.cotton_figure,
        result.object_preview,
        result.object_figure,
        result.point_cloud_file,
        result.mesh_file,
        result.depth_npy_file,
    )


def create_app() -> gr.Blocks:
    initial_phase = "pre-deflation"
    initial_choices = build_dataset_choices(initial_phase)
    initial_value = initial_choices[0] if initial_choices else None
    build_label = get_build_label()

    with gr.Blocks(title="Cotton 3D Reconstruction", css=APP_CSS) as demo:
        gr.HTML(
            f"""
            <div class="hero-panel">
              <div class="hero-title">Cotton 3D Reconstruction Interface <span class="hero-build">Build {html.escape(build_label)}</span></div>
              <div class="hero-subtitle">
                UAV cotton analytics workspace with scene-scale depth reconstruction, cotton isolation, and a dedicated object studio
                for rotatable 3D inspection. Built to read like a technical demo, not a classroom toy.
              </div>
              <div class="hero-grid">
                <div class="hero-card"><strong>Dataset Root</strong><span>{html.escape(str(DATASET_ROOT))}</span></div>
                <div class="hero-card"><strong>Pre-Deflation</strong><span>{len(list_dataset_images("pre-deflation"))} indexed frames</span></div>
                <div class="hero-card"><strong>Post-Deflation</strong><span>{len(list_dataset_images("post-deflation"))} indexed frames</span></div>
                <div class="hero-card"><strong>Studio Output</strong><span>Scene 3D, cotton-focused 3D, and isolated object viewer</span></div>
              </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="section-note">
              Upload a local UAV image or select a dataset frame. The system preserves the existing scene reconstruction,
              adds cotton isolation, and renders a separate object-style 3D studio view for closer inspection.
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel-shell"]):
                gr.Markdown("### Inputs")
                phase = gr.Radio(
                    choices=["pre-deflation", "post-deflation"],
                    value=initial_phase,
                    label="Acquisition phase",
                )
                dataset_image = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_value,
                    label="Indexed dataset frame",
                    interactive=True,
                )
                dataset_status = gr.Textbox(
                    value=f"Dataset root: {DATASET_ROOT}",
                    label="Dataset status",
                    interactive=False,
                )
                uploaded_image = gr.Image(
                    type="filepath",
                    label="Upload local UAV image",
                )
                max_points = gr.Slider(
                    minimum=2000,
                    maximum=40000,
                    value=12000,
                    step=1000,
                    label="Point cloud density",
                )
                depth_strategy = gr.Dropdown(
                    choices=["auto", "midas", "heuristic"],
                    value="auto",
                    label="Depth estimation backend",
                )
                run_button = gr.Button("Generate 3D Reconstruction", variant="primary")

            with gr.Column(scale=2, elem_classes=["results-shell"]):
                gr.Markdown("### Outputs")
                with gr.Tab("Input And Depth"):
                    preview_image = gr.Image(label="Input frame")
                    depth_preview = gr.Image(label="Estimated depth field")
                with gr.Tab("Interactive 3D"):
                    plot = gr.Plot(label="Scene-scale 3D reconstruction")
                with gr.Tab("Cotton Focus 3D"):
                    cotton_overlay = gr.Image(label="Cotton-candidate overlay")
                    cotton_plot = gr.Plot(label="Cotton-focused 3D viewer")
                with gr.Tab("Cotton Object Studio"):
                    gr.HTML(
                        """
                        <div class="studio-note">
                          Video-style object mode: isolated cotton preview on the left and a reliable in-app rotatable 3D object renderer on the right.
                        </div>
                        """
                    )
                    with gr.Row():
                        object_preview = gr.Image(label="Isolated cotton object preview")
                        object_plot = gr.Plot(label="Rotatable cotton object")
                with gr.Tab("Exports"):
                    point_cloud_file = gr.File(label="Point cloud export (.ply)")
                    mesh_file = gr.File(label="Surface mesh export (.obj)")
                    depth_npy_file = gr.File(label="Depth tensor export (.npy)")

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
                cotton_overlay,
                cotton_plot,
                object_preview,
                object_plot,
                point_cloud_file,
                mesh_file,
                depth_npy_file,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Cotton 3D app starting at {url}")
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
