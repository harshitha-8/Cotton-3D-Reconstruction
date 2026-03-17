from __future__ import annotations

import html
import os
import socket
from pathlib import Path
from typing import Optional

import gradio as gr

from src.data import DATASET_ROOT, list_dataset_images, resolve_dataset_image
from src.insights import generate_research_note
from src.reconstruction import ReconstructionConfig, reconstruct_image_to_assets


OUTPUT_ROOT = Path("outputs")
DEFAULT_PREFERRED_PORT = 8907
APP_CSS = """
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(58, 92, 74, 0.12), transparent 28%),
    linear-gradient(180deg, #f5f1e8 0%, #ece7dc 100%);
  color: #1d2822;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container {
  max-width: 1580px !important;
}

.hero-panel {
  background: linear-gradient(135deg, rgba(34, 73, 59, 0.96), rgba(59, 93, 74, 0.92));
  color: #f7f4ed;
  padding: 24px 28px;
  border-radius: 22px;
  box-shadow: 0 24px 60px rgba(20, 35, 28, 0.18);
  margin-bottom: 18px;
}

.hero-title {
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 10px;
}

.hero-subtitle {
  max-width: 980px;
  line-height: 1.65;
  color: rgba(247, 244, 237, 0.9);
  margin-bottom: 18px;
}

.hero-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.hero-card {
  background: rgba(247, 244, 237, 0.11);
  border: 1px solid rgba(247, 244, 237, 0.18);
  border-radius: 16px;
  padding: 14px 16px;
}

.hero-card strong {
  display: block;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(247, 244, 237, 0.72);
  margin-bottom: 6px;
}

.hero-card span {
  font-size: 1rem;
  line-height: 1.45;
}

.section-note {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(48, 71, 60, 0.14);
  border-radius: 18px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 12px 28px rgba(40, 51, 44, 0.08);
}

.panel-shell {
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid rgba(48, 71, 60, 0.12);
  border-radius: 22px;
  padding: 10px;
  box-shadow: 0 12px 32px rgba(34, 50, 42, 0.08);
}

.results-shell {
  background: rgba(250, 248, 242, 0.88);
  border-radius: 22px;
  border: 1px solid rgba(48, 71, 60, 0.12);
  padding: 10px;
  box-shadow: 0 12px 32px rgba(34, 50, 42, 0.08);
}

button.primary {
  background: linear-gradient(135deg, #2e5a45, #4f7b61) !important;
  border: none !important;
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
    display_summary = f"{summary}\n{result.cotton_metrics}"
    return (
        result.preview_image,
        result.depth_preview,
        result.figure,
        result.cotton_overlay,
        result.cotton_figure,
        result.object_preview,
        result.object_model_file,
        result.point_cloud_file,
        result.mesh_file,
        result.depth_npy_file,
        display_summary,
        summary,
        result.cotton_metrics,
        "",
    )


def create_app() -> gr.Blocks:
    initial_phase = "pre-deflation"
    initial_choices = build_dataset_choices(initial_phase)
    initial_value = initial_choices[0] if initial_choices else None

    with gr.Blocks(title="Cotton 3D Reconstruction", css=APP_CSS) as demo:
        gr.HTML(
            f"""
            <div class="hero-panel">
              <div class="hero-title">Cotton 3D Reconstruction Workspace</div>
              <div class="hero-subtitle">
                Academic interface for UAV-based cotton field analysis using pre-deflation and post-deflation imagery.
                This environment supports scene-level 3D reconstruction, cotton-focused object-style analysis, and exportable research artifacts.
              </div>
              <div class="hero-grid">
                <div class="hero-card"><strong>Dataset Root</strong><span>{html.escape(str(DATASET_ROOT))}</span></div>
                <div class="hero-card"><strong>Pre-Deflation Images</strong><span>{len(list_dataset_images("pre-deflation"))} indexed UAV frames</span></div>
                <div class="hero-card"><strong>Post-Deflation Images</strong><span>{len(list_dataset_images("post-deflation"))} indexed UAV frames</span></div>
                <div class="hero-card"><strong>Research Mode</strong><span>3D scene view, object-style cotton studio, and AI-assisted interpretation</span></div>
              </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="section-note">
              Upload a local UAV image or select a dataset frame. The system estimates depth, reconstructs a 3D point cloud,
              generates a surface mesh, isolates likely cotton structures, and adds a dedicated object-style cotton studio similar to the interaction shown in your reference video.
            </div>
            """
        )

        summary_store = gr.State("")
        cotton_metrics_store = gr.State("")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel-shell"]):
                gr.Markdown("### Reconstruction Inputs")
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
                gr.Markdown(
                    """
                    **Interpretation**

                    `auto`: tries MiDaS first if optional ML weights are installed.

                    `heuristic`: lightweight fallback that works immediately for local testing.
                    """
                )
                gr.Markdown("### AI Extension")
                ai_prompt = gr.Textbox(
                    label="Research-note prompt",
                    value="Summarize the cotton depth structure and explain how it could support boll-depth analysis for an academic project.",
                    lines=3,
                )
                ai_model = gr.Textbox(
                    label="OpenAI model",
                    value="gpt-5",
                )
                ai_button = gr.Button("Generate AI Research Note")

            with gr.Column(scale=2, elem_classes=["results-shell"]):
                gr.Markdown("### Reconstruction Outputs")
                with gr.Tab("Input And Depth"):
                    preview_image = gr.Image(label="Input frame")
                    depth_preview = gr.Image(label="Estimated depth field")
                with gr.Tab("Interactive 3D"):
                    plot = gr.Plot(label="3D reconstruction viewer")
                with gr.Tab("Cotton Focus 3D"):
                    cotton_overlay = gr.Image(label="Cotton-candidate overlay")
                    cotton_plot = gr.Plot(label="Cotton-focused 3D viewer")
                with gr.Tab("Cotton Object Studio"):
                    gr.Markdown(
                        """
                        This mode keeps the existing reconstruction and adds a video-inspired object view:
                        the likely cotton region is isolated on the left and exported as a rotatable 3D object on the right.
                        """
                    )
                    with gr.Row():
                        object_preview = gr.Image(label="Isolated cotton object preview")
                        object_model = gr.Model3D(
                            label="Rotatable cotton object model",
                            clear_color=[0.07, 0.07, 0.07, 1.0],
                        )
                with gr.Tab("Exports"):
                    point_cloud_file = gr.File(label="Point cloud export (.ply)")
                    mesh_file = gr.File(label="Surface mesh export (.obj)")
                    depth_npy_file = gr.File(label="Depth tensor export (.npy)")
                summary = gr.Textbox(label="Reconstruction summary", lines=8, interactive=False)
                ai_note = gr.Textbox(label="AI research note", lines=12, interactive=False)

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
                object_model,
                point_cloud_file,
                mesh_file,
                depth_npy_file,
                summary,
                summary_store,
                cotton_metrics_store,
                ai_note,
            ],
        )
        ai_button.click(
            fn=generate_research_note,
            inputs=[summary_store, cotton_metrics_store, ai_prompt, ai_model],
            outputs=[ai_note],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Cotton 3D app starting at {url}")
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
