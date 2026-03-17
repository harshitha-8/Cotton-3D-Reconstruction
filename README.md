# Cotton 3D Reconstruction

This repository contains a first academic prototype for interactive cotton-field 3D reconstruction from UAV imagery. The interface is built with Gradio and Plotly, and it is configured to browse the local image dataset stored at `/Volumes/T9/ICML`.

## What This Starter App Does

- loads pre-deflation and post-deflation image folders from the ICML dataset location
- accepts either a dataset image selection or a newly uploaded UAV image
- estimates monocular depth
- converts the depth map into a colored 3D point cloud
- exports reconstruction artifacts for later analysis

This is a basic building block, not a final research system. It is designed so we can later add:

- contour-specific measurements
- pre-vs-post comparison views
- segmentation-guided reconstruction
- improved surface fitting and quantitative evaluation

## Dataset Assumption

The app expects the following folders to exist:

- `/Volumes/T9/ICML/Part_one_pre_def_rgb`
- `/Volumes/T9/ICML/part 2_pre_def_rgb`
- `/Volumes/T9/ICML/205_Post_Def_rgb`
- `/Volumes/T9/ICML/Post_def_rgb_part1`
- `/Volumes/T9/ICML/part3_post_def_rgb`
- `/Volumes/T9/ICML/part4_post_def_rgb`

Files beginning with `._` are ignored automatically.

## Reconstruction Design

The default reconstruction route is:

1. load the selected image
2. estimate depth using MiDaS when the model is available
3. fall back to a lightweight heuristic depth prior when model weights are unavailable
4. convert the depth map to a colored point cloud and a simple surface mesh
5. render the result in Plotly inside Gradio

The Meta demo you shared is useful as product inspiration, but this project does not depend on it directly because that page is a browser demo rather than a stable public API backend.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional depth-model extras:

```bash
pip install -r requirements-ml.txt
```

## Run

```bash
python3 app.py
```

Then open:

- [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Tests

Run the lightweight smoke tests with:

```bash
pytest
```

## Outputs

Each reconstruction run writes a timestamped folder under `outputs/` containing:

- `input_preview.png`
- `depth_preview.png`
- `depth.npy`
- `point_cloud.ply`
- `surface_mesh.obj`

## GitHub Connection

To connect this local project to your GitHub repository:

```bash
git init
git branch -M main
git remote add origin https://github.com/harshitha-8/Cotton-3D-Reconstruction
git add .
git commit -m "Initial Gradio 3D reconstruction prototype"
git push -u origin main
```

If you want, the next step can be a stronger research-grade pipeline using segmentation, multi-view geometry, contour extraction, and pre/post-difflation comparison analytics.

## Best Backend For UAV 3D

If the goal is precise UAV reconstruction, the best next backend is not a single-image object demo. A stronger choice is a photogrammetry pipeline such as WebODM / OpenDroneMap using multiple overlapping drone images from the same plot. This project currently supports a single-image 3D approximation for interaction, but research-grade geometry should come from multi-view reconstruction.
