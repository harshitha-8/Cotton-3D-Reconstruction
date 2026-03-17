from __future__ import annotations

import argparse
from pathlib import Path

from src.reconstruction import ReconstructionConfig, reconstruct_image_to_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local 3D reconstruction on one image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum number of 3D points.")
    parser.add_argument(
        "--depth-strategy",
        choices=["auto", "midas", "heuristic"],
        default="heuristic",
        help="Depth estimation backend.",
    )
    args = parser.parse_args()

    result = reconstruct_image_to_assets(
        image_path=args.image,
        output_root=Path("outputs"),
        config=ReconstructionConfig(
            max_points=args.max_points,
            depth_strategy=args.depth_strategy,
        ),
    )

    print("Reconstruction complete")
    print(f"Input: {args.image}")
    print(f"Depth strategy: {result.depth_strategy_used}")
    print(f"Point cloud: {result.point_cloud_file}")
    print(f"Mesh: {result.mesh_file}")
    print(f"Depth array: {result.depth_npy_file}")
    print(f"Output folder: {result.output_dir}")


if __name__ == "__main__":
    main()
