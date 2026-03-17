from pathlib import Path

from src.data import list_dataset_images
from src.reconstruction import ReconstructionConfig, reconstruct_image_to_assets


def test_dataset_index_contains_pre_and_post_images():
    assert len(list_dataset_images("pre-deflation")) > 0
    assert len(list_dataset_images("post-deflation")) > 0


def test_heuristic_reconstruction_generates_expected_outputs(tmp_path: Path):
    sample = list_dataset_images("pre-deflation")[0].path
    result = reconstruct_image_to_assets(
        image_path=sample,
        output_root=tmp_path,
        config=ReconstructionConfig(depth_strategy="heuristic", max_points=3000),
    )

    assert result.depth_strategy_used == "heuristic"
    assert result.num_points == 3000
    assert Path(result.point_cloud_file).exists()
    assert Path(result.mesh_file).exists()
    assert Path(result.depth_npy_file).exists()
