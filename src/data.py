from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASET_ROOT = Path("/Volumes/T9/ICML")

PHASE_DIRECTORIES = {
    "pre-deflation": [
        DATASET_ROOT / "Part_one_pre_def_rgb",
        DATASET_ROOT / "part 2_pre_def_rgb",
    ],
    "post-deflation": [
        DATASET_ROOT / "205_Post_Def_rgb",
        DATASET_ROOT / "Post_def_rgb_part1",
        DATASET_ROOT / "part3_post_def_rgb",
        DATASET_ROOT / "part4_post_def_rgb",
    ],
}

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetImage:
    phase: str
    folder: str
    path: Path

    @property
    def display_name(self) -> str:
        return f"{self.folder} :: {self.path.name}"


def _is_valid_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_SUFFIXES and not path.name.startswith("._")


def list_dataset_images(phase: str) -> list[DatasetImage]:
    images: list[DatasetImage] = []
    for folder in PHASE_DIRECTORIES.get(phase, []):
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if _is_valid_image(path):
                images.append(DatasetImage(phase=phase, folder=folder.name, path=path))
    return images


def resolve_dataset_image(phase: str, display_name: str | None) -> Path:
    if not display_name:
        raise ValueError("No dataset image was selected.")
    for item in list_dataset_images(phase):
        if item.display_name == display_name:
            return item.path
    raise FileNotFoundError(f"Unable to find dataset image '{display_name}' in phase '{phase}'.")
