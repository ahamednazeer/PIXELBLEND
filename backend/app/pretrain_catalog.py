from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings


@dataclass(frozen=True)
class PretrainDataset:
    dataset_id: str
    name: str
    role: str
    source_url: str
    mirror_url: str
    license_note: str
    recommended_split: str
    recommended_sample_count: int
    recommended_resolution: int
    why_selected: str


@dataclass(frozen=True)
class PretrainedModelOption:
    model_id: str
    name: str
    source_url: str
    artifact_hint: str
    format_note: str
    why_selected: str


def get_recommended_datasets() -> list[PretrainDataset]:
    return [
        PretrainDataset(
            dataset_id="coco2017",
            name="COCO 2017",
            role="content",
            source_url="https://cocodataset.org/#download",
            mirror_url="https://huggingface.co/datasets/phiyodr/coco2017",
            license_note="COCO terms for dataset use apply.",
            recommended_split="train",
            recommended_sample_count=60000,
            recommended_resolution=512,
            why_selected=(
                "Strong object/layout diversity for preserving scene structure in style transfer content encoding."
            ),
        ),
        PretrainDataset(
            dataset_id="wikiart",
            name="WikiArt",
            role="style",
            source_url="https://huggingface.co/datasets/huggan/wikiart",
            mirror_url="https://www.wikiart.org/",
            license_note="Dataset card notes non-commercial usage restrictions for artwork images.",
            recommended_split="train",
            recommended_sample_count=80000,
            recommended_resolution=512,
            why_selected=(
                "Large artist/style diversity improves texture, color palette, and brush-stroke transfer quality."
            ),
        ),
    ]


def get_recommended_models() -> list[PretrainedModelOption]:
    return [
        PretrainedModelOption(
            model_id="adain",
            name="Adaptive Instance Normalization (AdaIN)",
            source_url="https://github.com/naoto0804/pytorch-AdaIN",
            artifact_hint="decoder/vgg weights released in project releases and checkpoints",
            format_note="Convert/export to TorchScript .pth for best backend compatibility.",
            why_selected=(
                "Supports arbitrary style transfer and style interpolation, which maps cleanly to PixelBlend dual-style blending."
            ),
        ),
        PretrainedModelOption(
            model_id="magenta_arbitrary_style",
            name="Magenta Arbitrary Image Stylization",
            source_url="https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization",
            artifact_hint="TensorFlow Hub SavedModel",
            format_note="Not a native .pth model; use only if you decide to support a TensorFlow runtime path.",
            why_selected=(
                "Widely used baseline with fast inference and reliable perceptual quality for production prototypes."
            ),
        ),
    ]


def build_pretrain_catalog_payload() -> dict[str, object]:
    datasets = [asdict(item) for item in get_recommended_datasets()]
    models = [asdict(item) for item in get_recommended_models()]
    return {
        "as_of_date": "2026-02-18",
        "recommended_datasets": datasets,
        "recommended_models": models,
    }


def local_pretrain_status(settings: Settings) -> dict[str, object]:
    content_count = _count_images(settings.pretrain_content_dir)
    style_count = _count_images(settings.pretrain_style_dir)

    return {
        "content_dir": str(settings.pretrain_content_dir),
        "style_dir": str(settings.pretrain_style_dir),
        "manifest_path": str(settings.pretrain_manifest_path),
        "content_images": content_count,
        "style_images": style_count,
        "manifest_exists": settings.pretrain_manifest_path.exists(),
        "ready_for_training": content_count > 0 and style_count > 0,
    }


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0

    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
