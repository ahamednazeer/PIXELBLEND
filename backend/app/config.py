from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIXELBLEND_",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    model_path: Path = Path(__file__).resolve().parents[2] / "model" / "pixelblend_pretrained_model.pth"
    uploads_dir: Path = Path(__file__).resolve().parents[2] / "uploads"
    outputs_dir: Path = Path(__file__).resolve().parents[2] / "outputs"
    frontend_dir: Path = Path(__file__).resolve().parents[2] / "frontend"
    data_root_dir: Path = Path(__file__).resolve().parents[2] / "data"
    pretrain_content_dir: Path = Path(__file__).resolve().parents[2] / "data" / "pretrain" / "content"
    pretrain_style_dir: Path = Path(__file__).resolve().parents[2] / "data" / "pretrain" / "style"
    pretrain_manifest_path: Path = Path(__file__).resolve().parents[2] / "data" / "pretrain" / "manifest.json"

    max_upload_size_mb: int = Field(default=15, ge=1, le=100)
    output_image_size: int = Field(default=768, ge=128, le=2048)
    output_quality: int = Field(default=95, ge=60, le=100)

    model_device: str = "cpu"
    model_threads: int = Field(default=4, ge=1, le=64)
    allow_fallback: bool = True

    auto_download_pretrain: bool = True
    auto_download_content_count: int = Field(default=5000, ge=1, le=200000)
    auto_download_style_count: int = Field(default=5000, ge=1, le=200000)
    auto_download_max_edge: int = Field(default=512, ge=128, le=2048)
    auto_download_seed: int = Field(default=42, ge=0, le=1_000_000)
    auto_download_streaming: bool = True
    auto_download_reset: bool = False

    @model_validator(mode="after")
    def normalize_paths(self) -> "Settings":
        root = Path(self.project_root).resolve()
        self.project_root = root

        self.model_path = _normalize_path(
            self.model_path,
            default=root / "model" / "pixelblend_pretrained_model.pth",
            project_root=root,
        )
        self.uploads_dir = _normalize_path(self.uploads_dir, default=root / "uploads", project_root=root)
        self.outputs_dir = _normalize_path(self.outputs_dir, default=root / "outputs", project_root=root)
        self.frontend_dir = _normalize_path(self.frontend_dir, default=root / "frontend", project_root=root)
        self.data_root_dir = _normalize_path(self.data_root_dir, default=root / "data", project_root=root)
        self.pretrain_content_dir = _normalize_path(
            self.pretrain_content_dir,
            default=root / "data" / "pretrain" / "content",
            project_root=root,
        )
        self.pretrain_style_dir = _normalize_path(
            self.pretrain_style_dir,
            default=root / "data" / "pretrain" / "style",
            project_root=root,
        )
        self.pretrain_manifest_path = _normalize_path(
            self.pretrain_manifest_path,
            default=root / "data" / "pretrain" / "manifest.json",
            project_root=root,
        )
        return self


def _normalize_path(raw_path: Path, default: Path, project_root: Path) -> Path:
    path = Path(raw_path)

    if not path.is_absolute():
        return (project_root / path).resolve()

    if str(path).startswith(("/data/", "/model/")):
        return (project_root / path.relative_to("/")).resolve()

    return path.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
