from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import Settings, get_settings
from .model_runner import ModelLoadError, PixelBlendModelRunner
from .pretrain_catalog import build_pretrain_catalog_payload, local_pretrain_status
from .pretrain_downloader import get_download_runtime_state

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

settings = get_settings()
model_runner = PixelBlendModelRunner(settings)

app = FastAPI(title="PixelBlend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    _ensure_directories(settings)
    try:
        model_runner.load()
    except ModelLoadError as exc:
        raise RuntimeError(f"Failed to initialize pretrained model: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": model_runner.status.loaded,
        "mode": model_runner.status.mode,
        "model_path": model_runner.status.model_path,
        "message": model_runner.status.message,
    }


@app.post("/api/generate")
async def generate(
    content_image: UploadFile = File(...),
    style_image_1: UploadFile = File(...),
    style_image_2: UploadFile = File(...),
    style_1_weight: float = Form(default=0.5),
    high_quality: bool = Form(default=False),
    detail_strength: float = Form(default=0.5),
    style_intensity: float = Form(default=0.5),
) -> JSONResponse:
    if not 0.0 <= style_1_weight <= 1.0:
        raise HTTPException(status_code=422, detail="style_1_weight must be between 0.0 and 1.0")
    if not 0.0 <= detail_strength <= 1.0:
        raise HTTPException(status_code=422, detail="detail_strength must be between 0.0 and 1.0")
    if not 0.0 <= style_intensity <= 1.0:
        raise HTTPException(status_code=422, detail="style_intensity must be between 0.0 and 1.0")

    content_pil, content_ext, content_bytes = await _load_upload(content_image, settings)
    style_1_pil, style_1_ext, style_1_bytes = await _load_upload(style_image_1, settings)
    style_2_pil, style_2_ext, style_2_bytes = await _load_upload(style_image_2, settings)

    job_id = uuid.uuid4().hex
    job_upload_dir = settings.uploads_dir / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    (job_upload_dir / f"content{content_ext}").write_bytes(content_bytes)
    (job_upload_dir / f"style1{style_1_ext}").write_bytes(style_1_bytes)
    (job_upload_dir / f"style2{style_2_ext}").write_bytes(style_2_bytes)

    stylized = model_runner.stylize(
        content_pil,
        style_1_pil,
        style_2_pil,
        style_1_weight,
        high_quality=high_quality,
        detail_strength=detail_strength,
        style_intensity=style_intensity,
    )

    output_name = f"stylized_{job_id}.jpg"
    output_path = settings.outputs_dir / output_name
    stylized.save(output_path, format="JPEG", quality=settings.output_quality)

    return JSONResponse(
        {
            "job_id": job_id,
            "output_path": f"/outputs/{output_name}",
            "download_path": f"/outputs/{output_name}",
            "model_mode": model_runner.status.mode,
            "model_loaded": model_runner.status.loaded,
        }
    )


@app.get("/api/model-status")
def model_status() -> dict[str, str | bool]:
    return {
        "model_loaded": model_runner.status.loaded,
        "mode": model_runner.status.mode,
        "model_path": model_runner.status.model_path,
        "message": model_runner.status.message,
    }


@app.get("/api/pretrain/catalog")
def pretrain_catalog() -> dict[str, object]:
    return build_pretrain_catalog_payload()


@app.get("/api/pretrain/status")
def pretrain_status() -> dict[str, object]:
    status = local_pretrain_status(settings)
    status["auto_download_enabled"] = settings.auto_download_pretrain
    status["target_content_images"] = settings.auto_download_content_count
    status["target_style_images"] = settings.auto_download_style_count
    status["download_runtime"] = get_download_runtime_state()
    return status


def _ensure_directories(app_settings: Settings) -> None:
    app_settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    app_settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    app_settings.pretrain_content_dir.mkdir(parents=True, exist_ok=True)
    app_settings.pretrain_style_dir.mkdir(parents=True, exist_ok=True)


def _validate_type(upload: UploadFile) -> None:
    if upload.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{upload.content_type}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}."
            ),
        )


async def _load_upload(upload: UploadFile, app_settings: Settings) -> tuple[Image.Image, str, bytes]:
    _validate_type(upload)

    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail=f"{upload.filename or 'image'} is empty")

    max_bytes = app_settings.max_upload_size_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename or 'image'} exceeds {app_settings.max_upload_size_mb}MB limit",
        )

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}") from exc

    ext = _guess_extension(upload.filename, upload.content_type)
    return image, ext, payload


def _guess_extension(filename: str | None, content_type: str | None) -> str:
    if filename and "." in filename:
        suffix = Path(filename).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return ".jpg" if suffix == ".jpeg" else suffix

    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    return ".jpg"


app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")

if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
