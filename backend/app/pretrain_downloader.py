from __future__ import annotations

import io
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from .config import Settings
from .pretrain_catalog import local_pretrain_status

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_DATASET = "phiyodr/coco2017"
_DEFAULT_STYLE_DATASET = "huggan/wikiart"
_IMAGE_SUFFIX = ".jpg"
_DEFAULT_SCAN_FACTOR = 8
_DEFAULT_URL_TIMEOUT = 8
_MAX_HOST_FAILURES = 3
_host_failures: dict[str, int] = {}

_download_lock = threading.Lock()
_runtime_state: dict[str, Any] = {
    "started": False,
    "in_progress": False,
    "completed": False,
    "last_error": "",
    "last_started_at": None,
    "last_finished_at": None,
}


def start_pretrain_download_if_needed(settings: Settings) -> None:
    if not settings.auto_download_pretrain:
        logger.info("Startup pretrain download disabled (PIXELBLEND_AUTO_DOWNLOAD_PRETRAIN=false).")
        return

    status = local_pretrain_status(settings)
    content_images = int(status["content_images"])
    style_images = int(status["style_images"])

    if (
        content_images >= settings.auto_download_content_count
        and style_images >= settings.auto_download_style_count
        and settings.pretrain_manifest_path.exists()
    ):
        logger.info(
            "Pretrain dataset already available (content=%s, style=%s). Skipping startup download.",
            content_images,
            style_images,
        )
        return

    if _runtime_state["in_progress"]:
        logger.info("Pretrain download is already in progress.")
        return

    thread = threading.Thread(
        target=_download_worker,
        name="pixelblend-pretrain-downloader",
        args=(settings,),
        daemon=True,
    )
    thread.start()


def get_download_runtime_state() -> dict[str, Any]:
    return dict(_runtime_state)


def _download_worker(settings: Settings) -> None:
    if not _download_lock.acquire(blocking=False):
        return

    _runtime_state["started"] = True
    _runtime_state["in_progress"] = True
    _runtime_state["completed"] = False
    _runtime_state["last_error"] = ""
    _runtime_state["last_started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        logger.info("Starting pretrain dataset download (content=%s, style=%s).", settings.auto_download_content_count, settings.auto_download_style_count)
        _download_pretrain_dataset(settings)
        _runtime_state["completed"] = True
        logger.info("Pretrain dataset download completed successfully.")
    except Exception as exc:  # noqa: BLE001
        _runtime_state["last_error"] = str(exc)
        logger.exception("Pretrain dataset download failed: %s", exc)
    finally:
        _runtime_state["in_progress"] = False
        _runtime_state["last_finished_at"] = datetime.now(timezone.utc).isoformat()
        _download_lock.release()


def _download_pretrain_dataset(settings: Settings) -> None:
    load_dataset = _get_load_dataset()
    tqdm = _get_tqdm()

    content_dir = settings.pretrain_content_dir
    style_dir = settings.pretrain_style_dir
    manifest_path = settings.pretrain_manifest_path

    if settings.auto_download_reset:
        _reset_dir(content_dir)
        _reset_dir(style_dir)

    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    content_total = _save_dataset_images(
        load_dataset=load_dataset,
        tqdm=tqdm,
        dataset_name=_DEFAULT_CONTENT_DATASET,
        config_name=None,
        split="train",
        output_dir=content_dir,
        target_count=settings.auto_download_content_count,
        max_edge=settings.auto_download_max_edge,
        seed=settings.auto_download_seed,
        streaming=settings.auto_download_streaming,
        filename_prefix="content",
        scan_factor=_DEFAULT_SCAN_FACTOR,
        url_timeout=_DEFAULT_URL_TIMEOUT,
    )

    style_total = _save_dataset_images(
        load_dataset=load_dataset,
        tqdm=tqdm,
        dataset_name=_DEFAULT_STYLE_DATASET,
        config_name=None,
        split="train",
        output_dir=style_dir,
        target_count=settings.auto_download_style_count,
        max_edge=settings.auto_download_max_edge,
        seed=settings.auto_download_seed,
        streaming=settings.auto_download_streaming,
        filename_prefix="style",
        scan_factor=_DEFAULT_SCAN_FACTOR,
        url_timeout=_DEFAULT_URL_TIMEOUT,
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": "2026-02-18",
        "content": {
            "dataset": _DEFAULT_CONTENT_DATASET,
            "split": "train",
            "saved_images": content_total,
            "output_dir": str(content_dir),
        },
        "style": {
            "dataset": _DEFAULT_STYLE_DATASET,
            "split": "train",
            "saved_images": style_total,
            "output_dir": str(style_dir),
        },
        "sources": {
            "coco": "https://cocodataset.org/#download",
            "wikiart": "https://huggingface.co/datasets/huggan/wikiart",
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_dataset_images(
    load_dataset,
    tqdm,
    dataset_name: str,
    config_name: str | None,
    split: str,
    output_dir: Path,
    target_count: int,
    max_edge: int,
    seed: int,
    streaming: bool,
    filename_prefix: str,
    scan_factor: int,
    url_timeout: int,
) -> int:
    dataset = _load_hf_dataset(load_dataset, dataset_name, config_name=config_name, split=split, streaming=streaming)

    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    iterator = iter(dataset)
    saved = 0
    scanned = 0
    skipped = 0
    max_scan = max(target_count, target_count * max(1, scan_factor))

    with tqdm(total=target_count, desc=f"{dataset_name}:{split}", unit="img") as progress:
        while saved < target_count and scanned < max_scan:
            try:
                row = next(iterator)
            except StopIteration:
                break

            scanned += 1
            image = _extract_image(row, url_timeout=url_timeout)
            if image is None:
                skipped += 1
                if scanned % 25 == 0:
                    progress.set_postfix(scanned=scanned, skipped=skipped)
                continue

            prepared = _prepare_image(image, max_edge=max_edge)
            file_path = output_dir / f"{filename_prefix}_{saved:06d}{_IMAGE_SUFFIX}"
            prepared.save(file_path, format="JPEG", quality=95)

            saved += 1
            progress.update(1)
            if saved % 25 == 0:
                progress.set_postfix(scanned=scanned, skipped=skipped)

    if saved == 0:
        logger.warning(
            "%s:%s saved 0 images after scanning %s rows. Check network access and dataset schema.",
            dataset_name,
            split,
            scanned,
        )

    return saved


def _load_hf_dataset(load_dataset, dataset_name: str, config_name: str | None, split: str, streaming: bool):
    if config_name:
        try:
            return load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        except Exception:  # noqa: BLE001
            pass

    return load_dataset(dataset_name, split=split, streaming=streaming)


def _get_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install -r backend/requirements.txt"
        ) from exc
    return load_dataset


def _get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'tqdm'. Install with: pip install -r backend/requirements.txt") from exc
    return tqdm


def _extract_image(row: dict[str, object], url_timeout: int) -> Image.Image | None:
    candidate = row.get("image")
    if candidate is None:
        for value in row.values():
            if isinstance(value, Image.Image):
                candidate = value
                break
            if isinstance(value, dict) and {"bytes", "path"}.intersection(value.keys()):
                candidate = value
                break

    if isinstance(candidate, Image.Image):
        return candidate.convert("RGB")

    if isinstance(candidate, dict):
        blob = candidate.get("bytes")
        if isinstance(blob, (bytes, bytearray)):
            return Image.open(io.BytesIO(blob)).convert("RGB")

        path = candidate.get("path")
        if isinstance(path, str) and path:
            return Image.open(path).convert("RGB")

    for key in ("coco_url", "image_url", "url", "flickr_url"):
        url = row.get(key)
        if isinstance(url, str) and url:
            image = _download_image(url, timeout=url_timeout)
            if image is not None:
                return image

    return None


def _download_image(url: str, timeout: int) -> Image.Image | None:
    host = urlparse(url).netloc.lower()
    if host and _host_failures.get(host, 0) >= _MAX_HOST_FAILURES:
        return None

    try:
        request = Request(url, headers={"User-Agent": "pixelblend-bootstrap/1.0"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
            if host:
                _host_failures.pop(host, None)
            return image
    except Exception:  # noqa: BLE001
        if host:
            _host_failures[host] = _host_failures.get(host, 0) + 1
        return None


def _prepare_image(image: Image.Image, max_edge: int) -> Image.Image:
    image = image.convert("RGB")
    if max_edge <= 0:
        return image

    width, height = image.size
    largest_edge = max(width, height)
    if largest_edge <= max_edge:
        return image

    ratio = max_edge / float(largest_edge)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _reset_dir(directory: Path) -> None:
    if not directory.exists():
        return

    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_path.unlink()
