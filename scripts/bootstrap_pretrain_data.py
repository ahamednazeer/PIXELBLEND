#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_CONTENT_DATASET = "phiyodr/coco2017"
DEFAULT_STYLE_DATASET = "huggan/wikiart"
IMAGE_SUFFIX = ".jpg"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_HOST_FAILURES = 3
_host_failures: dict[str, int] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a curated COCO + WikiArt pretraining set for PixelBlend development."
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "pretrain")
    parser.add_argument("--content-count", type=int, default=5000)
    parser.add_argument("--style-count", type=int, default=5000)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode to avoid full dataset cache.")
    parser.add_argument("--reset", action="store_true", help="Delete previously downloaded images first.")
    parser.add_argument(
        "--scan-factor",
        type=int,
        default=8,
        help="Maximum scanned rows is target_count * scan_factor before stopping.",
    )
    parser.add_argument(
        "--url-timeout",
        type=int,
        default=8,
        help="Timeout in seconds when downloading images from URL fields.",
    )
    parser.add_argument(
        "--max-total-gb",
        type=float,
        default=0.0,
        help="Stop downloading when output-root reaches this size in GB (0 disables cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.content_count < 0 or args.style_count < 0:
        raise ValueError("content-count and style-count must be >= 0")
    if args.content_count == 0 and args.style_count == 0:
        raise ValueError("At least one of content-count or style-count must be > 0")
    if args.max_total_gb < 0:
        raise ValueError("max-total-gb must be >= 0")

    content_dir = args.output_root / "content"
    style_dir = args.output_root / "style"
    manifest_path = args.output_root / "manifest.json"
    max_total_bytes = int(args.max_total_gb * 1024 * 1024 * 1024) if args.max_total_gb > 0 else 0

    if args.reset:
        _reset_dir(content_dir)
        _reset_dir(style_dir)

    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    content_total = 0
    if args.content_count > 0:
        content_total = _save_dataset_images(
            dataset_name=DEFAULT_CONTENT_DATASET,
            config_name=None,
            split="train",
            output_dir=content_dir,
            target_count=args.content_count,
            max_edge=args.max_edge,
            seed=args.seed,
            streaming=args.streaming,
            filename_prefix="content",
            scan_factor=args.scan_factor,
            url_timeout=args.url_timeout,
            size_root=args.output_root,
            max_total_bytes=max_total_bytes,
        )

    style_total = 0
    if args.style_count > 0:
        style_total = _save_dataset_images(
            dataset_name=DEFAULT_STYLE_DATASET,
            config_name=None,
            split="train",
            output_dir=style_dir,
            target_count=args.style_count,
            max_edge=args.max_edge,
            seed=args.seed,
            streaming=args.streaming,
            filename_prefix="style",
            scan_factor=args.scan_factor,
            url_timeout=args.url_timeout,
            size_root=args.output_root,
            max_total_bytes=max_total_bytes,
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": "2026-02-18",
        "content": {
            "dataset": DEFAULT_CONTENT_DATASET,
            "split": "train",
            "saved_images": content_total,
            "output_dir": str(content_dir),
        },
        "style": {
            "dataset": DEFAULT_STYLE_DATASET,
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

    print(f"Saved manifest: {manifest_path}")
    print(f"Content images: {content_total}")
    print(f"Style images:   {style_total}")


def _save_dataset_images(
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
    size_root: Path,
    max_total_bytes: int,
) -> int:
    dataset = _load_dataset(dataset_name, config_name=config_name, split=split, streaming=streaming)
    tqdm = _load_tqdm()

    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    iterator = iter(dataset)
    saved = 0
    scanned = 0
    skipped = 0
    max_scan = max(target_count, target_count * max(1, scan_factor))

    with tqdm(total=target_count, desc=f"{dataset_name}:{split}", unit="img") as progress:
        while saved < target_count and scanned < max_scan:
            if max_total_bytes > 0 and scanned % 25 == 0:
                current_size = _dir_size_bytes(size_root)
                if current_size >= max_total_bytes:
                    print(
                        f"[info] Size cap reached for {size_root}: "
                        f"{current_size / (1024 ** 3):.2f}GB >= {max_total_bytes / (1024 ** 3):.2f}GB"
                    )
                    break

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
            file_path = output_dir / f"{filename_prefix}_{saved:06d}{IMAGE_SUFFIX}"
            prepared.save(file_path, format="JPEG", quality=95)

            saved += 1
            progress.update(1)
            if saved % 25 == 0:
                progress.set_postfix(scanned=scanned, skipped=skipped)

    if saved == 0:
        print(
            f"[warn] {dataset_name}:{split} saved 0 images after scanning {scanned} rows. "
            "Check network access and dataset schema."
        )

    return saved


def _dir_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _load_dataset(dataset_name: str, config_name: str | None, split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install dependencies with: pip install -r backend/requirements.txt"
        ) from exc

    if config_name:
        try:
            return load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        except Exception:  # noqa: BLE001
            pass

    return load_dataset(dataset_name, split=split, streaming=streaming)


def _load_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "The 'tqdm' package is required. Install dependencies with: pip install -r backend/requirements.txt"
        ) from exc
    return tqdm


def _extract_image(row: dict[str, object], url_timeout: int) -> Image.Image | None:
    candidate = row.get("image")
    if candidate is None:
        for key, value in row.items():
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
    if host and _host_failures.get(host, 0) >= MAX_HOST_FAILURES:
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


if __name__ == "__main__":
    main()
