#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PixelBlend pretrained model artifact.")
    parser.add_argument("--url", required=True, help="Direct URL to model .pth file")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "model" / "pixelblend_pretrained_model.pth",
        help="Target model path",
    )
    parser.add_argument(
        "--sha256",
        default="",
        help="Optional expected SHA256 checksum for integrity verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    print(f"Downloading model from: {args.url}")
    with urlopen(args.url) as response:  # noqa: S310
        with temp_path.open("wb") as out:
            shutil.copyfileobj(response, out)

    if args.sha256:
        digest = _sha256(temp_path)
        if digest.lower() != args.sha256.lower():
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch. expected={args.sha256} actual={digest}")

    shutil.move(str(temp_path), str(args.output))
    print(f"Saved model to: {args.output}")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
