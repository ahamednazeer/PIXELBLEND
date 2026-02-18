#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PixelBlend web application.")
    parser.add_argument("--host", default=os.getenv("PIXELBLEND_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PIXELBLEND_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"])
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    project_root = env_path.resolve().parents[1]
    legacy_path_overrides = {
        "PIXELBLEND_MODEL_PATH": project_root / "model" / "pixelblend_pretrained_model.pth",
        "PIXELBLEND_PRETRAIN_CONTENT_DIR": project_root / "data" / "pretrain" / "content",
        "PIXELBLEND_PRETRAIN_STYLE_DIR": project_root / "data" / "pretrain" / "style",
        "PIXELBLEND_PRETRAIN_MANIFEST_PATH": project_root / "data" / "pretrain" / "manifest.json",
    }

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        if key in legacy_path_overrides and value.startswith(("/model/", "/data/")):
            value = str(legacy_path_overrides[key])

        os.environ.setdefault(key, value)

    _normalize_legacy_pixelblend_paths(project_root)


def _normalize_legacy_pixelblend_paths(project_root: Path) -> None:
    replacements = {
        "PIXELBLEND_MODEL_PATH": project_root / "model" / "pixelblend_pretrained_model.pth",
        "PIXELBLEND_PRETRAIN_CONTENT_DIR": project_root / "data" / "pretrain" / "content",
        "PIXELBLEND_PRETRAIN_STYLE_DIR": project_root / "data" / "pretrain" / "style",
        "PIXELBLEND_PRETRAIN_MANIFEST_PATH": project_root / "data" / "pretrain" / "manifest.json",
    }
    for key, fallback in replacements.items():
        current = os.environ.get(key, "")
        if current.startswith(("/data/", "/model/")):
            os.environ[key] = str(fallback)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / "backend" / ".env")

    args = parse_args()

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
