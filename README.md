# PixelBlend (Pretrained Model App)

PixelBlend is a web app for dual-style neural style transfer using a pretrained model artifact (`.pth`) and an inference-only backend.

## Best Pretrain Data (As of 2026-02-18)

Recommended pair for PixelBlend training/fine-tuning:

- Content data: **COCO 2017 train**
  - Official source: https://cocodataset.org/#download
  - HF mirror used by bootstrap script: https://huggingface.co/datasets/phiyodr/coco2017
- Style data: **WikiArt train**
  - Dataset card used by bootstrap script: https://huggingface.co/datasets/huggan/wikiart

Why this pair:

- COCO gives dense real-world structure/object diversity for content preservation.
- WikiArt gives strong artist/style diversity for texture/color/style transfer quality.

## Project Structure

```
PIXELBLEND/
  backend/
    app/
      main.py
      config.py
      model_runner.py
      pretrain_catalog.py
    requirements.txt
  frontend/
    index.html
    styles.css
    app.js
  scripts/
    bootstrap_pretrain_data.py
    download_pretrained_model.py
  model/
    pixelblend_pretrained_model.pth
  data/
    pretrain/
      content/
      style/
      manifest.json
  uploads/
  outputs/
```

## API Endpoints

- `GET /api/health`
- `GET /api/model-status`
- `POST /api/generate`
- `GET /api/pretrain/catalog`
- `GET /api/pretrain/status`

## Quick Start

1. Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. Download model artifact into `model/pixelblend_pretrained_model.pth`:

```bash
python3 scripts/download_pretrained_model.py --url "<direct-model-url>"
```

3. Start backend (serves frontend + outputs):

```bash
python3 run.py --reload
```

4. Open browser:

- http://localhost:8000

At startup, backend automatically starts background download of required pretrain datasets (COCO + WikiArt) into `data/pretrain/`.

## Generate Flow

1. User uploads `content_image`, `style_image_1`, `style_image_2`.
2. Backend validates file type/size and stores source files under `uploads/<job_id>/`.
3. Pretrained model is loaded in memory at startup from `model/pixelblend_pretrained_model.pth`.
4. Images are preprocessed (RGB + resize + tensor conversion).
5. Inference runs dual-style blending.
6. Output written to `outputs/stylized_<job_id>.jpg`.
7. Frontend previews and provides download.

## Runtime Notes

- CPU mode is supported (`PIXELBLEND_MODEL_DEVICE=cpu`).
- If model loading fails and `PIXELBLEND_ALLOW_FALLBACK=true`, backend uses a lightweight fallback blend so development can continue.
- Path vars in `backend/.env` can be relative (for example `data/pretrain/content`); they are resolved dynamically against the project root.
- Dataset auto-download starts at startup by default (`PIXELBLEND_AUTO_DOWNLOAD_PRETRAIN=true`).
- Download progress/status is visible at `GET /api/pretrain/status` under `download_runtime`.
- Main startup controls:
  - `PIXELBLEND_AUTO_DOWNLOAD_CONTENT_COUNT`
  - `PIXELBLEND_AUTO_DOWNLOAD_STYLE_COUNT`
  - `PIXELBLEND_AUTO_DOWNLOAD_STREAMING`
  - `PIXELBLEND_AUTO_DOWNLOAD_RESET`
- To force strict pretrained-only startup:

```bash
export PIXELBLEND_ALLOW_FALLBACK=false
```

Optional manual dataset bootstrap remains available:

```bash
python3 scripts/bootstrap_pretrain_data.py --streaming --content-count 5000 --style-count 5000
```
