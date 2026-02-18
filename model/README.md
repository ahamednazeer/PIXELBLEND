# Pretrained Model Directory

Place the pretrained style transfer file here:

- `pixelblend_pretrained_model.pth`

The backend loads this file at startup from `/model/pixelblend_pretrained_model.pth`.

You can download a model artifact directly into this path:

```bash
python3 scripts/download_pretrained_model.py --url "<direct-model-url>"
```
