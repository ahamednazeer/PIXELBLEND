from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .config import Settings


class ModelLoadError(RuntimeError):
    """Raised when the pretrained model cannot be loaded."""


@dataclass
class ModelStatus:
    mode: str
    loaded: bool
    model_path: str
    message: str


class PixelBlendModelRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device(settings.model_device)
        self.model: Any | None = None
        self.model_mode: str = "fallback"
        self.status = ModelStatus(
            mode="fallback",
            loaded=False,
            model_path=str(settings.model_path),
            message="Model not loaded yet.",
        )

    def load(self) -> None:
        torch.set_num_threads(self.settings.model_threads)
        model_path = Path(self.settings.model_path)

        if not model_path.exists():
            message = f"Pretrained model not found at {model_path}."
            if not self.settings.allow_fallback:
                raise ModelLoadError(message)
            self.status = ModelStatus(
                mode="fallback",
                loaded=False,
                model_path=str(model_path),
                message=f"{message} Falling back to lightweight blending pipeline.",
            )
            return

        load_errors: list[str] = []

        try:
            scripted = torch.jit.load(str(model_path), map_location=self.device)
            scripted.eval()
            self.model = scripted
            self.model_mode = "torchscript"
            self.status = ModelStatus(
                mode="torchscript",
                loaded=True,
                model_path=str(model_path),
                message="Loaded TorchScript pretrained model.",
            )
            return
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"TorchScript load failed: {exc}")

        try:
            checkpoint = torch.load(str(model_path), map_location=self.device)
            if isinstance(checkpoint, torch.nn.Module):
                checkpoint.eval()
                self.model = checkpoint
                self.model_mode = "nn_module"
                self.status = ModelStatus(
                    mode="nn_module",
                    loaded=True,
                    model_path=str(model_path),
                    message="Loaded serialized nn.Module pretrained model.",
                )
                return

            if isinstance(checkpoint, dict):
                candidate = checkpoint.get("model") or checkpoint.get("generator")
                if isinstance(candidate, torch.nn.Module):
                    candidate.eval()
                    self.model = candidate
                    self.model_mode = "checkpoint_module"
                    self.status = ModelStatus(
                        mode="checkpoint_module",
                        loaded=True,
                        model_path=str(model_path),
                        message="Loaded model from checkpoint dictionary.",
                    )
                    return

                if _looks_like_fast_style_state_dict(checkpoint):
                    self.model = None
                    self.model_mode = "fallback"
                    self.status = ModelStatus(
                        mode="fallback_dual_style",
                        loaded=False,
                        model_path=str(model_path),
                        message=(
                            "Fixed-style checkpoint detected (.pth state_dict). "
                            "Switching to dual-style fallback so uploaded style images are respected."
                        ),
                    )
                    return

            load_errors.append("Checkpoint did not contain a runnable model object.")
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"torch.load failed: {exc}")

        message = " | ".join(load_errors)
        if not self.settings.allow_fallback:
            raise ModelLoadError(message)

        self.status = ModelStatus(
            mode="fallback",
            loaded=False,
            model_path=str(model_path),
            message=f"Failed to load pretrained model. {message}",
        )
        self.model_mode = "fallback"

    def stylize(
        self,
        content_image: Image.Image,
        style_image_1: Image.Image,
        style_image_2: Image.Image,
        style_1_weight: float,
        high_quality: bool = False,
        detail_strength: float = 0.5,
        style_intensity: float = 0.5,
    ) -> Image.Image:
        style_1_weight = float(np.clip(style_1_weight, 0.0, 1.0))

        if self.model is None:
            stylized = self._fallback_blend(content_image, style_image_1, style_image_2, style_1_weight)
            return self._content_preserving_finish(
                content_image,
                stylized,
                style_image_1,
                style_image_2,
                style_1_weight,
                high_quality=high_quality,
                detail_strength=detail_strength,
                style_intensity=style_intensity,
            )

        target_size = 1024 if high_quality else self.settings.output_image_size
        content_tensor = self._to_tensor(content_image, target_size=target_size)
        style_tensor_1 = self._to_tensor(style_image_1, target_size=target_size)
        style_tensor_2 = self._to_tensor(style_image_2, target_size=target_size)

        with torch.inference_mode():
            output = self._call_model(content_tensor, style_tensor_1, style_tensor_2, style_1_weight)

        stylized = self._to_image(output, target_size=content_image.size)
        return self._content_preserving_finish(
            content_image,
            stylized,
            style_image_1,
            style_image_2,
            style_1_weight,
            high_quality=high_quality,
            detail_strength=detail_strength,
            style_intensity=style_intensity,
        )

    def _content_preserving_finish(
        self,
        content_image: Image.Image,
        stylized_image: Image.Image,
        style_image_1: Image.Image,
        style_image_2: Image.Image,
        style_1_weight: float,
        high_quality: bool = False,
        detail_strength: float = 0.5,
        style_intensity: float = 0.5,
    ) -> Image.Image:
        """Align colors to style reference while preventing style-image geometry bleed-through."""
        content = content_image.convert("RGB")
        stylized = stylized_image.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)
        style_1 = style_image_1.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)
        style_2 = style_image_2.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)
        mixed_style_img = Image.blend(style_2, style_1, style_1_weight)

        content_arr = np.asarray(content, dtype=np.float32) / 255.0
        stylized_arr = np.asarray(stylized, dtype=np.float32) / 255.0
        mixed_style_arr = np.asarray(mixed_style_img, dtype=np.float32) / 255.0

        content_yuv = _rgb_to_yuv(content_arr)
        stylized_yuv = _rgb_to_yuv(stylized_arr)
        mixed_style_yuv = _rgb_to_yuv(mixed_style_arr)

        # 1. Luminance Blending (Structure & Contrast)
        c_y = content_yuv[:, :, 0]
        s_y = mixed_style_yuv[:, :, 0]
        st_y = stylized_yuv[:, :, 0] # The neural output luminance

        # Determine target stats based on Style Intensity
        # 0.0 -> Match Content Stats (Original Photo Lighting)
        # 1.0 -> Match Style Stats (Ink/Comic Contrast)
        target_mean = (1.0 - style_intensity) * c_y.mean() + style_intensity * s_y.mean()
        target_std = (1.0 - style_intensity) * c_y.std() + style_intensity * s_y.std()

        # Helper to remap a channel to target stats
        def remap_channel(source, t_mean, t_std):
            s_mean = source.mean()
            s_std = source.std()
            return ((source - s_mean) / (s_std + 1e-6)) * t_std + t_mean

        # A. Base Background Structure construction
        # We blend Content Structure vs Neural Output Structure
        # 0.0 Intensity -> 0.7*Content + 0.3*Stylized (Default/Historical behavior)
        # 1.0 Intensity -> 0.0*Content + 1.0*Stylized (Pure Neural Style)
        # This ensures we actually SEE the style strokes at 100%
        content_w = 0.7 * (1.0 - style_intensity)
        stylized_w = 1.0 - content_w
        
        base_y = content_w * c_y + stylized_w * st_y
        
        # Remap the base structure to the target luminance stats
        out_y = np.clip(remap_channel(base_y, target_mean, target_std), 0.0, 1.0)
        
        # 2. Color Blending (Chrominance)
        # Match colors to style
        matched_u = _match_channel_stats(stylized_yuv[:, :, 1], mixed_style_yuv[:, :, 1])
        matched_v = _match_channel_stats(stylized_yuv[:, :, 2], mixed_style_yuv[:, :, 2])
        
        # Blend Color Target
        # 0.0 -> Stylized Output Colors (Usually close to content/style mix)
        # 1.0 -> Strict Style Palette
        # Actually, let's blend target U/V
        target_u = (1.0 - style_intensity) * matched_u + style_intensity * mixed_style_yuv[:, :, 1]
        target_v = (1.0 - style_intensity) * matched_v + style_intensity * mixed_style_yuv[:, :, 2]
        
        out_yuv = np.empty_like(stylized_yuv)
        out_yuv[:, :, 0] = out_y
        out_yuv[:, :, 1] = target_u
        out_yuv[:, :, 2] = target_v
        
        combined_rgb = np.clip(_yuv_to_rgb(out_yuv), 0.0, 1.0)

        # Mild palette anchoring
        palette = _extract_palette(mixed_style_img, num_colors=18)
        palette_snapped = _snap_to_palette(combined_rgb, palette)
        
        palette_weight = 0.16 + (style_intensity * 0.20)
        palette_aligned = np.clip((1.0 - palette_weight) * combined_rgb + palette_weight * palette_snapped, 0.0, 1.0)

        # 3. Detail Preservation (The Fix for "Looks like Original")
        # When we inject "Detail" (Original Content), we MUST tone-map it to the style first!
        # Otherwise, injecting raw content pixels overrides the style we just created.
        
        # Create a "Style-Matched Content" version
        # It has Content Structure (Luminance) but mapped to Style Stats
        c_y_mapped = remap_channel(c_y, target_mean, target_std)
        c_u_matched = (1.0 - style_intensity) * content_yuv[:,:,1] + style_intensity * target_u
        c_v_matched = (1.0 - style_intensity) * content_yuv[:,:,2] + style_intensity * target_v
        
        content_matched_yuv = np.dstack([c_y_mapped, c_u_matched, c_v_matched])
        content_matched_rgb = np.clip(_yuv_to_rgb(content_matched_yuv), 0.0, 1.0)

        # Calculate Edge Mask
        gy, gx = np.gradient(c_y)
        edge_energy = np.sqrt(gx * gx + gy * gy)
        edge_energy = edge_energy / (edge_energy.max() + 1e-6)

        # Base offset keeps faint details; multiplier rewards strong edges
        # Refined formula: Lower base_offset to let style pattern shows in flat areas.
        # Higher multiplier to aggressively keep edges.
        base_offset = 0.05 * detail_strength  # Max 0.05 base retention (was 0.50!)
        multiplier = 4.0 * detail_strength    # Strong reaction to edges
        
        # Allow keeping up to 95% of original structure at max strength
        max_keep = 0.62 + (detail_strength * 0.35) if detail_strength < 0.8 else 0.95

        edge_keep = np.clip(base_offset + multiplier * edge_energy, 0.0, max_keep)[:, :, None]
        
        # BLEND: Use content_matched_rgb instead of raw content_arr
        structure_preserved = palette_aligned * (1.0 - edge_keep) + content_matched_rgb * edge_keep

        # Add detail from model output (Texture)
        model_detail_weight = 0.12 * (1.0 - detail_strength * 0.8)
        blur_radius = 1.0 if high_quality else 2.2
        smooth_stylized = (
            np.asarray(stylized.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32) / 255.0
        )
        stylized_detail = stylized_arr - smooth_stylized
        textured = np.clip(structure_preserved + model_detail_weight * stylized_detail, 0.0, 1.0)

        if high_quality:
            return Image.fromarray((textured * 255).astype(np.uint8))
        
        poster_levels = 16.0
        poster_weight = 0.10

        posterized = np.round(textured * (poster_levels - 1.0)) / (poster_levels - 1.0)
        final_rgb = np.clip((1.0 - poster_weight) * textured + poster_weight * posterized, 0.0, 1.0)
        return Image.fromarray((final_rgb * 255).astype(np.uint8))

    def _call_model(
        self,
        content_tensor: torch.Tensor,
        style_tensor_1: torch.Tensor,
        style_tensor_2: torch.Tensor,
        style_1_weight: float,
    ) -> torch.Tensor:
        assert self.model is not None

        style_weights = torch.tensor(
            [style_1_weight, 1.0 - style_1_weight], dtype=torch.float32, device=self.device
        )

        call_attempts = [
            lambda: self.model(content_tensor, style_tensor_1, style_tensor_2, style_weights),
            lambda: self.model(content_tensor, style_tensor_1, style_tensor_2),
            lambda: self.model(
                {
                    "content": content_tensor,
                    "style_1": style_tensor_1,
                    "style_2": style_tensor_2,
                    "weights": style_weights,
                }
            ),
        ]

        last_error: Exception | None = None
        for attempt in call_attempts:
            try:
                return self._extract_tensor(attempt())
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"Model forward pass failed for all supported signatures: {last_error}")

    def _extract_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)) and output:
            return self._extract_tensor(output[0])

        if isinstance(output, dict):
            for key in ("stylized", "output", "image", "prediction"):
                if key in output:
                    return self._extract_tensor(output[key])

        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    def _to_tensor(self, image: Image.Image, target_size: int | None = None) -> torch.Tensor:
        size = target_size or self.settings.output_image_size
        resized = image.convert("RGB").resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _to_image(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> Image.Image:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor[0].detach().cpu().clamp(0.0, 1.0)
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(arr)
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def _fallback_blend(
        self,
        content_image: Image.Image,
        style_image_1: Image.Image,
        style_image_2: Image.Image,
        style_1_weight: float,
    ) -> Image.Image:
        content = content_image.convert("RGB")
        style_1 = style_image_1.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)
        style_2 = style_image_2.convert("RGB").resize(content.size, Image.Resampling.LANCZOS)

        mixed_style = Image.blend(style_2, style_1, style_1_weight)

        content_arr = np.asarray(content, dtype=np.float32) / 255.0
        style_arr = np.asarray(mixed_style, dtype=np.float32) / 255.0
        style_soft_arr = np.asarray(
            mixed_style.filter(ImageFilter.GaussianBlur(radius=6)),
            dtype=np.float32,
        ) / 255.0
        style_strong_arr = np.asarray(
            mixed_style.filter(ImageFilter.GaussianBlur(radius=18)),
            dtype=np.float32,
        ) / 255.0

        content_yuv = _rgb_to_yuv(content_arr)
        style_yuv = _rgb_to_yuv(style_arr)
        style_soft_yuv = _rgb_to_yuv(style_soft_arr)
        style_tone_y = _rgb_to_yuv(style_strong_arr)[:, :, 0]

        c_y = content_yuv[:, :, 0]
        c_u = content_yuv[:, :, 1]
        c_v = content_yuv[:, :, 2]
        s_u = style_yuv[:, :, 1]
        s_v = style_yuv[:, :, 2]
        s_soft_y = style_soft_yuv[:, :, 0]

        matched_u = _match_channel_stats(c_u, s_u)
        matched_v = _match_channel_stats(c_v, s_v)
        matched_tone = _match_channel_stats(style_tone_y, c_y)
        texture_detail = s_soft_y - style_tone_y

        k_chroma = 0.9
        k_tone = 0.42
        k_texture = 0.10
        out_yuv = np.empty_like(content_yuv)
        out_yuv[:, :, 0] = np.clip((1.0 - k_tone) * c_y + k_tone * matched_tone + k_texture * texture_detail, 0.0, 1.0)
        out_yuv[:, :, 1] = (1.0 - k_chroma) * c_u + k_chroma * matched_u
        out_yuv[:, :, 2] = (1.0 - k_chroma) * c_v + k_chroma * matched_v

        stylized = np.clip(_yuv_to_rgb(out_yuv), 0.0, 1.0)

        result = Image.fromarray((stylized * 255).astype(np.uint8))
        return result.filter(ImageFilter.DETAIL)


def _looks_like_fast_style_state_dict(checkpoint: dict[str, Any]) -> bool:
    required = {
        "conv1.conv2d.weight",
        "conv2.conv2d.weight",
        "conv3.conv2d.weight",
        "deconv1.conv2d.weight",
        "deconv2.conv2d.weight",
        "deconv3.conv2d.weight",
    }
    keys = set(checkpoint.keys())
    return required.issubset(keys)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(self.reflection_pad(x))


class UpsampleConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, upsample: int | None = None
    ) -> None:
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.conv2d(self.reflection_pad(x))


class InstanceNormalization(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-9) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(channels))
        self.shift = nn.Parameter(torch.FloatTensor(channels))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        t = x.view(n, c, h * w)
        mean = t.mean(dim=2).view(n, c, 1, 1).expand(n, c, h, w)
        var = t.var(dim=2, unbiased=False).view(n, c, 1, 1).expand(n, c, h, w)
        scale_broadcast = self.scale.view(1, c, 1, 1).expand(n, c, h, w)
        shift_broadcast = self.shift.view(1, c, 1, 1).expand(n, c, h, w)
        return (x - mean) / torch.sqrt(var + self.eps) * scale_broadcast + shift_broadcast


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class FastStyleTransformerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = InstanceNormalization(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = F.relu(self.in4(self.deconv1(y)))
        y = F.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


def _extract_palette(image: Image.Image, num_colors: int = 16) -> np.ndarray:
    quantized = image.convert("RGB").quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
    palette_raw = quantized.getpalette()
    used = quantized.getcolors(maxcolors=max(256, num_colors * 4)) or []

    if not palette_raw or not used:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.array([[float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean())]])

    entries: list[list[float]] = []
    for _, idx in sorted(used, reverse=True):
        base = idx * 3
        if base + 2 >= len(palette_raw):
            continue
        entries.append(
            [
                palette_raw[base] / 255.0,
                palette_raw[base + 1] / 255.0,
                palette_raw[base + 2] / 255.0,
            ]
        )

    if not entries:
        return np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    return np.asarray(entries[:num_colors], dtype=np.float32)


def _snap_to_palette(image_arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if palette.size == 0:
        return image_arr
    flat = image_arr.reshape(-1, 3)
    distances = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(distances, axis=1)
    snapped = palette[nearest]
    return snapped.reshape(image_arr.shape)


def _match_channel_stats(content_channel: np.ndarray, style_channel: np.ndarray) -> np.ndarray:
    c_mean = float(content_channel.mean())
    c_std = float(content_channel.std())
    s_mean = float(style_channel.mean())
    s_std = float(style_channel.std())
    denom = c_std if c_std > 1e-6 else 1e-6
    return ((content_channel - c_mean) / denom) * s_std + s_mean


def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    return np.stack([y, u, v], axis=2)


def _yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    return np.stack([r, g, b], axis=2)
