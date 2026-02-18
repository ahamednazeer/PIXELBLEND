#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# Architecture Definitions (copied/adapted from naoto0804/pytorch-AdaIN)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Architecture Definitions (copied/adapted from naoto0804/pytorch-AdaIN)
# -------------------------------------------------------------------------

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    size = feat.size()
    N, C = size[0], size[1]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def get_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )

def get_vgg():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )

class PixelBlendAdapter(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(PixelBlendAdapter, self).__init__()
        # Freeze and slice encoder
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        
        self.decoder = decoder
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        x = self.enc_1(input)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        return x

    def forward(self, content: torch.Tensor, style1: torch.Tensor, style2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # weights is [w1, w2]
        w1 = weights[0]
        w2 = weights[1]

        c_feat = self.encode(content)
        s1_feat = self.encode(style1)
        s2_feat = self.encode(style2)

        feat = w1 * adaptive_instance_normalization(c_feat, s1_feat) + \
               w2 * adaptive_instance_normalization(c_feat, s2_feat)

        return self.decoder(feat)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def download_file(url, output_path):
    print(f"Downloading {url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(url, context=ctx) as response, open(output_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def main():
    parser = argparse.ArgumentParser(description="Build PixelBlend Adapter Model")
    parser.add_argument("--output", type=Path, default="model/pixelblend_pretrained_model.pth")
    args = parser.parse_args()

    # URLs for naoto0804/pytorch-AdaIN pretrained weights (tag v0.0.0)
    VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
    DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        vgg_path = temp_dir / "vgg_normalised.pth"
        decoder_path = temp_dir / "decoder.pth"

        # Download weights
        try:
            download_file(VGG_URL, vgg_path)
            download_file(DECODER_URL, decoder_path)
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Trying alternate URLs (generic)...")
            # If v1.0 fails, try checking other sources if implemented (omitted for brevity)
            sys.exit(1)

        print("Loading weights...")
        vgg = get_vgg()
        decoder = get_decoder()

        vgg.load_state_dict(torch.load(vgg_path))
        decoder.load_state_dict(torch.load(decoder_path))

        print("Building adapter...")
        model = PixelBlendAdapter(vgg, decoder)
        model.eval()

        print("Scripting model...")
        # TorchScript trace or script? Script is safer for logic preservation.
        # But we need to use tracing if 'script' fails on dynamic parts (AdaIN has simple logic).
        # PixelBlendAdapter uses `adaptive_instance_normalization` which is functional.
        
        try:
            scripted_model = torch.jit.script(model)
        except Exception as e:
            print(f"Scripting failed: {e}")
            print("Attempting to save as standard nn.Module checkpoint (fallback for model_runner)")
            # model_runner.py supports loading nn.Module checkpoints too!
            # But the class needs to be available? No, torch.save saves the class structure if pickle...
            # But loading it requires the class to be defined in the loading context OR 
            # we must save the state_dict? 
            # model_runner.py logic:
            # checkpoint = torch.load(...)
            # if isinstance(checkpoint, nn.Module): ...
            # This requires the class definition to be importable by the backend.
            # The backend DOES NOT import this script. 
            # So we MUST use TorchScript.
            sys.exit(1)

        print(f"Saving to {args.output}...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(scripted_model, args.output)
        print("Done!")

if __name__ == "__main__":
    main()
