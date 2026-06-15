#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbone registry for STGRAPH:
- UNetBackbone: original UNet feature/logit extractor
- DINOBackbone: DINOv3 feature extractor (native-size input), with optional freeze
"""

from typing import Dict, Tuple
import sys
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetFeature(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feat = x
        logits = self.outc(x)
        return feat, logits


class UNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNetFeature(n_channels=1, n_classes=1, bilinear=True)
        self.out_dim = 64

    def forward_features(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        feat, logits = self.unet.forward_features(frames)
        return feat, logits, {"backbone_type": "unet", "out_dim": self.out_dim}


class DINOBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str,
        repo_dir: str,
        backbone_weights: str,
        freeze: bool,
        use_autocast_bf16: bool,
        gray_to_rgb: bool,
        dino_layer_select: int = -1,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.repo_dir = str(repo_dir)
        self.freeze = bool(freeze)
        self.use_autocast_bf16 = bool(use_autocast_bf16)
        self.gray_to_rgb = bool(gray_to_rgb)
        self.dino_layer_select = int(dino_layer_select)
        self.out_dim = int(hidden_dim)

        # Load from dinov3.hub.backbones directly to avoid importing hubconf.py,
        # which may pull segmentation deps incompatible with local torch.
        self.dino = self._load_dino_backbone_direct(
            repo_dir=self.repo_dir,
            model_name=self.model_name,
            backbone_weights=str(backbone_weights or ""),
        )

        self.patch_size = int(getattr(self.dino, "patch_size", 16))
        self.token_dim = int(getattr(self.dino, "embed_dim", hidden_dim))

        self.feat_proj = nn.Conv2d(self.token_dim, self.out_dim, kernel_size=1)
        self.point_head = nn.Conv2d(self.out_dim, 1, kernel_size=1)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        if self.freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()

    @staticmethod
    def _resolve_layer_index(depth: int, sel: int) -> int:
        if sel == 0:
            raise ValueError("dino_layer_select=0 is invalid (1-based positive index required, or negative index from tail).")
        if sel < 0:
            idx = int(depth + sel)
        else:
            idx = int(sel - 1)
        if idx < 0 or idx >= int(depth):
            raise ValueError(f"dino_layer_select out of range: sel={sel}, depth={depth}, resolved_idx={idx}")
        return idx

    @staticmethod
    def _load_dino_backbone_direct(repo_dir: str, model_name: str, backbone_weights: str) -> nn.Module:
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from dinov3.hub import backbones as dino_backbones  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import dinov3.hub.backbones from {repo_dir}: {e}") from e

        if not hasattr(dino_backbones, model_name):
            raise ValueError(f"Unknown DINO backbone '{model_name}' in dinov3.hub.backbones")
        ctor = getattr(dino_backbones, model_name)
        weights_path = str(backbone_weights or "").strip()
        if not weights_path:
            raise ValueError(
                f"DINO backbone requires a local weights path. "
                f"Please set --dino_backbone_weights for model '{model_name}'."
            )
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"DINO weights file not found: {weights_path}")

        # dinov3_vitl16's hub loader requires the weights filename to carry an
        # 8-character hash suffix (e.g. '...-8aa4cbdd.pth'). We ship the weights
        # file already named that way and point the config directly at it, so no
        # symlink/alias is created here — the explicit path is handed straight to
        # the constructor.
        if model_name == "dinov3_vitl16":
            import re

            if re.search(r"-(.{8})\.pth$", weights_path) is None:
                raise ValueError(
                    "dinov3_vitl16 expects a weights file whose name ends with an "
                    "8-character hash suffix, e.g. "
                    "'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'. "
                    f"Got: {weights_path}. Rename the weights file accordingly "
                    "(the canonical LVD-1689M hash is 8aa4cbdd)."
                )

        return ctor(pretrained=True, weights=weights_path)

    def _prepare_input(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (T,1,H,W) in [0,1]
        if self.gray_to_rgb:
            x = frames.repeat(1, 3, 1, 1)
        else:
            if frames.shape[1] != 3:
                raise ValueError("DINO expects 3-channel input when --no_dino_gray_to_rgb is used.")
            x = frames
        x = (x - self.pixel_mean.to(x.device, x.dtype)) / self.pixel_std.to(x.device, x.dtype)
        return x

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: (B, N, C)
        b, n, c = tokens.shape
        hp = max(1, h // self.patch_size)
        wp = max(1, w // self.patch_size)
        if hp * wp != n:
            # Fallback for uncommon shape mismatch.
            hp = int(math.sqrt(n))
            hp = max(1, hp)
            wp = max(1, n // hp)
            if hp * wp != n:
                hp, wp = n, 1
        feat = tokens.transpose(1, 2).reshape(b, c, hp, wp)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return feat

    def forward_features(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # frames: (T,1,H,W)
        t, _, h, w = frames.shape
        x = self._prepare_input(frames)

        if self.freeze:
            self.dino.eval()

        use_amp = (x.device.type == "cuda") and self.use_autocast_bf16
        amp_dtype = torch.bfloat16

        if not hasattr(self.dino, "get_intermediate_layers"):
            raise RuntimeError(f"{self.model_name} does not support get_intermediate_layers")
        depth = int(getattr(self.dino, "n_blocks", len(getattr(self.dino, "blocks", []))))
        if depth <= 0:
            raise RuntimeError(f"Failed to infer transformer depth for {self.model_name}")
        idx = self._resolve_layer_index(depth, self.dino_layer_select)

        with torch.set_grad_enabled(not self.freeze):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outs = self.dino.get_intermediate_layers(
                    x,
                    n=[idx],
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )
        if not isinstance(outs, (list, tuple)) or len(outs) != 1:
            raise RuntimeError(f"Unexpected get_intermediate_layers output for {self.model_name}: type={type(outs)} len={len(outs) if hasattr(outs, '__len__') else 'NA'}")
        tok = outs[0]  # (T, N, C)

        feat_map = self._tokens_to_map(tok, h, w)
        feat_proj = self.feat_proj(feat_map)
        logits_like = self.point_head(feat_proj)
        return feat_proj, logits_like, {
            "backbone_type": "dino",
            "model_name": self.model_name,
            "freeze": bool(self.freeze),
            "out_dim": int(self.out_dim),
            "patch_size": int(self.patch_size),
            "dino_depth": int(depth),
            "dino_layer_select": int(self.dino_layer_select),
            "dino_layer_index_0based": int(idx),
        }


def build_backbone(
    backbone_type: str,
    hidden_dim: int,
    dino_model_name: str,
    dino_repo_dir: str,
    dino_backbone_weights: str,
    dino_freeze: bool,
    dino_use_autocast_bf16: bool,
    dino_gray_to_rgb: bool,
    dino_layer_select: int = -1,
) -> nn.Module:
    btype = str(backbone_type).lower()
    if btype == "unet":
        return UNetBackbone()
    if btype == "dino":
        return DINOBackbone(
            hidden_dim=int(hidden_dim),
            model_name=str(dino_model_name),
            repo_dir=str(dino_repo_dir),
            backbone_weights=str(dino_backbone_weights or ""),
            freeze=bool(dino_freeze),
            use_autocast_bf16=bool(dino_use_autocast_bf16),
            gray_to_rgb=bool(dino_gray_to_rgb),
            dino_layer_select=int(dino_layer_select),
        )
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")
