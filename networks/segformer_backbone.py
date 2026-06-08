"""Frozen SegFormer-B0 backbone for semantic plane gating.

Wraps ``nvidia/segformer-b0-finetuned-cityscapes-512-1024`` from HuggingFace
Transformers. All parameters are permanently frozen — it is a pure feature
extractor. The output is 19-class Cityscapes logits bilinearly upsampled to a
quarter of the input resolution (matching the stride-4 output of the ResNet
encoder so the SemanticPlaneGate sees the same spatial scale as the depth
decoder feature maps).

The model is downloaded once from HuggingFace and cached in the default
Transformers cache (~/.cache/huggingface). No manual download needed.

Usage::

    backbone = SegFormerBackbone()
    logits = backbone(img_01)   # img_01: (B, 3, H, W) normalised to [0, 1]
    # → (B, 19, H//4, W//4)
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

# ImageNet mean / std used by SegFormer feature extractor
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NUM_CLASSES = 19  # Cityscapes-19


class SegFormerBackbone(nn.Module):
    """Frozen SegFormer that returns Cityscapes-19 logits at H/4 × W/4.

    Args:
        model_id: HuggingFace model identifier (default: SegFormer-B0 Cityscapes)
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL):
        super().__init__()

        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "The `transformers` package is required for SegFormer: "
                "  pip install transformers>=4.20.0"
            ) from e

        model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.num_classes = NUM_CLASSES

        # Keep normalisation constants as buffers so they move with .to(device)
        self.register_buffer("mean", _IMAGENET_MEAN.clone())
        self.register_buffer("std",  _IMAGENET_STD.clone())

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) float in [0, 1]

        Returns:
            logits: (B, 19, H//4, W//4)
        """
        H, W = img.shape[-2:]
        img_norm = (img - self.mean) / self.std

        out = self.model(pixel_values=img_norm)
        # SegformerForSemanticSegmentation returns logits at ~H/4 × W/4
        logits = out.logits   # (B, 19, H/4-ish, W/4-ish)

        target_h, target_w = H // 4, W // 4
        if logits.shape[-2:] != (target_h, target_w):
            logits = F.interpolate(
                logits, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
        return logits
