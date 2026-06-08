"""Frozen SegFormer-B0 backbone for semantic plane gating.

Wraps ``nvidia/segformer-b0-finetuned-cityscapes-512-1024`` from HuggingFace
Transformers. All parameters are permanently frozen — it is a pure feature
extractor. The output is 19-class Cityscapes logits bilinearly upsampled to a
quarter of the input resolution (matching the stride-4 output of the ResNet
encoder so the SemanticPlaneGate sees the same spatial scale as the depth
decoder feature maps).

Offline / air-gapped machines
------------------------------
The model is downloaded once from HuggingFace and cached in
~/.cache/huggingface.  On machines without internet access, pre-download
on a connected machine first::

    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('nvidia/segformer-b0-finetuned-cityscapes-512-1024')
    "

Then copy the cache directory to the offline machine, OR pass
``--segformer_model /path/to/local/directory`` pointing to a folder that
contains ``config.json`` and ``pytorch_model.bin``.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NUM_CLASSES = 19  # Cityscapes-19


class SegFormerBackbone(nn.Module):
    """Frozen SegFormer that returns Cityscapes-19 logits at H/4 × W/4.

    Args:
        model_id: HuggingFace model ID **or** path to a local directory
                  containing config.json + pytorch_model.bin.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL):
        super().__init__()

        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "The `transformers` package is required for SegFormer:\n"
                "  pip install transformers>=4.12.0"
            ) from e

        try:
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        except OSError:
            # Network unavailable — retry from local cache only
            try:
                model = SegformerForSemanticSegmentation.from_pretrained(
                    model_id, local_files_only=True)
            except OSError as e:
                raise OSError(
                    f"Cannot load SegFormer from '{model_id}'.\n\n"
                    "Cache not found. Pre-download on a connected machine:\n\n"
                    "  python -c \"\n"
                    "  from huggingface_hub import snapshot_download\n"
                    "  snapshot_download('nvidia/segformer-b0-finetuned-cityscapes-512-1024')\n"
                    "  \"\n\n"
                    "Copy ~/.cache/huggingface/hub/models--nvidia--segformer* "
                    "to the same path on this machine, or pass a local directory "
                    "via --segformer_model /path/to/model.\n"
                ) from e

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.num_classes = NUM_CLASSES

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
        logits = out.logits  # (B, 19, H/4-ish, W/4-ish)

        target_h, target_w = H // 4, W // 4
        if logits.shape[-2:] != (target_h, target_w):
            logits = F.interpolate(
                logits, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
        return logits
