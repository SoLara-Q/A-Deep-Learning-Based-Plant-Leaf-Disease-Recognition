from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a lightweight MobileNetV2 classifier.

    MobileNetV2 is small enough for normal laptops and suitable for a course demo.
    """
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
        except Exception:
            # Older torchvision fallback
            model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
