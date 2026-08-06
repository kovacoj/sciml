"""Model factory: build FlowUNet or SimpleFlowNet from name."""
from __future__ import annotations

import torch.nn as nn


def build_model(architecture: str, **kwargs) -> nn.Module:
    if architecture == "simple":
        from .simple_model import SimpleFlowNet
        return SimpleFlowNet(**kwargs)
    elif architecture == "unet":
        from .models import FlowUNet
        return FlowUNet(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_model_from_checkpoint(path: str):
    import torch
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(checkpoint["architecture"], **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
