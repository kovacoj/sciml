"""Shared U-Net architecture for both supervised and physics-trained flow prediction.

Input:  [batch, 1, 64, 64]  (lambda field, 0=fluid, 1=solid)
Output: [batch, 3, 64, 64]  (ux, uy, p)

The same architecture is used for:
  - Supervised training (MSE against converged HFDIB fields)
  - Physics training (DAFoam HFDIB residual loss, no labels)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlowUNet(nn.Module):
    """U-Net for topology-to-flow prediction.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 = lambda only).
    out_channels : int
        Number of output channels (3 = ux, uy, p).
    base_filters : int
        Width of the first encoder layer.
    depth : int
        Number of encoder/decoder stages.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_filters: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        prev_ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(prev_ch, out_ch))
            prev_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottlenecks.append(ConvBlock(prev_ch, bottleneck_ch))
        prev_ch = bottleneck_ch

        # Decoder
        for i in range(depth):
            skip_ch = base_filters * (2 ** (depth - 1 - i))
            out_ch = base_filters * (2 ** (depth - 1 - i))
            self.upsamples.append(nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(out_ch + skip_ch, out_ch))
            prev_ch = out_ch

        # Output: 1x1 conv
        self.output = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

        # Optional Gaussian smoothing (initialized, not trained by default)
        self.use_smoothing = False
        k = 5
        sigma = 1.0
        ax = torch.arange(k) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("smooth_kernel", kernel.view(1, 1, k, k).expand(out_channels, 1, k, k).contiguous())
        self.smooth_conv = nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=k//2,
                                      groups=out_channels, bias=False, padding_mode="replicate")
        with torch.no_grad():
            self.smooth_conv.weight.copy_(self.smooth_kernel)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottlenecks[0](x)

        # Decoder
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = skips[self.depth - 1 - i]
            # Handle size mismatch (if input not power of 2)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        x = self.output(x)

        if self.use_smoothing:
            x = self.smooth_conv(x)

        return x
