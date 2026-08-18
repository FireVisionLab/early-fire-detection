from __future__ import annotations

import torch
import torch.nn as nn

class FireFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class EYTNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        anchors_per_scale=3,
        channels=(32, 64, 128, 256, 512),
        kernel_size=3,
        blocks_per_stage=2,
        downsample_mode="conv",
        upsample_mode="nearest",
    ):

        super().__init__()

        if len(channels) != 5:
            raise ValueError("channels 5 değer içermeli.")
        if kernel_size not in (3,5):
            raise ValueError("kernel_size 3 veya 5 olmalı.")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage 1 veya daha büyük olmalı.")

        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError("upsample_mode 'nearest' veya 'bilinear' olmalı.")
        

        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.num_outputs = 5 + num_classes

        c1, c2, c3,c4,c5 = channels

        def stage(in_channels, out_channels):
            if downsample_mode == "conv":
                layers = [
                    FireFeatureBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=2,
                    )
                ]
            else:
                layers = [
                    nn.MaxPool2d(2, 2),
                    FireFeatureBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                    ),
                ]

            for _ in range(blocks_per_stage - 1):
                layers.append(
                    FireFeatureBlock(
                        out_channels,
                        out_channels,
                        kernel_size,
                    )
                )

            return nn.Sequential(*layers)

        self.stem = FireFeatureBlock(3, c1, kernel_size, stride=2)
        self.stage1 = stage(c1,c2)
        self.stage2 = stage(c2,c3)
        self.stage3 = stage(c3,c4)
        self.stage4 = stage(c4,c5)

        self.p5_reduce = FireFeatureBlock(c5,c4, kernel_size=1)
        self.p5_out = FireFeatureBlock(c4, c4, kernel_size,)
        if upsample_mode == "nearest":
            self.upsample = nn.Upsample(
                    scale_factor=2,
                    mode="nearest",
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        self.p4_fuse = nn.Sequential(FireFeatureBlock(2*c4, c4, kernel_size), FireFeatureBlock(c4, c4, kernel_size))

        head_channels = anchors_per_scale*self.num_outputs
        self.head_p4 = nn.Conv2d(c4, head_channels, kernel_size=1)
        self.head_p5 = nn.Conv2d(c4, head_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,
                    nonlinearity="leaky_relu",
                )

        with torch.no_grad():
            for head in (
                self.head_p4,
                self.head_p5,
            ):
                nn.init.normal_(
                    head.weight,
                    std=0.01,
                )
                head.bias.zero_()

                head.bias.view(
                    self.anchors_per_scale,
                    self.num_outputs,
                )[:, 4] = -4.0

    def _reshape(self, raw):
            batch, _, height, width = raw.shape

            raw = raw.view(
                batch,
                self.anchors_per_scale,
                self.num_outputs,
                height,
                width,
            )

            return raw.permute(
                0,
                1,
                3,
                4,
                2,
            ).contiguous()

    def forward(self, x):
        x = self.stage2(
            self.stage1(
                self.stem(x)
            )
        )

        p4 = self.stage3(x)
        p5 = self.stage4(p4)

        p5_reduced = self.p5_reduce(p5)

        out_p5 = self.head_p5(
            self.p5_out(p5_reduced)
        )

        fused = torch.cat([self.upsample(p5_reduced), p4], dim=1)
        out_p4 = self.head_p4(self.p4_fuse(fused))

        return [self._reshape(out_p4), self._reshape(out_p5)]

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

def build_model(config):
    return EYTNet(
        num_classes=config.num_classes,
        anchors_per_scale=config.anchors_per_scale,
        channels=tuple(
            getattr(
                config,
                "channels",
                (32, 64, 128, 256, 512),
            )
        ),
        kernel_size=int(
            getattr(config, "kernel_size", 3)
        ),
        blocks_per_stage=int(
            getattr(config, "blocks_per_stage", 2)
        ),
        downsample_mode=str(
            getattr(config, "downsample_mode", "conv")
        ),
        upsample_mode=str(
            getattr(config, "upsample_mode", "nearest")
        ),
    )
        