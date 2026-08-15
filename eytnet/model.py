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
    def __init__(self, num_classes=2, anchors_per_scale=3, channels=(32,64,128,256,512)):

        super().__init__()

        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.num_outputs = 5 + num_classes

        c1, c2, c3,c4,c5 = channels

        def stage(in_ch, out_ch):
            return nn.Sequential(FireFeatureBlock(in_ch, out_ch, stride=2), FireFeatureBlock(out_ch, out_ch))

        self.stem = FireFeatureBlock(3, c1, stride=2)
        self.stage1 = stage(c1,c2)
        self.stage2 = stage(c2,c3)
        self.stage3 = stage(c3,c4)
        self.stage4 = stage(c4,c5)

        self.p5_reduce = FireFeatureBlock(c5,c4, kernel_size=1)
        self.p5_out = FireFeatureBlock(c4, c4)
        self.upsample = nn.Upsample(scale_factor = 2, mode="nearest")
        self.p4_fuse = nn.Sequential(FireFeatureBlock(2*c4, c4), FireFeatureBlock(c4, c4))

        head_channels = anchors_per_scale*self.num_outputs
        self.head_p4 = nn.Conv2d(c4, head_channels, kernel_size=1)
        self.head_p5 = nn.Conv2d(c4, head_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
        with torch.no_grad():
            for head in (self.head_p4, self.head_p5):
                head.bias.view(self.anchors_per_scale, self.num_outputs)[:,4] = -4.0

    def _reshape(self,raw):
        b, _, h, w = raw.shape

        raw = raw.view(b, self.anchors_per_scale, self.num_outputs, h,w)
        return raw.permute(0, 1,3,4,2).contiguous()

    def forward(self, x):
        x = self.stage2(self.stage1(self.stem(x)))
        p4 = self.stage3(x)
        p5 = self.stage4(p4)

        p5r = self.p5_reduce(p5)
        out_p5 = self.head_p5(self.p5_out(p5r))
        fused = self.p4_fuse(torch.cat([self.upsample(p5r), p4], dim=1))
        out_p4 = self.head_p4(fused)

        return [self._reshape(out_p4), self._reshape(out_p5)]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def build_model(config) -> EYTNet:
    return EYTNet(num_classes=config.num_classes, anchors_per_scale=config.anchors_per_scale,
                  channels=tuple(getattr(config, "channels", (32,64,128,256,512))))
        