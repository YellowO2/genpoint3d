"""Turn RGB frames into a 128-wide feature map, one cell per 4 pixels."""

import torch
import torch.nn.functional as F
from torch import nn

# Pixels per cell of the CNN's output, fixed by its two stride-2 stages.
STRIDE = 4


class _ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride), nn.InstanceNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """(M, 3, H, W) in [-1, 1] -> (M, 128, H / 4, W / 4).

    Four stages at strides 2, 4, 8, 16, all resampled to stride 4 and mixed by
    two convolutions. Module names and shapes are CoTracker3's `fnet`, so its
    weights load as they are. Instance norm has no weights and no running
    statistics: a frame's features do not depend on what else is in the batch,
    which is what lets `FineFinder` run it a clip at a time.
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        self.conv1 = nn.Conv2d(3, self.in_planes, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)
        self.conv2 = nn.Conv2d(output_dim * 3 + output_dim // 4, output_dim * 2, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, dim: int, stride: int) -> nn.Sequential:
        layers = (_ResidualBlock(self.in_planes, dim, stride), _ResidualBlock(dim, dim))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = (x.shape[-2] // STRIDE, x.shape[-1] // STRIDE)
        x = self.relu1(self.norm1(self.conv1(x)))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        up = lambda t: F.interpolate(t, size, mode="bilinear", align_corners=True)
        x = self.conv2(torch.cat([up(a), up(b), up(c), up(d)], dim=1))
        return self.conv3(self.relu2(self.norm2(x)))
