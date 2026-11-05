"""Small randomly initialized ResNet18, using only the supplied input convention."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels)))

    def forward(self, images):
        residual = self.shortcut(images)
        output = self.relu(self.bn1(self.conv1(images)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class SmallResNet18(nn.Module):
    """C32 CIFAR stem, two basic blocks per stage, global pool and ten logits.

    Construction is random initialization only. Normalization is inside the
    model; callers supply unnormalized [0,1] RGB pixels, normally 64 by 64.
    """
    base_channels = 32

    def __init__(self, num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor((.4625, .4580, .4295)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((.2813, .2766, .3006)).view(1, 3, 1, 1))
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stages = nn.Sequential(
            nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32)),
            nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64)),
            nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128)),
            nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        output = self.stem((images - self.mean) / self.std)
        output = self.stages(output)
        return self.classifier(self.pool(output).flatten(1))
