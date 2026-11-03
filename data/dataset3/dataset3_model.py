"""The classifier provided to students for Dataset 3 (COMP5577 project).

Load it with:
    import torch
    from dataset3_model import ProvidedNet
    model = ProvidedNet()
    model.load_state_dict(torch.load('provided_model.pt', map_location='cpu'))
    model.eval()

Input: a float tensor of shape (N, 3, 64, 64) with pixel values in [0, 1] (just divide the PNG pixels by 255).
The model normalises its input internally. Output: 10 class scores (logits).
"""
import torch
import torch.nn as nn

MEAN = (0.4625, 0.4580, 0.4295)
STD = (0.2813, 0.2766, 0.3006)


class ProvidedNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))  # 64 -> 4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features((x - self.mean) / self.std))
