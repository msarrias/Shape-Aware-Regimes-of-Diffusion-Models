import torch
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 3, grayscale: bool = False):
        """
        ResNet18-based classifier. Supports grayscale and RGB inputs.

        Args:
            num_classes: number of output classes
            grayscale:   if True, use a single-channel input conv (e.g. MNIST)
        """
        super().__init__()
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        if grayscale:
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.base = nn.Sequential(*list(base.children())[1:-1])
        else:
            self.conv1 = None
            self.base = nn.Sequential(*list(base.children())[:-1])

        in_features = base.fc.in_features
        self.drop   = nn.Dropout()
        self.final  = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv1 is not None:
            x = self.conv1(x)
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        return self.final(x)
        