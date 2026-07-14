import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self, num_classes=2, dataset='MNIST'):
        super().__init__()
        self.dataset = dataset 
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        if self.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base = nn.Sequential(*list(base.children())[1:-1])
        else:   # For others
            self.base = nn.Sequential(*list(base.children())[:-1])
        
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        if self.dataset == 'MNIST':
            x = self.conv1(x)
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        return self.final(x)