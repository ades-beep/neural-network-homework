import torch
import torch.nn as nn
from torchvision import models
from typing import cast
from torchvision.models import AlexNet_Weights, VGG19_Weights, ResNet50_Weights

class mlp1(nn.Module):
    def __init__(self):
        super().__init__()
        # 把Sequential放进类的成员变量
        self.net = nn.Sequential(
            nn.Flatten(), #
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 前向传播：数据流过网络
        return self.net(x)

class mlp2(nn.Module):
    def __init__(self):
        super().__init__()
        # 把Sequential放进类的成员变量
        self.net = nn.Sequential(
            nn.Flatten(), #
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # 前向传播：数据流过网络
        return self.net(x)

class mlp3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 896),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(896, 768),
            nn.ReLU(),

            nn.Linear(768, 640),
            nn.ReLU(),


            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 384),
            nn.ReLU(),

            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, pretrain=False):
        super().__init__()
        if pretrain:
            self.net = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        else:
            self.net = models.alexnet(weights=None)
        # 替换最后一层，适配分类类别
        last_layer = cast(nn.Linear, self.net.classifier[-1])
        in_feature = last_layer.in_features
        self.net.classifier[-1] = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        return self.net(x)

class VGG19(nn.Module):
    def __init__(self, num_classes=10, pretrain=False):
        super().__init__()
        if pretrain:
            self.net = models.vgg19(weights=VGG19_Weights.DEFAULT)
        else:
            self.net = models.vgg19(weights=None)
        last_layer = cast(nn.Linear, self.net.classifier[-1])
        in_feature = last_layer.in_features
        self.net.classifier[-1] = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        return self.net(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrain=False):
        super().__init__()
        if pretrain:
            self.net = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.net = models.resnet50(weights=None)
        in_feature = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        return self.net(x)

    