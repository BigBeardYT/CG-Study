import torch.nn as nn
import torch


# 定义AlexNet网络模型
class AlexNet(nn.Module):
    def __init__(self,  in_features=1, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # (1, 3, 32, 32) -> (32-11+10)/3 = 10 + 1 -- 11 --> (1, 64, 8, 8)
            nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=5),
            nn.ReLU(inplace=True),
            # (1, 64, 5, 5)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (1, 192, 4, 4)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # (1, 192, 2, 2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.rg = RenormalizationGroup(8, 256)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = self.rg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

