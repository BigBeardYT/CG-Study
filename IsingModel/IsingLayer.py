import torch.nn as nn
import torch


class IsingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(IsingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        # 应用符号激活函数，使得输出为-1或1，加一个小小的偏置，保证torch.sign()永远不会返回0
        x = torch.sign(x + 1e-10)
        return x


class IsingLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(IsingLeNet, self).__init__()
        self.out_kernels_size = 5
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5)
        # Ising模型
        self.IsingConv1 = IsingLayer(32, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5)
        # Ising
        self.IsingConv2 = IsingLayer(128, 128, 3, 1, 1)

        # 定义全连接层
        self.fc1 = nn.Linear(128 * self.out_kernels_size * self.out_kernels_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # 定义激活函数
        self.relu = nn.ReLU()

        # 定义池化层
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # 卷积层1 (1, 1, 32, 32) -> (1, 6, 14, 14)
        x = self.conv1(x)
        x = self.IsingConv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 卷积层2 (1, 6, 14, 14) -> (1, 16, 5, 5)
        x = self.conv2(x)
        x = self.IsingConv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展开
        x = x.view(-1, 128 * self.out_kernels_size * self.out_kernels_size)

        # 全连接层1
        x = self.fc1(x)
        x = self.relu(x)

        # 全连接层2
        x = self.fc2(x)
        x = self.relu(x)

        # 输出层
        x = self.fc3(x)

        return x


# net = IsingLeNet()
# x = torch.rand(128, 1, 28, 28)
# y = net(x)
# print(y.shape)




