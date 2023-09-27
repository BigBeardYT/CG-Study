import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloPooling(nn.Module):
    def __init__(self, pool_size=(2, 2)):
        super(MonteCarloPooling, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        # 获取输入尺寸
        batch_size, channels, height, width = x.size()

        # 选择要采样的块
        pooled_height = height // self.pool_size[0]
        pooled_width = width // self.pool_size[1]

        # 初始化输出
        pooled = torch.zeros((batch_size, channels, pooled_height, pooled_width)).to(x.device)

        # 对输入进行采样
        for i in range(pooled_height):
            for j in range(pooled_width):
                block = x[:, :, i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                        j * self.pool_size[1]:(j + 1) * self.pool_size[1]]
                block_flat = block.reshape(batch_size, channels, -1)
                selected = torch.multinomial(block_flat, 1).squeeze()
                pooled[:, :, i, j] = selected

        return pooled


class MC_LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(MC_LeNet, self).__init__()
        self.out_kernels_size = 4
        if in_channels == 3:
            self.out_kernels_size = 5
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5)

        # 定义全连接层
        self.fc1 = nn.Linear(128 * self.out_kernels_size * self.out_kernels_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # 定义激活函数
        self.relu = nn.ReLU()

        # 定义池化层
        self.pool = MonteCarloPooling((2, 2))

    def forward(self, x):
        # 卷积层1 (1, 1, 32, 32) -> (1, 6, 14, 14)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 卷积层2 (1, 6, 14, 14) -> (1, 16, 5, 5)
        x = self.conv2(x)
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


x = torch.rand(2, 1, 28, 28)
net = MC_LeNet()
y = net(x)
print(y)

