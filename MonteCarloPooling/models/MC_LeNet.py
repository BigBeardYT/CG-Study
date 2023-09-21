import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloPooling(nn.Module):
    def __init__(self, pool_size=(2, 2)):
        super(MonteCarloPooling, self).__init__()
        self.pool_size = pool_size  # 设置Monte Carlo池化的池大小

    def forward(self, inputs):
        # 获取输入张量的形状（批大小，通道数，高度，宽度）
        shape = inputs.shape

        # 初始化一个空的张量，用于存放池化后的输出
        pooled = torch.zeros((shape[0], shape[1], shape[2] // self.pool_size[0], shape[3] // self.pool_size[1]))

        # 按照pool_size的步长遍历输入张量
        for i in range(0, shape[2], self.pool_size[0]):
            for j in range(0, shape[3], self.pool_size[1]):
                # 从输入张量中提取出大小为(pool_size[0], pool_size[1])的块
                block = inputs[:, :, i:i+self.pool_size[0], j:j+self.pool_size[1]]

                # 对这个块进行Monte Carlo采样
                # 随机从每个块中选择一个元素，对每个通道和批中的每个样本进行此操作
                # 注意：因为PyTorch没有torch.choice，我们使用torch.randint进行随机索引
                rand_idx = torch.randint(low=0, high=block.numel() // shape[0], size=(shape[1],), dtype=torch.long)
                block_flat = block.reshape(shape[0], -1)
                pooled[:, :, i // self.pool_size[0], j // self.pool_size[1]] = block_flat[torch.arange(shape[1]), rand_idx]

        # 最终的池化张量
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


