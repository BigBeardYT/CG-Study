import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# 定义 LeNet-5 模型
from my_utils.load_datasets import get_datasets


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义自定义的损失函数
def custom_loss(outputs, targets, model, alpha=1.0, beta=1.0):
    # 鲁棒性损失：这里我们只使用交叉熵损失作为示例，但在实际应用中，应使用对抗损失
    robustness_loss = nn.CrossEntropyLoss()(outputs, targets)

    # 泛化性损失：为简单起见，我们使用模型的 L2 范数，但更复杂的度量也可以使用
    generalization_loss = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))

    # 总损失
    total_loss = alpha * robustness_loss + beta * generalization_loss
    return total_loss


batch_size = 32
data_name = 'MNIST'
# 加载数据集
train_datasets, test_datasets, train_loader, test_loader = get_datasets(batch_size=batch_size, data_name=data_name)
# 创建模型，优化器和设备
model = LeNet5()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        # _, pred = torch.max(outputs, 1)


        # 计算损失
        loss = custom_loss(outputs, labels, model)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
