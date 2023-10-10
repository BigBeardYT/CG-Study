import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# 定义 LeNet-5 模型
from my_utils.load_datasets import get_datasets
from LeNet5 import LeNet5


# 定义自定义的损失函数
def custom_loss(outputs, targets, model, alpha=1.0, beta=1.0):
    # 鲁棒性损失：为简单起见，我们使用模型的 L2 范数，但更复杂的度量也可以使用，但在实际应用中，应使用对抗损失
    # robustness_loss = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))
    robustness_loss = nn.CrossEntropyLoss()(outputs, targets)

    # 泛化性损失：使用交叉熵损失作为示例
    generalization_loss = nn.CrossEntropyLoss()(outputs, targets)

    # 总损失
    total_loss = alpha * robustness_loss + beta * generalization_loss
    return total_loss


batch_size = 32
num_epochs = 5
learning_rate = 0.01
data_name = 'MNIST'
# 加载数据集
train_datasets, test_datasets, train_loader, test_loader = get_datasets(batch_size=batch_size, data_name=data_name)
# 创建模型，优化器和设备
model = LeNet5()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 训练

best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        correct_pred = (pred == labels).sum()
        acc = correct_pred / labels.shape[0]
        # 计算损失
        loss = custom_loss(outputs, labels, model)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc.item():.4f}')

    # 模型测试
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct_pred = torch.sum(pred == labels)
            acc = correct_pred / labels.shape[0]

        print('Epoch: [{}/{}], Test_Acc: {:.4f}'.format(epoch+1, num_epochs, acc.item()))
        save_name = 'MNIST_GinzburgLandau'
        if acc.item() > best_acc and epoch > num_epochs*0.5:
            best_acc = acc.item()
            # 存储模型
            best_model_params_path = '../savemodel/' + save_name + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                                     '_lr' + str(learning_rate) + 'seedNone2' + '.pth'
            torch.save(model.state_dict(), best_model_params_path)


