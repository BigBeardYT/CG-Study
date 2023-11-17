""" 自定义双损失函数的训练方法 """
import torch
import torch.nn as nn
import torch.optim as optim
from my_utils.load_models import get_model
from my_utils.pgd import generate_pgd_noise
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def my_custom_loss_train(data_name, model_name, num_classes, train_loader, test_loader,
             batch_size, num_epochs, learning_rate, epsilon, start, end):
    save_name = data_name + '_' + model_name + '_GinzburgLandau'

    if data_name != 'CiFar10' and data_name != 'SVHN':
        in_features = 1
    else:
        in_features = 3

    # 加载模型
    model = get_model(model_name=model_name, in_features=in_features).to(device)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model.to(device)
    for i in range(start, end):

        print(f'第{i}次训练, 模型: {model_name}, 数据集: {data_name}')

        best_acc = 0.0
        for epoch in range(num_epochs):
            model.train()   # 转换成训练模式
            for idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                correct_pred = (pred == labels).sum()
                train_acc = correct_pred / labels.shape[0]
                # 计算损失
                loss = custom_loss(outputs, images, labels, model, epsilon=epsilon)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
                          f'Accuracy: {train_acc.item():.4f}')

            # 模型测试
            model.eval()
            with torch.no_grad():
                # 训练精度、训练损失以及测试的精度和损失
                correct_valid_pred = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    valid_outputs = model(images)
                    _, valid_pred = torch.max(valid_outputs, 1)
                    correct_valid_pred = correct_valid_pred + (valid_pred == labels).sum()

                valid_acc = correct_valid_pred / len(test_loader)
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} Train Acc: {train_acc.item():.2f}%'
                      f' | Validation Acc: {valid_acc:.2f}%')

            # if valid_acc > best_acc:
            print("Saving Model ... ")
            # best_acc = valid_acc
            # 存储模型
            best_model_params_path = 'D:/Python_CG_Project/Study_Stage/savemodel/' + save_name + '_bz' + str(
                batch_size) + '_ep' + str(num_epochs) \
                                     + '_lr' + str(learning_rate) + '_seedNone' + str(i) + '.pth'
            torch.save(model.state_dict(), best_model_params_path)

            # 动态更改学习率
            # if (epoch + 1) == (int)(num_epochs * 0.75) or (epoch + 1) == (int)(num_epochs * 0.90):
            #     for params_group in optimizer.param_groups:
            #         params_group['lr'] *= 0.1
            #         print('更改学习率为{}:'.format(params_group['lr']))


# 定义自定义的损失函数
def custom_loss(outputs, images, labels, model, epsilon, alpha=0.618, beta=0.382):
    """
    :param outputs: 输出
    :param images: 原始图像输入
    :param labels: 标签
    :param model: 模型
    :param epsilon: 对抗训练的扰动大小
    :param alpha: 对抗损失的参数
    :param beta: 泛化损失的参数
    :return:
    """
    # 鲁棒性损失：为简单起见，可以先使用模型的L2范数，或者更复杂的度量，但在实际应用中，应使用对抗损失
    # robustness_loss = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()]))
    # robustness_loss = nn.CrossEntropyLoss()(outputs, targets)
    adv_criterion = nn.CrossEntropyLoss()
    gen_criterion = nn.CrossEntropyLoss()

    # 生成PGD噪声 默认20步长攻击
    iters = generate_pgd_noise(model, images, labels, adv_criterion, device, epsilon, num_iter=20, minv=0, maxv=1)
    # 攻击之后的样本
    eta, adv_images = iters
    # 将对抗样本进行重新训练
    adv_outputs = model(adv_images)
    # 计算对抗损失
    robustness_loss = adv_criterion(adv_outputs, labels)
    # 泛化性损失：使用交叉熵损失作为示例
    generalization_loss = gen_criterion(outputs, labels)

    # 总损失
    total_loss = alpha * robustness_loss + beta * generalization_loss
    return total_loss


# 计算精确度和损失
def compute_accuracy_and_loss(model, data_loader, model_name, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)

        _, predicted_labels = torch.max(outputs, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


