import sys

import torch

from my_utils.fgsm import generate_fgsm_noise
from my_utils.load_models import get_model
from my_utils.pgd import generate_pgd_noise, generate_bim_noise
import torch.nn as nn
sys.path.append("..")
from my_utils.load_datasets import get_datasets
from models.LeNet5 import LeNet5
# 导包，自定义的攻击函数


""" ######## 以下参数训练之前手动设置 ######### """
attacked_batch_size = 128
attacked_num_epochs = 30
lr = 0.01
data_name = 'MNIST'
model_name = 'LeNet5'
num_classes = 10
""" ######## 以上参数训练之前手动设置 ######### """
# 第2次攻击, 模型: ResNet18, 数据集: CiFar10, 攻击方式: PGD
# 模型参数所在位置
# D:/Python_CG_Project/Study_Stage/savemodel/CiFar10_ResNet18_GinzburgLandau_bz128_ep30_lr0.01_seedNone2.pth
# Epsilon: 0.01	Test Accuracy = 6875 / 10016 = 68.64
# Epsilon: 0.05	Test Accuracy = 5772 / 10016 = 57.63

# 攻击时使用的数据集大小
attack_used_batch_size = 32

# 加载数据集
train_dataset, test_dataset, \
    train_loader, test_loader = \
    get_datasets(batch_size=attack_used_batch_size, data_name=data_name)
epsilons = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]
in_features = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
num_epochs = 5
# 攻击方法
noise_name = 'PGD'
for i in range(1, 3):
    # 模型对抗攻击
    print('第{}次攻击, 模型: {}, 数据集: {}, 攻击方式: {}'.format(i, model_name, data_name, noise_name))
    # 加载模型
    attacked_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)
    # attacked_model = LeNet5().to(device)
    # 普通模型直接存储的路径
    # attacked_model_params_path = 'D:/Python_CG_Project/Study_Stage/savemodel/' + data_name + '_' + model_name + '_' + \
    #                              'GinzburgLandau' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
    #                              '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'

    # 对抗样本训练后的路径
    attacked_model_params_path = './trained_model/' + data_name + '_' + model_name + '_' + 'PGD' \
                                 + '_train' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                                 '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'

    # attacked_model_params_path = '../trades_trained_model/CiFar10_ResNet_Trades_train_bz128_ep100_lr0
    # .01_seedNone1.pth'

    print('模型参数所在位置\n' + attacked_model_params_path)

    # 加载参数
    attacked_model.load_state_dict(torch.load(attacked_model_params_path))

    """ 扰动攻击 """
    for epsilon in epsilons:

        criterion = nn.CrossEntropyLoss()
        total_correct = 0
        total_samples = 0
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)
            # 原始模型
            init_outputs = attacked_model(images)
            # 预测结果
            _, pred = torch.max(init_outputs, 1)

            total_samples += init_outputs.shape[0]
            # 生成PGD噪声 默认20步长攻击
            if noise_name == 'PGD':
                iters = generate_pgd_noise(attacked_model, images, labels, criterion, device,
                                           epsilon=epsilon, num_iter=20, minv=0, maxv=1)
            # BIM噪声
            elif noise_name == 'BIM':
                iters = generate_bim_noise(attacked_model, images, labels, criterion, device,
                                           epsilon=epsilon, iters=5, minv=0, maxv=1)
            # FGSM攻击噪声
            elif noise_name == 'FGSM':
                iters = generate_fgsm_noise(attacked_model, images, labels, criterion, device,
                                            epsilon=epsilon, minv=0, maxv=1)

            eta, adv_images = iters

            # 攻击后的图片的预测结果
            final_outputs = attacked_model(adv_images)
            _, final_preds = torch.max(final_outputs, 1)

            final_preds_list = final_preds.tolist()
            src_labels = labels.tolist()
            for j in range(len(final_preds_list)):
                if final_preds_list[j] == src_labels[j]:
                    total_correct += 1

        final_acc = total_correct / float(len(test_loader) * attack_used_batch_size)  # 计算整体准确率
        print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(
            epsilon, total_correct, len(test_loader) * attack_used_batch_size, final_acc * 100))

