import sys
sys.path.append("..")
import torch
from my_utils.load_models import get_model
from my_utils.load_datasets import get_datasets


device = 'cuda'
# 数据集
Mnist_data_name = 'Mnist'
KMnist_data_name = 'KMnist'
CiFar10_data_name = 'CiFar10'
Svhn_data_name = 'SVHN'

# 模型
Ising_LeNet_model_name = 'IsingLeNet'


# 预测，加载对应数据集 -> 加载模型 -> 预测
""" ######## 以下参数 手动设置 ######### """
batch_size = 128
num_epochs = 10
lr = 0.01
# 数据集
data_name = Mnist_data_name
# 模型
model_name = Ising_LeNet_model_name
""" ######## 以上参数 手动设置 ######### """
num_classes = 10

# 加载数据集
train_dataset, test_dataset, \
    train_loader, test_loader = \
    get_datasets(batch_size=128, data_name=data_name)

# 模型路径
for i in range(1, 3):
    print('第{}次预测, 模型: {}, 数据集: {}'.format(i, model_name, data_name))
    in_features = 1
    if data_name == 'CiFar10' or data_name == 'SVHN':
        in_features = 3

    # 加载模型
    predicted_model = get_model(model_name, in_features=in_features, num_classes=num_classes).to(device)

    # 对抗训练的路径
    # attacked_model_params_path = '../trained_model/' + data_name + '_' + model_name \
    #                              + '_PGD' + '_train' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
    #                              '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'

    # 正常训练的路径
    attacked_model_params_path = '../savemodel/' + data_name + '_' + model_name \
                                 + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                                 '_lr' + str(lr) + '_seedNone' + str(i) + '.pth'
    print('模型参数所在位置: {}'.format(attacked_model_params_path))

    # 加载参数
    predicted_model.load_state_dict(torch.load(attacked_model_params_path))

    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # 原始预测输出
        outputs = predicted_model(images)
        # 预测结果
        _, pred = torch.max(outputs, 1)
        total_samples += outputs.shape[0]
        total_correct += torch.sum(pred == labels)

    print('预测准确率为: {} / {} = {:.2f}'.format(total_correct, total_samples, (total_correct / total_samples)*100))


