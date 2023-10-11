import torch
from models.LeNet5 import LeNet5
from Models.resnet import ResNet18
from my_utils.load_datasets import get_datasets


batch_size = 128
num_epochs = 100
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_name = 'CiFar10'
model_name = 'ResNet18'
train_dataset, test_dataset, train_loader, test_loader = get_datasets(batch_size=batch_size, data_name=data_name)

for i in range(1, 2):
    # 读取模型
    predicted_model = ResNet18().to(device)
    model_path = 'D:/Python_CG_Project/Study_Stage/savemodel/' + data_name + '_' + model_name + '_GinzburgLandau' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                 '_lr' + str(learning_rate) + '_seedNone' + str(i) + '.pth'
    # 加载参数
    predicted_model.load_state_dict(torch.load(model_path))

    total_correct = 0
    total_samples = 0

    predicted_model.load_state_dict(torch.load(model_path))
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # 原始预测输出
        outputs = predicted_model(images)
        # 预测结果
        _, pred = torch.max(outputs, 1)
        total_samples += outputs.shape[0]
        total_correct += torch.sum(pred == labels)

    print('第{}次预测, 模型路径: {}, 预测准确率: {} / {} = {:.2f}'.format(
        i, model_path, total_correct, total_samples, (total_correct / total_samples)*100))




