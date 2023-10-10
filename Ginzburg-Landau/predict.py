import torch
from LeNet5 import LeNet5
from my_utils.load_datasets import get_datasets
batch_size = 32
num_epochs = 5
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_name = 'MNIST'
train_dataset, test_dataset, train_loader, test_loader = get_datasets(batch_size=batch_size, data_name=data_name)

for i in range(1, 3):
    # 读取模型
    model_path = '../savemodel/' + data_name + '_GinzburgLandau' + '_bz' + str(batch_size) + '_ep' + str(num_epochs) + \
                 '_lr' + str(learning_rate) + '_seedNone' + str(i) + '.pth'

    model = LeNet5().to(device)
    model.load_state_dict(torch.load(model_path))

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        correct_pred = torch.sum(pred == labels)

        acc = correct_pred / labels.shape[0]
    print(f'第{i}次预测, 路径:{model_path}, 准确率: {acc.item():.4f}')



