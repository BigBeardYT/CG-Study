from custom_loss_train import my_custom_loss_train
from my_utils.load_datasets import get_datasets

# 训练配置
device = 'cuda'

# 加载模型
model_name = 'AlexNet'
# 数据集
data_name = 'CiFar10'
num_classes = 10
batch_size = 128
num_epochs = 100
lr = 0.01

train_datasets, test_datasets, train_loader, test_loader = get_datasets(batch_size=batch_size, data_name=data_name)


my_custom_loss_train(data_name, model_name, num_classes,
         train_loader, test_loader,
         batch_size, num_epochs, lr,
         1, 3)

