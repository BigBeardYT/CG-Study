import sys
sys.path.append("..")
from my_utils.load_datasets import get_datasets
# 导包，自定义的攻击函数
from my_utils.adversarial_train import adversarial_noise_train


""" ######## 以下参数训练之前手动设置 ######### """
attacked_batch_size = 32
attacked_num_epochs = 5
lr = 0.01
data_name = 'MNIST'
model_name = 'LeNet5'
num_classes = 10
""" ######## 以上参数训练之前手动设置 ######### """

# 攻击时使用的数据集大小
attack_used_batch_size = 32

# 加载数据集
train_dataset, test_dataset, \
train_loader, test_loader = \
    get_datasets(batch_size=attack_used_batch_size, data_name=data_name)

# 攻击方法
noise_name = 'PGD'

adversarial_noise_train(noise_name, data_name, model_name,
                        test_loader, num_classes, lr,
                        attacked_batch_size, attacked_num_epochs,
                        1, 3)
