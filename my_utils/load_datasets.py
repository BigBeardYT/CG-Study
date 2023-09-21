import torch
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")


# Mnist数据集的数据优化
transform_Mnist_train_data = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_Mnist_test_data = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 其他
transform_EKQMnist_data = transforms.Compose([transforms.ToTensor(), ])

# FahsionMnist数据集的数据初始化
transform_fashionMnist_train_data = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_fashionMnist_test_data = transforms.Compose([
    transforms.ToTensor()
])

# cifar10数据集的预处理
transform_CiFar10_train_data = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 在四周填充0，在把图片随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
])
transform_CiFar10_test_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

norm_mean = 0
norm_var = 1
transform_svhn_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])

transform_svhn_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])


# 加载数据
def get_datasets(batch_size=32, data_name='MNIST'):
    if data_name == 'EMNIST' or data_name == 'EMnist':
        # 加载EMNIST数据集
        train_dataset = datasets.EMNIST(root='E:\\Python_Code\\data\\emnist',
                                        split='letters', train=True,
                                        transform=transform_EKQMnist_data,
                                        download=True)
        test_dataset = datasets.EMNIST(root='E:\\Python_Code\\data\\emnist',
                                       split='letters', train=False,
                                       transform=transform_EKQMnist_data,
                                       download=True)

    elif data_name == 'MNIST' or data_name == 'Mnist':
        # 加载MNIST数据集
        train_dataset = datasets.MNIST(root='E:\\Python_Code\\data\\mnist',
                                       train=True,
                                       transform=transform_Mnist_train_data,
                                       download=True)
        test_dataset = datasets.MNIST(root='E:\\Python_Code\\data\\mnist',
                                      train=False,
                                      transform=transform_Mnist_test_data,
                                      download=True)

    elif data_name == 'KMNIST' or data_name == 'KMnist':
        # 加载KMNIST数据集
        train_dataset = datasets.KMNIST(root='E:\\Python_Code\\data\\kmnist',
                                        train=True,
                                        transform=transform_EKQMnist_data,
                                        download=True)
        test_dataset = datasets.KMNIST(root='E:\\Python_Code\\data\\kmnist',
                                       train=False,
                                       transform=transform_EKQMnist_data,
                                       download=True)

    elif data_name == 'QMNIST' or data_name == 'QMnist':
        # 加载QMNIST数据集
        train_dataset = datasets.QMNIST(root='/data/home/scy0467/dataset/qmnist',
                                        train=True,
                                        transform=transform_EKQMnist_data,
                                        download=True)
        test_dataset = datasets.QMNIST(root='/data/home/scy0467/dataset/qmnist',
                                       train=True,
                                       transform=transform_EKQMnist_data,
                                       download=True)
    elif data_name == 'FashionMnist':
        # 加载FashionMnist数据集
        train_dataset = datasets.FashionMNIST(root='/data/home/scy0467/dataset/fashionmnist',
                                              train=True,
                                              transform=transform_fashionMnist_train_data,
                                              download=True)
        test_dataset = datasets.FashionMNIST(root='/data/home/scy0467/dataset/fashionmnist',
                                             train=False,
                                             transform=transform_fashionMnist_test_data,
                                             download=True)
    elif data_name == 'CiFar10':
        # 加载CIFAR10数据集
        train_dataset = datasets.CIFAR10(root='E:\\Python_Code\\data\\cifar10',
                                         train=True,
                                         transform=transform_CiFar10_train_data,
                                         download=False)
        test_dataset = datasets.CIFAR10(root='E:\\Python_Code\\data\\cifar10',
                                        train=False,
                                        transform=transform_CiFar10_test_data,
                                        download=False)
    elif data_name == 'SVHN':
        # 加载SVHN
        train_dataset = datasets.SVHN(root="E:\\Python_Code\\data\\svhn",
                                      split='train',
                                      transform=transform_svhn_train,
                                      download=False)
        test_dataset = datasets.SVHN(root="E:\\Python_Code\\data\\svhn",
                                     split='test',
                                     transform=transform_svhn_test,
                                     download=False)
    else:
        print('数据提取错误，检查拼写是否正确！')
        return None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader


# train_dataset, test_dataset, train_loader, test_loader = load_datasets(32, 'EMNIST')


