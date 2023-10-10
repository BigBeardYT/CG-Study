""" 根据函数传入的实参，返回相应的模型 """
import sys

sys.path.append("..")
from MonteCarloPooling.models.MC_LeNet import MC_LeNet
from IsingModel.IsingLayer import IsingLeNet
from Models.LeNet5 import LeNet5
from Models.alexnet import AlexNet
from Models.resnet import ResNet18


def get_model(model_name, in_features=1, num_classes=10):
    """ 传入模型名称，以及分类数 """
    if model_name == 'MC_LeNet':
        return MC_LeNet()
    if model_name == 'IsingLeNet':
        return IsingLeNet()
    if model_name == 'LeNet':
        return LeNet5()
    elif model_name == 'AlexNet':
        return AlexNet(num_classes=num_classes, in_features=in_features)
    elif model_name == 'ResNet18':
        return ResNet18()

    else:
        print("输入的模型有误!!!")
        return None
