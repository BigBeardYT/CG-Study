""" 根据函数传入的实参，返回相应的模型 """
import sys

sys.path.append("..")
from MonteCarloPooling.models.MC_LeNet import MC_LeNet
from IsingModel.IsingLayer import IsingLeNet


def get_model(model_name, in_features=1, num_classes=10):
    """ 传入模型名称，以及分类数 """
    if model_name == 'MC_LeNet':
        return MC_LeNet()
    if model_name == 'IsingLeNet':
        return IsingLeNet()

    else:
        print("输入的模型有误!!!")
        return None
