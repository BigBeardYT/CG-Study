import torch
import numpy as np


# 传入混淆矩阵用于计算多个类别情况的四项指标
from scipy.optimize._lsq.common import EPS


def calculate_multiclass_convnet_metrics(self, confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    num_classes = confusion_matrix.shape[0]
    total = np.sum(confusion_matrix)
    tp = np.diag(confusion_matrix)  # 真阳性
    fp = np.sum(confusion_matrix, axis=0) - tp  # 假阳性
    fn = np.sum(confusion_matrix, axis=1) - tp  # 假阴性
    tn = total - tp - fp - fn  # 真阴性
    # 准确率
    accuracy = np.sum(tp) / total
    # 精确率
    precision = np.sum(tp / (np.sum(confusion_matrix, axis=0) + EPS)) / num_classes
    # 灵敏度，也叫作召回率
    recall = np.sum(tp / (np.sum(confusion_matrix, axis=1) + EPS)) / num_classes
    # 三级指标，统计学中的F1-score F1-Score的取值范围（0~1），越接近1说明模型预测效果越好。
    f1_score = 2 * precision * recall / (precision + recall + EPS)

    return accuracy * 100, precision * 100, recall * 100, f1_score * 100


def calculate_confusion_mtx(self, eval_loader):
    # 使用同义引用，用于简化代码的逻辑
    args, model, _, _, criterion, _, device = self.setting_list

    model.eval()
    total, avg_loss = 0, 0
    confusion_mtx = np.array([[0] * args.num_class for _ in range(args.num_class)], dtype=np.int32)  # 创建一个混淆矩阵

    with torch.no_grad():
        for i, (images, labels) in enumerate(eval_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_mtx[t, p] += 1

            avg_loss += loss.item()

    avg_loss /= total
    return avg_loss, confusion_mtx


