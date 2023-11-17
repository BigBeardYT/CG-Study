import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimSun'

epsilon = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]
GL_resnet18_cifar10_1 = [72.31, 60.10, 46.86, 27.78, 23.37, 20.42]
# GL_resnet18_cifar10_2 = [98.76, 98.48, 97.96, 96.20, 95.19, 93.78]
# GL_resnet18_cifar10_3 = [98.90, 98.68, 98.27, 97.18, 96.31, 95.15]

plt.plot(epsilon, GL_resnet18_cifar10_1, 'green', label='Ginzburg_Landau_Train')
# plt.plot(epsilon, GL_resnet18_cifar10_2, 'green')
# plt.plot(epsilon, GL_resnet18_cifar10_3, 'green')

NM_lenet5_cifar10_1 = [99.55, 98.93, 93.13, 72.47, 60.47, 48.04]
# NM_lenet5_cifar10_2 = [99.65, 99.43, 99.22, 98.14, 97.08, 95.34]
# NM_lenet5_cifar10_3 = [99.82, 99.68, 99.54, 98.39, 97.33, 96.21]
plt.plot(epsilon, NM_lenet5_cifar10_1, 'red', label='Noram_And_Adversarial_Train')
# plt.plot(epsilon, NM_lenet5_cifar10_2, 'red')
# plt.plot(epsilon, NM_lenet5_cifar10_3, 'red')

plt.title("ResNet18-CiFar10-红色代表正常训练+对抗训练，绿色代表同时正常训练和对抗训练")
plt.xlabel('epsilon')
plt.ylabel('accuracy')
plt.legend()

plt.show()


