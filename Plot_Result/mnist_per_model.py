import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimSun'

epsilon = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]
GL_lenet5_mnist1 = [98.69, 98.33, 97.80, 96.60, 95.59, 94.53]
GL_lenet5_mnist2 = [98.76, 98.48, 97.96, 96.20, 95.19, 93.78]
GL_lenet5_mnist3 = [98.90, 98.68, 98.27, 97.18, 96.31, 95.15]

plt.plot(epsilon, GL_lenet5_mnist1, 'green')
plt.plot(epsilon, GL_lenet5_mnist2, 'green')
plt.plot(epsilon, GL_lenet5_mnist3, 'green')

NM_lenet5_mnist1 = [99.71, 99.59, 99.34, 98.31, 97.23, 96.12]
NM_lenet5_mnist2 = [99.65, 99.43, 99.22, 98.14, 97.08, 95.34]
NM_lenet5_mnist3 = [99.82, 99.68, 99.54, 98.39, 97.33, 96.21]
plt.plot(epsilon, NM_lenet5_mnist1, 'red')
plt.plot(epsilon, NM_lenet5_mnist2, 'red')
plt.plot(epsilon, NM_lenet5_mnist3, 'red')

plt.title("MNIST-LeNet5-红色代表正常训练+对抗训练，绿色代表同时正常训练和对抗训练")
plt.xlabel('epsilon')
plt.ylabel('accuracy')


plt.show()


