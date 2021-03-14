import numpy as np
import random


# ReLU激活函数
def relu(z):
    z[z < 0] = 0
    return z


# ReLU激活函数的导数
def d_relu(z):
    z[z > 0] = 1
    z[z <= 0] = 0
    return z


# softmax函数
def softmax(z):
    t = np.exp(z)
    a = np.nan_to_num(np.exp(z) / np.sum(t))
    return a


# 神经网络的类
class Network(object):
    # 构造函数初始化网络
    def __init__(self, sizes):   # sizes(784, 15, 10) 输入784 隐层15 输出10
        # 神经网络层数
        self.layer_nums = len(sizes)
        self.sizes = sizes
        # randn(j, i) 可以生成y行x列的随机数矩阵，是均值为0,标准差为1的高斯分布
        # bias向量维度由下一层神经元个数决定 15*1 10*1
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        # weights矩阵维度由输入层和下一层共同决定，15*784 10*15 w^T方便运算
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]  # [784 15] [15,10]

    # 前向传播
    # z1=W1^T*x+B1 a1 = relu(z1) z2=W2^T*a1+B2 a2= softmax(z2) a2=y^
    def forward(self, a):
        # 中间隐藏层的激活函数选择relu，输出层的激活函数为softmax
        num_forward = 1
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a)+b
            if num_forward < (self.layer_nums - 1):  # relu激活
                a = relu(z)
                num_forward = num_forward + 1
            else:
                a = softmax(z)
        return a

    # 随机梯度下降(训练数据,迭代次数,batch大小,学习率,是否有测试集)
    def SGD(self, train_data, epochs, mini_batch_size, learning_rate, test_data=None):
        # 迭代过程
        print("Training.........")
        n = len(train_data)  # 训练数据大小
        for j in range(epochs):
            # 打乱训练集
            random.shuffle(train_data)
            # mini_batches是分批后的mini_batch列表
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 每个mini_batch都更新一次,重复整个数据集
            self.update_mini_batch(mini_batches, learning_rate)
            # 若有测试数据,则在屏幕上打印训练进度
            if test_data:
                len_test = len(test_data)
                correct_num = self.evaluate(test_data)
                print("Epoch{0}:{1}/{2}".format(j, correct_num, len_test))
            else:
                print("Epoch {0} complete:".format(j))

    # 更新mini_batch
    def update_mini_batch(self, mini_batches, learning_rate):
        for mini_batch in mini_batches:
            # 存储对于各个参数的偏导，格式和self.biases和self.weights是一样的
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            eta = learning_rate / len(mini_batch)
            # mini_batch中的一个实例调用梯度下降得到各个参数的偏导  mini_batch(x,y)元组
            for x, y in mini_batch:
                # 从一个实例得到的梯度
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 依次取出mini_batch中的（x,y)输入backprop
                # nabla_w,nabla_b 表示整个mini_batch 所有训练样本的总代价函数梯度
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 每一个mini_batch更新一下参数
            self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]
            self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]

    # 反向传播(对于每一个实例)
    def backprop(self, x, y):
        # 生成权重矩阵形状和偏置矩阵形状的零矩阵用于存放每层的梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播
        activation = x  # activation存储激活值
        activations = [x]  # 存储每层的激活值a
        z_save = []   # 存储前向传播的z
        current_layer = 1  # 当前层数
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            z_save.append(z)
            # 最后一层使用softmax,前几层使用relu
            if current_layer < (self.layer_nums - 1):
                activation = relu(z)
                current_layer = current_layer + 1
            else:
                activation = softmax(z)
            activations.append(activation)
        # 计算loss反向传播
        delta = self.d_cost(activations[-1], y)  # 输出层的a^L
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 倒数第二层开始求偏导
        for l in range(2, self.layer_nums):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*d_relu(z_save[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    # 代价函数偏导
    def d_cost(self, output_activations, y):  # 输出的激活值a, label_y, z^L
        return output_activations-y   # 交叉熵代价函数  E = -ln(a^L) ai^L=softmax(zi^L) E对ai^L求偏导

    # 验证准确率
    def evaluate(self, test_data):
        # 神经网络的输出结果是输出层激活值最大的一个神经元所对应的结果，使用numpy的argmax方法来找到该输出层神经元的编号
        # 将测试得到的结果以二元组(神经网络判断结果,正确结果) 形式存储
        test_result = [(np.argmax(self.forward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(i == j) for (i, j) in test_result)

