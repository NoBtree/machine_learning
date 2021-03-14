import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import classification_report
import pandas as pd


# 实验目标：使用逻辑回归模型训练一个分类器，并对结果进行可视化

# 读取于文档“ex2data2.txt”中的数据
def read_data(path):
    raw_data = pd.read_csv(path, header=None, names=['x1', 'x2', 'y'])
    return raw_data


# 绘制原始数据散点图
def draw_scatter(data):
    # 将样本分为正负样本
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    # 绘制x1和x2的散点图
    plt.scatter(positive['x1'], positive['x2'], s=50, c='green', marker='o', label='accepted')
    plt.scatter(negative['x1'], negative['x2'], s=50, c='red', marker='x', label='rejected')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 注释的显示位置：右上角
    plt.legend(loc='upper right')
    # 设置坐标轴上刻度的精度
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    return plt


# 特征映射 x1 x2映射到power阶特征
def feature_mapping(x1, x2, power):
    data_map = {}
    for i in range(power+1):
        for j in range(i+1):
            data_map["f{}{}".format(j, i-j)] = np.power(x1, j)*np.power(x2, i-j)
    return pd.DataFrame(data_map)


# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 正则化代价函数
def regularized_cost_function(theta, x, y, lam):
    m = x.shape[0]  # m-样本数量
    # 使用交叉熵损失函数
    j = ((y.dot(np.log(sigmoid(x.dot(theta)))))+((1-y).dot(np.log(1-sigmoid(x.dot(theta))))))/-m
    # L2正则项
    penalty = lam*(theta.dot(theta))/(2*m)
    return j+penalty


# 梯度函数
def regularized_gradient_descent(theta, x, y, lam):
    m = x.shape[0]
    # 损失函数对theta_j求导
    partial_j = ((sigmoid(x.dot(theta))-y).T).dot(x)/m   # .T表示转置
    partial_penalty = lam*theta/m
    # 不惩罚第一项
    partial_penalty[0] = 0
    return partial_j+partial_penalty


# 预测函数
def predict(theta, x):
    h = x.dot(theta)  # 矩阵相乘
    return [1 if x >= 0.5 else 0 for x in h]


# 绘制决策边界
def draw_boundary(theta, data):
    x = np.linspace(-1, 1.5, 200)
    x1, x2 = np.meshgrid(x, x)

    # 生成高维特征数据
    z = feature_mapping(x1.flatten(), x2.flatten(), 6).values  # flatten()展平
    z = z.dot(theta)
    # 保持维度一致
    z = z.reshape(x1.shape)
    # 绘制散点图
    plt = draw_scatter(data)
    # 绘制高度为0的等高线
    plt.contour(x1, x2, z, 0)
    plt.title('boundary')
    plt.show()


# 主函数
if __name__ == '__main__':
    # 读取原始数据
    raw_data = read_data('ex2data2.txt')
    # print(raw_data)
    # plt = draw_scatter(raw_data)
    # plt.show()

    # 由散点图可知决策边界非线性，正则化逻辑回归，采用多项式回归，6阶
    # 构造从原始特征的多项式中得到的特征
    processed_data = feature_mapping(raw_data['x1'], raw_data['x2'], power=6)
    print(processed_data)
    x = processed_data.values  # 118*28 矩阵
    # print(x)
    y = raw_data['y']  # 118*1 label
    # print(y.shape)

    # 初始化theta矩阵  规格28*1  0填充
    theta = np.zeros(x.shape[1])

    # 设置正则化参数lambda
    lam = 0.01

    print(regularized_cost_function(theta, x, y, lam))
    # 使用minimize函数求解
    theta = opt.minimize(fun=regularized_cost_function, x0=theta, args=(x, y, lam), method='tnc', jac=regularized_gradient_descent).x

    print(regularized_cost_function(theta, x, y, lam))

    # sklearn classification_report方法 评估分类器性能
    print(classification_report(predict(theta, x), y))
    # 可视化决策边界
    draw_boundary(theta, raw_data)

