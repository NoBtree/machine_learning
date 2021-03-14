import numpy as np
from scipy.io import loadmat
import math

# 加载训练集和测试集数据
def load_data():
    # 加载mat格式的字典文件
    spam_train = loadmat(file_name="spamTrain.mat")
    print(spam_train.keys())
    spam_train_x = spam_train["X"]
    spam_train_y = spam_train["y"]
    # 一个数据的长度是1899，也就是说垃圾邮件一共有1899个特征
    spam_train_y = [math.pow(-1, i+1) for i in spam_train_y]
    spam_train_y = np.array(spam_train_y, dtype=int).reshape(-1, 1)
    # print(spam_train_y)
    # 同样的方式对测试集进行处理
    spam_test = loadmat(file_name="spamTest.mat")
    print(spam_test.keys())
    spam_test_x = spam_test["Xtest"]
    spam_test_y = spam_test["ytest"]
    spam_test_y = [math.pow(-1, i + 1) for i in spam_test_y]
    spam_test_y = np.array(spam_test_y, dtype=int).reshape(-1, 1)
    for x in spam_train_x:
        print("训练集特征长度:{}".format(len(x)))
        break
    for x in spam_test_x:
        print("测试集特征长度:{}".format(len(x)))
        break
    print("训练集样本数量:{}".format(spam_train_y.shape[0]))
    print("测试集样本数量:{}".format(spam_test_y.shape[0]))
    return spam_train_x, spam_train_y, spam_test_x, spam_test_y


# 批量Pegasos算法，参数分别是数据，数据标签，C=0.1，训练轮数，batch大小
def batchPegasos(x, y, C, T, k):
    lam = 1 / (k * C)
    m, n = np.shape(x)
    w = np.zeros(n)
    dataIndex = np.array([i for i in range(m)])
    for t in range(1, T + 1):
        wDelta = np.zeros(n)
        eta = 1.0 / (lam * t)
        np.random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, x[i, :])
            if y[i][0] * p < 1:
                wDelta += y[i] * x[i, :]
        w = (1.0 - 1 / t) * w + (eta / k) * wDelta
    return w


# 预测 wx+b
def predict(w, x):
    return w.T @ x


# 对测试集进行测试
def test(x, y, w):
    predict_y = []
    label_y = y.reshape(-1)
    # print(label_y)
    for x_i, y_i in zip(x, label_y):
        tmp = predict(w, x_i)
        if tmp <= 0:
            predict_y.append(-1)
        else:
            predict_y.append(1)
    predict_y = np.asarray(predict_y)
    # print(np.sum(predict_y == label_y))
    print("正确率为{}/{}".format(np.sum(predict_y == label_y), len(predict_y)))


# 主函数
if __name__ == '__main__':
    spam_train_x, spam_train_y, spam_test_x, spam_test_y = load_data()
    # 训练
    c = 0.1
    epochs = 100
    batch_size = 100
    w = batchPegasos(spam_train_x, spam_train_y, c, epochs, batch_size)
    # 测试
    test(spam_test_x, spam_test_y, w)