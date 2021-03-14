import pickle
import gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 解压数据集并读取
def load_data():
    file = gzip.open('mnist.pkl.gz', 'rb')
    train_data, validation_data, test_data = pickle.load(file, encoding="bytes")
    # train_data (50000*784,5000*1)
    # print(train_data[0].shape)
    file.close()
    return train_data, validation_data, test_data


# 处理读取的初始数据，转换为常用格式
def raw_data_process():
    raw_tr_data, raw_val_data, raw_te_data = load_data()
    # 训练集
    train_inputs = [np.reshape(x, (784, 1)) for x in raw_tr_data[0]]  # input 784*50000
    train_labels = [one_hot_transform(x) for x in raw_tr_data[1]]  # label 10*50000
    # input和label一一对应
    train_data = list(zip(train_inputs, train_labels))
    # print("训练集大小：{}".format(len(train_labels)))
    # 验证集
    validation_inputs = [np.reshape(x, (784, 1)) for x in raw_val_data[0]]  # 784*10000
    validation_labels = [one_hot_transform(x) for x in raw_tr_data[1]]  # 输出 10*10000
    validation_data = list(zip(validation_inputs, validation_labels))
    # 测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in raw_te_data[0]]  # 784*10000
    test_labels = [one_hot_transform(x) for x in raw_te_data[1]]  # 输出 10*10000
    test_data = list(zip(test_inputs, test_labels))
    return train_data, validation_data, test_data


# 生成one-hot向量
def one_hot_transform(j):
    # shape 10*1
    label_y = np.zeros((10, 1))
    label_y[j] = 1.0
    return label_y


# 使用PIL图像api，将图像显示出来 784*1 28*28
def show_image(vector):
    vector.resize((28, 28))
    img = Image.fromarray(np.uint8(vector * 255)).convert("1")  # 二值化
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    train_data, validation_data, test_data = raw_data_process()
    # train_data[0] 784*50000 train_data[1] 10*50000
    # print(train_data[0][1])  # 784*1
    show_image(train_data[0][1])