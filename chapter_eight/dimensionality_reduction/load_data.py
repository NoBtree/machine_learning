import numpy as np
import matplotlib.pyplot as plt

# 读取单个pgm，转成行向量
def load_single_pgm(file_path):
    f = open(file_path, 'rb')
    # p5格式pgm
    f.readline()  # P5\n
    (width, height) = [int(i) for i in f.readline().split()]  # 92 112
    depth = int(f.readline())
    # 按行读取像素
    data = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(ord(f.read(1)))
        data.append(row)
    data = np.array(data)  # list->ndarray
    data = data.reshape(width * height)
    return data   # 返回一维数组/向量 1*10304


# 将orl_faces所有数据转换成矩阵
def get_data():
    x_matrix = []
    for i in range(40):  # 40个文件夹
        for j in range(10):  # 每个文件夹10个pgm文件
            f_path = "orl_faces/s{}/{}.pgm".format(i+1, j+1)
            x_ij = load_single_pgm(f_path)
            x_matrix.append(x_ij)
    x_matrix = np.array(x_matrix).T  # x 10304*400
    return x_matrix


if __name__ == "__main__":
    f_path = "orl_faces/s1/1.pgm"
    data = load_single_pgm(f_path)
    print(data.shape)
    x = get_data()
    print(x.shape)
