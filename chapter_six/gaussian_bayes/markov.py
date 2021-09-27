import numpy as np


# 返回矩阵的n阶结果
def get_matrix_pow(matrix, n):
    ret = matrix
    for i in range(n-1):
        ret = np.dot(ret, matrix)
    return ret


def main():
    A = np.array([[0.8, 0.2],
                  [0.5, 0.5]])
    A_3 = get_matrix_pow(A, 3)
    print("第一次没中且第四次射中的概率为：%.3f" % A_3[1][0])


main()