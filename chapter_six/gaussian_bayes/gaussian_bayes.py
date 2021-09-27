import numpy as np
import read_data


# 求均值 x[x1,x2]
def mean_value(x):
    x1_mean = np.mean(x[:, 0])
    x2_mean = np.mean(x[:, 1])
    return np.array([x1_mean, x2_mean])


# 求协方差 x[x1 x2] 2D数据
def cov(x):
    x_cov = np.cov(x.astype(float).T)  # 默认无偏估计，即分母为n-1
    return x_cov


# 方差 x[x1,x2]
def var_value(x):
    x1_var = np.var(x[:, 0])
    x2_var = np.var(x[:, 1])
    return np.array([x1_var, x2_var])


#  高斯贝叶斯/高斯朴素贝叶斯
def gaussian_bayes(x_mean, x_cov, x_input):
    x_cov_det = np.linalg.det(x_cov)  # 协方差矩阵行列式
    x_cov_inv = np.linalg.inv(x_cov)  # 协方差矩阵逆矩阵
    temp = x_input - x_mean
    gaussian = 1 / (2 * np.pi) * 1 / (pow(x_cov_det, 0.5)) * np.exp(-(1/2 * np.dot(np.dot(temp, x_cov_inv), temp)))
    return gaussian


def main():
    x_good, x_bad = read_data.split_data()  # x[x1,x2]
    # 均值
    x_good_mean = mean_value(x_good)
    x_bad_mean = mean_value(x_bad)
    # 协方差矩阵
    x_good_cov = cov(x_good)
    x_bad_cov = cov(x_bad)
    x = np.array([0.5, 0.3])
    # 先验概率
    p_good = len(x_good)/(len(x_good) + len(x_bad))
    p_bad = len(x_bad)/(len(x_good) + len(x_bad))

    # 高斯贝叶斯
    good_gaussian = gaussian_bayes(x_good_mean, x_good_cov, x)
    bad_gaussian = gaussian_bayes(x_bad_mean, x_bad_cov, x)
    print("good_gaussian:%s, bad_gaussian:%s" % (good_gaussian, bad_gaussian))
    good = good_gaussian * p_good
    bad = bad_gaussian * p_bad
    print("gaussian_bayes: good:%s, bad:%s" % (good, bad))

    # 高斯朴素贝叶斯
    # 方差
    var_good = var_value(x_good)
    var_bad = var_value(x_bad)
    print(var_good, var_bad)
    # 协方差矩阵
    cov_good_naive = np.zeros((2, 2))
    cov_good_naive[0][0] = var_good[0]
    cov_good_naive[1][1] = var_good[1]
    cov_bad_naive = np.zeros((2, 2))
    cov_bad_naive[0][0] = var_good[0]
    cov_bad_naive[1][1] = var_good[1]

    good_naive_gaussian = gaussian_bayes(x_good_mean, cov_good_naive, x)
    bad_naive_gaussian = gaussian_bayes(x_bad_mean, cov_bad_naive, x)
    print("good__naive_gaussian:%s, bad_naive_gaussian:%s" % (good_naive_gaussian, bad_naive_gaussian))
    good_naive = good_naive_gaussian * p_good
    bad_naive = bad_naive_gaussian * p_bad
    print("gaussian_naive_bayes: good_naive:%s, bad_naive:%s" % (good_naive, bad_naive))


main()