from load_data import *
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


# 使用SVD分解计算人脸图像的低维表示
def svd_pca(X):  # 10304*400
    p, m = X.shape
    x_mean = np.mean(X, axis=1).reshape((p, 1))
    print(x_mean.shape)
    X = X - x_mean  # 均值归一化
    A = np.dot(X.T, X) / m  # (400, 400)协方差矩阵
    lamda, V = np.linalg.eig(A)  # A的特征值以列的形式显示
    for i in range(m):
        V[:, i:i + 1] /= np.dot(V[:, i:i + 1].T, V[:, i:i + 1])
    sorted_indices = np.argsort(-lamda)
    chance = [8, 20, 50, 100, 150, 200, 250, 300]  # 降维列表
    data = []  # 用来保留降维后重构的数据
    for k in chance:
        print("降到{}维，信息量保留为{}".format
              (k, np.sum([lamda[i] for i in range(k)] / np.sum(list(lamda)))))
        U = np.ones((10304, k))
        for i, j in zip(sorted_indices[0:k], range(k)):
            U[:, j:j + 1] = (X @ V[:, i:i + 1]) / np.sqrt(lamda[i])
        Z = U.T @ X
        X1 = U @ Z + x_mean  # 数据还原
        data.append(X1[:, 0])
    data = np.array(data)
    return data


# 使用最大似然估计计算人脸图像的低维表示
def max_likelihood_estimation_pca(data, k):
    p, m = data.shape
    mu = np.mean(data, axis=1).reshape((p, 1))
    data = data - mu
    S = (data @ data.T) / m  # 协方差矩阵
    vector_U, value, vector_V = np.linalg.svd(data)
    sort_indices = np.argsort(-value)
    I = np.eye(k)
    sigma2 = sum(value[sort_indices[k:]]) / (p - k)
    diag_sorted = np.diag(value[sort_indices[:k]])
    W = vector_U[:, 0:k] @ ((diag_sorted - sigma2 * I) ** 0.5)
    Z = np.zeros((k, m))
    for i in range(m):
        Z[:, i:i + 1] = np.linalg.inv(W.T @ W + sigma2 * I) @ W.T @ (data[:, i:i + 1] - mu)
    recon_data = (W @ Z + mu)
    return Z, recon_data


# 使用简化的EM算法计算人脸图像的低维表示
def em_pca(data, k):
    p, m = data.shape
    # 初始化
    W = np.random.randn(p, k)
    Z = np.random.randn(k, m)
    x_mean = np.mean(data, axis=1).reshape(p, 1)
    for epoch in range(50):
        print(epoch)
        # E步
        x_mean = np.mean(data, axis=1).reshape(p, 1)
        data = data - x_mean
        Z = np.linalg.inv(W.T @ W) @ W.T @ data
        # M步
        W = data @ Z.T @ np.linalg.inv(Z @ Z.T)
    recon_data = (W @ Z + x_mean)
    return Z, recon_data


def main():
    X = get_data()
    print(X.shape)
    # SVD
    data = svd_pca(X)
    x = data[5].reshape(112, 92)
    plt.title("SVD-200维")
    plt.imshow(x, cmap='gray')
    plt.show()

    # 最大似然估计
    Z, recon_X = max_likelihood_estimation_pca(X, 300)
    x = recon_X[:, 0].reshape(112, 92)
    plt.title("最大似然估计-300维")
    plt.imshow(x, cmap='gray')
    plt.show()

    # em
    Z, recon_X = em_pca(X, 300)
    x = recon_X[:, 0].reshape(112, 92)
    plt.title("EM-300维")
    plt.imshow(x, cmap='gray')
    plt.show()


main()