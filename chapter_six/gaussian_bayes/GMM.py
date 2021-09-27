import numpy as np


class GMM:
    def __init__(self, k=2):
        self.k = k  # 定义聚类个数,默认值为2
        self.p = None  # 样本维度
        self.n = None  # 样本个数
        # 声明变量
        self.params = {
            "πi": None,  # 混合系数1*k
            "μ": None,  # 均值k*p
            "cov": None,  # 协方差k*p*p
            "pji": None  # 后验分布n*k
        }

    # 初始化参数
    def init_params(self, init_μ):
        πi = np.ones(self.k) / self.k
        μ = init_μ
        cov = np.ones((self.k, self.p, self.p))
        pji = np.zeros((self.n, self.k))
        self.params = {
            "πi": πi,  # 混合系数1*k
            "μ": μ,  # 均值k*p
            "cov": cov,  # 协方差k*p*p
            "pji": pji  # 后验分布n*k
        }

    # 高斯公式
    def gaussian_function(self, x_j, μ_k, cov_k):
        one = -((x_j - μ_k) @ np.linalg.inv(cov_k) @ (x_j - μ_k).T) / 2
        two = -self.p * np.log(2 * np.pi) / 2
        three = -np.log(np.linalg.det(cov_k)) / 2
        return np.exp(one + two + three)

    # 计算Pji隐变量概率
    def E_step(self, x):
        πi = self.params["πi"]
        μ = self.params["μ"]
        cov = self.params["cov"]
        for j in range(self.n):
            x_j = x[j]
            pji_list = []
            for i in range(self.k):
                πi_k = πi[i]
                μ_k = μ[i]
                cov_k = cov[i]
                pji_list.append(πi_k * self.gaussian_function(x_j, μ_k, cov_k))
            self.params['pji'][j, :] = np.array([v / np.sum(pji_list) for v in pji_list])

    # 更新参数
    def M_step(self, x):
        μ = self.params["μ"]
        pji = self.params["pji"]
        for i in range(self.k):
            μ_k = μ[i]  # p
            pji_k = pji[:, i]  # n
            pji_k_j_list = []
            mu_k_list = []
            cov_k_list = []
            for j in range(self.n):
                x_j = x[j]  # p
                pji_k_j = pji_k[j]
                pji_k_j_list.append(pji_k_j)
                mu_k_list.append(pji_k_j * x_j)
            self.params['μ'][i] = np.sum(mu_k_list, axis=0) / np.sum(pji_k_j_list)
            for j in range(self.n):
                x_j = x[j]  # p
                pji_k_j = pji_k[j]
                cov_k_list.append(pji_k_j * np.dot((x_j - μ_k).T, (x_j - μ_k)))
            self.params['cov'][i] = np.sum(cov_k_list, axis=0) / np.sum(pji_k_j_list)
            self.params['πi'][i] = np.sum(pji_k_j_list) / self.n
        print("均值为：", self.params["μ"].T[0], end=" ")
        print("方差为：", self.params["cov"].T[0][0], end=" ")
        print("混合系数为：", self.params["πi"])

    # 迭代，返回聚类结果
    def fit(self, x, μ, max_iter=10):
        x = np.array(x)
        self.n, self.p = x.shape
        self.init_params(μ)

        for i in range(max_iter):
            print("第{}次迭代".format(i+1))
            self.E_step(x)
            self.M_step(x)
        return np.argmax(np.array(self.params["pji"]), axis=1)


def main():
    dataset = np.array([[1.0], [1.3], [2.2], [2.6], [2.8], [5.0], [7.3], [7.4], [7.5], [7.7], [7.9]])
    μ = np.array([[6], [7.5]])
    my_model = GMM(2)
    result = my_model.fit(dataset, μ, max_iter=8)
    print(result)


main()

