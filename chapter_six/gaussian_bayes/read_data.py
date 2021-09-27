import pandas as pd
import numpy as np


def read_data(path):
    # 密度  含糖率 好瓜
    data = pd.read_csv(path, header=None, names=['x1', 'x2', 'c'])
    return data


#  将c=c1和 c=c2两种情况下的x1 x2分离
def split_data(path="data.txt"):
    data = np.array(read_data(path))
    x1_x2_good = np.array([x[0:2] for x in data if x[2] == '是'])
    x1_x2_bad = np.array([x[0:2] for x in data if x[2] == '否'])
    return x1_x2_good, x1_x2_bad
