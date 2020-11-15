# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.linear_model import LinearRegression

# 新建因子与股票的相关data，假设3个因子，100支股票

N = 100
f = 3
factor1 = np.random.normal(size=(100))
factor2 = np.random.normal(size=(100))
factor3 = np.random.normal(size=(100))

factor_table = pd.DataFrame({"factor1":factor1,
                             "factor2":factor2,
                             "factor3":factor3})

# 新建portfolio（股票data），假设100只股票，1000天的数据
stock = np.random.normal(size=(100, 1000))
stock_table = pd.DataFrame(stock)
stock_table.head()

# 设置因子收益率 f 3*1
f = np.random.uniform(0, 1, 3)

# 设置股票残差 u 100*1
u = stock_table.std()

# portfolio 相关系数
stock_table.corr()

# 设置权重 weight
h_tmpt = np.array(np.random.randint(1, 11, 100))
h = h_tmpt/h_tmpt.sum()


# h = h.reshape((100, 1))
# f = f.reshape((100, 1))

X = np.array(factor_table)

# 组合的收益率
r_P = h.T.dot(X).dot(f) + h.T.dot(u)


# 多因子模型风险预测
r_V = X.dot(f) + u


# 拟合股票模型
lr_list = []
date_num = 1000
for i in range(100):
    # i = 0;
    x = np.array(range(date_num)).reshape(-1, 1)
    y = stock[i, :]
    lr = LinearRegression()



    lr.fit(np.array(range(date_num)).reshape(-1, 1), stock[i,:])
    lr_list.append(lr)

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3]])


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(f)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/