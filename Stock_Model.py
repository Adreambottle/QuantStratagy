import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import  datetime
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima_model import ARIMA
from Build_Stock_Pool import Read_One_Stock
import arch

# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# pylab.rcParams['figure.figsize'] = (10, 6)   #设置输出图片大小
sns.set(color_codes=True) #seaborn设置背景


# fig = plt.figure()
# ax = plt.axes()
# ax.plot(list(range(100)), np.random.uniform(0, 1, 100))

def change_stock_1(df:pd.DataFrame):
    if "trade_date" in df.columns:
        date = pd.to_datetime(df["trade_date"], format = '%Y%m%d')
        df.insert(df.shape[1], 'date', date)
        df.drop("trade_date")
        return df
    else:
        print("There is no DataFrame named trade_date")
        return 0

def change_stock_2(df:pd.DataFrame):
    if "trade_date" in df.columns:
        df.index = pd.to_datetime(df["trade_date"], format = '%Y%m%d')
        df.drop("trade_date", axis=1)
        return df

    elif "date" in df.columns:
        df.index = df["date"]
        df.drop("date", axis=1)
        return df

    else:
        print("There is no DataFrame named trade_date")
        return 0

data = Read_One_Stock("000021.SZ").select_close_data()
data = change_stock_2(data)
data = data.sort_index()

# data.insert(2, "test", list(map(lambda x : x**2, data["close"].to_list())))

data
plot_acf(data).show()
plot_pacf(data).show()



# 自相关图显示自相关系数长期大于零，说明时间序列有很强的相关性



"""
动态加权的函数
n 是我们在计算动态加权的选择范围，默认选择 n = 10
为了在数据长度没有超过选择范围的时候仍然能够适应改函数，所以取数据长度和键入值中较小的值
power 是关于期差 j 的多次项的函数指数， 建议取 [0, 2] 的区间内，默认是1
这里只考虑函数是多次项函数的形式
"""
def Moving_weight(df, n = 10, power = 1):

    # df = data["close"]
    new_obj_value_list = []

    # 我们用到的真实值是建入值和 DataFrame 长度中较为小的那个
    num = min(n, df.shape[0])

    if isinstance(df, pd.Series):
        # i = 0
        # 设置元期差和经过权重函数变化后的期差数据
        meta_data = list(range(1, num + 1))
        func_data = list(map(lambda x: x ** power, meta_data))

        # 获取最终的权重向量
        weight = np.array(func_data) / sum(func_data)

        # 获取位于数据最后n位的部分

        data_n = np.array(df.tail(num))

        # 用权重向量和数据尾部点乘得到最终期望值
        obj_value = weight.dot(data_n)

        new_obj_value_list.append(obj_value)

    elif isinstance(df, pd.DataFrame):
        for i in range(df.shape[1]):

            # i = 0
            # 设置元期差和经过权重函数变化后的期差数据
            meta_data = list(range(1, num + 1))
            func_data = list(map(lambda x : x**power, meta_data))

            # 获取最终的权重向量
            weight = np.array(func_data) / sum(func_data)

            # 获取位于数据最后n位的部分

            data_n = np.array(df.iloc[:,i].tail(num))

            # 用权重向量和数据尾部点乘得到最终期望值
            obj_value = weight.dot(data_n)

            new_obj_value_list.append(obj_value)

    return new_obj_value_list


Moving_weight(data["close"])


# 进行实践序列分析
def time_series_analysis(data):

    # 进行ADF检验
    # p-value小于显著性水平，因此序列是平稳的，接下来我们建立AR(p)模型，先判定阶次
    t = sm.tsa.stattools.adfuller(data)  # ADF检验
    print("p-value: ", t[1])

    # fig = plt.figure(figsize=(20, 5))
    # ax1 = fig.add_subplot(111)
    # fig = plot_pacf(data, lags=20, ax=ax1)

    sm.tsa.stattools.pacf(data, nlags=24)

Moving_weight(data)
data = Read_One_Stock("000021.SZ").select_close_data()


datan.loc["date"] = pd.to_datetime(datan["trade_date"], format = '%Y%m%d')

fig = plt.figure()
ax = plt.axes()
ax.plot(data.date, data.close)

plt.show()

pd.to_datetime(datan["trade_date"], format = '%Y%m%d')

datan["trade_datedata"]
datetime.strptime(datan["trade_date"], '%Y%m%d')

