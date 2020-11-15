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


data
plot_acf(data).show()
plot_pacf(data).show()



# 自相关图显示自相关系数长期大于零，说明时间序列有很强的相关性



# 动态加权的函数

def Moving_weight(df:pd.DataFrame, n = 10):
    new_obj_value_list = []
    for i in range(df.shape[1]):
        if df.shape[i] >= n:
            weight = np.array(list(range(1, n+1))) / sum(list(range(1, n+1)))
            data_n = np.array(data.tail(n))
            obj_value = weight.dot(data_n)[0]
        else:
            n = df.shape[i]
            weight = np.array(list(range(1, n + 1))) / sum(list(range(1, n + 1)))
            data_n = np.array(data.tail(n))
            obj_value = weight.dot(data_n)[0]
        new_obj_value_list.append(obj_value)
    return new_obj_value_list

ARIMA()





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

