import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

from pmdarima import auto_arima

from arch import arch_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


import seaborn as sns

# sns.set(color_codes=True)


# seaborn设置背景

def change_stock_1(df: pd.DataFrame):
    """
    规范化因子和股票数据信息，将时间作为index
    :param df: 数据
    :return: 新的数据
    """
    if "trade_date" in df.columns:
        date = pd.to_datetime(df["trade_date"], format='%Y%m%d')
        df.insert(df.shape[1], 'date', date)
        df.drop("trade_date")
        return df
    else:
        print("There is no DataFrame named trade_date")
        return 0


def change_stock_2(df: pd.DataFrame):
    """
    规范化因子和股票数据信息，将时间作为index
    :param df: 数据
    :return: 新的数据
    """
    if "trade_date" in df.columns:
        df.index = pd.to_datetime(df["trade_date"], format='%Y%m%d')
        df.drop("trade_date", axis=1)
        return df

    elif "date" in df.columns:
        df.index = df["date"]
        df.drop("date", axis=1)
        return df

    else:
        print("There is no DataFrame named trade_date")
        return 0


def Moving_weight(df, n=10, power=0.5):
    """
    动态加权的函数
    n 是我们在计算动态加权的选择范围，默认选择 n = 10
    为了在数据长度没有超过选择范围的时候仍然能够适应改函数，所以取数据长度和键入值中较小的值
    power 是关于期差 j 的多次项的函数指数， 建议取 [0, 2] 的区间内，默认是1
    这里只考虑函数是多次项函数的形式
    :param df: 数据
    :param n: 对n次之前的代码进行叠加
    :param power: 加权采用的适应函数
    :return:
    """
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
            func_data = list(map(lambda x: x ** power, meta_data))

            # 获取最终的权重向量
            weight = np.array(func_data) / sum(func_data)

            # 获取位于数据最后n位的部分

            data_n = np.array(df.iloc[:, i].tail(num))

            # 用权重向量和数据尾部点乘得到最终期望值
            obj_value = weight.dot(data_n)

            new_obj_value_list.append(obj_value)

    return new_obj_value_list


# a = pd.DataFrame(np.arange(100))
# Moving_weight(a)

def time_series_analysis(data):
    """
    对因子数据进行时间序列分析
    :param data:
    :return:
    """
    # 需要data是一个时间序列，我觉得这样是最好的
    # 用折线图查看data

    t = sm.tsa.stattools.adfuller(data)  # ADF检验
    print("p-value: ", t[1])
    # p-value小于显著性水平，因此序列是平稳的

    # 对matplotlib的配置进行设置
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # 画出原图
    ax1.plot(data)

    # 计算data的一阶差分，查看相关的图像
    data_diff = data.diff()
    ax2.plot(data_diff)

    # # 计算data的二阶差分
    # data_diff_2 = data_diff.diff()
    # data_diff_2.plot(figsize=(15, 5))

    fig2 = plt.figure(figsize=(15, 8))
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    # 画出不同order的ACF和PACF两张与
    fig2 = sm.graphics.tsa.plot_acf(data, lags=20, ax=ax3)
    fig2 = sm.graphics.tsa.plot_pacf(data, lags=20, ax=ax4)
    plt.show()


def pred_by_auto_arima(data):
    """
    用arima模型对新的因子值仅需预测
    用已有的历史数据作为train data
    然后自动寻找自己的arima模型的order
    建设模型，然后对新的值进行预测
    :param data: 因子数据
    :return: 新的预测值
    """
    model = auto_arima(data, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(data)
    forecast = model.predict(n_periods=1)
    return forecast

def pred_by_arch(data):
    """
    用arch模型对variance进行预测
    :param data:
    :return:
    """
    # 计算最大的自相关order
    lags = sm.tsa.acf(data, nlags=20)[1:].tolist()
    max_lag_index = lags.index(max(lags)) + 1

    model = sm.tsa.ARIMA()

    model = arch_model(data, mean='AR', vol='ARCH', p=max_lag_index)
    model_fit = model.fit(show_warning=False, )
    yhat = model_fit.forecast(horizon=1)
    yhat.variance.values[-1, :]


# 在这里选用LSTM进行新的因子值的预测
from sklearn.preprocessing import MinMaxScaler

def pred_by_LSTM_total(data):
    """
    用LSTM模型对新的因子值进行预测
    用已有的历史数据作为t
    :param data: 因子数据
    :return: 新的预测值
    """
    # data = f1[:52]
    data = np.array(data)
    x_train = []
    y_train = []
    back = 10
    for i in range(back, len(data)):
        x_train.append(data[i-back:i])
        y_train.append(data[i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train_new = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_new.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train_new, y_train, epochs=1, batch_size=1, verbose=2)

    x_test = np.array(data[len(data)-back:len(data)])
    x_test_new = np.reshape(x_test, (1, x_test.shape[0], 1))
    predict_value = model.predict(x_test_new)

    return predict_value