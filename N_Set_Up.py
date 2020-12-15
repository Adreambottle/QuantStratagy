import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts

from datetime import datetime

import statsmodels.api as sm




# 读取从 Wind 上分类的信息技术板块
Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

# 将数据重新按照英文命名
Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

# 提取在股票池中的股票代码
SC_in_pool = Stock_code_pool['Stock_Code']


token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

pro = ts.pro_api(token)

SC = '000021.SZ'

data = SF.get_list_date()
data

def current_stocks(df_total, t):
    """
    获取一个交易时间中，股票池中有哪些股票
    将上市时间在 t 之前的股票作为股票池中的股票

    :param df_total: 一个 DataFrame{"ts_code", "list_date"}
    :param t: 一个时间点
    :return: 返回一个列表
    """
    t = "20100101"
    t = datetime.strptime(t, '%Y%m%d')
    df_in_pool = df_total[df_total["list_date"] < t]["ts_code"].copy()

    return list(df_in_pool)



factor = FF.read_factor("000021.SZ")
index = FF.Read_Index().index_data.resample('W', on="date").mean().index
Sta_Time = "20100101"
Sta_Time = datetime.strptime(Sta_Time, '%Y%m%d')
index_valid = index[index > Sta_Time].copy()  # 每次调仓的时间点


for t_i in range(len(index_valid)):
    t_i = 0
    time_tp = index_valid[t_i]

    df_total = SF.get_list_date()
    df_total.index = df_total["ts_code"]
    stock_in_pool = current_stocks(df_total, Sta_Time)


    # len(stock_in_pool)
    for i in range(len(stock_in_pool)):
        # i = 0

        SC = stock_in_pool[i]
        Return_Stock = SF.Read_One_Stock(SC)
        Return_Stock.select_pct_chg()

        factor = FF.read_factor(SC)
        factor_index = factor.index
        factor_index_np = np.array(factor_index)
        time_tp = np.datetime64(time_tp)
        date_index = np.where(factor_index_np == time_tp)[0][0]  #date_index是滴几个时间


        factor_columns = factor.columns
        factor_columns_selected = factor_columns[0:20]

        factor_sub = factor.iloc[date_index-52:date_index, 0:20]


        # 开始因子检验
        # 开始因子检验
        # 开始因子检验

        for f_i in range(factor_sub.shape[1]):
            # f_i = 4
            x = factor.iloc[:,f_i]
            x_new = MF.pred_by_LSTM_total(x)[0][0]

