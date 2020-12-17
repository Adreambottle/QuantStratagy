import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
import random

from datetime import datetime


import pymysql
from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as st

import random

import time
# 读取从 Wind 上分类的信息技术板块
Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

# 将数据重新按照英文命名
Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

# 提取在股票池中的股票代码
SC_in_pool = Stock_code_pool['Stock_Code']
# len(SC_in_pool)

token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

pro = ts.pro_api(token)

SC = '002657.SZ'

# 获取每次调仓的时间
factor_sample = FF.read_factor("000021.SZ")
time_total_index = FF.Read_Index().index_data.resample('W', on="date").mean().index
time_total_index_np = np.array(time_total_index)

Sta_Time = "20100101"
Sta_Time = datetime.strptime(Sta_Time, '%Y%m%d')
time_valid_index = time_total_index[time_total_index > Sta_Time].copy()  # 每次调仓的时间点

# End_Time = "20110101"
# time_valid_index = time_valid_index[time_valid_index < End_Time].copy()

# 获取股票上市时间
df_total = SF.get_list_date()
df_total.index = df_total["ts_code"]
df_total = df_total.loc[SC_in_pool, :]
df_total = df_total.dropna(axis=0)
df_total["list_date"] = pd.to_datetime(df_total["list_date"])

# f_time_list = []

stock_relocate_dic = {}





"""
对调仓日期进行循环
循环的内容是调仓的日期
"""
time_sta = time.time()
repo_dict = {}
for t_i in range(len(time_valid_index)):

    time_sta = time.time()
    if t_i == 0:


        time_tp = time_valid_index[t_i]
        time_tp = np.datetime64(time_tp)
        time_on_tp_index_order = np.where(time_total_index_np == time_tp)[0][0]

        # 确定调仓日期，定下时间格式的Index
        time_in_use_index = time_total_index[time_on_tp_index_order - 100:time_on_tp_index_order]

        # 选取股票池中可行的有多少支股票，确定可以选取的股票
        stock_in_pool = SF.current_stocks(df_total, Sta_Time)
        # stock_in_pool = current_stocks(df_total, Sta_Time)


        stock_sample_order = random.sample(list(range(len(stock_in_pool))), 20)
        stock_sample_order_s = pd.Series(stock_in_pool)
        stock_selected = stock_sample_order_s[stock_sample_order]
        repo_dict[time_tp] = stock_selected

        random_num = random.randint(0, 10)
        stock_dropped = stock_selected.iloc[:20-random_num]

    else:
        time_tp = time_valid_index[t_i]
        time_tp = np.datetime64(time_tp)
        time_on_tp_index_order = np.where(time_total_index_np == time_tp)[0][0]

        # 选取股票池中可行的有多少支股票，确定可以选取的股票
        stock_in_pool = SF.current_stocks(df_total, Sta_Time)
        stock_in_pool_s = pd.Series(stock_in_pool)
        # stock_in_pool = current_stocks(df_total, Sta_Time)

        repo_num = len(stock_dropped)

        add_num = 20 - repo_num

        add_index = random.sample(list(range(len(stock_in_pool))), add_num)

        stock_add = stock_in_pool_s[add_index]

        stock_selected = pd.Series(list(stock_dropped) + list(stock_add))
        repo_dict[time_tp] = stock_selected

        random_num = random.randint(0, 10)
        stock_dropped = stock_selected.iloc[:20 - random_num]

repo_df = pd.DataFrame()
for i in range(len(time_valid_index)):

    repo_df[list(repo_dict.keys())[i]] = list(list(repo_dict.values())[i])

repo_df = repo_df.iloc[:, :-1]



close_data = pd.DataFrame()
for i in range(len(SC_in_pool)):
    SC = SC_in_pool[i]
    close = SF.Read_One_Stock(SC).select_col("close")
    close.index = close["trade_date"]
    print(i)
    if i == 0:
        close_data = pd.DataFrame(close.copy())
    else:
        close_data = pd.merge(close_data, close.loc[:, "close"], left_index=True, right_index=True, how="outer")
close_data.columns = ["trade_date"] + list(SC_in_pool)
# close_data.to_excel("close_data.xlsx")

new_data = pd.DataFrame({"date":pd.date_range("20070101", "20191231", freq="D")},
                        index = pd.date_range("20070101", "20191231", freq="D"))
new_data = pd.merge(new_data, close_data, how="outer", left_index=True, right_index=True)
new_data.fillna(axis=0, inplace=True, method="bfill")
import N_Stock_Functions as SF

repo_cash = np.zeros(repo_df.shape)

for i in range(repo_df.shape[1]):
    for j in range(repo_df.shape[0]):
        stock_name = repo_df.iloc[j, i]
        date = repo_df.columns[i]
        repo_cash[j, i] = new_data.loc[date, stock_name]


repo_df.columns[0]


