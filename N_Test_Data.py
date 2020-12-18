import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
import random
import math

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
        stock_dropped = stock_selected.iloc[:20 - random_num]

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

new_data = pd.DataFrame({"date": pd.date_range("20070101", "20191231", freq="D")},
                        index=pd.date_range("20070101", "20191231", freq="D"))
new_data = pd.merge(new_data, close_data, how="outer", left_index=True, right_index=True)
new_data.fillna(axis=0, inplace=True, method="bfill")
import N_Stock_Functions as SF

repo_cash = np.zeros(repo_df.shape)

for i in range(repo_df.shape[1]):
    for j in range(repo_df.shape[0]):
        stock_name = repo_df.iloc[j, i]
        date = repo_df.columns[i]
        repo_cash[j, i] = new_data.loc[date, stock_name]

time_valid_index = time_valid_index[:-1]
repo_df_valid = repo_df.loc[:, time_valid_index]

vol = (2500000 / repo_cash)
vol_real = vol
for i in range(vol.shape[0]):
    for j in range(vol.shape[1]):
        vol_real[i, j] = int(math.floor(vol[i, j]))
vol_real_df = pd.DataFrame(vol_real, columns=repo_df.columns)
vol_real_df_valid = vol_real_df.loc[:, time_valid_index]

position_dic_list = []

for i in range(len(time_valid_index)):
    position_dic = {}
    for SC in SC_in_pool:
        position_dic[SC] = 0
    for j in range(repo_df_valid.shape[0]):
        position_dic[repo_df_valid.iloc[j, i]] = vol_real_df_valid.iloc[j, i]
    position_dic_list.append(position_dic)

position_dic_list[0]

order_dic_list = []
for i in range(len(time_valid_index) - 1):
    # position_dic = {}
    order_dic = {}
    for SC in SC_in_pool:
        # position_dic[SC] = 0

        order_dic[SC] = {'buy': 0, 'sell': 0}
    for SC in SC_in_pool:
        # position_dic[repo_df_valid.iloc[j, i]] =  vol_real_df_valid.iloc[j, i]
        stock_bs_dic = {'buy': 0, 'sell': 0}
        diff = position_dic_list[i + 1][SC] - position_dic_list[i][SC]
        if diff >= 0:
            stock_bs_dic['buy'] = abs(diff)
            stock_bs_dic['sell'] = 0
        else:
            stock_bs_dic['sell'] = abs(diff)
            stock_bs_dic['buy'] = 0
        order_dic[SC] = stock_bs_dic
    order_dic_list.append(order_dic)

# order_dic_list[0]
#
#
# order_dic_list[1]


# 回测信息
position_dic_dic = {}
for i in range(len(position_dic_list)):
    position_dic_dic[i] = position_dic_list[i]

order_dic_dic = {}
for i in range(len(order_dic_list)):
    order_dic_dic[i] = order_dic_list[i]

np.save('position.npy', position_dic_dic)
np.save('order.npy', order_dic_dic_new)

order_dic_dic[0]

order_dic_dic_new = {}
for i in range(len(order_dic_dic.keys())):
    order_dic_dic_new[i+1] = order_dic_dic[i]
order_dic_dic_new[0] = order_dic_dic_new[1]

for i in range(len(order_dic_dic_new[0].keys())):
    stock_name = list(order_dic_dic_new[0].keys())[i]
    order_dic_dic_new[0][stock_name]['buy'] = position_dic_dic[0][stock_name]
    order_dic_dic_new[0][stock_name]['sell'] = 0



df = pro.index_daily(ts_code='399300.SZ',
                     start_date='20100101',
                     end_date='20191231')
df.index = pd.to_datetime(df["trade_date"])

df_new = pd.DataFrame(pd.date_range('20100101', '20191231', freq='D'),
                      index = pd.date_range('20100101', '20191231', freq='D'))
df_new = pd.merge(df_new, df.loc[:,"pct_chg"], left_index=True, right_index=True, how='outer')
df_new.fillna(method='bfill', inplace=True)

hs300 = df_new.loc[time_valid_index, 'pct_chg']
hs300.to_csv("index300.csv")

pd.date_range('20100101', '20191231', freq='D')
np.save('position.npy', position_dic_dic)
np.save('order.npy', order_dic_dic_new)
np.save('time.npy', np.array(time_valid_index))
np.save('index300.npy', hs300)









def find_weekday(test_date):
    return datetime.strptime(test_date, "%Y%m%d").weekday()


trade_date = pro.trade_cal(start_date='20100101', end_date='20191231')
trade_date_on = trade_date[trade_date["is_open"] == 1]["cal_date"]

trade_date_weekday = []
for i in range(trade_date_on.shape[0]):
    # i = 0
    weekday = datetime.strptime(trade_date_on.iloc[i], "%Y%m%d").weekday()
    trade_date_weekday.append(weekday)

trade_date_on = pd.DataFrame({"date": trade_date_on,
                              "weekday": trade_date_weekday})
date_index = pd.to_datetime(trade_date_on[trade_date_on["weekday"] == 4]["date"])
date_index.index = date_index

