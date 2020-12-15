# 导入需要用到的模块
import numpy as np
import pandas as pd
import datetime
import pymysql
from sqlalchemy import create_engine


class Read_One_Stock_Factor():

    # 初始化数据，定义 MySQL 访问链接参数
    def __init__(self, SC_Code):
        self.conn = pymysql.connect(
            host="localhost",
            database="factor",
            user="root",
            password="zzzzzzzz",
            port=3306,
            charset='utf8'
        )
        self.SC_Code = SC_Code

    # 获取每天的收盘价
    def select_all_data(self):
        # 读取每天的收盘价
        sqlcmd = "SELECT * FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    # 获取想要获取的交易数据，可以自定义
    def select_col(self, *args):
        col_list = args
        sqlcmd = "SELECT trade_date, "
        for arg in args:
            sqlcmd = sqlcmd + arg + ", "
        sqlcmd = sqlcmd[:-2]
        sqlcmd = sqlcmd + " FROM `{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table


class Read_Index():

    # 初始化数据，定义 MySQL 访问链接参数
    def __init__(self):
        self.conn = pymysql.connect(
            host="localhost",
            database="factor",
            user="root",
            password="zzzzzzzz",
            port=3306,
            charset='utf8'
        )
        self.select_all_data()

    # 获取每天的收盘价
    def select_all_data(self):
        # 读取每天的收盘价
        sqlcmd = "SELECT * FROM Index_daily_data"
        table = pd.read_sql(sqlcmd, self.conn)

        self.index_data = table


def column_add_max(column: list):
    column_max = []
    for ele in column:
        ele = ele + "_max"
        column_max.append(ele)
    return column_max


def column_add_min(column: list):
    column_min = []
    for ele in column:
        ele = ele + "_min"
        column_min.append(ele)
    return column_min


def column_add_avg(column: list):
    column_avg = []
    for ele in column:
        ele = ele + "_avg"
        column_avg.append(ele)
    return column_avg


def factor_formulate(data: pd.DataFrame, time='W'):
    # data = factor_td
    column_diff_in_date = ["date",
                           "BP_LF",
                           "EBIT2EV",
                           "OCFP",
                           "SP",
                           "NCFP",
                           "FCFP",
                           "dv_ratio",
                           "PPReversal_1",
                           "PPReversal_5",
                           "PPReversal_20",
                           "TO_5d",
                           "TO_20d",
                           "TO_100d",
                           "turnover_vol_5d",
                           "turnover_vol_20d",
                           "turnover_vol_100d",
                           "VSTD_6d",
                           "VSTD_30d",
                           "VSTD_100d",
                           "DAVOL5",
                           "DAVOL20",
                           "DAVOL100"]

    column_same_in_date = ["date",
                           "return_1m",
                           "return_3m",
                           "return_6m",
                           "return_12m",
                           "wgt_return_1m",
                           "wgt_return_3m",
                           "wgt_return_6m",
                           "wgt_return_12m",
                           "turn_1m",
                           "turn_3m",
                           "turn_6m",
                           "wgt_turn_1m",
                           "wgt_turn_3m",
                           "wgt_turn_6m",
                           "std_1m",
                           "std_3m",
                           "std_6m",
                           "std_12m",
                           "id_std_1m",
                           "id_std_3m",
                           "id_std_6m",
                           "id_std_12m",
                           "roe",
                           "opincome_of_ebt",
                           "ocf_to_or",
                           "currentdebt_to_debt",
                           "current_ratio",
                           "assets_turn",
                           "roe_q",
                           "revenue",
                           "n_income_attr_p",
                           "revenue_q",
                           "n_income_attr_p_q",
                           "n_cashflow_act",
                           "n_cashflow_act_q",
                           "q_profit_yoy"]

    factor_diff_in_date = data.loc[:, column_diff_in_date]

    column_diff_in_date_max = column_add_max(column_diff_in_date)
    column_diff_in_date_min = column_add_min(column_diff_in_date)
    column_diff_in_date_avg = column_add_avg(column_diff_in_date)

    factor_diff_in_date_w_max = factor_diff_in_date.resample('W', on='date').max()
    factor_diff_in_date_w_max.columns = column_diff_in_date_max

    factor_diff_in_date_w_min = factor_diff_in_date.resample('W', on='date').min()
    factor_diff_in_date_w_min.columns = column_diff_in_date_min

    factor_diff_in_date_w_avg = factor_diff_in_date.resample('W', on='date').mean()
    factor_diff_in_date_w_avg.columns = column_diff_in_date_avg[1:]

    factor_same_in_date = data.loc[:, column_diff_in_date]
    factor_same_in_date_w = factor_same_in_date.resample('W', on='date').mean()

    factor_w = pd.merge(factor_same_in_date_w, factor_diff_in_date_w_avg, how='outer',
                        left_index=True, right_index=True)
    factor_w = pd.merge(factor_w, factor_diff_in_date_w_max, how='outer',
                        left_index=True, right_index=True)
    factor_w = pd.merge(factor_w, factor_diff_in_date_w_min, how='outer',
                        left_index=True, right_index=True)

    factor_w = factor_w.dropna(axis=0)
    factor_w.drop(["date_max", "date_min"], axis=1, inplace=True)

    return factor_w


def mad(series):
    n = 10
    median = series.quantile(0.5)
    diff_median = ((series - median).abs()).quantile(0.5)
    max_range = median + n * diff_median
    min_range = median - n * diff_median
    return np.clip(series, min_range, max_range)


def three_sigma(series, n):
    mean = series.mean()
    std = series.std()
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)


def standard_z_score(series):
    std = series.std()
    mean = series.mean()
    return (series - mean) / std

def normalization_data(data: pd.DataFrame):
    data = data.apply(mad, axis=0)
    data = data.apply(standard_z_score, axis=0)
    return data



def read_factor(SC):

    SC = "000021.SZ"

    # 这部分是主函数
    factor = Read_One_Stock_Factor(SC).select_all_data()

    factor.index = factor["index"]

    RI = Read_Index()

    trade_date = RI.index_data["trade_date"]

    sta_date = datetime.datetime.strptime("20080101", "%Y%m%d")

    trade_date = trade_date[trade_date > sta_date]

    factor_td = factor.loc[trade_date, :]

    factor_f = factor_formulate(factor_td)

    factor_n = normalization_data(factor_f)

    return factor_n
