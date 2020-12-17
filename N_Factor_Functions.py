# 导入需要用到的模块
import numpy as np
import pandas as pd
import datetime
import pymysql
from sqlalchemy import create_engine


class Read_One_Stock_Factor():
    """
    从SQL里面读取一只股票的全部因子数据
    """

    # 初始化数据，定义 MySQL 访问链接参数
    def __init__(self, SC_Code):
        """
        定义MySQL的连接方式
        获取访问权限
        :param SC_Code: SC是stock code
                        根据股票的信息
        """
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
        """
        编写SQL语句，送SQL中选取全部的数据
        :return:
        """
        # 读取每天的收盘价
        sqlcmd = "SELECT * FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    # 获取想要获取的交易数据，可以自定义
    def select_col(self, *args):
        """
        可以读取股票中的任意多个column
        :param args: 可以增添的column的
        :return:
        """
        col_list = args
        sqlcmd = "SELECT trade_date, "
        for arg in args:
            sqlcmd = sqlcmd + arg + ", "
        sqlcmd = sqlcmd[:-2]
        sqlcmd = sqlcmd + " FROM `{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table


class Read_Index():
    """
    获取指数信息，用于确定交易日还有其他和指数信息相关的信息
    """

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
        """
        读取每天的收盘价
        :return:
        """

        sqlcmd = "SELECT * FROM Index_daily_data"
        table = pd.read_sql(sqlcmd, self.conn)

        self.index_data = table


"""
为了尽可能多保留因子的信息，选择将每日都不一样的因子进行拆分
从5日的不同数据中提取出最大值，最小值和中间值，作为新的因子
"""
def column_add_max(column: list):
    """
    在名称后添加 max 词缀
    :param column: 需要传入的column名称
    :return:
    """
    column_max = []
    for ele in column:
        ele = ele + "_max"
        column_max.append(ele)
    return column_max


def column_add_min(column: list):
    """
    在名称后添加 min 词缀
    :param column: 需要传入的column名称
    :return:
    """
    column_min = []
    for ele in column:
        ele = ele + "_min"
        column_min.append(ele)
    return column_min


def column_add_avg(column: list):
    """
    在名称后添加 avg 词缀
    :param column: 需要传入的column名称
    :return:
    """
    column_avg = []
    for ele in column:
        ele = ele + "_avg"
        column_avg.append(ele)
    return column_avg


def factor_formulate(data: pd.DataFrame, time='W'):
    """
    规范化因子库，
    为了尽可能多保留因子的信息
    将因子分成
        1.每日都不一样的
        2.一周内都是一样的
    这两种因子


    :param data: 需要传入的数据
    :param time: 设定最小调仓时间单位，默认是'W'一周
    :return:
    """
    # data = factor_td
    column_diff_in_date = ["date",
                           "BP_LF",
                           "EBIT2EV",
                           "OCFP",
                           "SP",
                           "NCFP",
                           "FCFP",
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

    # 选出每日都不一样的因子
    factor_diff_in_date = data.loc[:, column_diff_in_date]

    # 在因子名称后面添加后缀
    column_diff_in_date_max = column_add_max(column_diff_in_date)
    column_diff_in_date_min = column_add_min(column_diff_in_date)
    column_diff_in_date_avg = column_add_avg(column_diff_in_date)

    # 按照周将因子进行groupby的重筛选操作
    factor_diff_in_date_w_max = factor_diff_in_date.resample(time, on='date').max()
    factor_diff_in_date_w_max.columns = column_diff_in_date_max

    factor_diff_in_date_w_min = factor_diff_in_date.resample(time, on='date').min()
    factor_diff_in_date_w_min.columns = column_diff_in_date_min

    factor_diff_in_date_w_avg = factor_diff_in_date.resample(time, on='date').mean()
    factor_diff_in_date_w_avg.columns = column_diff_in_date_avg[1:]

    # 选出周时间段内每日一样的因子
    # 计算出一周之内的平均数
    factor_same_in_date = data.loc[:, column_same_in_date]
    factor_same_in_date_w = factor_same_in_date.resample(time, on='date').mean()

    # 将重新计算好的因子merge在一起
    factor_w = pd.merge(factor_same_in_date_w, factor_diff_in_date_w_avg, how='outer',
                        left_index=True, right_index=True)
    factor_w = pd.merge(factor_w, factor_diff_in_date_w_max, how='outer',
                        left_index=True, right_index=True)
    factor_w = pd.merge(factor_w, factor_diff_in_date_w_min, how='outer',
                        left_index=True, right_index=True)

    # 删去重复的column
    # factor_w = factor_w.dropna(axis=0)
    factor_w.drop(["date_max", "date_min"], axis=1, inplace=True)

    return factor_w


def mad(series):
    """
    采用MAD的模式对因子进行去尾处理
    :param series: 需要传入的Series
    :return:
    """
    n = 10
    median = series.quantile(0.5)
    diff_median = ((series - median).abs()).quantile(0.5)
    max_range = median + n * diff_median
    min_range = median - n * diff_median
    return np.clip(series, min_range, max_range)


def three_sigma(series, n):
    """
    采用three_sigma的模式对因子进行去尾处理
    :param series: 需要传入的Series
    :return:
    """
    mean = series.mean()
    std = series.std()
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)


def standard_z_score(series):
    """
    采用z_score的模式对因进行标准化处理
    :param series: 需要传入的Series
    :return:
    """
    std = series.std()
    mean = series.mean()
    return (series - mean) / std

def normalization_data(data: pd.DataFrame):
    """
    采用normalization_data的模式对因进行标准化处理
    :param series: 需要传入的Series
    :return:
    """
    data = data.apply(mad, axis=0)
    data = data.apply(standard_z_score, axis=0)
    return data

"""
因为在计算因子的时候需要用到回滚两年的历史数据
所以在获取数据的时候选择了提前三年的数据
"""

def read_factor(SC):
    """
    读取一只股票的所有因子
    :param SC: 股票代码
    :return:
    """
    # SC = "000561.SZ"

    # 从MySQL中读取为规整的因子的全部数据
    factor = Read_One_Stock_Factor(SC).select_all_data()

    # 将"index"列作为DataFrame的Index
    factor.index = factor["index"]

    # 读取指数信息
    RI = Read_Index()

    # 获取可行的交易日
    trade_date = RI.index_data["trade_date"]

    # 选出2008年以后的交易日
    sta_date = datetime.datetime.strptime("20080101", "%Y%m%d")

    trade_date = trade_date[trade_date > sta_date]

    # 选出在交易日出现的数据
    factor_td = factor.loc[trade_date, :]

    # 重新构建成按周分组处理后后的数据
    factor_f = factor_formulate(factor_td)

    # 将数据规范化
    factor_n = normalization_data(factor_f)

    return factor_n



# 读取一直股票的全部因子
# factor_df = read_factor(SC)