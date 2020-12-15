# 导入需要用到的模块
import numpy as np
import pandas as pd
import datetime
import pymysql
import tushare as ts
from sqlalchemy import create_engine

class Read_One_Stock():

    # 初始化数据，定义 MySQL 访问链接参数
    def __init__(self, SC_Code):
        self.conn = pymysql.connect(
            host="localhost",
            database="stock",
            user="root",
            password="zzzzzzzz",
            port=3306,
            charset='utf8'
        )
        self.SC_Code = SC_Code

    # 获取每天的收盘价
    def select_all_data(self):

        sqlcmd = "SELECT * FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    def select_pct_chg(self):
        # 读取每天的收盘价
        sqlcmd = "SELECT trade_date, pct_chg FROM`{}`".format(self.SC_Code)
        # sqlcmd = "SELECT trade_date, pct_chg FROM`{}`".format("000021.SZ")
        table = pd.read_sql(sqlcmd, self.conn)
        # table = pd.read_sql(sqlcmd, conn)
        table["pct_chg"] = table["pct_chg"]/100
        table["trade_date"] = pd.to_datetime(table["trade_date"],
                                                format='%Y%m%d')
        table.index = table["trade_date"]

        table.sort_index(inplace=True)

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


def current_stocks(df_total, t):
    """
    获取一个交易时间中，股票池中有哪些股票
    将上市时间在 t 之前的股票作为股票池中的股票

    :param df_total: 一个 DataFrame{"ts_code", "list_date"}
    :param t: 一个时间点
    :return: 返回一个列表
    """
    t = "20100101"
    t = datetime.datetime.strptime(t, '%Y%m%d')
    df_in_pool = df_total[df_total["list_date"] < t]["ts_code"].copy()

    return list(df_in_pool)



def get_list_date():
    """
    获取市场上所有股票的基本信息
    返回的是一个 DataFrame，上市时间是有有效信息
    :return:
    """
    token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
    pro = ts.pro_api(token)
    data = pro.stock_basic(exchange='',
                           list_status='L',
                           fields='ts_code,'
                                  'list_date')
    return data

