# 导入需要用到的模块
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime,timedelta
import tushare as ts
from tushare import pro
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

class Data(object):
    def __init__(self,
                 start ='20050101',
                 end ='20200101',
                 table_name = 'fator'):
        self.start = start
        self.end = end
        self.token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
        self.table_name = table_name
        self.codes = self.get_code()
        self.cals = self.get_cals()
        self.pro = ts.pro_api(self.token)

    #获取股票交易日历
    def get_cals(self):
        #获取交易日历
        cals = pro.trade_cal(exchange = '')
        cals = cals[cals.is_open == 1].cal_date.values
        return cals

    #每日行情数据
    # def daily_data(self,code):
    #     try:
    #         df0 = pro.daily(ts_code = code, start_date = self.start, end_date = self.end)
    #         df1 = pro.adj_factor(ts_code = code, trade_date = '')
    #         #复权因子
    #         df = pd.merge(df0, df1)  #合并数据
    #
    #     except Exception as e:
    #         print(code)
    #         print(e)
    #     return df

    def factor(self,code):
        try:
            df0 = pro.income(ts_code=self.code, start_date=self.start, end_date=self.end)
            df1 = pro.balancesheet(ts_code=self.code, start_date=self.start, end_date=self.end)
            df2 = pro.cashflow(ts_code=self.code, start_date=self.start, end_date=self.end)
            df3 = pro.express(ts_code=self.code, start_date=self.start, end_date=self.end)
            # df2 = pro.cashflow(ts_code=self.code, start_date=self.start, end_date=self.end)
            # df2 = pro.cashflow(ts_code=self.code, start_date=self.start, end_date=self.end)
            # df2 = pro.cashflow(ts_code=self.code, start_date=self.start, end_date=self.end)



        except Exception as e:
            print(code)
            print(e)
        return df0



def formulate_factor_full(df):
    """
    在tushare上截取相关的时间信息，然后补全每一天的
    :param df: 需要input的DataFrame，需要有 "trade_date" 或 "ann_date"
    :return: 返回一个补全所有自然日的DataFrame
    """
    start = '20100101'
    end = '20200101'
    # df = pro.income(ts_code=code, start_date=start, end_date=end)
    # date_index = pd.date_range(start, end, freq='D')

    if "date" in df.columns:
        df.drop(["date"], axis=1, inplace=True)
    if "trade_date" in df.columns:
        date = pd.to_datetime(df["trade_date"], format = '%Y%m%d')
        df.insert(df.shape[1], 'date', date)
    if "ann_date" in df.columns:
        date = pd.to_datetime(df["ann_date"], format='%Y%m%d')
        df.insert(df.shape[1], 'date', date)
    df_new = pd.DataFrame({'date':pd.date_range(start, end, freq='D')})
    df_new = pd.merge(df_new, df, how='outer', on='date')
    df_new.index = df_new.date
    df_new = df_new.fillna(method='ffill')
    return df_new



def select_trading_date(df):
    Parallel(n_jobs=10)(delayed(my_fun)(i, j) for i in range(3) for j in range(3))


