# 导入需要用到的模块
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta
import tushare as ts
from tushare import pro
from scipy.stats import linregress
import pymysql
from sqlalchemy import create_engine


class Factor_Data(object):
    def __init__(self,
                 start='20100101',
                 end='20191231',
                 code='000021.SZ', ):
        self.start = start
        self.end = end
        self.token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
        self.pro = ts.pro_api(self.token)
        # self.table_name = table_name
        self.code = code
        self.factors = pd.DataFrame({'date': pd.date_range(self.start, self.end, freq='D')},
                                    index=pd.date_range(self.start, self.end, freq='D'))

    def formulate_factor_full(self, df):
        """
        在tushare上截取相关的时间信息，然后补全每一天的
        :param df: 需要input的DataFrame，需要有 "trade_date" 或 "ann_date"
        :return: 返回一个补全所有自然日的DataFrame
        """
        # start = '20100101'
        # end = '20200101'
        # df = pro.income(ts_code=code, start_date=start, end_date=end)
        # date_index = pd.date_range(start, end, freq='D')

        # 删去可能重复出现的column，主要是时间数据
        if "date" in df.columns:
            df.drop(["date"], axis=1, inplace=True)
        if "trade_date" in df.columns:
            date = pd.to_datetime(df["trade_date"], format='%Y%m%d')
            df.insert(df.shape[1], 'date', date)
        if "ann_date" in df.columns:
            date = pd.to_datetime(df["ann_date"], format='%Y%m%d')
            df.insert(df.shape[1], 'date', date)

        # 构建新的index，用"date"命名，并准备将其作为索引
        df_new = pd.DataFrame({'date': pd.date_range(self.start, self.end, freq='D')})

        # 将原来的 DataFrame merge 到新的 df 上，用的是外连接
        df_new = pd.merge(df_new, df, how='outer', on='date')

        # 将所有数据的索引全部改为用index索引
        df_new.index = df_new.date

        # 对于缺失值，采用向前填充的方法
        df_new = df_new.fillna(method='bfill')
        return df_new

    # 个股每日行情数据
    def get_daily_data(self):
        try:
            # 获取每日交易数据
            df0 = pro.daily(ts_code=self.code,
                            start_date=self.start,
                            end_date=self.end)

            # 获取每日指标
            df1 = pro.daily_basic(ts_code=self.code,
                                  start_date=self.start,
                                  end_date=self.end,
                                  fields='ts_code,trade_date,turnover_rate')

            # # 获取复权行情
            # df2 = pro.pro_bar(ts_code=self.code, start_date=self.start, end_date=self.end)
            #
            # # 获取复权因子
            # df3 = pro.adj_factor(ts_code=self.code, start_date=self.start, end_date=self.end)

            df_daily = pd.merge(df0, df1)  # 合并数据

            df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"],
                                                    format='%Y%m%d')
            df_daily.index = df_daily["trade_date"]

            self.daily_data = df_daily


        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    # 指数每日行情数据
    def get_index_daily_data(self):
        try:
            # 获取每日交易数据
            df_index = pro.index_daily(ts_code="000300.SH",
                                  start_date=self.start,
                                  end_date=self.end,
                                  fields='ts_code,trade_date,close,pre_close,pct_chg')
            df_index["trade_date"] = pd.to_datetime(df_index["trade_date"],
                                                    format='%Y%m%d')
            df_index.index = df_index["trade_date"]

            self.df_index = df_index

        except Exception as e:
            print("index 000300.SH is failed!")
            print(e)

    # 个股每月行情数据
    def get_monthly_data(self):
        try:
            df_monthly = pro.monthly(ts_code=self.code,
                                     start_date=self.start,
                                     end_date=self.end,
                                     fields='ts_code,trade_date,close,pre_close,pct_chg')
            # 月涨跌幅，叫做动量因子
            df_monthly["trade_date"] = pd.to_datetime(df_monthly["trade_date"],
                                                      format='%Y%m%d')
            df_monthly.index = df_monthly["trade_date"]
            self.monthly_data = df_monthly

        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    def fa_monthly_return(self):
        df_monthly = self.monthly_data.copy()
        df_monthly["return_1m"] = df_monthly['pct_chg']
        df_monthly["return_3m"] = df_monthly['close'].pct_change(periods=3)
        df_monthly["return_6m"] = df_monthly['close'].pct_change(periods=6)
        df_monthly["return_12m"] = df_monthly['close'].pct_change(periods=12)

        df_tmpt = df_monthly.iloc[:, -4:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_monthly_avg_turnover_return(self):
        mon_avg_turnover = self.daily_data.loc[:, ["trade_date", "turnover_rate"]]. \
            resample('M', on='trade_date').mean()

        df_monthly = self.monthly_data.copy()

        mon_avg_turnover.index = df_monthly.index
        mon_avg_turnover = mon_avg_turnover.iloc[:, 0]

        df_monthly["wgt_return_1m"] = df_monthly['pct_chg'] / mon_avg_turnover  # 一个月
        df_monthly["wgt_return_3m"] = df_monthly['close'].pct_change(periods=3) / mon_avg_turnover  # 三个月
        df_monthly["wgt_return_6m"] = df_monthly['close'].pct_change(periods=6) / mon_avg_turnover  # 六个月
        df_monthly["wgt_return_12m"] = df_monthly['close'].pct_change(periods=12) / mon_avg_turnover

        df_tmpt = df_monthly.iloc[:, -4:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_turnover(self):
        df_daily_basic = self.daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        mon_avg_turnover = df_daily_basic.resample('M', on='trade_date').mean()
        mon_avg_turnover["turn_1m"] = mon_avg_turnover['turnover_rate']
        mon_avg_turnover["turn_3m"] = mon_avg_turnover['turnover_rate'].pct_change(periods=3)
        mon_avg_turnover["turn_6m"] = mon_avg_turnover['turnover_rate'].pct_change(periods=6)

        df_tmpt = mon_avg_turnover.iloc[:, -3:]
        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_wgt_turnover(self):
        pass
        # df_daily_basic = self.daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        # mon_avg_turnover = df_daily_basic.resample('M', on='trade_date').mean()
        # year_avg_turnover = df_daily_basic.resample('Y', on='trade_date').mean()  # 最近一年内日均换手率
        #
        # wgt_turn_1m = mon_avg_turnover['turnover_rate'] / year_avg_turnover
        # wgt_turn_3m = pd.DataFrame({'month': mon_avg_turnover['trade_date'],
        #                             i: mon_avg_turnover['turnover_rate'].pct_change(periods=3) / year_avg_turnover});
        # wgt_turn_6m = pd.DataFrame({'month': mon_avg_turnover['trade_date'],
        #                             i: mon_avg_turnover['turnover_rate'].pct_change(periods=6)} / year_avg_turnover);

    def fa_daily_return(self):
        df_daily = self.daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        # df_daily = daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
        daily_return.columns = ["daily_return"]
        daily_return["trade_date"] = df_daily["trade_date"]
        # 因为袁老师的原因 这些代码要修改

        daily_return_1m = daily_return.resample('M', on='trade_date').mean()
        daily_return_3m = daily_return.resample('3M', on='trade_date').mean()
        daily_return_6m = daily_return.resample('6M', on='trade_date').mean()
        daily_return_12m = daily_return.resample('Y', on='trade_date').mean()

        df_tmpt = pd.merge(daily_return_1m, daily_return_3m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, daily_return_6m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, daily_return_12m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt.columns = ["daily_return_1m", "daily_return_3m",
                           "daily_return_6m", "daily_return_12m"]
        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_std_Nm(self):
        df_daily = self.daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
        daily_return.columns = ["daily_return"]
        daily_return["trade_date"] = df_daily["trade_date"]

        std_1m = daily_return.resample('M', on='trade_date').std()
        std_3m = daily_return.resample('3M', on='trade_date').std()
        std_6m = daily_return.resample('6M', on='trade_date').std()
        std_12m = daily_return.resample('Y', on='trade_date').std()

        df_tmpt = pd.merge(std_1m, std_3m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, std_6m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, std_12m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt.columns = ["std_1m", "std_3m",
                           "std_6m", "std_12m"]
        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def find_residual_std(self, arr):
        x = arr[:, 0]
        y = arr[:, 1]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_pred = x * slope + intercept
        residual = y - y_pred
        red_std_err = residual.std()
        return red_std_err

    def fa_id_std_Nm(self):
        df_daily = self.daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        df_index = self.df_index.loc[:, ["trade_date", "close", "pre_close"]].copy()

        daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
        index_return = pd.DataFrame(df_index['close'] - df_index['pre_close'])

        daily_return.columns = ["daily_return"]
        index_return.columns = ["index_return"]

        combine_data = pd.merge(daily_return["daily_return"], index_return["index_return"],
                                left_index=True, right_index=True)
        combine_data["trade_date"] = combine_data.index
        combine_data.resample('M', on='trade_date').apply(find_residual_std)

        daily_return["trade_date"] = df_daily["trade_date"]



        daily_return.resample('M', on='trade_date').apply(self.find_residual_std)

    def process(self):
        self.get_daily_data()
        self.get_monthly_data()

        self.fa_monthly_return()
        self.fa_monthly_avg_turnover_return()
        self.fa_turnover()
        # self.fa_wgt_turnover()
        self.fa_volatility()
        self.fa_std_Nm()

    # 保存数据到数据库
    def finance_data(self, code):
        try:
            # 获取每日交易数据
            df0 = pro.income(ts_code=code, start_date=self.start, end_date=self.end,
                             fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
            df1 = pro.cashflow(ts_code=code, start_date=self.start, end_date=self.end)

            df = pd.merge(df0, df1)  # 合并数据

            return df

        except Exception as e:
            print(code + " is failed!")
            print(e)
