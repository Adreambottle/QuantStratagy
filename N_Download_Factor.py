# 导入需要用到的模块
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime, timedelta
import tushare as ts
from scipy.stats import linregress
import pymysql
from sqlalchemy import create_engine


class Factor_Data(object):

    def __init__(self,
                 start='20090101',
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
        self.process()
        self.factors = self.factors.fillna(method='bfill')

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
        """
        获取单股的每日交易数据
        分为 pro.daily pro.daily_basic 两个接口
        :return: 将下载的数据保存到 self.daily_data
        """
        try:
            # 获取每日交易数据
            df0 = self.pro.daily(ts_code=self.code,
                                 start_date=self.start,
                                 end_date=self.end)

            # 获取每日指标
            df1 = self.pro.daily_basic(ts_code=self.code,
                                       start_date=self.start,
                                       end_date=self.end,
                                       fields='ts_code,'
                                              'trade_date,'
                                              'turnover_rate,'
                                              'total_mv,'
                                              'turnover_rate_f,'
                                              'dv_ratio')

            df_daily = pd.merge(df0, df1)  # 合并数据

            df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"],
                                                    format='%Y%m%d')
            df_daily.index = df_daily["trade_date"]

            df_daily.sort_index(inplace=True)

            self.daily_data = df_daily


        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    # 指数每日行情数据
    def get_index_daily_data(self):
        """
        获取指数的每日交易数据
        :return: 将下载的数据保存到 self.index_data
        """
        try:
            # 获取每日交易数据
            df_index = self.pro.index_daily(ts_code="000300.SH",
                                            start_date=self.start,
                                            end_date=self.end,
                                            fields='ts_code,'
                                                   'trade_date,'
                                                   'close,'
                                                   'pre_close,'
                                                   'pct_chg')
            df_index["trade_date"] = pd.to_datetime(df_index["trade_date"],
                                                    format='%Y%m%d')
            df_index.index = df_index["trade_date"]
            df_index.sort_index(inplace=True)

            self.index_data = df_index

        except Exception as e:
            print("index 000300.SH is failed!")
            print(e)

    # 个股每月行情数据
    def get_monthly_data(self):
        """
        获取单股的每月交易数据
        :return: 将下载的数据保存到 self.monthly_data
        """
        try:
            df_monthly = self.pro.monthly(ts_code=self.code,
                                          start_date=self.start,
                                          end_date=self.end,
                                          fields='ts_code,'
                                                 'trade_date,'
                                                 'close,'
                                                 'pre_close,'
                                                 'pct_chg')

            df_monthly["trade_date"] = pd.to_datetime(df_monthly["trade_date"],
                                                      format='%Y%m%d')
            df_monthly.index = df_monthly["trade_date"]
            df_monthly.sort_index(inplace=True)

            self.monthly_data = df_monthly

        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    def get_finance_data(self):
        """
        获取单股的财务数据
        分为 [pro.income, pro.fina_indicator, pro.cashflow, pro.balancesheet] 四个接口
        分别调取之后，merge 在一起
        :return: 将下载的数据保存到 self.finance_data
        """
        try:
            """
            # df0  ==>  income
            # 利润表
            """
            df0 = self.pro.income(ts_code=self.code,
                                  start_date=self.start,
                                  end_date=self.end,
                                  fields='ts_code,'
                                         'end_date,'
                                         'basic_eps,'
                                         'diluted_eps,'
                                         'ebit,'
                                         'revenue')

            df0["end_date"] = pd.to_datetime(df0["end_date"],
                                             format='%Y%m%d')
            df0.index = df0["end_date"]
            df0.sort_index(inplace=True)
            df0.drop(["end_date"], axis=1, inplace=True)


            """
            # df1  ==>  fina_indicator
            # 财务指标数据
            """
            df1 = self.pro.fina_indicator(ts_code=self.code,
                                          start_date=self.start,
                                          end_date=self.end,
                                          fields='ts_code,end_date,'
                                                 'q_profit_yoy,'
                                                 'ocfps,'
                                                 'roe,'
                                                 'opincome_of_ebt,'
                                                 'ocf_to_or,'
                                                 'currentdebt_to_debt,'
                                                 'current_ratio,'
                                                 'assets_turn')
            df1["end_date"] = pd.to_datetime(df1["end_date"],
                                             format='%Y%m%d')
            df1.index = df1["end_date"]
            df1.sort_index(inplace=True)
            df1.drop(["end_date"], axis=1, inplace=True)


            """
            # df2  ==>  cashflow
            # 现金流量表
            """
            df2 = self.pro.cashflow(ts_code=self.code,
                                    start_date=self.start,
                                    end_date=self.end,
                                    fields='ts_code,'
                                           'end_date,'
                                           'n_cashflow_act,'
                                           'free_cashflow,'
                                           'n_cashflow_act')
            df2["end_date"] = pd.to_datetime(df2["end_date"],
                                             format='%Y%m%d')
            df2.index = df2["end_date"]
            df2.sort_index(inplace=True)
            df2.drop(["end_date"], axis=1, inplace=True)



            """
            # df3  ==>  balancesheet
            # 资产负债表
            """
            df3 = self.pro.balancesheet(ts_code=self.code,
                                        start_date=self.start,
                                        end_date=self.end,
                                        fields='ts_code,'
                                               'end_date,'
                                               'total_assets,'
                                               'total_liab')

            df3["end_date"] = pd.to_datetime(df3["end_date"],
                                             format='%Y%m%d')
            df3.index = df3["end_date"]
            df3.sort_index(inplace=True)
            df3.drop(["end_date"], axis=1, inplace=True)

            """
            将数据拼接在一起
            """
            df_finance = pd.merge(df0, df1, how='outer',
                                  left_index=True, right_index=True)
            df_finance = pd.merge(df_finance, df2, how='outer',
                                  left_index=True, right_index=True)
            df_finance = pd.merge(df_finance, df3, how='outer',
                                  left_index=True, right_index=True)
            df_finance["end_date"] = df_finance.index
            df_finance = df_finance[~df_finance.index.duplicated(keep='first')]

            self.finance_data = df_finance

        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    """
    计算因子，并且拼合在 self.factor 上
    """

    # 动量因子
    def fa_monthly_return(self):
        """
        计算个股N个月的收益率
        因子: return_1m
             return_3m
             return_6m
             return_12m
        :return: 添加以上因子
        """
        df_monthly = self.monthly_data.copy()
        df_monthly["return_1m"] = df_monthly['pct_chg']
        df_monthly["return_3m"] = df_monthly['close'].pct_change(periods=3)
        df_monthly["return_6m"] = df_monthly['close'].pct_change(periods=6)
        df_monthly["return_12m"] = df_monthly['close'].pct_change(periods=12)

        df_tmpt = df_monthly.iloc[:, -4:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_monthly_avg_turnover_return(self):
        """
        个股最近N个月内以每日换手率作为权重对每日收益率求算术平均值
        因子: wgt_return_1m
             wgt_return_3m
             wgt_return_6m
             wgt_return_12m
        :return:添加以上因子
        """
        mon_avg_turnover = self.daily_data.loc[:, ["trade_date", "turnover_rate"]]. \
            resample('M', on='trade_date').mean()

        df_monthly = self.monthly_data.copy()
        df_monthly = pd.merge(df_monthly, mon_avg_turnover, how='outer',
                                left_index=True, right_index=True)
        df_monthly.iloc[:,:6] = df_monthly.iloc[:,:6].fillna(method="ffill")
        df_monthly = df_monthly.loc[mon_avg_turnover.index].copy()

        mon_avg_turnover = df_monthly["turnover_rate"]

        df_monthly["wgt_return_1m"] = df_monthly['pct_chg'] / mon_avg_turnover  # 一个月
        df_monthly["wgt_return_3m"] = df_monthly['close'].pct_change(periods=3) / mon_avg_turnover  # 三个月
        df_monthly["wgt_return_6m"] = df_monthly['close'].pct_change(periods=6) / mon_avg_turnover  # 六个月
        df_monthly["wgt_return_12m"] = df_monthly['close'].pct_change(periods=12) / mon_avg_turnover

        df_tmpt = df_monthly.iloc[:, -4:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    # 换手率因子
    def fa_turnover(self):
        """
        个股最近N个月内日均换手率
        个股最近N个月内以每日换手率作为权重对每日收益率求算术平均值
        因子: turn_1m
             turn_3m
             turn_6m
        :return:添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        mon_avg_turnover = df_daily_basic.resample('M', on='trade_date').mean()
        mon_avg_turnover["turn_1m"] = mon_avg_turnover['turnover_rate']
        mon_avg_turnover["turn_3m"] = mon_avg_turnover['turnover_rate'].pct_change(periods=3)
        mon_avg_turnover["turn_6m"] = mon_avg_turnover['turnover_rate'].pct_change(periods=6)

        df_tmpt = mon_avg_turnover.iloc[:, -3:]
        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_wgt_turnover(self):
        """
        个股最近N个月内月度换手率对年度换手率的比
        因子: wgt_turn_1m
             wgt_turn_3m
             wgt_turn_6m
        :return:添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        # df_daily_basic = daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        mon_avg_turnover = df_daily_basic.resample('M', on='trade_date').mean()
        mon3_avg_turnover = df_daily_basic.resample('3M', on='trade_date').mean()
        mon6_avg_turnover = df_daily_basic.resample('6M', on='trade_date').mean()

        year_avg_turnover = df_daily_basic.resample('Y', on='trade_date').mean()  # 最近一年内日均换手率
        df_tmpt = pd.merge(mon_avg_turnover, mon3_avg_turnover, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, mon6_avg_turnover, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, year_avg_turnover, how='outer',
                           left_index=True, right_index=True)

        df_tmpt = df_tmpt.fillna(method='bfill')
        df_tmpt.columns = ["turnover_rate_M",
                           "turnover_rate_3M",
                           "turnover_rate_6M",
                           "turnover_rate_Y"]

        df_tmpt["wgt_turn_1m"] = df_tmpt["turnover_rate_M"] / df_tmpt["turnover_rate_Y"]
        df_tmpt["wgt_turn_3m"] = df_tmpt["turnover_rate_3M"] / df_tmpt["turnover_rate_Y"]
        df_tmpt["wgt_turn_6m"] = df_tmpt["turnover_rate_6M"] / df_tmpt["turnover_rate_Y"]

        self.factors = pd.merge(self.factors, df_tmpt.iloc[:, -3:], how='outer',
                                left_index=True, right_index=True)

    def fa_daily_return(self):
        """
        月度营收率
        因子：
            daily_return_1m
            daily_return_3m
            daily_return_6m
            daily_return_12m
        :return:
        """
        df_daily = self.daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        # df_daily = daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
        daily_return.columns = ["daily_return"]
        daily_return["trade_date"] = df_daily["trade_date"]

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

    # 波动率因子
    def fa_std_Nm(self):
        """
        std_Nm
        普通波动率：个股最近N个月内日的标准差
        因子: std_1m
             std_3m
             std_6m
             std_12m

        :return: 添加以上因子
        """
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
        """
        从 scipy.stats 模块里面调用 linregress 简单线性回归
        把个股数据作为 x ，指数数据作为 y ，建立线性回归模型
        因为特质波动率需要残差的标准差
        需要用到 apply 的方式广播这个自建函数

        :param arr: input arr 是一个 DataFrame，
                    第一列是个股return，第二列是指数return
        :return: 返回的是残差residual的标准差
        """
        if arr.empty:
            return 0
        else:
            arr = np.array(arr.iloc[:, [0, 1]])
            x = arr[:, 0]
            y = arr[:, 1]

            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            y_pred = x * slope + intercept
            residual = y - y_pred
            red_std_err = residual.std()
            return red_std_err

    def fa_id_std_Nm(self):
        """
        id_std_Nm
        特质波动率：个股最近N个月内日收益率对沪深300收益率序列进行一元线性回归的残差的标准差
        因子: id_std_1m
             id_std_3m
             id_std_6m
             id_std_12m
        :return: 添加以上因子
        """
        df_daily = self.daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
        df_index = self.index_data.loc[:, ["trade_date", "close", "pre_close"]].copy()

        daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
        index_return = pd.DataFrame(df_index['close'] - df_index['pre_close'])

        daily_return.columns = ["daily_return"]
        index_return.columns = ["index_return"]

        combine_data = pd.merge(daily_return["daily_return"], index_return["index_return"],
                                left_index=True, right_index=True)
        combine_data["trade_date"] = combine_data.index
        id_std_1m = pd.DataFrame(combine_data.resample('M', on='trade_date').apply(self.find_residual_std))
        id_std_3m = pd.DataFrame(combine_data.resample('3M', on='trade_date').apply(self.find_residual_std))
        id_std_6m = pd.DataFrame(combine_data.resample('6M', on='trade_date').apply(self.find_residual_std))
        id_std_12m = pd.DataFrame(combine_data.resample('Y', on='trade_date').apply(self.find_residual_std))

        df_tmpt = pd.merge(id_std_1m, id_std_3m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, id_std_6m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = pd.merge(df_tmpt, id_std_12m, how='outer',
                           left_index=True, right_index=True)
        df_tmpt.columns = ["id_std_1m", "id_std_3m",
                           "id_std_6m", "id_std_12m"]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    # 情绪因子
    def fa_turnover_volatility(self):
        """
        换手率想对波动率
        因子：
            turnover_vol_5d
            turnover_vol_20d
            turnover_vol_100d
        :return:添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
        df_daily_basic["turnover_vol_5d"] = df_daily_basic["turnover_rate"].rolling(5, axis=0).std()
        df_daily_basic["turnover_vol_20d"] = df_daily_basic["turnover_rate"].rolling(20, axis=0).std()
        df_daily_basic["turnover_vol_100d"] = df_daily_basic["turnover_rate"].rolling(100, axis=0).std()
        df_tmpt = df_daily_basic.iloc[:, -3:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_VSTD(self):
        """
        N日成交量标准差
        VSTD N
        因子：
            VSTD_6d
            VSTD_30d
            VSTD_100d
        :return:添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["trade_date", "amount"]].copy()
        df_daily_basic["VSTD_6d"] = df_daily_basic["amount"].rolling(5, axis=0).std()
        df_daily_basic["VSTD_30d"] = df_daily_basic["amount"].rolling(30, axis=0).std()
        df_daily_basic["VSTD_100d"] = df_daily_basic["amount"].rolling(100, axis=0).std()
        df_tmpt = df_daily_basic.iloc[:, -3:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_DAVOL_N(self):
        """
        DAVOLN
        5日平均换手率 与 120日平均换手率 的比
        因子：
            DAVOL5
            DAVOL20
            DAVOL100
        :return:添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["turnover_rate"]].copy()

        df_daily_basic["turnover_rate_5D"] = df_daily_basic["turnover_rate"].rolling(5, axis=0).mean()
        df_daily_basic["turnover_rate_20D"] = df_daily_basic["turnover_rate"].rolling(20, axis=0).mean()
        df_daily_basic["turnover_rate_100D"] = df_daily_basic["turnover_rate"].rolling(100, axis=0).mean()
        df_daily_basic["turnover_rate_120D"] = df_daily_basic["turnover_rate"].rolling(120, axis=0).mean()

        df_daily_basic["DAVOL5"] = df_daily_basic["turnover_rate_5D"] / df_daily_basic["turnover_rate_120D"]
        df_daily_basic["DAVOL20"] = df_daily_basic["turnover_rate_20D"] / df_daily_basic["turnover_rate_120D"]
        df_daily_basic["DAVOL100"] = df_daily_basic["turnover_rate_100D"] / df_daily_basic["turnover_rate_120D"]

        df_tmpt = df_daily_basic.iloc[:, -3:]

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    # 财务因子
    def fa_fin_fat(self):
        """
        添加财务因子
        因子：
            roe 单季度ROE
            opincome_of_ebt      单季度经营活动净收益/利润总额
            ocf_to_or            单季度经营现金净流量/营业收入
            currentdebt_to_debt  流动负债/负债合计
            current_ratio        流动比率
            assets_turn          总资产周转率

        :return:添加以上因子
        """
        df_finance = self.finance_data.loc[:, ["roe",
                                               "opincome_of_ebt",
                                               "ocf_to_or",
                                               "currentdebt_to_debt",
                                               "current_ratio",
                                               "assets_turn"]].copy()

        self.factors = pd.merge(self.factors, df_finance, how='outer',
                                left_index=True, right_index=True)

    # 成长因子
    def fa_grow_fat(self):
        """
        添加成长因子
        因子:
            roe_q               当季ROE同比增长率
            n_income_attr_p     当季净利润
            n_income_attr_p_q   当季净利润同比增长率
            revenue             当季营业收入
            revenue_q           当季营业收入同比增长率
            n_cashflow_act      当季经营性现金流
            n_cashflow_act_q    当季经营性现金流同比增长率
        :return:添加以上因子
        """
        # 获取 roe roe_q
        roe = self.pro.fina_indicator(ts_code=self.code,
                                      start_date=self.start,
                                      end_date=self.end,
                                      fields='end_date,roe')
        roe["end_date"] = pd.to_datetime(roe["end_date"],
                                         format='%Y%m%d')
        roe.index = roe["end_date"]
        roe.sort_index(inplace=True)
        roe["roe_q"] = roe["roe"].pct_change(periods=1)
        self.factors = pd.merge(self.factors, roe["roe_q"], how='outer',
                                left_index=True, right_index=True)

        # 获取 n_income_attr_p, n_income_attr_p_q, revenue, revenue_q
        pd_income = self.pro.income(ts_code=self.code,
                                    start_date=self.start,
                                    end_date=self.end,
                                    fields='end_date,n_income_attr_p,revenue')

        pd_income["end_date"] = pd.to_datetime(pd_income["end_date"],
                                               format='%Y%m%d')
        pd_income.index = pd_income["end_date"]
        pd_income.sort_index(inplace=True)
        pd_income["revenue_q"] = pd_income["revenue"].pct_change(periods=1)
        pd_income["n_income_attr_p_q"] = pd_income["n_income_attr_p"].pct_change(periods=1)

        self.factors = pd.merge(self.factors, pd_income.iloc[:, -4:], how='outer',
                                left_index=True, right_index=True)

        # 获取 n_cashflow_act, n_cashflow_act_q
        pd_cashflow = self.pro.cashflow(ts_code=self.code,
                                        start_date=self.start,
                                        end_date=self.end,
                                        fields='end_date,n_cashflow_act')
        pd_cashflow["end_date"] = pd.to_datetime(pd_cashflow["end_date"],
                                                 format='%Y%m%d')
        pd_cashflow.index = pd_cashflow["end_date"]
        pd_cashflow.sort_index(inplace=True)
        pd_cashflow["n_cashflow_act_q"] = pd_cashflow["n_cashflow_act"].pct_change(periods=1)
        self.factors = pd.merge(self.factors, pd_cashflow.iloc[:, -2:], how='outer',
                                left_index=True, right_index=True)

    def fa_ProfitGrowth_YOY(self):
        """
        因子：
            ProfitGrowth_YOY
            净利润增长率（季度同比）
        :return: 添加以上因子
        """
        df_finance = self.finance_data["q_profit_yoy"].copy()
        self.factors = pd.merge(self.factors, df_finance, how='outer',
                                left_index=True, right_index=True)

    # 估值因子
    def fa_BP_LF(self):
        """
        最近财报的净资产 / 总市值
        BP_LF
        :return: 添加以上因子
        """
        df_daily_basic = self.daily_data.loc[:, ["total_mv"]].copy()
        df_finance = self.finance_data.loc[:, ["total_assets", "total_liab"]].copy()
        # df_daily_basic = daily_data.loc[:, ["total_mv"]].copy()
        # df_finance = finance_data.loc[:, ["total_assets", "total_liab"]].copy()
        df_finance["total_equity"] = df_finance["total_assets"] - df_finance["total_liab"]
        df_tmpt = pd.merge(df_finance, df_daily_basic, how='outer',
                           left_index=True, right_index=True)

        df_tmpt = df_tmpt.fillna(method='bfill')
        df_tmpt["BP_LF"] = df_tmpt['total_equity'] / df_tmpt['total_mv']

        self.factors = pd.merge(self.factors, df_tmpt["BP_LF"], how='outer',
                                left_index=True, right_index=True)

    def fa_appraisement(self):
        """
        Value Factor（估值因子)

        因子：
            EBIT2EV
            过去12个月息税前利润/总市值

            OCFP：经营现金流/总市值
            经营现金流：pro.fina_indicator [“ocfps”]
            总市值：pro.daily_basic [“total_mv”]


            SP：营业收入/总市值
            营业收入：pro.income [“revenue”]
            总市值：pro.daily_basic [“total_mv”]


            NCFP：净现金流/总市值
            净现金流：pro.cashflow [“n_cashflow_act”]
            总市值：pro.daily_basic [“total_mv”]


            DP：近 12 个月现金红利(按除息日计)/总市值
            pro.daily_basic [“dv_ratio”]

            FCFP：自由现金流(最新年报)/总市值
            自由现金流：pro.cashflow [“free_cashflow”]
            总市值：pro.daily_basic [“total_mv”]

        :return: 添加以上因子
        """
        df_finance = self.finance_data.loc[:, ["ebit",
                                               "ocfps",
                                               "revenue",
                                               "n_cashflow_act",
                                               "free_cashflow"]].copy()

        df_daily = self.daily_data.loc[:, ["total_mv"]].copy()
        df_tmpt = pd.merge(df_finance, df_daily, how='outer',
                           left_index=True, right_index=True)
        df_tmpt = df_tmpt.fillna(method='bfill')

        df_tmpt["EBIT2EV"] = df_tmpt["ebit"] / df_tmpt['total_mv']

        df_tmpt["OCFP"] = df_tmpt["ocfps"] / df_tmpt['total_mv']

        df_tmpt["SP"] = df_tmpt["revenue"] / df_tmpt['total_mv']

        df_tmpt["NCFP"] = df_tmpt["n_cashflow_act"] / df_tmpt['total_mv']

        df_tmpt["FCFP"] = df_tmpt["free_cashflow"] / df_tmpt['total_mv']

        df_tmpt_new = df_tmpt.iloc[:, -5:].copy()

        self.factors = pd.merge(self.factors, df_tmpt_new, how='outer',
                                left_index=True, right_index=True)

        dv_ratio = self.daily_data.loc[:, ["dv_ratio"]].copy()
        self.factors = pd.merge(self.factors, dv_ratio, how='outer',
                                left_index=True, right_index=True)

    def fa_MV(self):
        """
        因子：
            MV 总市值
        :return:
        """
        df_daily = self.daily_data.loc[:, ["total_mv"]].copy()

        self.factors = pd.merge(self.factors, df_daily, how='outer',
                                left_index=True, right_index=True)

    # 技术因子
    def fa_PPReversal(self):
        """
        因子：
            PPReversal_1     1日均价/60日成交均价
            PPReversal_5     5日均价/60日成交均价
            PPReversal_20    20日均价/60日成交均价

        :return:添加以上因子
        """
        df_daily = self.daily_data.loc[:, ["close"]].copy()
        # df_daily = daily_data.loc[:,["close"]].copy()

        df_daily["avg_return_1"] = df_daily["close"]
        df_daily["avg_return_5"] = df_daily["close"].rolling(5, axis=0).mean()
        df_daily["avg_return_20"] = df_daily["close"].rolling(20, axis=0).mean()
        df_daily["avg_return_60"] = df_daily["close"].rolling(60, axis=0).mean()

        df_daily["PPReversal_1"] = df_daily["avg_return_1"] / df_daily["avg_return_60"]
        df_daily["PPReversal_5"] = df_daily["avg_return_5"] / df_daily["avg_return_60"]
        df_daily["PPReversal_20"] = df_daily["avg_return_20"] / df_daily["avg_return_60"]

        df_tmpt = df_daily.iloc[:, -3:].copy()

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)

    def fa_TO_Nd(self):
        """
        因子：
            TO_5d    以流通股本计算的5日日均换手率
            TO_20d   以流通股本计算的20日日均换手率
            TO_100d  以流通股本计算的100日日均换手率
        :return:
        """
        df_daily = self.daily_data.loc[:, ["turnover_rate_f"]].copy()
        # df_daily = daily_data["turnover_rate_f"].copy()

        df_daily["TO_5d"] = df_daily["turnover_rate_f"].rolling(5, axis=0).mean()
        df_daily["TO_20d"] = df_daily["turnover_rate_f"].rolling(20, axis=0).mean()
        df_daily["TO_100d"] = df_daily["turnover_rate_f"].rolling(100, axis=0).mean()

        df_tmpt = df_daily.iloc[:, -3:].copy()

        self.factors = pd.merge(self.factors, df_tmpt, how='outer',
                                left_index=True, right_index=True)



    def process(self):
        """
        调用成员函数，下载因子数据，计算因子值
        :return:
        """

        # 下载因子数据
        self.get_daily_data()
        if self.daily_data.empty:
            self.factor = pd.DataFrame()
        else:
            self.get_monthly_data()
            self.get_index_daily_data()
            self.get_finance_data()

            # 计算因子值
            self.fa_monthly_return()
            self.fa_monthly_avg_turnover_return()
            self.fa_turnover()
            self.fa_wgt_turnover()
            # self.fa_daily_return()
            self.fa_std_Nm()
            self.fa_id_std_Nm()
            self.fa_turnover_volatility()
            self.fa_VSTD()
            self.fa_DAVOL_N()
            self.fa_fin_fat()
            self.fa_grow_fat()
            self.fa_ProfitGrowth_YOY()
            self.fa_BP_LF()
            self.fa_appraisement()
            self.fa_PPReversal()
            self.fa_TO_Nd()

