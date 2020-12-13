from N_Download_Factor import Factor_Data
import tushare as ts
import numpy as np
import pandas as pd

start = '20090101'
end = '20191231'
code = '002976.SZ'
token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
pro = ts.pro_api(token)
pro.daily(ts_code=code,
          start_date=start,
          end_date=end)

FD = Factor_Data(code="000021.SZ")
FD.get_daily_data()
FD.get_monthly_data()
FD.get_finance_data()
FD.get_index_daily_data()

daily_data = FD.daily_data
monthly_data = FD.monthly_data
finance_data = FD.finance_data
index_data = FD.index_data

factor = FD.factors
factor.to_excel("/Users/meron/Desktop/factor.xlsx")


df_daily_basic = daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()
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

factors = pd.merge(factors, df_tmpt.iloc[:, -3:], how='outer',
                        left_index=True, right_index=True)
