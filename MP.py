import tushare as ts
import pandas as pd

token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
start = '20100101'
end = '20200101'
code = "000021.SZ"
pro = ts.pro_api(token)


def formulate_factor_full(df):
    """
    在tushare上截取相关的时间信息，然后补全每一天的
    :param df: 需要input的DataFrame，需要有 "trade_date" 或 "ann_date"
    :return: 返回一个补全所有自然日的DataFrame
    """
    # start = '20100101'
    # end = '20200101'
    # df = pro.income(ts_code=code, start_date=start, end_date=end)
    # date_index = pd.date_range(start, end, freq='D')

    if "date" in df.columns:
        df.drop(["date"], axis=1, inplace=True)
    if "trade_date" in df.columns:
        date = pd.to_datetime(df["trade_date"], format='%Y%m%d')
        df.insert(df.shape[1], 'date', date)
    if "ann_date" in df.columns:
        date = pd.to_datetime(df["ann_date"], format='%Y%m%d')
        df.insert(df.shape[1], 'date', date)
    df_new = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    df_new = pd.merge(df_new, df, how='outer', on='date')
    df_new.index = df_new.date
    df_new = df_new.fillna(method='bfill')
    return df_new


df_monthly = pro.monthly(ts_code=code, start_date=start, end_date=end,
                         fields='ts_code,trade_date,close,pre_close,pct_chg')
df_monthly["trade_date"] = pd.to_datetime(df_monthly["trade_date"],
                                          format='%Y%m%d')
df_monthly.index = df_monthly["trade_date"]

df_monthly["return_1m"] = df_monthly['pct_chg']
df_monthly["return_3m"] = df_monthly['close'].pct_change(periods=3)
df_monthly["return_6m"] = df_monthly['close'].pct_change(periods=6)
df_monthly["return_12m"] = df_monthly['close'].pct_change(periods=12)
df_tmpt = df_monthly.iloc[:, -4:]

df_new = pd.DataFrame({'date': pd.date_range(start, end, freq='D')},
                      index=pd.date_range(start, end, freq='D'))
df_new = pd.merge(df_new, df_tmpt, how='outer',
                  left_index=True, right_index=True)

df_monthly = formulate_factor_full(df_monthly)
return_3m = df_monthly['close'].pct_change(periods=3)  # 三个月

factors = pd.DataFrame({'date': pd.date_range(start, end, freq='D')},
                       index=pd.date_range(start, end, freq='D'))

mon_avg_turnover = daily_data.loc[:, ["trade_date", "turnover_rate"]]. \
    resample('M', on='trade_date').mean()
fd = Factor_Data()
fd.process()
factors = fd.factors
fd.get_daily_data()
daily_data = fd.daily_data.copy()
monthly_data = fd.monthly_data.copy()
daily_data.columns

fd.monthly_data()
fd.factors
mon_avg_turnover = mon_avg_turnover.iloc[:, 0]
type(df_monthly['pct_chg'])
type(mon_avg_turnover)

test = pd.DataFrame(df_daily["trade_date"].copy())
test["trade_date"] = df_daily["trade_date"]
test["ha"] = 1
test.resample('M', on='trade_date').std()

df_daily = daily_data.loc[:, ["close", "pre_close"]].copy()
df_daily_basic = daily_data.loc[:, ["trade_date", "turnover_rate"]].copy()

df_index = pro.index_daily(ts_code="000300.SH",
                           start_date=start,
                           end_date=end,
                           fields='ts_code,trade_date,close,pre_close,pct_chg')
df_index["trade_date"] = pd.to_datetime(df_index["trade_date"],
                                        format='%Y%m%d')
df_index.index = df_index["trade_date"]

arr = combine_data.copy()


def find_residual_std(arr):
    arr = np.array(arr.iloc[:, [0, 1]])
    x = arr[:, 0]
    y = arr[:, 1]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = x * slope + intercept
    residual = y - y_pred
    red_std_err = residual.std()
    return red_std_err


find_residual_std(combine_data)

y = daily_return["daily_return"].copy() + 1
test = pd.merge(daily_return["daily_return"], y)

combine_data.rolling(3, axis=0).std()

df_daily_basic = daily_data.loc[:, ["trade_date", "amount"]].copy()

df_daily = daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
df_index = df_index.loc[:, ["trade_date", "close", "pre_close"]].copy()

daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
index_return = pd.DataFrame(df_index['close'] - df_index['pre_close'])

df_daily = daily_data.loc[:, ["trade_date", "close", "pre_close"]].copy()
df_index = df_index.loc[:, ["trade_date", "close", "pre_close"]].copy()

daily_return = pd.DataFrame(df_daily['close'] - df_daily['pre_close'])
index_return = pd.DataFrame(df_index['close'] - df_index['pre_close'])

daily_return.columns = ["daily_return"]
index_return.columns = ["index_return"]

combine_data = pd.merge(daily_return["daily_return"], index_return["index_return"],
                        left_index=True, right_index=True)
combine_data["trade_date"] = combine_data.index
id_std_1m = pd.DataFrame(combine_data.resample('M', on='trade_date').apply(find_residual_std))
id_std_3m = pd.DataFrame(combine_data.resample('3M', on='trade_date').apply(find_residual_std))
id_std_6m = pd.DataFrame(combine_data.resample('6M', on='trade_date').apply(find_residual_std))
id_std_12m = pd.DataFrame(combine_data.resample('Y', on='trade_date').apply(find_residual_std))

df_tmpt = pd.merge(id_std_1m, id_std_3m, how='outer',
                   left_index=True, right_index=True)
df_tmpt = pd.merge(df_tmpt, id_std_6m, how='outer',
                   left_index=True, right_index=True)
df_tmpt = pd.merge(df_tmpt, id_std_12m, how='outer',
                   left_index=True, right_index=True)
df_tmpt.columns = ["id_std_1m", "id_std_3m",
                   "id_std_6m", "id_std_12m"]

factors = pd.merge(factors, df_tmpt, how='outer',
                   left_index=True, right_index=True)

# 获取财务指标
def get_finance_data():
    df_finance = pro.income(ts_code=code,
                            start_date=start,
                            end_date=end,
                            fields='ts_code,end_date,report_type,comp_type,basic_eps,diluted_eps,ebit')

    df_finance["end_date"] = pd.to_datetime(df_finance["end_date"],
                                            format='%Y%m%d')
    df_finance.index = df_finance["end_date"]
    df_finance.sort_index(inplace=True)

    return(df_finance)




# 获取每日指标
def get_daily_data():
    df0 = pro.daily(ts_code=code,
                    start_date=start,
                    end_date=end)

    df1 = pro.daily_basic(ts_code=code,
                          start_date=start,
                          end_date=end,
                          fields='ts_code,trade_date,turnover_rate, total_mv')
    df_daily = pd.merge(df0, df1)  # 合并数据
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"],
                                            format='%Y%m%d')
    df_daily.index = df_daily["trade_date"]
    df_daily.sort_index(inplace=True)
    return df_daily

daily_data = get_daily_data()
finance_data = get_finance_data()

df_finance = finance_data.loc[:, ["ebit"]].copy()
df_daily = daily_data.loc[:, ["total_mv"]].copy()
df_tmpt = pd.merge(df_finance, df_daily, how='outer',
                   left_index=True, right_index=True)
df_tmpt = df_tmpt.fillna(method='bfill')
df_tmpt["EBIT2EV"] = df_tmpt["ebit"] / df_tmpt['total_mv']


start = '20100101'
end = '20191231'
code = '000021.SZ'
pro = ts.pro_api(token)
fd = Factor_Data()
fd.process()
factors = fd.factors
