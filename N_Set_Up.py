import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts

from datetime import datetime

import statsmodels.api as sm
import scipy.stats as st



# 读取从 Wind 上分类的信息技术板块
Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

# 将数据重新按照英文命名
Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

# 提取在股票池中的股票代码
SC_in_pool = Stock_code_pool['Stock_Code']


token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

pro = ts.pro_api(token)

SC = '000021.SZ'


# 获取每次调仓的时间
factor_sample = FF.read_factor("000021.SZ")
time_total_index = FF.Read_Index().index_data.resample('W', on="date").mean().index
time_total_index_np = np.array(time_total_index)

Sta_Time = "20100101"
Sta_Time = datetime.strptime(Sta_Time, '%Y%m%d')
time_valid_index = time_total_index[time_total_index > Sta_Time].copy()  # 每次调仓的时间点

# 获取股票上市时间
df_total = SF.get_list_date()
df_total.index = df_total["ts_code"]
df_total = df_total.loc[SC_in_pool, :]
df_total = df_total.dropna(axis=0)
df_total["list_date"] = pd.to_datetime(df_total["list_date"])

for t_i in range(len(time_valid_index)):
    # t_i = 0
    time_tp = time_valid_index[t_i]
    time_tp = np.datetime64(time_tp)
    time_on_tp_index_order = np.where(time_total_index_np == time_tp)[0][0]
    time_in_use_index = time_total_index[time_on_tp_index_order-100:time_on_tp_index_order]


    stock_in_pool = SF.current_stocks(df_total, Sta_Time)
    # stock_in_pool = current_stocks(df_total, Sta_Time)




    # len(stock_in_pool)
    for s_i in range(len(stock_in_pool)):
        # s_i = 0

        SC = SC_in_pool[s_i]
        ROS = SF.Read_One_Stock(SC)
        return_of_stock = ROS.select_pct_chg()
        return_of_total = pd.DataFrame({'date': pd.date_range('20070101', '20191231', freq='D')},
                                    index=pd.date_range('20070101', '20191231', freq='D'))

        return_of_total = pd.merge(return_of_total, return_of_stock, how='outer',
                                  left_index=True, right_index=True)
        return_of_total = return_of_total.fillna(method='bfill')
        return_df_of_stock_in_use = return_of_total.loc[time_in_use_index, :]
        return_of_stock_in_use = return_df_of_stock_in_use["pct_chg"]
        # return_of_stock_in_use.shape


        factor_df = FF.read_factor(SC)
        factor_df_in_use = factor_df.loc[time_in_use_index, :]  #选取100次历史数据
        # factor_df_in_use.shape
        # factor_df_in_use.columns

        # 开始因子检验
        factor_columns = factor_df.columns
        factor_index_list = []
        for f_i in range(len(factor_columns)):
            # f_i = 0
            factor_name = factor_columns[f_i]
            factor_X = factor_df_in_use[factor_name]
            factor_data_input = pd.merge(return_of_stock_in_use, factor_X, how='outer',
                        left_index=True, right_index=True)
            factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
            factor_data_input = factor_data_input.dropna(axis=0).copy()


            # t检验
            ols = sm.OLS(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])
            output = ols.fit()
            OLS_params = output.params[-1]       # 这个是什么东西
            OLS_t_test = output.tvalues[-1]      # 这个是t的值
            if not True:
                pass


            # IRIC检验
            IC = st.pearsonr(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])[0]
            if not True:
                pass

            # 平稳性检验
            t = sm.tsa.stattools.adfuller(factor_data_input["Factor_Value"])
            if not True:
                pass

        # 相关性检验



        for f_i in range(factor_sub.shape[1]):
            # f_i = 4
            x = factor.iloc[:,f_i]
            x_new = MF.pred_by_LSTM_total(x)[0][0]



f0 = factor.iloc[:, 0]
f1 = factor.iloc[:, 1]
import statsmodels.api as sm
def t_test(result,period,start_date,end_date,factor):
    #获取申万一级行业数据

    #生成空的dict，存储t检验、IC检验结果
    WLS_params = {}
    WLS_t_test = {}
    IC = {}

    date_period = get_period_date(period,start_date,end_date)

    for date in date_period[:-2]:
        temp = result[result['date'] == date]
        X = f0
        Y = return_of_stock_valid
        # WLS回归
        wls = sm.WLS(Y, X, weights=temp['Weight'])
        ols = sm.OLS(Y, X)
        output = ols.fit()
        WLS_params[date] = output.params[-1]
        WLS_t_test[date] = output.tvalues[-1]
        #IC检验
        IC[date]=st.pearsonr(Y, temp[factor])[0]
        print date+' getted!!!'

    return WLS_params,WLS_t_test,IC



from scipy import stats
def get_ic(datas):
    factors_name=[i for i in datas.columns.tolist() if i not in ['next_ret']]  #得到因子名
    ic = datas.groupby(level=0).apply(lambda data: [stats.spearmanr(data[factor],data['next_ret'])[1] for factor in factors_name])  #得到的是以列表为值的序列
    ic = pd.DataFrame(ic.tolist(), index=ic.index, columns=factors_name)  #得到各因子IC值,一个list为一个列
    return ic
