import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
import random

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

SC = '002657.SZ'


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

# f_time_list = []

for t_i in range(len(time_valid_index)):
    # t_i = 0
    time_tp = time_valid_index[t_i]
    time_tp = np.datetime64(time_tp)
    time_on_tp_index_order = np.where(time_total_index_np == time_tp)[0][0]
    time_in_use_index = time_total_index[time_on_tp_index_order-100:time_on_tp_index_order]


    stock_in_pool = SF.current_stocks(df_total, Sta_Time)
    # stock_in_pool = current_stocks(df_total, Sta_Time)


    # 开始因子检验
    factor_columns = factor_sample.columns
    factor_index_drop_list = []
    factor_name_drop_list = []

    # f_for_one_stock = {}




    # 建立打分dict
    factor_score = {}
    for name in factor_sample.columns:
        factor_score[name] = 0



    """
    只是部分的股票用于选出因子
    """
    stock_sample_order = random.sample(list(range(len(stock_in_pool))), 3)
    # len(stock_in_pool)
    for s_i in stock_sample_order:
        # s_i = 0

        SC = SC_in_pool[s_i]
        # SC = "000561.SZ"
        print("这是股票", SC)

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
        factor_df_in_use.dropna(axis=1, inplace=True, how='all')

        # factor_df_in_use.shape
        # factor_df_in_use.columns

        # column_test = np.random.randint(0, 100, 20)
        # factor_test = factor_df_in_use.iloc[:, column_test]


        # # 建立打分dict
        # factor_score = {}
        # for name in factor_sample.columns:
        #     factor_score[name] = 0


        # 开始因子检验
        factor_columns = factor_df_in_use.columns
        factor_index_drop_list = []
        factor_name_drop_list = []

        # f_for_one_stock = {}


        for f_i in range(len(factor_columns)):
            # f_i = 0
            print("这是股票", SC, "这是因子", factor_columns[f_i])


            factor_name = factor_columns[f_i]
            # print(f_i, factor_name)
            factor_X = factor_df_in_use[factor_name]
            factor_data_input = pd.merge(return_of_stock_in_use, factor_X, how='outer',
                        left_index=True, right_index=True)
            factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
            factor_data_input = factor_data_input.dropna(axis=0).copy()

            # factor_name = factor_columns[f_i]
            # # print(f_i, factor_name)
            # factor_X = factor_df_in_use[factor_name]
            # factor_new = MF.pred_by_LSTM_total(factor_X)
            # factor_data_input = pd.merge(return_of_stock_in_use, factor_X, how='outer',
            #                              left_index=True, right_index=True)
            # factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
            # factor_data_input = factor_data_input.dropna(axis=0).copy()
            #
            # slope, intercept, r_value, p_value, std_err = \
            #     st.linregress(factor_data_input["Factor_Value"], factor_data_input["Factor_Value"])
            # f = slope
            #
            # return_i_pred = slope * factor_new
            # f_for_one_stock[factor_name] = f
            # f_hist_list = f_time_list[t_i][]


            # t检验
            ols = sm.OLS(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])
            output = ols.fit()
            OLS_params = output.params[-1]       # 这个是什么东西
            OLS_t_test = output.tvalues[-1]      # 这个是t的值
            OLS_p_value = output.pvalues[-1]      # 这个是t的值

            # if abs(OLS_p_value):
            #     factor_index_drop_list.append(f_i)
            #     factor_name_drop_list.append(factor_name)

            # IRIC检验
            IC = st.pearsonr(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])[0]

            if IC < 0.03:
                factor_index_drop_list.append(f_i)
                factor_name_drop_list.append(factor_name)

            # 平稳性检验
            t = sm.tsa.stattools.adfuller(factor_data_input["Factor_Value"])
            # if False:
                # factor_index_drop_list.append(f_i)
                # factor_name_drop_list.append(factor_name)

            # print(f_i, factor_name, OLS_p_value, IC, t[1])



        factor_name_stay_list = []
        for x in range(len(factor_columns)):
            if factor_columns[x] not in factor_name_drop_list:
                factor_name_stay_list.append(factor_columns[x])
        factor_name_stay_list_reverse = factor_name_stay_list[::-1]
        factor_df_stay_first = factor_df_in_use.loc[:, factor_name_stay_list]
        factor_df_stay_first_reverse = factor_df_stay_first.loc[:,factor_name_stay_list_reverse]
        # 相关性检验

        factor_name_valid_list = MF.get_var_no_colinear(0.9, factor_df_stay_first_reverse)
        factor_df_valid = factor_df_in_use.loc[:, factor_name_valid_list]


        # 建立打分dict
        # factor_score = {}
        # for name in factor_name_valid_list:
        #     factor_score[name] = 0

        factor_pool = {}

        # 用模型开始预测
        for f_ii in range(factor_df_valid.shape[1]):
            # f_ii = 0
            factor_name = factor_name_valid_list[f_ii]
            # print(f_i, factor_name)

            factor_X = factor_df_in_use[factor_name]

            factor_data_input = pd.merge(return_of_stock_in_use, factor_X, how='outer',
                                         left_index=True, right_index=True)
            factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
            factor_data_input = factor_data_input.dropna(axis=0).copy()

            slope, intercept, r_value, p_value, std_err = \
                st.linregress(factor_data_input["Factor_Value"], factor_data_input["Factor_Value"])
            f = slope

            factor_new = MF.pred_by_LSTM_total(factor_data_input["Factor_Value"])[0, 0]

            return_new = f * factor_new + intercept


            factor_pool[factor_name] = return_new
            # factor_score[factor_name] = factor_score[factor_name] + 1

        factor_pool_df = pd.DataFrame({"name":list(factor_pool.keys()),
                                       "value":list(factor_pool.values())})
        factor_pool_df.sort_values(by="value", ascending=False, inplace=True)
        if factor_pool_df.shape[0] >= 20:
            factor_name_selected = factor_pool_df.iloc[:20, 0]
        else:
            factor_name_selected = factor_pool_df.iloc[:, 0]

        for name in list(factor_name_selected):
            factor_score[name] = factor_score[name] + 1

