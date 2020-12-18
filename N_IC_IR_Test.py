import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF
import pandas as pd
import numpy as np
import random
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
"""
对剩余的因子进行循环
循环的内容是新的因子
用模型开始预测
"""
Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

# 将数据重新按照英文命名
Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

# 提取在股票池中的股票代码
SC_in_pool = Stock_code_pool['Stock_Code']

# 设置选用的时间
# time_in_use_index =

# 设置抽取的股票池
# stock_in_pool =

factor_name_valid_list = ['DAVOL20_min',
                          'DAVOL5_min',
                          'VSTD_6d_min',
                          'turnover_vol_100d_min',
                          'turnover_vol_5d_min',
                          'TO_100d_min',
                          'TO_20d_min',
                          'TO_5d_min',
                          'PPReversal_20_min',
                          'NCFP_min',
                          'BP_LF_min',
                          'VSTD_6d_max',
                          'turnover_vol_20d_max',
                          'turnover_vol_5d_max',
                          'n_cashflow_act',
                          'assets_turn',
                          'current_ratio',
                          'ocf_to_or',
                          'opincome_of_ebt',
                          'wgt_turn_6m',
                          'wgt_turn_3m',
                          'turn_6m',
                          'turn_3m',
                          'wgt_return_12m',
                          'wgt_return_6m',
                          'wgt_return_3m',
                          'wgt_return_1m',
                          'return_3m',
                          'return_1m']

stock_sample_order = random.sample(list(range(len(stock_in_pool))), 20)

factor_pool = {}

for f_ii in range(len(factor_name_valid_list)):
    # f_ii = 0

    factor_name = factor_name_valid_list[f_ii]
    stock_order_dict = {}
    factor_name = factor_name_valid_list[f_ii]
    for s_i in stock_sample_order:
        # SC = "000021.SZ"
        SC = SC_in_pool[s_i]

        ROS = SF.Read_One_Stock(SC)
        return_of_stock = ROS.select_pct_chg()
        return_of_total = pd.DataFrame({'date': pd.date_range('20070101', '20191231', freq='D')},
                                       index=pd.date_range('20070101', '20191231', freq='D'))

        # 选取对应时间断的股票回报率
        return_of_total = pd.merge(return_of_total, return_of_stock, how='outer',
                                   left_index=True, right_index=True)
        return_of_total = return_of_total.fillna(method='bfill')
        return_df_of_stock_in_use = return_of_total.loc[time_in_use_index, :]
        return_of_stock_in_use = return_df_of_stock_in_use["pct_chg"]
        # return_of_stock_in_use.shape

        # 选出本次调仓选用的因子部分数据，
        # axis = 0 的维度往期100次的因子值
        # axis = 1 的维度是所有可能有用的因子
        factor_df = FF.read_factor(SC)
        factor_df_in_use = factor_df.loc[time_in_use_index, :]  # 选取100次历史数据
        factor_df_in_use.dropna(axis=1, inplace=True, how='all')
        factor_df_in_use = factor_df_in_use.loc[:, factor_name_valid_list]



        # 记录因子的名称
        # factor_name = factor_name_valid_list[f_ii]
        # print(f_i, factor_name)

        # 选出单只因子
        factor_X = factor_df_in_use[factor_name]

        # 将单只因子和股票收益率merge在一起
        factor_data_input = pd.merge(return_of_stock_in_use, factor_X, how='outer',
                                     left_index=True, right_index=True)
        factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
        factor_data_input = factor_data_input.dropna(axis=0).copy()

        # 进行简单的线性回归，计算因子的收益率
        slope, intercept, r_value, p_value, std_err = \
            st.linregress(factor_data_input["Factor_Value"], factor_data_input["Stock_Return_Rate"])

        # 因子收益率选用的是斜率
        f = slope

        """
        用LSTM模型对新的因子值进行预测
        """
        # factor_new = MF.pred_by_LSTM_total(factor_data_input["Factor_Value"])[0, 0]

        # 在测试的时候选用简单的线性回归
        slope_f, intercept_f, r_value_f, p_value_f, std_err_f = \
            st.linregress(list(range(factor_data_input.shape[0])), factor_data_input["Factor_Value"])
        factor_new = slope_f * factor_data_input.shape[0] + intercept_f

        # 计算出新的收益率
        return_new = f * factor_new + intercept

        stock_order_dict[SC] = return_new
        # 将单因子模型对应的股票收益率
        # 存储进stock_order_dict这个dictionary中


    # 将factor_pool这个dictionary转换成DataFrame中
    stock_order_dict = pd.DataFrame({"name": list(stock_order_dict.keys()),
                                   "value": list(stock_order_dict.values())})

    factor_pool[factor_name] = stock_order_dict

factor_pool_df = pd.DataFrame({"name":[], "value":[]})

for factor in factor_pool.keys():
    factor_pool_df = pd.merge(factor_pool_df, factor_pool[factor],
                              on="name", how="outer")
factor_pool_df.index = factor_pool_df["name"]
factor_pool_df = factor_pool_df.iloc[:, 2:]
factor_pool_df.columns = list(factor_pool.keys())
factor_pool_df.fillna(0, inplace=True)






df_list = []
for k in range(10):
    SC = SC_in_pool[k]
    SC = "000021.SZ"
    print(k)
    factor_sample = FF.read_factor(SC)
    sta = factor_sample.index[300]
    index_use = factor_sample.index[300:500]

    factor_sub = factor_sample.loc[index_use, factor_name_valid_list]

    ROS = SF.Read_One_Stock(SC)
    return_of_stock = ROS.select_pct_chg()
    return_of_total = pd.DataFrame({'date': pd.date_range('20070101', '20191231', freq='D')},
                                   index=pd.date_range('20070101', '20191231', freq='D'))
    return_of_total = pd.merge(return_of_total, return_of_stock, how='outer',
                               left_index=True, right_index=True)
    return_of_total = return_of_total.fillna(method='bfill')

    return_of_total = return_of_total.loc[index_use, :]["pct_chg"]

    data_use = pd.merge(factor_sub, return_of_total, left_index=True, right_index=True)


    data_use.dropna(axis=0, inplace=True)
    index_use = data_use.index
    pccs_arr = np.empty((data_use.shape[0]-100, data_use.shape[1]-1))
    for i in range(data_use.shape[1]-1):
        # i 是 因子
        for j in range(data_use.shape[0]-100):
            # j 是 时间
            # i = 0
            # j = 0
            # j = 100
            time_index = index_use[j:j+100]
            X = data_use.iloc[:, i]
            X = X.loc[time_index]

            R = data_use.iloc[:, -1]
            R = R.loc[time_index]

            pccs = np.corrcoef(X, R)[0, 1]
            pccs_arr[j, i] = pccs

    df_list.append(pccs_arr)

a = np.zeros(df_list[0].shape)

for i in range(10):
    a += df_list[i]
df_mean = a/10

df_df_mean = pd.DataFrame(df_mean, columns=factor_name_valid_list)
df_df_mean.to_excel('df_mean.xlsx')
len(df_list)



for i in range(len(df_list)):
    # i = 0
    df = pd.DataFrame(df_list[i], columns=factor_name_valid_list)
    IC_mean = df.mean()
    IC_std_error = df.std()
    if i == 0:
        IC_mean_df = IC_mean
        IC_std_error_df = IC_std_error

    else:
        IC_mean_df = pd.concat([IC_mean_df, IC_mean], axis=1)
        IC_std_error_df = pd.concat([IC_std_error_df, IC_std_error], axis=1)


def larger_than_p(data:pd.Series):
    count = 0
    for i in range(len(data)):
        if abs(data[i]) >= 0.03:
            count += 1
    return count/len(data)


IC_mean_df = IC_mean_df.T
IC_std_error_df = IC_std_error_df.T


IC_mean_mean = IC_mean_df.mean()
IC_std_mean = IC_std_error_df.std()

IR_list = []
for i in range(len(IC_mean_mean)):
    # i = 1
    IR = IC_mean_mean[i] / IC_std_mean[i]
    IR_list.append(IR)

IR_list = pd.Series(IR_list, index=factor_name_valid_list)
larger_002 = IC_mean_df.apply(lambda x: sum(abs(x)>0.02)/10)

df_output = pd.concat([IC_mean_mean, IC_std_mean, IR_list, larger_002], axis=1)
df_output.to_excel("output.xlsx")


df_output = pd.merge(IC_mean_mean, IC_std_mean, left_index=True, right_index=True, how='outer')
df_output = pd.merge(df_output, IR_list, left_index=True, right_index=True, how='outer')
df_output = pd.merge(df_output, larger_002, left_index=True, right_index=True, how='outer')

IC_mean_mean.shape
IC_std_mean.shape
# IR = [IC_mean_mean.iloc[i]/IC_std_mean.iloc[i] for i in range(len(IC_mean_mean))]

# 将
pd.DataFrame(pccs_arr).plot.heatmap()
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
# sns.heatmap(pt, linewidths = 0.05, vmax=900, vmin=0, cmap=cmap)

# f, (ax1,ax2) = plt.subplots(figsize = (6,4),nrows=2)

sns.heatmap(pccs_arr, linewidths = 0.05, vmax=0.5, vmin=-0.5, cmap='RdYlGn_r')

pccs_arr_df = pd.DataFrame(pccs_arr, columns=factor_name_valid_list)
sns.heatmap(pccs_arr_df, linewidths = 0.1, vmax=0.5, vmin=-0.5, cmap='RdYlGn_r')



corr_list = []
for i in range(factor_sub.shape[1]-1):
    X = factor_sub.iloc[:-1, i]
    R = factor_sub.iloc[1:, factor_sub.shape[1]]
    corr = pearsonr(X, R)
    corr_list.append(corr)



def report_corr(df, table_name):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title='MPG Pandas Profiling Report')
    profile.to_file('report_cor_{}.html'.format(table_name))

def report_sv(df, table_name):
    import sweetviz as sv
    my_report = sv.analyze(df)
    my_report.show_html('report_sv_{}.html'.format(table_name))

report_sv(pccs_arr_df, "IC")
report_corr(factor_pool_df, "stock_return")

len(list(factor_pool.keys()))

pccs_arr_df.to_excel("IC_Value.xlsx")
pccs_arr_df.apply(std_err)

"""
画factor相关性热力图
"""
factor_index = random.sample(list(range(20)), 20)
data_report = data_use.iloc[:, factor_index]
report_corr(data_report, "correlation_20")
