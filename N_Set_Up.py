import N_Factor_Functions as FF
import N_Model_Functions as MF
import N_Stock_Functions as SF

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
import random

from datetime import datetime
from math import exp

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as st

import time
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

End_Time = "20110101"
time_valid_index = time_valid_index[time_valid_index < End_Time].copy()

# 获取股票上市时间
df_total = SF.get_list_date()
df_total.index = df_total["ts_code"]
df_total = df_total.loc[SC_in_pool, :]
df_total = df_total.dropna(axis=0)
df_total["list_date"] = pd.to_datetime(df_total["list_date"])

# f_time_list = []

stock_relocate_dic = {}


"""
对调仓日期进行循环
循环的内容是调仓的日期
"""
time_sta = time.time()
# for t_i in range(len(time_valid_index)):
for t_i in range(2):
    time_sta = time.time()
    # t_i = 0
    """
    选出调仓的日期，确定好调仓的因子池
    确定按照周调仓，调仓的日期是每一周的周日晚开始
    在进行因子分析的时候，选取的因子值是调仓日期向前滚动100周
    """

    time_tp = time_valid_index[t_i]
    time_tp = np.datetime64(time_tp)
    time_on_tp_index_order = np.where(time_total_index_np == time_tp)[0][0]

    # 确定调仓日期，定下时间格式的Index
    time_in_use_index = time_total_index[time_on_tp_index_order - 100:time_on_tp_index_order]

    # 选取股票池中可行的有多少支股票，确定可以选取的股票
    stock_in_pool = SF.current_stocks(df_total, Sta_Time)
    # stock_in_pool = current_stocks(df_total, Sta_Time)


    # 建立打分的字典，如果在本次股票循环中，
    # 该因子得到选用，则加一分
    factor_score = {}
    for name in factor_sample.columns:
        factor_score[name] = 0

    # 确定因子的名称
    # 以及需要保留或删除的的因子名称和序号
    factor_columns = factor_sample.columns
    factor_index_drop_list = []
    factor_name_drop_list = []


    """
    第一次对股票进行循环
    用到了random.sample的方法
    循环的是位于股票池中的部分股票
    这部分的股票用于选出因子
    """
    stock_sample_order = random.sample(list(range(len(stock_in_pool))), 20)
    # len(stock_in_pool)
    for s_i in stock_sample_order:
        # s_i = 0

        # SC是本次循环中选取股票的代码
        SC = SC_in_pool[s_i]
        # SC = "000561.SZ"
        print("选因子中，这是股票", s_i, SC)

        #
        # 从MySQL数据库中调用因子
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


        """
        PART 2 进行因子检验
        第一次对因子进行循环
        循环的内容是一只股票的所有因子
        """
        for f_i in range(len(factor_columns)):
            # f_i = 0
            print("选因子中，这是股票", SC, "这是因子", f_i, factor_columns[f_i])


            """
            读取单只股票的一个因子的信息
            将因子值和股票收益率拼接在一起
            """
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


            """
            进行因子检验，分别对单只因子和收益率进行
            t检验、IR-IC检验、还有平稳性ADF检验
            """

            """
            进行股票收益率和因子值的相关性t检验
            """
            ols = sm.OLS(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])
            output = ols.fit()
            OLS_params = output.params[-1]  # 这个是什么东西
            OLS_t_test = output.tvalues[-1]  # 这个是t的值
            OLS_p_value = output.pvalues[-1]  # 这个是t的值

            if abs(OLS_p_value) > 0.9:
                factor_index_drop_list.append(f_i)
                factor_name_drop_list.append(factor_name)


            """
            进行股票收益率和因子值的IR-IC检验
            """
            IC = st.pearsonr(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])[0]
            if IC < 0.03:
                factor_index_drop_list.append(f_i)
                factor_name_drop_list.append(factor_name)


            """
            进行因子值自身的值的ADF平稳性分析，
            如果采用LSTM模型进行预测则不需要平稳性分析
            """

            # 平稳性检验
            t = sm.tsa.stattools.adfuller(factor_data_input["Factor_Value"])
            # if False:
            #     factor_index_drop_list.append(f_i)
            #     factor_name_drop_list.append(factor_name)

            # print(f_i, factor_name, OLS_p_value, IC, t[1])

        # 设定需要保留的股票的list
        # 如果因子不在 factor_name_drop_list 中，则保留该因子
        factor_name_stay_list = []
        for x in range(len(factor_columns)):
            if factor_columns[x] not in factor_name_drop_list:
                factor_name_stay_list.append(factor_columns[x])

        # 按照进入筛选的方法，将每个因子放入候选中测试相关性
        factor_name_stay_list_reverse = factor_name_stay_list[::-1]
        factor_df_stay_first = factor_df_in_use.loc[:, factor_name_stay_list]
        factor_df_stay_first_reverse = factor_df_stay_first.loc[:, factor_name_stay_list_reverse]

        """
        对不同的因子进行相关性分析
        如果两个因子的相关性大于0.9，则删去这个因子
        """
        factor_name_valid_list = MF.get_var_no_colinear(0.9, factor_df_stay_first_reverse)
        factor_df_valid = factor_df_in_use.loc[:, factor_name_valid_list]

        # 建立打分dict
        # factor_score = {}
        # for name in factor_name_valid_list:
        #     factor_score[name] = 0


        """
        建立因子选取环节中的因子库
        对剩余因子进行单因子模型回归
        进行LSTM或ARIMA预测出新的因子值
        并用新的因子值计算出股票的收益率
        选取收益率表现最好的20值因子
        并将该单因子模型中的因子放入因子库中
        将在因子库中出现的因子 +1 分
        """
        factor_pool = {}

        """
        对剩余的因子进行循环
        循环的内容是新的因子
        用模型开始预测
        """
        for f_ii in range(factor_df_valid.shape[1]):
            # f_ii = 0

            # 记录因子的名称
            factor_name = factor_name_valid_list[f_ii]
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

            # 将单因子模型对应的股票收益率
            # 存储进factor_pool这个dictionary中
            factor_pool[factor_name] = return_new
            # factor_score[factor_name] = factor_score[factor_name] + 1

        # 将factor_pool这个dictionary转换成DataFrame中
        factor_pool_df = pd.DataFrame({"name": list(factor_pool.keys()),
                                       "value": list(factor_pool.values())})

        # 将DataFrame按照收益率大小排序
        # 选出单因子模型中收益率表现最好的前20只因子
        factor_pool_df.sort_values(by="value", ascending=False, inplace=True)
        if factor_pool_df.shape[0] >= 20:
            factor_name_selected = factor_pool_df.iloc[:20, 0]
        else:
            factor_name_selected = factor_pool_df.iloc[:, 0]

        # 将排名前20的因子加一分，
        # 最终选取排名前20的因子作为本次调仓的因子池
        for name in list(factor_name_selected):
            factor_score[name] = factor_score[name] + 1


    # 统计每个因子的分数
    factor_selected_df = pd.DataFrame({"name": list(factor_score.keys()),
                                       "value": list(factor_score.values())})

    # 排序
    factor_selected_df.sort_values(by="value", ascending=False, inplace=True)

    # 最终选取排名前20的因子作为本次调仓的因子池
    factor_name_selected = list(factor_pool_df.iloc[:20, 0])
    # factor_name_selected = list(factor_columns[2:12])


    """
    # Part 4 选取股票
    建立选取的股票集合
    将最后选定的股票和其预计收益率存储进这个dic中
    """
    stock_selected_dict = {}



    """
    在已经确认好选取的因子之后，
    再次对股票进行循环
    本次循环的内容是股票池中所有的股票
    用同样的20只因子对股票进行预测
    """
    # stock_in_pool = SC_in_pool[1:20]
    for s_i in range(len(stock_in_pool)):
        # s_i = 0

        SC = SC_in_pool[s_i]
        # SC = "000561.SZ"
        print("选股票中，这是股票", s_i, SC)

        # 从MySQL数据库中读取因子和股票的信息
        ROS = SF.Read_One_Stock(SC)
        return_of_stock = ROS.select_pct_chg()

        # 按照自然天数，将股票的收益率拼接到新的表格上
        return_of_total = pd.DataFrame({'date': pd.date_range('20070101', '20191231', freq='D')},
                                       index=pd.date_range('20070101', '20191231', freq='D'))

        # 读取数据库中单只股票的回报率
        return_of_total = pd.merge(return_of_total, return_of_stock, how='outer',
                                   left_index=True, right_index=True)

        # 填补缺失的值
        return_of_total = return_of_total.fillna(method='bfill')
        return_df_of_stock_in_use = return_of_total.loc[time_in_use_index, :]
        return_of_stock_in_use = return_df_of_stock_in_use["pct_chg"]
        # return_of_stock_in_use.shape


        # 读取数据库中单只股票的因子数据
        factor_df = FF.read_factor(SC)
        factor_df_in_use = factor_df.loc[time_in_use_index, factor_name_selected]  # 选取100次历史数据

        # 删去全部都是NA的数据
        factor_df_in_use.dropna(axis=1, inplace=True, how='all')

        # 确定因子的名称
        factor_columns = factor_df_in_use.columns

        # 为确保数据在时间维度上的一致性，
        factor_data_input_final = pd.merge(return_of_stock_in_use, factor_df_in_use, how='outer',
                                     left_index=True, right_index=True)
        factor_data_input_final.dropna(axis=0, inplace=True)

        # 建立储存因子预测值的diactionary
        factor_X_pred_dict = {}

        """
        对已经锁定的20个因子进行循环
        循环的内容是选定的因子
        用于预测新的因子值
        """
        for f_iii in range(len(factor_columns)):
            # f_iii = 0
            factor_name = factor_columns[f_iii]
            # print(f_i, factor_name)

            factor_X = factor_df_in_use[factor_name]

            """
            进行LSTM模型来预测新的因子值
            """
            # factor_new = MF.pred_by_LSTM_total(factor_data_input_final["pct_chg"])[0, 0]

            # 测试的时候采用简单线性模型进行预测
            slope_f, intercept_f, r_value_f, p_value_f, std_err_f = \
                st.linregress(list(range(factor_data_input.shape[0])), factor_data_input["Factor_Value"])
            factor_new = slope_f * factor_data_input.shape[0] + intercept_f

            # 存储单因子模型计算出的单只股票收益率
            factor_X_pred_dict[factor_name] = factor_new

        """
        采用多元线性回归的方法
        对选用的因子对股票的收益率进行预测
        选用fit的自变量是 100*20的 DataFrame
        打分最高的20个因子，回溯以前100周的历史信息
        """
        linreg = LinearRegression()
        model = linreg.fit(factor_data_input_final.iloc[:, 1:],
                           factor_data_input_final["pct_chg"])

        # f_list 是因子收益率
        f_list = linreg.coef_

        # X_test 是用LSTM或ARMA模型预测出的新的因子值
        # 将新的因子值放入模型，预测下一次股票的收益率
        X_test = pd.DataFrame(factor_X_pred_dict.values()).T

        # y_pred 是预测出的新的因子值
        y_pred = linreg.predict(X_test)[0]

        y_pred = 1 / (1 + exp((-1) * y_pred))
        # 将用多元线性回归算出的因子值添加到stock_selected_dict中
        stock_selected_dict[SC] = y_pred

    # 将选取的股票和股票的预期收益率按照DataFrame的形式返回
    stock_selected_df = pd.DataFrame({"stock_name": list(stock_selected_dict.keys()),
                                       "stock_pred_return": list(stock_selected_dict.values())})

    stock_selected_df.sort_values(by="stock_pred_return", ascending=False, inplace=True)

    # 按照收益率从小到大排序，
    # 并选取前20只预期收益率表现最好的股票作为本次调仓的选用股票
    stock_selected_df_cut = stock_selected_df.iloc[:20, :]

    # 将每次选出的信息存储在 stock_relocate_dic 这个字典中
    stock_relocate_dic[time_tp] = stock_selected_df_cut

# 计算时间
time_end = time.time()
time_cost = time_sta - time_sta
print(time_cost)

# 返回每次的调仓选用的股票
stock_relocate_dic

