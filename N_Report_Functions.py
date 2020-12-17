import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def factor_test():
    return_of_stock = ROS.select_pct_chg()
    return_of_total = pd.DataFrame({'date': pd.date_range('20070101', '20191231', freq='D')},
                                index=pd.date_range('20070101', '20191231', freq='D'))

    return_of_total = pd.merge(return_of_total, return_of_stock, how='outer',
                              left_index=True, right_index=True)
    return_of_total = return_of_total.fillna(method='bfill')
    return_of_test = return_of_total.loc[time_valid_index, :]
    factor_of_test = factor_df.loc[time_valid_index, :]
    # factor_of_test.to_excel("/Users/meron/Desktop/factor.xlsx")
    data_of_test = pd.merge(return_of_test["pct_chg"], factor_of_test,
                            left_index=True, right_index=True, how="outer")
    data_of_test = data_of_test.fillna(method='bfill')
    # pd.DataFrame(factor_columns).to_excel("/Users/meron/Desktop/factor_name.xlsx")




    factor_df_in_use_wona = factor_df_in_use.dropna()

    for f_i in range(len(factor_columns)):
        # f_i = 0

        factor_name = factor_columns[f_i]
        # print(f_i, factor_name)
        # factor_X = data_of_test[factor_name]
        factor_data_input = data_of_test.loc[:,["pct_chg", factor_name]]
        factor_data_input.columns = ["Stock_Return_Rate", "Factor_Value"]
        factor_data_input = factor_data_input.dropna(axis=0).copy()

        # t检验
        ols = sm.OLS(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])
        output = ols.fit()
        OLS_params = output.params[-1]  # 这个是什么东西
        OLS_t_test = output.tvalues[-1]  # 这个是t的值
        OLS_p_value = output.pvalues[-1]   # 这个是p的值


        # IRIC检验
        IC = st.pearsonr(factor_data_input["Stock_Return_Rate"], factor_data_input["Factor_Value"])[0]
        # if IC > 1:
        #     factor_index_drop_list.append(f_i)
        #     factor_name_drop_list.append(factor_name)
        #     break

def plot_ADF(factor_df_in_use_wona):
    """
    进行ADF数据的分析
    :return:
    """
    x_list = []
    p_list = []
    list_2 = []
    list_3 = []
    list_1p = []
    list_5p = []
    list_10p = []
    list_4 = []

    for i in range(factor_df_in_use_wona.shape[1]):
        data = factor_df_in_use_wona.iloc[:, i]
        t = sm.tsa.stattools.adfuller(data)
        x = t[0]
        p = t[1]
        x_list.append(x)
        p_list.append(p)
        list_2.append(t[2])
        list_3.append(t[3])
        list_1p.append(t[4]["1%"])
        list_5p.append(t[4]["5%"])
        list_10p.append(t[4]["10%"])
        list_4.append(t[5])
    df_ADF = pd.DataFrame({"Factor":factor_df_in_use_wona.columns,
                           "x":x_list,
                           "p_value":p_list,
                           "list_2":list_2,
                           "list_3":list_3,
                           "list_1p":list_1p,
                           "list_5p":list_5p,
                           "list_10p":list_10p,
                           "list_4":list_4})
    df_ADF.to_excel("/Users/meron/Desktop/ADF.xlsx")


# 生成数据可视化报告
def report_corr(df):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title='MPG Pandas Profiling Report')
    profile.to_file('report_cor.html')

def report_sv(df):
    import sweetviz as sv
    my_report = sv.analyze(df)
    my_report.show_html('report_sv.html')



def get_var_no_colinear(cutoff, df):
    corr_high = df.corr().applymap(lambda x: np.nan if x>cutoff else x).isnull()
    col_all = corr_high.columns.tolist()
    del_col = []
    i = 0
    while i < len(col_all)-1:
        ex_index = corr_high.iloc[:,i][i+1:].index[np.where(corr_high.iloc[:,i][i+1:])].tolist()
        for var in ex_index:
            col_all.remove(var)
        corr_high = corr_high.loc[col_all, col_all]
        i += 1
    return col_all


## 每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, col].values, ix)
               for ix in range(X.iloc[:, col].shape[1])]

        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=', X.columns[col[maxix]], '  ', 'vif=', maxvif)
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col])


