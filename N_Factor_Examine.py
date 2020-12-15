import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime,timedelta
import time
from sklearn import preprocessing
from scipy.stats import mstats
import scipy.stats as st



def get_all_data(start_date, end_date, stockPool, period):
    warnings.filterwarnings("ignore")

    # 获取日期数据
    date_period = get_period_date(period, start_date, end_date)

    # 获取申万一级行业数据
    indu_code = get_industries(name='sw_l1')
    indu_code = list(indu_code.index)

    data = pd.DataFrame()

    for date in date_period[:-1]:
        # 获取股票列表
        stockList = get_stock(stockPool, date, end_date)  # 获取date日的成份股列表

        # 获取横截面收益率
        df_close = get_price(stockList, date, date_period[date_period.index(date) + 1], 'daily', ['close'])
        df_pchg = df_close['close'].iloc[-1, :] / df_close['close'].iloc[0, :] - 1

        # 获取权重数据，流通市值的平方根为权重
        q = query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(stockList))
        R_T = get_fundamentals(q, date)
        R_T.set_index('code', inplace=True, drop=False)
        R_T['Weight'] = np.sqrt(R_T['circulating_market_cap'])  # 流通市值的平方根作为权重
        # 删除无用的code列和circulating_market_cap列
        del R_T['code']
        del R_T['circulating_market_cap']

        # 中证800指数收益率
        index_close = get_price('000906.XSHG', date, date_period[date_period.index(date) + 1], 'daily', ['close'])
        index_pchg = index_close['close'].iloc[-1] / index_close['close'].iloc[0] - 1
        R_T['pchg'] = df_pchg - index_pchg  # 每支股票在date日对中证800的超额收益率（Y）
        # 目前，R_T包含索引列code，权重列Weight，对中证800的超额收益率pchg

        # 获取行业暴露度、哑变量矩阵
        Linear_Regression = pd.DataFrame()
        for i in indu_code:
            i_Constituent_Stocks = get_industry_stocks(i, date)
            i_Constituent_Stocks = list(set(i_Constituent_Stocks).intersection(set(stockList)))
            try:
                temp = pd.Series([1] * len(i_Constituent_Stocks), index=i_Constituent_Stocks)
                temp.name = i
            except:
                print(i)
            Linear_Regression = pd.concat([Linear_Regression, temp], axis=1)
        Linear_Regression.fillna(0.0, inplace=True)

        Linear_Regression = pd.concat([Linear_Regression, R_T], axis=1)
        Linear_Regression = Linear_Regression.dropna()
        Linear_Regression['date'] = date
        Linear_Regression['code'] = Linear_Regression.index
        data = data.append(Linear_Regression)
        print
        date + ' getted!!'
    return data



# 获取新的一个因子数据并进行缩尾和标准化，因子一定要求是get_fundamentals里的
def get_factor_data(start_date, end_date, stockPool, period, factor):
    date_period = get_period_date(period, start_date, end_date)

    # 获取stockvaluaton格式的因子名
    sheet = get_sheetname(factor)
    str_factor = sheet + '.' + factor
    str_factor = eval(str_factor)

    factor_data = pd.DataFrame()
    for date in date_period[:-1]:
        # 获取股票列表
        stockList = get_stock(stockPool, date, end_date)  # 获取date日的成份股列表

        # 获取股票数据
        q = query(valuation.code, str_factor).filter(valuation.code.in_(stockList))
        temp = get_fundamentals(q, date)

        # 因子数据正态化
        temp[factor] = stats.boxcox(temp[factor])[0]  # mark!!!!!!!!!!!!!

        # 生成日期列
        temp['date'] = date

        # 缩尾处理 置信区间95%
        temp[factor] = mstats.winsorize(temp[factor], limits=0.025)

        # 数据标准化
        temp[factor] = preprocessing.scale(temp[factor])

        factor_data = factor_data.append(temp)
        print
        date + ' getted!!'

    return factor_data

def t_test(result,period,start_date,end_date,factor):
    #获取申万一级行业数据
    indu_code = get_industries(name = 'sw_l1')
    indu_code = list(indu_code.index)

    #生成空的dict，存储t检验、IC检验结果
    WLS_params = {}
    WLS_t_test = {}
    IC = {}

    date_period = get_period_date(period,start_date,end_date)

    for date in date_period[:-2]:
        temp = result[result['date'] == date]
        X = temp.loc[:,indu_code+[factor]]
        Y = temp['pchg']
        # WLS回归
        wls = sm.WLS(Y, X, weights=temp['Weight'])
        output = wls.fit()
        WLS_params[date] = output.params[-1]
        WLS_t_test[date] = output.tvalues[-1]
        #IC检验
        IC[date]=st.pearsonr(Y, temp[factor])[0]
        print date+' getted!!!'

    return WLS_params,WLS_t_test,IC

#参数设定
start_date = '2012-01-01'
end_date = '2019-01-31'
stockPool='中证800'
period='M'
Group=10
factor = 'pb_ratio'#这个地方用了get_fundamentals里面有的因子数据，如果是别的数据，可以不写这行

#获取市值权重、行业哑变量数据
data = get_all_data(start_date,end_date,stockPool,period)

#这部分为获取因子数据，如果因子数据为外部数据则可以忽略此步，导入你自己的因子数据即可
factor_data = get_factor_data(start_date,end_date,stockPool,period,factor)

#将因子数据与权重、行业数据合并。如果获取的因子数据用的是自己的因子数据，则保证code和date列可以确定一行观测即可
result = pd.merge(data,factor_data,how = 'left',on = ['code','date'])
result = result.dropna()

#t检验，IC检验
WLS_params,WLS_t_test,IC = t_test(result,period,start_date,end_date,factor)
WLS_params = pd.Series(WLS_params)
WLS_t_test = pd.Series(WLS_t_test)
IC = pd.Series(IC)

#t检验结果
n = [x for x in WLS_t_test.values if np.abs(x)>1.96]
print 't值序列绝对值平均值——判断因子的显著性是否稳定',np.sum(np.abs(WLS_t_test.values))/len(WLS_t_test)
print 't值序列绝对值大于1.96的占比——判断因子的显著性是否稳定',len(n)/float(len(WLS_t_test))
WLS_t_test.plot('bar',figsize=(20,8))

#IC检验结果
print 'IC 值序列的均值大小',IC.mean()
print 'IC 值序列的标准差',IC.std()
print 'IR 比率（IC值序列均值与标准差的比值）',IC.mean()/IC.std()
n_1 = [x for x in IC.values if x > 0]
print 'IC 值序列大于零的占比',len(n_1)/float(len(IC))

n_2 = [x for x in IC.values if np.abs(x) > 0.02]
print 'IC 值序列绝对值大于0.02的占比',len(n_2)/float(len(IC))
IC.plot('bar',figsize=(20,8))












