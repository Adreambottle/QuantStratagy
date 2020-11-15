from Build_Stock_Pool import Read_One_Stock

import pandas as pd
import numpy as np

from pyfinance import TSeries
import tushare as ts
def get_data(code,start='2011-01-01',end=''):
    df=ts.get_k_data(code,start,end)
    df.index=pd.to_datetime(df.date)
    ret=df.close/df.close.shift(1)-1
    #返回TSeries序列
    return TSeries(ret.dropna())
#获取中国平安数据
tss=get_data('601318')
#tss.head()
