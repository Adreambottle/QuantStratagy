from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch  # 条件异方差模型相关的库
import tushare as ts


IndexData = ts.get_hist_data('000001',start='2017-01-01',end='2019-01-01')
"""
IndexData = pd.read_csv(open("C:/Users/Administrator/Desktop/000001历史数据.csv"))
IndexData = IndexData.set_index(IndexData['date'])
"""

data = np.array(IndexData['p_change']) # 上证指数日涨跌
IndexData['p_change'].plot(figsize=(15,5))
t = sm.tsa.stattools.adfuller(data)  # ADF检验
print("p-value: ",t[1])  #p-value小于显著性水平，因此序列是平稳的，接下来我们建立AR(p)模型，先判定阶次

fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
fig = sm.graphics.tsa.plot_pacf(data,lags = 20,ax=ax1)

order = (9,0)
model = sm.tsa.ARMA(data,order).fit()


#计算均值方程残差
at = data -  model.fittedvalues
at2 = np.square(at)

plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(at,label = 'at')
plt.legend()
plt.subplot(212)
plt.plot(at2,label='at^2')
plt.legend(loc=0)


#序列进行混成检验
m = 25 # 我们检验25个自相关系数
acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
out = np.c_[range(1,26), acf[1:], q, p]
output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
output = output.set_index('lag')
output   #p-value小于显著性水平0.05，我们拒绝原假设，即认为序列具有相关性。因此具有ARCH效应。
#ARCH模型的阶次


fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
fig = sm.graphics.tsa.plot_pacf(at2, lags = 30, ax=ax1)
#可以粗略选择均值模型为AR(9)模型，波动率模型选择ARCH(2)模型

train = data[:-10]
test = data[-10:]
am = arch.arch_model(train, mean='AR',lags=9,vol='ARCH',p=2)
res = am.fit()
res.summary()

res.params
#预测
res.hedgehog_plot()
len(train)
pre = res.forecast(horizon=10,start=478).iloc[478]
plt.figure(figsize=(10,4))
plt.plot(test,label='realValue')
pre.plot(label='predictValue')
plt.plot(np.zeros(10),label='zero')
plt.legend(loc=0)

#GARCH模型建立
train = data[:-10]
test = data[-10:]
am = arch.arch_model(train, mean='AR',lags=9,vol='GARCH')
res = am.fit()
res.summary()
res.params
res.plot()
plt.plot(data)
res.hedgehog_plot()

ini = res.resid[-8:]
a = np.array(res.params[1:9])
w = a[::-1] # 系数
for i in range(10):
    new = test[i] - (res.params[0] + w.dot(ini[-8:]))
    ini = np.append(ini,new)
print(len(ini))
at_pre = ini[-10:]
at_pre2 = at_pre**2
at_pre2
#预测波动率
ini2 = res.conditional_volatility[-2:] #上两个条件异方差值
for i in range(10):
    new = 0.000007 + 0.1*at_pre2[i] + 0.88*ini2[-1]
    ini2 = np.append(ini2,new)
vol_pre = ini2[-10:]
vol_pre

plt.figure(figsize=(15,5))
plt.plot(data,label='origin_data')
plt.plot(res.conditional_volatility,label='conditional_volatility')
x=range(479,489)
plt.plot(x,vol_pre,'.r',label='predict_volatility')
plt.legend(loc=0)