# coding:utf-8 
"""
author:Sam
date：2020/12/18
"""

import numpy as np
import pandas as pd
import N_backtest_week
import time

start=time.perf_counter()  # 开始计时

"""第一步 获取行情数据"""
# 股票池
stock_pool = pd.read_excel('C:\\Users\\13035\\Desktop\\回测部分\\Wind_store_pool.xlsx')
Stock_code = stock_pool['stock_code'].tolist()
# print(Stock_code)
print("股票池共计" + str(len(Stock_code)) + "只股票")

# tushare API调取数据
# stock = backtest_week.BarData(Stock_code,'20100101','20191231')
# d = stock.read_data_API(Stock_code,'20100101','20191231')
# np.save('stock_data.npy',d)

# 直接读
d = np.load('stock_data.npy',allow_pickle=True).item()
# print(d['000021.SZ']['close'])

"""第二步 周末调仓"""
# 设置初始仓位0
initial_pos = {}.fromkeys(Stock_code,0)
pos_dict = {}
print("初始仓位：")
print(initial_pos)


# 读取交易订单
order = np.load('C:\\Users\\13035\\Desktop\\回测部分\\order.npy',allow_pickle=True).item()
print("第一次交易订单：")
print(order[0])   # 第X期的order(dict) 目前的order好像是521期

#读取交易日期
csv = pd.read_csv('C:\\Users\\13035\\Desktop\\回测部分\\index300.csv')
timelist = csv['date'].tolist()
tradedate = [str(i) for i in timelist ]
print("522次交易日期如下：")
print(tradedate)


# 设置初始资金
initial_cash = 500000000
cash_list = []

# 设置初始资产
asset = 500000000
asset_list = []

# 设置第一次交易（正常）
N = 522
first_trade = backtest_week.Trade(Stock_code,order[0],tradedate[0],initial_pos,initial_cash)
first_cash = first_trade.set_cash(order[0],tradedate[0])
first_pos = first_trade.set_position(order[0])
first_asset = first_trade.set_asset(order[0],tradedate[0],first_pos)
print(first_asset)

cash_list.append(first_cash)
pos_dict[tradedate[0]]=first_pos
asset_list.append(first_asset)
print(cash_list)
print(asset_list)
print(pos_dict)


# for i in range(1,N):
#     pre_date = tradedate[i-1]
#     tempt_trade = backtest_week.Trade(Stock_code,order[i],tradedate[i],pos_dict[pre_date],cash_list[i-1])
#     new_cash =  tempt_trade.set_cash(order[i],tradedate[i])
#     new_pos = tempt_trade.set_position(order[i])
#     new_asset = tempt_trade.set_asset(order[i],tradedate[i],new_pos)
#     cash_list.append(new_cash)
#     asset_list.append(new_asset)
#     pos_dict[tradedate[i]] = new_pos


# print(pos_dict[tradedate[521]])
# last_cash = cash_list[521]
# for i in Stock_code:
#     last_cash += pos_dict[tradedate[521]][i] * d[i].iloc[0]['close']
#
# cash_list.append(last_cash)

# dict1 = {"cash":cash_list}
# data1 = pd.DataFrame(dict1)
#
# dict2 = {'asset':asset_list}
# data2 = pd.DataFrame(dict2)
#
# # 循环结束 得到cash_list
# print(data1)
# print(data2)
# data1.to_csv("cash.csv")
# data2.to_csv("asset.csv")


# 读数据
cash = pd.read_csv('C:\\Users\\13035\\Desktop\\回测部分\\cash.csv')
cash_list = cash['CASH'].tolist()
asset = pd.read_csv('C:\\Users\\13035\\Desktop\\回测部分\\asset.csv')
asset_list = asset['ASSET'].tolist()
print(cash_list)
print(asset_list)


"""第三步 计算回测结果"""
# 读取基准收益率（周）
basic_return = csv ['return'].tolist()
basic_return.pop()

# 开始回测
result = backtest_week.Calculation(asset_list,basic_return)

# 回测数据
result.final_annualized_return()
result.max_drawback()
result.Sharpe_Ratio()
result.weekly_return()
print(len(result.weekly_return()))
result.alpha()
result.beta()
result.profit_count()
result.loss_count()
result.profit_ratio()
result.profit_loss_ratio()

# 回测结果可视化
result.cash_curve()
result.week_return_comp()
result.excess_return_curve()
result.drawback_curve()
result.pnl_curve()


end=time.perf_counter()
print("程序运行共计：" + str(end-start) + "秒")


