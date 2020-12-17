# coding:utf-8
"""
author:Sam
date：2020/12/9
"""

"""
搭建自己的多因子策略选股回测框架
一、交易环境初始化   √
输入参数：手续费、滑点、初始本金
二、获取行情数据     √
输入参数：股票代码，起始日期，截止日期
输出参数：至少包括收盘价、开盘价、最高价、最低价、成交量；
三、周末调仓交易     √
输入参数：当周交易股票代码及份额
输出参数：交易后股票仓位及剩余资金
四、输出策略回测报告   √
计算统计各种交易数据(夏普比、盈亏比、胜率、最大回撤、年化率/最大回撤、策略alpha等)
五、可视化策略回测效果   √
1. 资金净值曲线图（周）
2. 策略收益率（周）+与基准收益率对比 曲线图
3. 超额收益曲线图（周） 看看有没有办法和上一个图合并起来做
4. 回撤曲线（周）
5. 盈亏曲线（周）
 """

import numpy as np
import pandas as pd
import tushare as ts
import datetime
import matplotlib.pyplot as plt


# 一、交易环境初始化
class Broker(object):
    def __init__(self):
        super(Broker, self).__init__()

        # 手续费
        self.commission = 3 / 10000

        # 滑点率，默认为5/10000
        self.slipper_rate = 5 / 10000

        # 初始本金.
        self.cash = 100000000

        # 交易数据
        self.trades = []

        # 当前提交的订单
        self.active_orders = []

        # 回测的Dataframe数据
        self.backtest = None

        # 设置手续费
        def set_commission(self, commission: float):
            self.commission = commission
            return self.commission

        # 设置滑点率
        def set_slipper_rate(self, slipper_rate=float):
            self.slipper_rate = slipper_rate
            return self.slipper_rate

        # 设置初始资金
        def set_cash(self, cash):
            self.cash = cash
            return self.cash

        # 设置回测数据
        def set_backtest(self, data: pd.DataFrame):
            self.backtest = data
            return self.backtest


# 二、获取行情数据
class BarData():
    def __init__(self, stockpool: list, startdata: str, enddate: str):
        # 股票池用 list 起始日期和截止日期用8位str
        self.stock_pool = stockpool
        self.start_date = startdata
        self.end_date = enddate
        self.data = {}

    # 读取数据 好像太慢了
    def read_data_API(self, stockpool: list, startdate: str, enddate: str):
        ts.set_token('bf6851d0eb2cbe8ce6da2bb4ee4df880f70e018186d4adc166117059')
        pro = ts.pro_api()
        for i in stockpool:
            df = pro.daily(ts_code=i, start_date=startdate, end_date=enddate)
            self.data[i] = df
            # print(self.data)
        return self.data

    def read_data_CSV(self, stockpool: list, startdate: str, enddate: str):  # 还没有测试 不知道会不会出问题
        for i in stockpool:
            df1 = pd.read_csv("D:/pycharm project/quant_project/stock_data/" + i + "_daily_data.csv")
            df2 = df1.loc['trade_date'][startdate:enddate]
            df2.append(df1.loc['trade_date'][enddate])
            self.data[i] = df2
        return self.data


# 三、周末调仓交易
class Trade(Broker):  # 继承Brkoer类
    def __init__(self, stockpool: list, trade: dict, tradedate: str, pos: dict, cash_initial):
        # 月末调仓只需给出股票池、对应交易订单信息以及交易日期
        # 持仓量、资金由上期进行初始化
        self.stock_pool = stockpool
        self.trade_order = trade
        self.trade_date = tradedate
        self.position = pos
        self.cash = cash_initial

        self.commission = 3 / 10000
        self.slipper_rate = 5 / 10000
        # pos 持仓量应该长这样：
        # {'000021.SZ': q1
        #   .....
        #  '900941.SH': q2}  q1,q2为对应股票现持仓量 q1,q2>=0

        # trade 包含交易订单的字典应该长这样：
        # {'000021.SZ': {'buy':x1,'sell':y1}
        #   .....
        #  '900941.SH': {'buy':x2,'sell':y2}  x1,x2为对应股票的买入数量; y1,y2为对应股票的卖出数量
        #  暂时不允许做空 y1,y2需小于等于对应股票现有持仓量q1,q2

    def set_position(self, trade: dict):
        for i in self.stock_pool:
            self.position[i] = self.position[i] + trade[i]['buy'] - trade[i]['sell']
        return self.position

    def set_cash(self, trade: dict, tradetime: str):
        for i in self.stock_pool:
            d[i].index = d[i]['trade_date']
            if tradetime in d[i].index:
                close = d[i].loc[tradetime]['close']
                self.cash = self.cash + trade[i]['sell'] * close - trade[i]['buy'] * close - (
                            trade[i]['buy'] + trade[i]['sell']) * (self.commission + self.slipper_rate)  # 减去交易成本
                print(self.cash)
            else:
                self.cash = self.cash  # 停牌就不交易
                print(self.cash)
        return self.cash




# 四、五、 输出策略回测数据以及可视化图表
class Calculation():  # 计算统计各种交易数据
    def __init__(self, cashflow: list, basicreturn: list):
        # 输入：cashflow(包含120次交易后资金净值的列表) basicreturn(119个基准月收益率的列表）
        # 现在改成周了 多少次交易? 相应调整次数 120-->? 119-->?
        # 新输入：cashflow 500次 basic return 499个
        self.cash_flow = cashflow
        self.basic_return = basicreturn

    def weekly_return(self):  # 月收益率
        week_return = []
        for i in range(1, len(self.cash_flow)):
            tempt = (self.cash_flow[i] - self.cash_flow[i - 1]) / self.cash_flow[i - 1]  # 每一期的月收益率
            week_return.append(tempt)
        print(week_return)
        return week_return

    def final_annualized_return(self):  # 最终年化收益率（算数平均）
        weekly_avg_return = []
        for i in range(1, len(self.cash_flow)):
            tempt = (self.cash_flow[i] - self.cash_flow[i - 1]) / self.cash_flow[i - 1]  # 每一期的周收益率
            weekly_avg_return.append(tempt)
            annual_return = np.mean(weekly_avg_return) * 52  # 周收益率 乘以52
        print("平均年化收益率为：", annual_return)
        return annual_return

    def Sharpe_Ratio(self):  # 策略夏普比: 年化收益率/收益率标准差(年)
        r = self.final_annualized_return()
        std_weekly = np.std(self.cash_flow, ddof=1)
        std_yearly = std_weekly * (52 ** 0.5)  # 周收益率 乘以根号52
        ratio = r / std_yearly
        print("夏普比率为：", ratio)
        return ratio

    def max_drawback(self):  # 最大回撤率
        m_drawback = []
        for i in range(1, 500):  # 500次交易
            new_cashlist = self.cash_flow[0:i].copy()
            find_max = max(new_cashlist)
            find_min = min(new_cashlist)
            drawback = (find_max - find_min) / find_min
            m_drawback.append(drawback)
        m_drawback = max(m_drawback)
        print("最大回撤率为：", m_drawback)
        return m_drawback

    def excess_return(self):  # 超额收益率
        basic = 0.02  # 实际上要重新选取 先初始化一个值
        ex_return = self.final_annualized_return() - basic
        print("超额收益率为：", self.excess_return())
        return ex_return

    def profit_count(self):  # 盈利次数
        count = 0
        for i in range(1, len(self.cash_flow)):
            if self.cash_flow[i] > self.cash_flow[i - 1]:
                count += 1
        print("策略总盈利次数为：", count)
        return count

    def loss_count(self):  # 亏损次数
        count = 0
        for i in range(1, len(self.cash_flow)):
            if self.cash_flow[i] <= self.cash_flow[i - 1]:
                count += 1
        print("策略总亏损次数为：", count)
        return count

    def profit_loss_ratio(self):  # 盈亏比
        profit = 0
        loss = 0
        for i in range(1, len(self.cash_flow)):
            if self.cash_flow[i] > self.cash_flow[i - 1]:
                profit += self.cash_flow[i] - self.cash_flow[i - 1]
            else:
                loss += self.cash_flow[i - 1] - self.cash_flow[i]
        print("总盈利额为：", profit)
        print("总亏损额为：", loss)
        ratio = profit / loss
        print("盈亏比为：", ratio)
        return ratio

    def profit_ratio(self):  # 胜率
        ratio = self.profit_count() / 499
        print("胜率为：", ratio)
        return ratio

    def alpha(self):  # α值 = 实际收益率 -  CAPM收益率
        r = self.final_annualized_return()  # 实际平均年化收益率
        risk_free_interest = 0.01  # 假定的无风险收益率 后期按照国债去找
        beta = self.beta()
        CAPM_return = risk_free_interest + beta * (r - risk_free_interest)
        alpha = r - CAPM_return
        print("Alpha:", alpha)
        return alpha

    def beta(self):  # β值 = cov(策略收益率,市场收益率)/市场收益率标准差
        r = self.weekly_return()
        cov = np.cov(r, self.basic_return)  # 协方差矩阵
        cov = cov[0, 1]  # 协方差cov(x,y) = cov[0,1]
        basic_return_std = np.std(self.basic_return)
        beta = cov / basic_return_std
        print("Beta为：", beta)
        return beta

    def cash_curve(self):  # 资金净值曲线图（月）
        x = pd.date_range(start='2010-01', end='2020-01', freq='W')
        y = self.cash_flow
        plt.plot(x, y, label='cash netvalue')
        plt.legend()
        plt.show()

    def week_return(self):  # 策略收益率（月）+与基准收益率对比 曲线图
        x = pd.date_range(start='2010-02', end='2020-01', freq='W')
        y1 = []
        for i in range(1, 500):
            week_return = (self.cash_flow[i] - self.cash_flow[i - 1]) / self.cash_flow[i - 1]
            y1.append(week_return)
        plt.plot(x, y1, label='strategy return')
        plt.plot(x, self.basic_return, label='basic return')
        plt.legend()
        plt.show()

    def excess_return_curve(self):  # 超额收益曲线图（月）
        x = pd.date_range(start='2010-02', end='2020-01', freq='W')
        y1 = []
        for i in range(1, 500):
            week_return = (self.cash_flow[i] - self.cash_flow[i - 1]) / self.cash_flow[i - 1]
            y1.append(week_return)
        ex_return = y1 - self.basic_return
        plt.plot(x, ex_return, label='excess return')
        plt.legend()
        plt.show()

    def drawback_curve(self):  # 回撤曲线（月）
        x = pd.date_range(start='2010-02', end='2020-01', freq='W')
        cum_return = []
        max_cum_return = []
        drawback = []
        for i in range(1, 500):
            a = self.cash_flow[i] - self.cash_flow[0]
            cum_return.append(a)
            b = max(cum_return[0:i])
            max_cum_return.append(b)
            drawback.append((cum_return[i - 1] - max_cum_return[i - 1]) / (max_cum_return[i - 1]))

        plt.plot(x, drawback, label='drawback')
        plt.legend()
        plt.show()

    def pnl_curve(self):  # 盈亏曲线（月）
        x = pd.date_range(start='2010-02', end='2020-01', freq='W')
        net_profit = []
        for i in range(1, 500):
            a = self.cash_flow[i] - self.cash_flow[0]
            net_profit.append(a)

        plt.plot(x, net_profit, label='pnl_curve')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """
    使用说明：
    第一步 获取行情数据 BarData(stockpool:list,startdata:str,enddate:str)
        .read_data_API(stockpool:list,startdata:str,enddate:str) 读取数据

    第二步 周末调仓 Trade(stockpool:list, trade:dict, tradedate:str, pos:dict, cash_initial)
        .set_cash(trade:dict, tradedate:str)
        .set_position(trade:dict)
        当期计算得到的set_cash和set_position返回结果对应下一期的cash_initial和pos
            需要写一个for循环：
                1. 500次周调仓 除第一次外，每一次传入一个trade和tradedate
                2. 每一次循环调用Trade() set_cash set_position 作为下一期的传入参数
                3. 存储每一期的cash作为一个列表，作为第三部计算回测结果的输入参数
                4. 500次调仓结束，得到cashflow(List)

    第三步 计算回测结果 Calculation(cashflow:list,basicreturn:list)
        cashflow的list由第二步的周末调仓得到 basicreturn需要从读一个沪深300的周收益率序列
    # 回测数据
    # test.final_annualized_return()
    # test.max_drawback()
    # test.Sharpe_Ratio()
    # test.weekly_return() 
    # test.alpha()
    # test.beta()
    # test.profit_count()
    # test.loss_count()
    # test.profit_ratio()
    # test.profit_loss_ratio()

    # 回测结果可视化
    # test.cash_curve()
    # test.week_return()
    # test.excess_return_curve()
    # test.drawback_curve()
    # test.pnl_curve()
    """

    # """ Test data on BarData """
    # stock_pool = pd.read_excel('C:\\Users\\13035\\Desktop\\数据分析学习\\python\\Wind_store_pool.xlsx')
    # Stock_code = stock_pool['stock_code'].tolist()
    # print(Stock_code)
    # stock = BarData(Stock_code,'20100101','20101231')
    # d = stock.read_data_API(Stock_code,'20100101','20101231')    #调用数据好像太慢了 试试用CSV?
    # print(d['000021.SZ']['close'])

    # stock = BarData(['000021.SZ', '000032.SZ', '000034.SZ'],'20100101','20101231')
    # d = stock.read_data(['000021.SZ', '000032.SZ', '000034.SZ'],'20100101','20101231')

    # """ Test on Trade """
    # new_trade = {}
    # new_pos = {}
    # for i in Stock_code:
    #     new_pos[i] = 1000
    #     new_trade[i] = {'buy':200,'sell':100}
    #
    # print(new_trade)
    # print(new_pos)

    # new_trade = {}
    # new_pos = {}
    # for i in ['000021.SZ', '000032.SZ', '000034.SZ']:
    #     new_pos[i] = 1000
    #     new_trade[i] = {'buy': 200, 'sell': 100}
    #
    # print(new_trade)
    # print(new_pos)

    # tradetest = Trade(stockpool=Stock_code,trade=new_trade,tradedate='20100104',pos=new_pos,cash_initial=100000000)
    # print(tradetest.trade_date)
    # a = tradetest.set_position(new_trade)
    # b = tradetest.set_cash(new_trade,'20100104')
    # print(a)
    # print(b)

    """ Test on calculation """
    """ 真实情况下回测只需要传入资金净值list以及basic_return的list序列 """
    """ 现在的test只是模拟了资金净值和basic_return """
    # test_cash = np.random.normal(loc=100000000,scale=10000000,size=120)
    # print(test_cash)
    #
    # test_basic_return = np.random.normal(loc=0.02,scale=0.03,size=119)
    # print(test_basic_return)
    #
    # test = Calculation(test_cash,test_basic_return)
    # # 回测结果数据
    # test.final_annualized_return()
    # test.max_drawback()
    # test.Sharpe_Ratio()
    # test.monthly_return() #119个数据
    # test.alpha()
    # test.beta()
    # test.profit_count()
    # test.loss_count()
    # test.profit_ratio()
    # test.profit_loss_ratio()
    #
    # # 回测结果可视化
    # test.cash_curve()
    # test.month_return()
    # test.excess_return_curve()
    # test.drawback_curve()
    # test.pnl_curve()






















