# 导入需要用到的模块
import numpy as np
import pandas as pd
import time

import tushare as ts
from tushare import pro

import pymysql
from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()


# 爬取数据库
class Stock_Data(object):
    def __init__(self,
                 code='000021.SZ',
                 start='20070101',
                 end='20191231'):
        self.start = start
        self.end = end
        self.code = code
        self.token = "d44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c"
        self.pro = ts.pro_api(self.token)
        # self.daily_data = pd.DataFrame({'date': pd.date_range(self.start, self.end, freq='D')},
        #                                index=pd.date_range(self.start, self.end, freq='D'))
        # daily_data = pd.DataFrame({'date': pd.date_range(start, end, freq='D')},
        #                           index=pd.date_range(start, end, freq='D'))
        self.daily()
        # self.daily_data = self.daily_data.fillna(method='bfill')

    # 获取股票代码列表
    def get_code(self):
        codes = pro.stock_basic(list_status='L').ts_code.values
        return codes

    # 获取股票交易日历
    def get_cals(self):
        # 获取交易日历
        cals = pro.trade_cal(exchange='')
        cals = cals[cals.is_open == 1].cal_date.values
        return cals

    # 每日行情数据
    def daily(self):
        try:
            # 获取每日交易数据
            df_daily = pro.daily(ts_code=self.code, start_date=self.start, end_date=self.end)
            # df_daily = pro.daily(ts_code=code, start_date=start, end_date=end)

            df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"],
                                                    format='%Y%m%d')
            df_daily.index = df_daily["trade_date"]

            df_daily.sort_index(inplace=True)
            # # 获取每日指标
            # df1 = pro.daily_basic(ts_code=code, start_date=self.start, end_date=self.end)
            #
            # # 获取复权行情
            # df2 = pro.pro_bar(ts_code=code, start_date=self.start, end_date=self.end)
            #
            # # 获取复权因子
            # df3 = pro.adj_factor(ts_code=code, start_date=self.start, end_date=self.end)
            #
            # df = pd.merge(df0, df1, df2, df3)  # 合并数据
            # self.daily_data = pd.merge(self.daily_data, df_daily, how='outer',
            #                         left_index=True, right_index=True)
            # daily_data = pd.merge(daily_data, df_daily, how='outer',
            #                         left_index=True, right_index=True)

            self.daily_data = df_daily

        except Exception as e:
            print(self.code + " is failed!")
            print(e)

    # 获取最新交易日期
    def get_trade_date(self):
        # 获取当天日期时间
        pass

    # 更新数据库数据
    def update_sql(self):
        pass  # 代码省略

    # 查询数据库信息
    def info_sql(self):
        pass  # 代码省略


from N_Download_Factor import Factor_Data


# 将交易数据储存在 MySQL 里面
def store_factor_data():
    # 创建 MySQL 的链接
    # 数据库是本机的数据库，访问账号是root， 地址是localhost
    conn = create_engine('mysql+mysqldb://root:zzzzzzzz@localhost:3306/factor?charset=utf8')

    # 读取从 Wind 上分类的信息技术板块
    Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

    # 将数据重新按照英文命名
    Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

    # 提取在股票池中的股票代码
    SC_in_pool = Stock_code_pool['Stock_Code']

    # 标记并储存无法访问和获取数据的股票代码
    SC_unavailable = []
    SC_empty_data = []

    # 创建获取数据的api接口
    # pro = ts.pro_api(token)

    count = 0
    start = '20070101'
    end = '20191231'

    # 记录自己的token
    token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

    pro = ts.pro_api(token)

    """
    获取在交易时间范围内指数信息，并且存储在SQL中
    """

    df_index = pro.index_daily(ts_code='399300.SZ',
                               start_date=start,
                               end_date=end,
                               fields='ts_code,'
                                      'trade_date,'
                                      'close')
    df_index["trade_date"] = pd.to_datetime(df_index["trade_date"],
                                            format='%Y%m%d')
    df_index.index = df_index["trade_date"]
    df_index.sort_index(inplace=True)
    df_index.columns = ["index_code", "date", "price"]

    pd.io.sql.to_sql(df_index, "Index_daily_data", con=conn, schema='factor', if_exists='replace')

    """
    获取股票池中所有股票的上市时间，并且存储在SQL中
    """
    for i in range(len(SC_in_pool)):
        SC = SC_in_pool[i]
        df = pro.stock_basic(ts_code=SC,
                             fields='ts_code,'
                                    'list_date')
        if i == 0:
            df_total = df
        else:
            df_total = pd.concat([df_total, df], axis=0)
            time.sleep(0.5)
    df_total["list_date"] = pd.to_datetime(df_total["list_date"])
    pd.io.sql.to_sql(df_total, "Start_Data", con=conn, schema='factor', if_exists='replace')

    """
    获取股票池中所有的因子信息，并且存储在SQL中
    """
    # for SC in SC_in_pool:
    for i in range(len(SC_in_pool)):
        SC = SC_in_pool[i]

        print("This is the {} and the code is {}. \n".format(count, SC))
        # SC = SC_in_pool[0]
        FD = Factor_Data(start, end, SC)

        # 将所有在股票池中的数据储存在 MySQL 中
        try:
            tmp_data = FD.factors
            if tmp_data.empty:
                SC_empty_data.append(SC)
            else:
                pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='factor', if_exists='replace')

        # 如果访问失败，将数据储存在 SC_unavailable 的列表中
        except:
            SC_unavailable.append(SC)
        count += 1

    # 储存指数数据，用于获取自然交易日

# store_factor_data()

def store_stock_data():
    # 创建 MySQL 的链接
    # 数据库是本机的数据库，访问账号是root， 地址是localhost
    conn = create_engine('mysql+mysqldb://root:zzzzzzzz@localhost:3306/stock?charset=utf8')

    # 读取从 Wind 上分类的信息技术板块
    Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

    # 将数据重新按照英文命名
    Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

    # 提取在股票池中的股票代码
    SC_in_pool = Stock_code_pool['Stock_Code']

    # 标记并储存无法访问和获取数据的股票代码
    SC_unavailable = []
    SC_empty_data = []

    # 创建获取数据的api接口
    # pro = ts.pro_api(token)

    count = 0
    start = '20070101'
    end = '20191231'

    # 记录自己的token
    token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

    pro = ts.pro_api(token)

    """
    获取在交易时间范围内指数信息，并且存储在SQL中
    """

    df_index = pro.index_daily(ts_code='399300.SZ',
                               start_date=start,
                               end_date=end,
                               fields='ts_code,'
                                      'trade_date,'
                                      'close')
    df_index["trade_date"] = pd.to_datetime(df_index["trade_date"],
                                            format='%Y%m%d')
    df_index.index = df_index["trade_date"]
    df_index.sort_index(inplace=True)
    df_index.columns = ["index_code", "date", "price"]

    pd.io.sql.to_sql(df_index, "Index_daily_data", con=conn, schema='stock', if_exists='append')


    """
    获取股票池中所有股票的上市时间，并且存储在SQL中
    """
    for i in range(len(SC_in_pool)):
        SC = SC_in_pool[i]
        df = pro.stock_basic(ts_code=SC,
                             fields='ts_code,'
                                    'list_date')
        if i == 0:
            df_total = df
        else:
            df_total = pd.concat([df_total, df], axis=0)
            time.sleep(0.5)
    df_total["list_date"] = pd.to_datetime(df_total["list_date"])
    pd.io.sql.to_sql(df_total, "Start_Data", con=conn, schema='stock', if_exists='replace')


    """
    获取股票池中所有的因子信息，并且存储在SQL中
    """
    # for SC in SC_in_pool:
    for i in range(len(SC_in_pool)):
        # i = 0
        SC = SC_in_pool[i]

        print("This is the {} and the code is {}. \n".format(count, SC))
        # SC = SC_in_pool[0]
        SD = Stock_Data(SC, start, end)

        # 将所有在股票池中的数据储存在 MySQL 中
        try:
            tmp_data = SD.daily_data
            tmp_data.index.name = "index"
            if tmp_data.empty:
                SC_empty_data.append(SC)
            else:
                pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='stock', if_exists='replace')

        # 如果访问失败，将数据储存在 SC_unavailable 的列表中
        except:
            SC_unavailable.append(SC)
        count += 1

    # 储存指数数据，用于获取自然交易日

store_stock_data()




## 这部分要移到别的文件中
# 从 MySQL 中读取一直股票的数据，并将股票的数据以 DataFrame 的形式导出

class Read_One_Stock():

    # 初始化数据，定义 MySQL 访问链接参数
    def __init__(self, SC_Code):
        self.conn = pymysql.connect(
            host="localhost",
            database="Stock",
            user="root",
            password="zzzzzzzz",
            port=3306,
            charset='utf8'
        )
        self.SC_Code = SC_Code

    # 获取每天的收盘价
    def select_close_data(self):
        # 读取每天的收盘价
        sqlcmd = "SELECT trade_date, close FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    # 获取每天的开盘价
    def select_open_data(self):
        # 读取每天的开盘价
        sqlcmd = "SELECT trade_date, open FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    # 获取每天的交易数量
    def select_vol_amount(self):
        # 读取每天的交易数量
        sqlcmd = "SELECT trade_date, open FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    # 获取想要获取的交易数据，可以自定义
    def select_col(self, *args):
        col_list = args
        sqlcmd = "SELECT trade_date, "
        for arg in args:
            sqlcmd = sqlcmd + arg + ", "
        sqlcmd = sqlcmd[:-2]
        sqlcmd = sqlcmd + " FROM `{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table
