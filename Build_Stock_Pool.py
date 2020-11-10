# 导入需要用到的模块
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime,timedelta
import tushare as ts
from tushare import pro
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()



# 爬取数据库
class Data(object):
    def __init__(self,
                 start ='20050101',
                 end ='20200101',
                 table_name = 'daily_data'):
        self.start = start
        self.end = end
        self.table_name = table_name
        self.codes = self.get_code()
        self.cals = self.get_cals()

    #获取股票代码列表
    def get_code(self):
        codes = pro.stock_basic(list_status = 'L').ts_code.values
        return codes

    #获取股票交易日历
    def get_cals(self):
        #获取交易日历
        cals = pro.trade_cal(exchange = '')
        cals = cals[cals.is_open == 1].cal_date.values
        return cals

    #每日行情数据
    def daily_data(self,code):
        try:
            df0 = pro.daily(ts_code = code, start_date = self.start, end_date = self.end)
            df1 = pro.adj_factor(ts_code = code, trade_date = '')
            #复权因子
            df = pd.merge(df0, df1)  #合并数据

        except Exception as e:
            print(code)
            print(e)
        return df

    #保存数据到数据库
    def save_sql(self):
       pass


    #获取最新交易日期
    def get_trade_date(self):
        #获取当天日期时间
        pass
    #更新数据库数据
    def update_sql(self):
        pass #代码省略
    #查询数据库信息
    def info_sql(self):
        pass #代码省略



# 将交易数据储存在 MySQL 里面
def store_daily_data():

    # 创建 MySQL 的链接
    # 数据库是本机的数据库，访问账号是root， 地址是localhost
    conn = create_engine('mysql+mysqldb://root:zzzzzzzz@localhost:3306/Stock?charset=utf8')

    # 读取从 Wind 上分类的信息技术板块
    Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")

    # 将数据重新按照英文命名
    Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']

    # 提取在股票池中的股票代码
    SC_in_pool = Stock_code_pool['Stock_Code']

    # 标记并储存无法访问和获取数据的股票代码
    SC_unavailable = []

    # 记录自己的token
    token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'

    # 创建获取数据的api接口
    pro = ts.pro_api(token)

    # 创建数据对象 Data()
    Stock_data = Data()

    # 获取全部A股市场中的股票
    Stock_code = Stock_data.get_code()
    Stock_code_df = pd.DataFrame({"Stock_name": Stock_code})

    count = 0
    for SC in SC_in_pool:
        print("This is the {} and the code is {}.".format(count, SC))

        # 将所有在股票池中的数据储存在 MySQL 中
        try:
            tmp_data = Stock_data.daily_data(SC)
            pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='Stock', if_exists='append')

        # 如果访问失败，将数据储存在 SC_unavailable 的列表中
        except:
            SC_unavailable.append(SC)
        count += 1
