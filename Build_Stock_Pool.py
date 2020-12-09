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
                 start ='20100101',
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
    def daily_data(self, code):
        try:
            # 获取每日交易数据
            df0 = pro.daily(ts_code = code, start_date = self.start, end_date = self.end)

            # 获取每日指标
            df1 = pro.daily_basic(ts_code = code, start_date = self.start, end_date = self.end)

            # 获取复权行情
            df2 = pro.pro_bar(ts_code = code, start_date = self.start, end_date = self.end)

            # 获取复权因子
            df3 = pro.adj_factor(ts_code = code, start_date = self.start, end_date = self.end)

            df = pd.merge(df0, df1, df2, df3)  #合并数据

            return df

        except Exception as e:
            print(code + " is failed!")
            print(e)



    #保存数据到数据库
    def finance_data(self, code):
        try:
            # 获取每日交易数据
            df0 = pro.income(ts_code = code, start_date = self.start, end_date = self.end,
                             fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
            df1 = pro.cashflow(ts_code = code, start_date = self.start, end_date = self.end)

            df = pd.merge(df0, df1)  #合并数据

            return df

        except Exception as e:
            print(code + " is failed!")
            print(e)


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


    # 将每日数据储存到SQL中

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





    # 将财务数据储存在SQL中

    for SC in SC_in_pool:
        print("This is the {} and the code is {}.".format(count, SC))

        # 将所有在股票池中的数据储存在 MySQL 中
        try:
            tmp_data = Stock_data.finance_data(SC)
            pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='Finance', if_exists='append')

        # 如果访问失败，将数据储存在 SC_unavailable 的列表中
        except:
            SC_unavailable.append(SC)
        count += 1






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

def select_stock_pool():
    conn = pymysql.connect(
        host="localhost",
        database="Stock",
        user="root",
        password="zzzzzzzz",
        port=3306,
        charset='utf8'
    )
    sqlcmd = "SELECT * FROM `Stock_Pool`"
    table = pd.read_sql(sqlcmd, conn)
    return table