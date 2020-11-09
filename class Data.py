#导入需要用到的模块
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime,timedelta
import tushare as ts
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

from pyecharts.charts import Kline
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Grid, Tab


# #使用python3自带的sqlite数据库
#
# file = "sqlite:///Users/meron/PycharmProjects/5250Pjt/db_stock"
# # 数据库名称
# db_name = 'stock_data.db'
# engine = create_engine(file + db_name)


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
        # for code in self.codes:
        #     data = self.daily_data(code)
        #     data.to_sql(self.table_name, engine, index=False, if_exists = 'append')

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
    SC_in_pool = Stock_code_pool['Stock_Code']
    SC_unavailable = []
    count = 210
    for SC in SC_in_pool[210:]:
        print("This is the {} and the code is {}.".format(count, SC))
        try:
            tmp_data = Stock_data.daily_data(SC)
            pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='Stock', if_exists='append')
        except:
            SC_unavailable.append(SC)
        count += 1



# 从 MySQL 中读取数据

class Read_One_Stock():

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

    def select_close_data(self):
        # 读取每天的收盘价
        sqlcmd = "SELECT trade_date, close FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    def select_open_data(self):
        # 读取每天的开盘价
        sqlcmd = "SELECT trade_date, open FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

    def select_vol_amount(self):
        # 读取每天的
        sqlcmd = "SELECT trade_date, open FROM`{}`".format(self.SC_Code)
        table = pd.read_sql(sqlcmd, self.conn)
        return table

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

# 将数据拼接好连在一起


class Join_Table():
    def __init__(self):
        self.Df_StockPoll = select_stock_pool()
        self.Ar_SC = self.Df_StockPoll["Stock_Code"]

    def set_SC_list(self, SC_list):
        self.Ar_SC = np.array(SC_list)

    def get_SC_list(self):
        return self.Ar_SC

    def join_close(self):
        join_table = pd.DataFrame({"trade_date":[]})
        for SC in self.Ar_SC:
            Stock_data = Read_One_Stock(SC).select_close_data()
            Stock_data.columns = ["trade_date", SC]
            join_table = pd.merge(join_table, Stock_data, how='outer', on="trade_date")
        return join_table

    def join_open(self):
        join_table = pd.DataFrame({"trade_date":[]})
        for SC in self.Ar_SC:
            Stock_data = Read_One_Stock(SC).select_open_data()
            Stock_data.columns = ["trade_date", SC]
            join_table = pd.merge(join_table, Stock_data, how='outer', on="trade_date")
        return join_table

    def join_others:
        pass


# def save_table_to_mysql(table:pd.DataFrame, table_name):
#     conn = create_engine('mysql+mysqldb://root:zzzzzzzz@localhost:3306/Factor?charset=utf8')
#     pd.io.sql.to_sql(table, table_name, con=conn, schema='Factor', if_exists='append')
#
# miao = Join_Table()
# test = miao.join_close()
# save_table_to_mysql(test, "Daily_close")

# class Create_Factor():
#     def __init__(self, ):
#         self.Df_StockPoll = select_stock_pool()
#         self.Ar_SC = self.Df_StockPoll["Stock_Code"]
#         self.conn = pymysql.connect(
#             host="localhost",
#             database="Stock",
#             user="root",
#             password="zzzzzzzz",
#             port=3306,
#             charset='utf8'
#         )
#
#     def save_close_to_mysql(self, JD:Join_Table):
#         tmp_data = Stock_data.daily_data(
#         pd.io.sql.to_sql(tmp_data, SC, con=conn, schema='Stock', if_exists='append')

def calculate_ma_n(data, n):
    ma_value = []
    for i in range(n):
        ma_value.append(None)
    for i in range(n, len(data)):
        value = round(sum(data[(i-n):i])/n, 5)
        ma_value.append(value)
    return ma_value


def calculate_ma_n(data, n):
    ma_value = []
    for i in range(n):
        ma_value.append(None)
    for i in range(n, len(data)):
        value = round(sum(data[(i - n):i]) / n, 5)
        ma_value.append(value)
    return ma_value


class Draw_Plots(object):
    def __init__(self, SC):
        self.SC = SC

    def draw_Kline(self):
        Df_s1 = Read_One_Stock(self.SC).select_col("open", "high", "low", "close", "vol", "amount")
        length = Df_s1.shape[0]
        Df_s1.sort_values("trade_date", inplace=True)
        Df_s1.index = list(range(length))
        price = np.array(Df_s1[["open", "close", "high", "low"]]).tolist()
        date = np.array(Df_s1["trade_date"], dtype=np.string_).tolist()
        ma_value_5 = calculate_ma_n(list(Df_s1['close']), 5)
        ma_value_10 = calculate_ma_n(list(Df_s1['close']), 10)
        ma_value = np.array([ma_value_5, ma_value_10]).tolist()

        kline = Kline()
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[opts.DataZoomOpts()],
            title_opts=opts.TitleOpts(title="K-Line of {}".format(self.SC)),
        )
        kline.add_xaxis(date)
        kline.add_yaxis('K-Line', price)

        line = Line()
        line.add_xaxis(date)
        line.add_yaxis(
            series_name="ma5",
            y_axis=ma_value[0],
            label_opts=opts.LabelOpts(is_show=False)
        )
        line.add_yaxis(
            series_name="ma10",
            y_axis=ma_value[1],
            label_opts=opts.LabelOpts(is_show=False)

        )
        line.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[opts.DataZoomOpts()],
            title_opts=opts.TitleOpts(title="")
        )

        kline.overlap(line)
        kline.render("./Plots/{} Candle Plot.html".format(self.SC))

    def draw_amount_bar(self):
        # SC = "000021.SZ"
        Df_s1 = Read_One_Stock(self.SC).select_col("vol", "amount")
        length = Df_s1.shape[0]
        Df_s1.sort_values("trade_date", inplace=True)
        Df_s1.index = list(range(length))
        amount = np.array(Df_s1["amount"]).tolist()
        date = np.array(Df_s1["trade_date"], dtype=np.string_).tolist()

        bar = Bar()
        bar.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[opts.DataZoomOpts()],
            title_opts=opts.TitleOpts(title="K-Line of {}".format(self.SC))
        )
        bar.add_xaxis(date)
        bar.add_yaxis("Amounts", amount, label_opts=opts.LabelOpts(is_show=False))

        bar.render("./Plots/{} Amount Bar Plot.html".format(self.SC))


Draw_Plots("000021.SZ").draw_Kline()
Draw_Plots("000021.SZ").draw_amount_bar()

#
#
# def draw_Kline(SC):
#     # SC = "000021.SZ"
#     Df_s1 = Read_One_Stock(SC).select_col("open", "high", "low", "close", "vol", "amount")
#     length = Df_s1.shape[0]
#     Df_s1.sort_values("trade_date", inplace=True)
#     Df_s1.index = list(range(length))
#     price = np.array(Df_s1[["open", "close", "high", "low"]]).tolist()
#
#     ma_value = []
#     ma_value_5 = calculate_ma_n(list(Df_s1['close']), 5)
#     ma_value_10 = calculate_ma_n(list(Df_s1['close']), 10)
#     ma_value = np.array([ma_value_5, ma_value_10]).tolist()
#
#     amount = np.array(Df_s1["amount"]).tolist()
#     date = np.array(Df_s1["trade_date"],dtype=np.string_).tolist()
#
#
#     grid = Grid()
#     kline = Kline()
#     kline.set_global_opts(
#         xaxis_opts=opts.AxisOpts(is_scale=True),
#         yaxis_opts=opts.AxisOpts(
#             is_scale=True,
#             splitarea_opts=opts.SplitAreaOpts(
#                 is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
#             ),
#         ),
#         datazoom_opts=[opts.DataZoomOpts()],
#         title_opts=opts.TitleOpts(title="K-Line of {}".format(SC)),
#     )
#     kline.add_xaxis(date)
#     kline.add_yaxis('K-Line', price)
#
#     line = Line()
#     line.add_xaxis(date)
#     line.add_yaxis(
#             series_name="ma5",
#             y_axis=ma_value[0],
#             label_opts=opts.LabelOpts(is_show=False)
#         )
#     line.add_yaxis(
#             series_name="ma10",
#             y_axis=ma_value[1],
#             label_opts=opts.LabelOpts(is_show=False)
#
#     )
#     line.set_global_opts(
#         xaxis_opts=opts.AxisOpts(is_scale=True),
#         yaxis_opts=opts.AxisOpts(
#             is_scale=True,
#             splitarea_opts=opts.SplitAreaOpts(
#                 is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
#             ),
#         ),
#         datazoom_opts=[opts.DataZoomOpts()],
#         title_opts=opts.TitleOpts(title="K-Line of {}".format(SC))
#     )
#
#
#     bar = Bar()
#     bar.set_global_opts(
#         xaxis_opts=opts.AxisOpts(is_scale=True),
#         yaxis_opts=opts.AxisOpts(
#             is_scale=True,
#             max_=(max(amount) * 5),
#             splitarea_opts=opts.SplitAreaOpts(
#                 is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
#             ),
#         ),
#         datazoom_opts=[opts.DataZoomOpts()],
#         title_opts=opts.TitleOpts(title="")
#     )
#     bar.add_xaxis(date)
#     bar.add_yaxis("", amount, label_opts=opts.LabelOpts(is_show=False))
#
#
#     kline.overlap(line)
#
#     grid.add(kline,
#              grid_opts=opts.GridOpts(),
#              is_control_axis_index=True)
#     grid.add(bar,
#              grid_opts=opts.GridOpts(),
#              is_control_axis_index=False)
#
#     grid.render()
#
# draw_Kline("000021.SZ")
#
#

# 创建数据库

token = 'd44cbc9ab3e7c25e5dfcbe6437ac061b125395567ad582806d02d38c'
pro = ts.pro_api(token)

Stock_data = Data()
Stock_code = Stock_data.get_code()
Stock_code_df = pd.DataFrame({"Stock_name":Stock_code})


# 链接 MySQL
conn = create_engine('mysql+mysqldb://root:zzzzzzzz@localhost:3306/Stock?charset=utf8')
pd.io.sql.to_sql(Stock_code_df,'Stock_Code', con=conn, schema='Stock', if_exists='append')

# 修改数据内容
Stock_code_pool = pd.read_excel("/Users/meron/PycharmProjects/5250Pjt/Wind_store_pool.xlsx")
Stock_code_pool.columns = ['Stock_Code', 'Stock_Name']
pd.io.sql.to_sql(Stock_code_pool,'Stock_Pool', con=conn, schema='Stock', if_exists='append')

store_daily_data()
# connection = pymysql.connect(host = 'localhost',
#                              port = 3306,
#                              user = 'root',
#                              password = 'zzzzzzzz',
#                              db = 'DS',
#                              charset = 'utf8')
# cursor = connection.cursor()
# cursor.execute("select * from student;")
# cursor.fetchmany(10)