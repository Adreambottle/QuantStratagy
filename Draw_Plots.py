import numpy as np
import pandas as pd
from N_Build_Stock_Pool import Read_One_Stock

from pyecharts.charts import Kline
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Grid, Tab


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