import talib
import tushare
from pyecharts.charts import Line, Kline, Bar, Overlap, Grid

# get 300ETF from tushare
data = tushare.get_k_data('600519', ktype='D', autype='None', start='2015-01-01', end='2018-07-01')
# 计算并画出cci
cci = talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
# 简单的一个择时策略，当cci>50则持仓，当cci<50则空仓
position = [50 if idx >= 50 else 0 for idx in cci]

# 定义k线图的提示框的显示函数
def show_kline_data(params, pos):
    param = params[0]
    if param.data[4]:
        return "date = " + param.name + "<br/>" + "open = " + param.data[1] + "<br/>" +
               "close = " + param.data[2] + "<br/>" + "high = " + param.data[3] + "<br/>"
                + "low = " + param.data[4] + "<br/> "
    else:
        return "date = " + param.name + "<br/>" + "cci = " + param.value + "<br/>"
# 绘制cci
cci_line = Line()
cci_line.add("cci", x_axis=data['date'], y_axis=cci, is_datazoom_show=True,
             datazoom_xaxis_index=[0, 1],
             tooltip_tragger='axis',
             is_toolbox_show=True,
             yaxis_force_interval=100,
             legend_top="70%",
             legend_orient='vertical',
             legend_pos='right',
             yaxis_pos='left',
             is_xaxislabel_align=True,
             tooltip_formatter=show_kline_data,
             )

# 绘制持仓
bar = Bar()
bar.add('持仓', data['date'], position, is_datazoom_show=True)
# 将持仓和cci重叠在一个图中
cci_overlap = Overlap()
cci_overlap.add(cci_line)
cci_overlap.add(bar)
cci_overlap.render()

# 画出K线图
price = [[open, close, lowest, highest] for open, close, lowest, highest in
         zip(data['open'], data['close'], data['low'], data['high'])]
kline = Kline("贵州茅台", title_pos='center')
kline.add('日线', x_axis=data['date'], y_axis=price, is_datazoom_show=True,
          is_xaxislabel_align=True,
          tooltip_tragger='axis',
          yaxis_pos='left',
          legend_top="20%",
          legend_orient='vertical',
          legend_pos='right',
          is_toolbox_show=True,
          tooltip_formatter=show_kline_data)

# 将cci折线图和K线图合并到一张图表中
grid = Grid()
grid.add(cci_overlap, grid_top="70%")
grid.add(kline, grid_bottom="40%")

grid.render()