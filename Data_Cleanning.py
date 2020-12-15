import numpy as np
import pandas as pd
from N_Build_Stock_Pool import Read_One_Stock, select_stock_pool

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

    def join_others(self):
        pass