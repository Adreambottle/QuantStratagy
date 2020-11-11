import numpy as np
import pandas as pd

import Build_Stock_Pool as BSP
import Draw_Plots as DP
import Data_Cleanning as DC


Stock_Codes_Df = BSP.select_stock_pool()
Stock_Codes = Stock_Codes_Df["Stock_Code"]

test = [i+"This a test" for i in Stock_Codes]
test = list(map(lambda x : x**2, [0, 1, 2, 3, 4]))
