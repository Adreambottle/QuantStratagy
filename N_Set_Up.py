import N_Factor_Function as FF

import N_Model_Functions as MF

import matplotlib.pyplot as plt
import pandas as pd

factor = FF.read_factor("000021.SZ")

f1 = factor.iloc[:, 0]

cao = f1.rolling(24, axis=0).apply(MF.pred_by_auto_arima)


newdata = pd.DataFrame({"real":f1, "pred":cao})
newdata.plot()
fig = plt.figure()
fig.plot(f1.index, f1)
fig.plot