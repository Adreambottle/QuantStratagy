
def ADF_test():
    x_list = []
    p_list = []
    list_2 = []
    list_3 = []

    list_1p = []
    list_5p = []
    list_10p = []

    list_4 = []

    for i in range(factor.shape[1]):
        data = factor.iloc[:, i]
        t = sm.tsa.stattools.adfuller(data)
        x = t[0]
        p = t[1]
        x_list.append(x)
        p_list.append(p)
        list_2.append(t[2])
        list_3.append(t[3])
        list_1p.append(t[4]["1%"])
        list_5p.append(t[4]["5%"])
        list_10p.append(t[4]["10%"])
        list_4.append(t[5])
    df_ADF = pd.DataFrame({"Factor":factor.columns,
                           "x":x_list,
                           "p_value":p_list,
                           "list_2":list_2,
                           "list_3":list_3,
                           "list_1p":list_1p,
                           "list_5p":list_5p,
                           "list_10p":list_10p,
                           "list_4":list_4})
    df_ADF.to_excel("/Users/meron/Desktop/ADF.xlsx")


