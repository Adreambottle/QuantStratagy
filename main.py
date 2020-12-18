import numpy as np
import pandas as pd

import N_Build_Stock_Pool as BSP
import N_Draw_Plots as DP
import Data_Cleanning as DC



import time
from joblib import Parallel , delayed
from multiprocessing import Pool


def run(fn):
    # fn: 函数参数是数据列表的一个元素
    time.sleep(1)
    print(fn * fn)


if __name__ == "__main__":
    testFL = [1, 2, 3, 4, 5, 6]
    print('shunxu:')  # 顺序执行(也就是串行执行，单进程)
    s = time.time()
    for fn in testFL:
        run(fn)
    t1 = time.time()
    print("顺序执行时间：", int(t1 - s))

    print('concurrent:')  # 创建多个进程，并行执行
    # pool = Pool(8)  # 创建拥有3个进程数量的进程池
    # # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    # pool.map(run, testFL)
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出

    Parallel(n_jobs=10)(delayed(run)(i) for i in testFL)

    t2 = time.time()
    print("并行执行时间：", int(t2 - t1))

