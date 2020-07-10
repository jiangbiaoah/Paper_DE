# -*- coding:utf-8 -*-
import numpy as np
import time
import threading
import multiprocessing

from DS import decoding_result, store2file, get_fa
import configuration


mutex = threading.Lock()


# =================================================================================
# input:    从configura中获取用户分布ax,iternum_ps; 配备的天线数n从键盘读取
# process:  针对选择的分布ax，以此分布，计算在配备不同的天线下，获取不同用户数下的每次传输成功率
# output:   输出ps,n*1维数组，表示不同用户数[1,n]对应的解码成功率
# =================================================================================


def ps_m_per(q, ax, m, n, iternum, start_time):
    """ transmission successful probability of per number m
    返回m个用户同时传输时的成功概率：成功概率 = 1 - 失败用户数/总用户数
    :param q:进程队列
    :param ax:
    :param m:
    :param n:
    :param iternum:10000
    :return:1*iternum维数组，对于每个m,共iternum个成功概率
    """
    print("Computing for m = {}/{}...".format(m, n))

    key = str(m)
    value = np.zeros([1, iternum])

    for i in range(iternum):
        num_failed = decoding_result(ax, m, n)
        value[0, i] = 1 - num_failed / m    # 成功概率 = 1 - 失败用户数/总用户数
        # if num_failed > 0:
        #     value[0, i] = 0
        # else:
        #     value[0, i] = 1
    q.put({key: value})     # {m:[ps1, ps2, ..., ps_iternum]}
    print("     Done for m = {}/{}...".format(m, n))

    # 阶段数据存储
    filename = "ps_maxcopies={}_n={}_iternum={}_m={}".format(len(ax)-1, n, iternum, m)
    store2file({key: value}, filename, 0)

    end_time = time.time()
    total_time = end_time - start_time
    print("已用 {:.3f}秒, {:.3f}分钟, {:.3f}小时".format(total_time, total_time / 60, total_time / (60 * 60)))


if __name__ == "__main__":
    start_time = time.time()
    ax = configuration.a_dict[10016]
    n_input = input("Input equiped antennae n:")
    n = int(n_input)
    m_th_input = input("Input m-th (compute from m-th user, default 1):")
    m_th = int(m_th_input)
    iternum = configuration.iternum_ps

    ps = np.zeros([n, 1])   # 存储结果：表示不同用户数m∈[1,n]对应的解码成功率

    q = multiprocessing.Manager().Queue()   # 记录每个用户的iternum个成功率
    p = multiprocessing.Pool(configuration.multiprocesspoolnum)     # 最多同时执行3个进程，为了不让电脑卡死
    for m in range(m_th, n + 1):
        p.apply_async(ps_m_per, args=(q, ax, m, n, iternum, start_time,))

    p.close()
    p.join()

    for i in range(1, q.qsize() + 1):
        result = q.get()
        ps[i-1, 0] = result[str(i)].mean(axis=1)

    filename = "ps_maxcopies=" + str(len(ax)-1) + "_n=" + str(n) + "_iternum=" + str(iternum)
    store2file(ps, filename, 0)

    end_time = time.time()
    total_time = end_time - start_time
    print("耗时 {:.3f}秒, {:.3f}分钟, {:.3f}小时".format(total_time, total_time / 60, total_time / (60 * 60)))
