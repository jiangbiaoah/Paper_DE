# -*- coding:utf-8 -*-
import numpy as np
import time

import multiprocessing

import DS
import configuration


# =========================================================================
# input:    从configura中获取用户分布ax,iternum_ds; 配备的天线数n从键盘读取
# process:  针对选择的分布ax，以此分布，计算在配备不同的天线下，系统时延随用户m的变化
# outpur:   ds_avg[] n*1维数组，保存了n个用户的延迟信息
# =========================================================================
def ds_per_m(q, ax, m, n, iternum, m_max):
    """
    计算m个用户的延时，计算iternum次
    :param q:
    :param ax:
    :param m:
    :param n:
    :param iternum:
    :return:返回1*iternum维矩阵，分别为每次计算的延时
    """
    print("Computing for m = {}/{}...".format(m, m_max))

    key = str(m)
    value = None
    ds_dist = np.zeros([1, iternum])    # 保存m个用户每次迭代的延时组成，将其平均值作为此m个用户的延时代表
    ds_avg_per = 0
    for i in range(iternum):
        ds = DS.avg_ds(ax, m, n)  # 获取不同用户数下的系统延时
        ds_dist[0, i] = ds[0]
    ds_avg_per = ds_dist.mean(axis=1)
    value = ds_avg_per
    q.put({key: value})
    print("--Done for m = {}/{}...".format(m, m_max))

    filename = "ds_avg_maxcopies=" + str(len(ax) - 1) + "_n=" + str(n) + "_mmax=" + str(m_max) + '_m=' + m
    DS.store2file(ds_avg, filename, 0)


if __name__ == "__main__":
    start_time = time.time()
    # ax = [0, 0, 0.52, 0.26, 0, 0, 0, 0, 0.22]
    ax = configuration.a_dict[16]
    n_input = input("Input equiped antennae n:")
    n = int(n_input)
    # m_max = configuration.m_max
    m_max = n + int(n/4)    # 多计算n/4个用户
    iternum = configuration.iternum_ds
    ds_avg = np.zeros([m_max, 1])

    p = multiprocessing.Pool(configuration.multiprocesspoolnum)
    q = multiprocessing.Manager().Queue()

    for m in range(1, m_max + 1):
        p.apply_async(ds_per_m, args=(q, ax, m, n, iternum, m_max,))
    p.close()
    p.join()

    for i in range(1, q.qsize() + 1):
        result = q.get()
        ds_avg[i-1, 0] = result[str(i)]

    filename = "ds_avg_maxcopies=" + str(len(ax)-1) + "_n=" + str(n) + "_mmax=" + str(m_max)
    DS.store2file(ds_avg, filename, 0)

    end_time = time.time()
    total_time = end_time - start_time
    print("耗时 {:.3f}秒, {:.3f}分钟, {:.3f}小时".format(total_time, total_time / 60, total_time / (60 * 60)))
