# -*- coding:utf-8 -*-
import numpy as np
import numpy.matlib
import random
import time
import pickle

import configuration


# 已使用并行运算替代
def ds_m(ax, n):
    """获取ds与m的关系，m = 1 ~ n
    在天线数一定的情况下，获取系统平均时延ds与用户数m的关系
    :param ax:
    :param n:
    :return:返回n*iternum维数组，每行记录了每个用户仿真1000次的ds结果，一共最多n个用户。
    """
    iternum = configuration.iternum
    ds_dist = np.zeros([n, iternum])
    ds_avg = np.zeros([n, 1])

    for m in range(1, n + 1):
        print("Computing for m = {}/{}...".format(m, n))
        for i in range(iternum):
            ds = avg_ds(ax, m, n)       # 需要并行运算加速
            ds_dist[m-1, i] = ds[0]
        # print("m = ", m)
        # print("dis_dist[", m-1, ", :] = ")
        # print(ds_dist[m-1, :])

    ds_avg = ds_dist.mean(axis=1)       # axis=1横向的均值，axis=0纵向的均值

    return ds_dist, ds_avg


def avg_ds(ax, m, n):
    """ average delay of systems
    求系统平均时延
    :param ax:
    :param m:
    :param n:
    :return:ds, delay, delay_str即延时组成情况
    """
    if n < len(ax):
        print("错误：天线数应大于分布函数指定的最大副本数")
        return
    # if m > n:
    #     print("错误：天线数应大于用户数")
    #     return

    t = 0  # t ∈ T, T = max j
    b = np.zeros([m, n])    # b[j] = B[i][j][l]
    num_failed = m          # 解码失败的用户数
    delay = 0               # 系统总延时
    delay_str = ''          # 系统延时的组成情况

    j = 0
    while num_failed != 0:
        num_usr = num_failed

        # 获取解码结果情况，返回解码失败的用户数
        num_failed = decoding_result(ax, num_usr, n)

        delay = delay + (num_usr - num_failed) * (j + 1)
        delay_str = delay_str + str(num_usr - num_failed) + ' * ' + str(j + 1) + " + "
        j = j + 1

    # print("ds = ", delay, " / ", m, " = ", delay / m, " || delay = ", delay_str[:-2])
    return delay / m, delay, delay_str


def decoding_result(ax, m, n):
    """
    获取每个TTI中的用户天线分布，并使用IC解码该分布，并返回解码失败的用户数
    :param ax:
    :param m:
    :param n:
    :return:返回解码失败的用户数
    """
    # 获取每个TTI中的用户天线分布
    b = _atdist_pertti(ax, m, n)
    # 使用IC解码该分布，并返回解码情况
    num_failed = _icdecoding(b, m, n)
    return num_failed


def _atdist_pertti(ax, m, n):
    """ antennae distribution of per TTI
    获取每个TTI中的用户天线分布
    :param ax:
    :param m:
    :param n:
    :return:返回该TTI中的用户天线分布矩阵 b ∈ B
    """
    b = np.zeros([m, n])  # b[j] = B[i][j][l]

    # 将多项式中的分布概率转化为概率的一维分布
    scale = [0]  # 概率的一维分布
    for i in range(len(ax)):
        if i == 0:
            continue
        scale.append(scale[i - 1] + ax[i])
    # print('ax = ', ax)
    # print('scale = ', scale)

    # 使用0~1之间的随机数判断其在一维分布中的位置，以确定m个用户所发送的副本数copies[]。
    copies = []  # 记录每个用户的副本数
    for i in range(m):
        x = random.random()
        for j in range(len(ax)):
            if scale[j] <= x < scale[j + 1]:
                copy = j + 1
        copies.append(copy)
    # print('m个用户的副本数: copies = ', copies)

    # 确定用户的天线分配
    antennae_list = [i for i in range(n)]  # range(5) = [0, 1, 2, 3, 4]
    for i in range(m):
        copy = copies[i]  # 第i个用户有copy个副本，随机分布在n个天线上
        random.shuffle(antennae_list)
        for j in range(copy):
            b[i][antennae_list[j]] = 1
    # print('b = "天线分布矩阵"', b)

    return b


def _icdecoding(b, m, n):
    """
    使用IC解码机制解码,返回解码失败的用户数
    :param b:天线分布矩阵
    :return:返回解码失败的用户数
    """
    num_failed = m  # 解码失败的用户数
    btemp = np.asmatrix(b.copy())           # 在原始数组的拷贝上操作，将其转换为矩阵，以方便使用行列式操作
    onestemp = np.matlib.ones((1, m))       # 1*m 全1矩阵
    onestemp_1 = np.matlib.ones((n, 1))     # m*1 全1矩阵

    while True:
        if np.all(btemp == 0):
            # print('--decoding success')
            return 0    # 解码成功

        btemp_1 = np.dot(onestemp, btemp)
        # print('btemp_1 = ', btemp_1)
        ch_seq = np.where(btemp_1[0, :] == 1)     # 找出无冲突的所有信道的序号ch

        num_suc = len(ch_seq[0])        # ch_seq 是tuple类型
        if num_suc == 0:
            # print('--decoding failed')
            btemp_2 = np.dot(btemp, onestemp_1)
            failed_seq = np.where(btemp_2[:, 0] > 0)
            num_failed = len(failed_seq[0])
            # print('num_failed = ', num_failed)
            return num_failed

        # num_failed = m - num_suc
        for ch in np.nditer(ch_seq[1]):
            usr_seq = np.where(btemp[:, ch])
            if len(usr_seq[0]) == 0:
                continue
            btemp[usr_seq[1][0], :] = 0

        # print()
        # print('---------------------------')
        # print()

    return num_failed


# 已使用并行运算替代
def ps_m(ax, n):
    """ transmissioin successful probability
    天线数n一定的情况下，...，供show_ps_m(ax, n)函数调用
    :param ax:
    :param n:
    :return: 本次传输成功的概率
    """
    iternum = configuration.iternum
    result = np.zeros([n, iternum])
    ps = np.zeros([n, 1])

    for m in range(1, n + 1):       # 需要并行加速
        print("Computing for m = {}/{}...".format(m, n))
        for i in range(iternum):
            num_failed = decoding_result(ax, m, n)
            if num_failed > 0:
                result[m - 1, i] = 0
            else:
                result[m - 1, i] = 1
    ps = result.mean(axis=1)

    return ps


def store2file(data, filename, adddate=1, type=1):
    """
    封装了对文件的操作
    :param data:
    :param filename:
    :param adddate: 1:文件名中加入时间字符串，0：不加入时间字符串
    :param type: 1：保存成.pkl，0：保存成txt
    :return:保存文件并打印出文件名
    """
    filenames = "./data_0/" + filename
    if adddate is 1:
        time_str = str(time.strftime("_%Y%m%d_%H%M%S", time.localtime()))
        if type is 1:
            filenames = filenames + time_str + ".pkl"
        else:
            filenames = filenames + time_str + ".txt"
    else:
        if type is 1:
            filenames = filenames + ".pkl"
        else:
            filenames = filenames + ".txt"

    if type is 1:
        with open(filenames, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(filenames, "w") as f:
            f.write(str(data))
    print("-----文件写入成功!-----")
    print("[filepath:{}]".format(filenames))


def merging_data():
    """
    整合计算成功率的中间文件
    :return:
    """
    maxcopies = 16
    n = 512

    filepaths = []
    # {m:[ps1, ps2, ..., ps_iternum]}
    for i in range(1, n+1):
        filepath = "./data{}_{}/ps_maxcopies={}_n={}_iternum=10000_m={}.pkl"\
            .format(n, maxcopies, maxcopies, n, i)
        filepaths.append(filepath)
    # print("filepaths = {}".format(filepaths))

    ps = np.zeros([n, 1])  # 存储结果：表示不同用户数m∈[1,n]对应的解码成功率
    i = 0
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            res = pickle.load(f)
            for key in res:
                # print("key = {}, type(key) = {}".format(key, type(key)))
                # print("res[key] = {}".format(res[key]))
                print("i = {}".format(i))
                ps[i, 0] = res[key].mean(axis=1)
                i = i + 1
    print("len(ps) = {}, ps = {}".format(len(ps), ps))

    filename = "ps_maxcopies=" + str(maxcopies) + "_n=" + str(n) + "_iternum=" + str(10000)
    store2file(ps, filename, 0)


def get_m_with_given_rs():
    """
    已知某一分布，获取在满足成功率rs大于一定值的情况下，最大能够支持的并行用户数
    输入的数据来自成功率文件 *.pkl
    :return:
    """
    filepaths = ["./data/ps_maxcopies=4_n=20_iternum=10000.pkl",    # 0
                 "./data/ps_maxcopies=4_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=20_iternum=10000.pkl",    # 8
                 "./data/ps_maxcopies=8_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=20_iternum=10000.pkl",   # 16
                 "./data/ps_maxcopies=16_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=512_iternum=10000.pkl"]
    N = [20, 50, 80, 100, 120, 150, 200, 512, 20, 50, 80, 100, 120, 150, 200, 512, 20, 50, 80, 100, 120, 150, 200, 512, ]
    n_i = 0
    for filepath in filepaths:
        n = N[n_i]
        n_i = n_i + 1
        # filepath = filepaths[10]
        print("filepath = {}".format(filepath))
        y = []
        with open(filepath, "rb") as f:
            ps = pickle.load(f)  # ps是一个n*1维数组，保存了n=20,50,100,200个用户下的系统延时
            y = ps.reshape(1, len(ps))[0]
        # print("type(y) = {}".format(type(y)))
        # print("y = {}".format(y))

        max_m = 0
        for yi in y:
            if yi >= 0.995:
                max_m = max_m + 1
        print("max_m = {}, n = {}, m/n = {}".format(max_m, n, max_m/n))
        print("")


def get_fa(ax):
    """
    将ax返回成数学表达式字符串
    :param ax:
    :return:
    """
    poly_str = ""
    iter_arr = np.arange(len(ax))
    for i in iter_arr:
        if ax[i] != 0:
            poly_str = poly_str + str(ax[i]) + "x^" + str(i) + " + "
    return poly_str[:-3]


def get_avg_copies():
    """
    针对每个最优分布，获取其平均副本数，代表着能量的消耗量，即 ∑lΛl
    :return:
    """
    for key in configuration.a_dict:
        maxcopies = key
        dist = configuration.a_dict[key]
        a = np.arange(len(dist))
        avg = numpy.vdot(dist, a)
        print("---------------------------")
        print("{}:{}".format(key, dist))
        print("avg_copies = {}".format(avg))

# =========================
# 调用上面的函数


if __name__ == "__main__":
    # get_avg_copies()
    # merging_data()
    get_m_with_given_rs()
