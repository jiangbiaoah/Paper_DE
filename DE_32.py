# -*-coding:utf-8-*-
import numpy as np
import random
import time
import DS
import DE
import configuration


def de(m=8, n=8):
    """Differential Evolution (DE)
    主函数：差分进化算法
    :return:返回天线数量的分度函数
    """
    time_start = time.time()

    GMAX = 10000    # gm:最大迭代次数
    f0 = 0.5        # 变异率
    NP = m          # 用户数m,一般取 5 ≤ m ≤ 10
    D = n + 1       # 天线数（每个个体的维度）：下标最大为8，则D=9
    CR = 0.9        # 交叉概率，一般在[0,2]之间选择，通常取0.5
    print("starting computing for n = {}".format(n))

    gmax = np.zeros(GMAX)    # 各代目标函数的最优值，也就是myfunc()中G的最大值G* ggstar
    bestuser = np.zeros((GMAX, D))   # 各代的最优个体

    x0 = np.random.rand(NP, D)       # 随机产生NP个D维向量 x0∈[0, 1)
    x0[:, 0:2] = 0        # 度为0和1的概率为0

    # 归一化
    g0 = normalize(x0)      # 第一代个体

    # ----------------
    # -----初始化------
    gnext_1_1 = np.zeros((NP, D))
    gnext_1 = np.zeros((NP, D))       # 变异后：满足边界条件
    gnext_2_1 = np.zeros((NP, D))
    gnext_2 = np.zeros((NP, D))       # 变异后：满足边界条件 & 满足交叉操作；
    gnext = np.zeros((NP, D))         # 变异后：满足边界条件 & 满足交叉操作 & 函数值小于目标函数

    g = 1  # 初始化代数：第一代
    # ----------------改动DE_1------------------------
    # 改动为修改终止条件
    value_max = 0
    while g <= GMAX and value_max < (64-1)/64:        # 退出条件1:达到最大迭代次数；条件2:最优值不再变化(此条件不成立)
        print('------------------------------------------------')
        print('generation = {}'.format(g))
        # ----------------
        # ------变异------
        for i in range(NP):
            dx = np.arange(0, NP)
            random.shuffle(dx)   # [0,1,2,3,...,NP-1]的随机序列

            j = dx[0]
            k = dx[1]
            p = dx[2]

            if j == i:
                j = dx[3]
            elif k == i:
                k = dx[3]
            elif p == i:
                p = dx[3]

            # 此为自适应变异算法
            ftemp = np.exp(1 - GMAX/(GMAX + 1 - g))
            f = f0 * pow(2, ftemp)      # f∈[f0, 2f0]

            son = g0[p, :] + f * (g0[j, :] - g0[k, :])
            for m in range(D):
                if 0 <= son[m] <= 1:
                    gnext_1_1[i, m] = son[m]
                else:
                    gnext_1_1[i, m] = random.random()
        gnext_1 = normalize(gnext_1_1)

        # ----------------
        # ------交叉------
        for i in range(NP):
            # rnbr = np.resize(np.arange(0, D), np.ceil(NP / D) * D)  # ceil(x):比x大的最小整数
            rnbr = np.arange(0, D)
            random.shuffle(rnbr)
            for j in range(D):
                if random.random() > CR and j != rnbr[i]:
                    gnext_2_1[i, j] = g0[i, j]
                else:
                    gnext_2_1[i, j] = gnext_1[i, j]

        gnext_2 = normalize(gnext_2_1)

        # ----------------
        # ------选择------
        for i in range(NP):
            if gmax_func(gnext_2[i, :], n) >= gmax_func(g0[i, :], n):
                gnext[i, :] = gnext_2[i, :]
            else:
                gnext[i, :] = g0[i, :]

        print('ax = {}'.format(gnext))

        # 找出此代各个个体的G值
        value = np.zeros(NP)  # 此代中每个用户对应的G*，共NP个用户
        for i in range(NP):
            value[i] = gmax_func(gnext[i, :], n)

        print('第{}代各个个体的G值：{}'.format(g, value))

        # 第g代中的目标函数的最大值
        value_max = max(value)
        gmax[g] = value_max
        # 保留最优个体
        pos_max = np.where(value == value_max)[0][0]  # np.where()返回的是元组，通过下标[0][0]访问
        bestuser[g, :] = gnext[pos_max, :]

        print('第{}代个体最优值：{}'.format(g, value_max))
        print('第{}代最优个体分布为：{}'.format(g, gnext[pos_max, :]))

        g0 = gnext
        g = g + 1

        time_end = time.time()
        total_time = time_end - time_start
        print("--Done of generation = {}, n = {}, total_time = {:.3f}s = {:.3f}min = {:.3f}h"
              .format(g, n, total_time, total_time/60, total_time/3600))
    # pause = input("pause...")


def normalize(x):
    """
    使得每行的和即概率和为1，并且将概率值小于0.01的值归为0。
    :param x:NP*D维数组
    :return:归一化后的NP*D维数组，每行的和满足概率为1。
    """
    m, n = x.shape      # 获取数组的行列数

    onestemp = np.ones((n, 1))
    x0 = np.matmul(x, onestemp)
    res = np.zeros((m, n))

    for i in range(m):
        res[i, :] = x[i, :] / x0[i]
        # for j in range(n):
        #     if res[i, j] < 0.01:
        #         res[i, j] = 0
    # 此处需要再次使得概率和为1，但由于小于0.01部分的占比太小，影响太小，考虑到减少计算量，不再归一化处理
    return res


def gmax_func(a, n):
    """成本函数
    根据输入的序列计算相应的G的最大值Gmax，并返回；
    :param a:概率分布多项式
    :return:Gmax
    """
    ggstar = 0  # G* g的最大值(gg:G, ggstar:G*， g:generation)
    xmin = 0
    xmax = 1

    # ----------------改动DE_1------------------------
    n = 64
    for m in range(1, n+1):
        gg = m / n
        iterbool, q_last = q_iterative(a, gg)
        if iterbool is True:
            ggstar = gg
    # -----------------------------------------------

    # while xmax - xmin >= 0.001:
    #     gg = xmin + (xmax - xmin) / 2
    #     # 将多项式a和G的每个值gg带入pq迭代方程，计算出多项式a对应的最优G:ggstar
    #     # iterbool为True,表示gg能够成功迭代。
    #     iterbool, q_last = q_iterative(a, gg)
    #     if iterbool is True:
    #         ggstar = gg
    #         xmin = gg
    #     else:
    #         xmax = gg
    #     # print("gg = {}, ggstar = {}".format(gg, ggstar))

    return ggstar


def q_iterative(a, gg):
    """
    若给定G*值，通过多次迭代，q能趋于0,则返回True;否则返回False
    :param a:
    :param gg:
    :return:True or False:迭代是否成功
    """
    q = [1]
    result_bool = False

    fa = np.polynomial.Polynomial(a)  # 数组a的多项式表示
    fader = fa.deriv()  # 数组a的导数的多项式表示
    # fader1 = np.polynomial.polyval(fader, 1)  # a的导数在1处的取值

    while True:
        qnext = fader(1 - np.exp(-q[-1] * gg * fader(1))) / fader(1)
        q.append(qnext)

        if qnext < 1e-6:
            result_bool = True
            break
        if len(q) >= 10000:  # 迭代10000次还未收敛，则判断在该G情况下不收敛
            result_bool = False
            break
    q_last = qnext
    return result_bool, q_last


if __name__ == "__main__":
    # a = configuration.a_dict[18]
    # get_gstar(a)

    # n = input("Please input antennae number n:")
    n = 8
    # m = input("Please input user number m (default 8):")
    m = 8
    print("input m = {}, n = {}".format(m, n))
    de(int(m), int(n))
    # if int(m) <= int(n):
    #     de(int(m), int(n))
    # else:
    #     print("condition:m <= n")

