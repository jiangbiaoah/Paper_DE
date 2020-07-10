# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from brokenaxes import brokenaxes
import time
import pickle
import DE
import DS
import configuration


# =========================================
# --------------图示展示函数-----------------
# =========================================
def show_ds_m_all():
    """ 仿真结果
    确定一个最优分布configuration.ax，在配备不同的天线下，展示ds随m的变化
    数据来源于文件ds_avg_maxcopies=xxx_n=xxx.pkl，由ds_m_parallel.py并行计算得到
    :return:
    """
    print("runing show_ds_m_all() ...")
    filepaths = ["./data/ds_avg_maxcopies=4_n=20_mmax=25.pkl",
                 "./data/ds_avg_maxcopies=4_n=50_mmax=62.pkl",
                 "./data/ds_avg_maxcopies=4_n=80_mmax=100.pkl",
                 "./data/ds_avg_maxcopies=4_n=100_mmax=125.pkl",
                 "./data/ds_avg_maxcopies=4_n=120_mmax=150.pkl",
                 "./data/ds_avg_maxcopies=4_n=150_mmax=187.pkl",
                 "./data/ds_avg_maxcopies=4_n=200_mmax=250.pkl",
                 "./data/ds_avg_maxcopies=8_n=20_mmax=25.pkl",
                 "./data/ds_avg_maxcopies=8_n=50_mmax=62.pkl",
                 "./data/ds_avg_maxcopies=8_n=80_mmax=100.pkl",
                 "./data/ds_avg_maxcopies=8_n=100_mmax=125.pkl",
                 "./data/ds_avg_maxcopies=8_n=120_mmax=150.pkl",
                 "./data/ds_avg_maxcopies=8_n=150_mmax=187.pkl",
                 "./data/ds_avg_maxcopies=8_n=200_mmax=250.pkl",
                 "./data/ds_avg_maxcopies=16_n=20_mmax=25.pkl",
                 "./data/ds_avg_maxcopies=16_n=50_mmax=62.pkl",
                 "./data/ds_avg_maxcopies=16_n=80_mmax=100.pkl",
                 "./data/ds_avg_maxcopies=16_n=100_mmax=125.pkl",
                 "./data/ds_avg_maxcopies=16_n=120_mmax=150.pkl",
                 "./data/ds_avg_maxcopies=16_n=150_mmax=187.pkl",
                 "./data/ds_avg_maxcopies=16_n=200_mmax=250.pkl"]
    m_max = configuration.m_max
    x = np.arange(1, m_max+1)
    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ds_avg = pickle.load(f)     # ds_avg是一个m_max*1维数组，保存了m_max个用户下的系统延时
            y0 = ds_avg.reshape(1, len(ds_avg))[0]
            y.append(y0)
    # 计算的最多的用户数为 n + n/4
    # 显示所有计算的用户的延时
    fig = plt.figure(figsize=(16, 8))

    # 显示部分计算的用户的延时
    plt.plot(np.arange(1, 25+1), y[0], "b+-", label=r'$AN=4, n=20$')
    plt.plot(np.arange(1, 62+1), y[1], "bx-", label=r'$AN=4, n=50$')
    plt.plot(np.arange(1, 100+1), y[2], "b--", label=r'$AN=4, n=80$')
    plt.plot(np.arange(1, 125+1), y[3], "b-.", label=r'$AN=4, n=100$')
    plt.plot(np.arange(1, 150+1), y[4], "b-", label=r'$AN=4, n=120$')
    plt.plot(np.arange(1, 187+1), y[5], "b--.", label=r'$AN=4, n=150$')
    plt.plot(np.arange(1, 250+1), y[6], "b.-", label=r'$AN=4, n=200$')

    plt.plot(np.arange(1, 25+1), y[7], "k+-", label=r'$AN=8, n=20$')
    plt.plot(np.arange(1, 62+1), y[8], "kx-", label=r'$AN=8, n=50$')
    plt.plot(np.arange(1, 100+1), y[9], "k--", label=r'$AN=8, n=80$')
    plt.plot(np.arange(1, 125+1), y[10], "k-.", label=r'$AN=8, n=100$')
    plt.plot(np.arange(1, 150+1), y[11], "k-", label=r'$AN=8, n=120$')
    plt.plot(np.arange(1, 187+1), y[12], "k--.", label=r'$AN=8, n=150$')
    plt.plot(np.arange(1, 250+1), y[13], "k.-", label=r'$AN=8, n=200$')

    plt.plot(np.arange(1, 25+1), y[14], "r+-", label=r'$AN=16, n=20$')
    plt.plot(np.arange(1, 62+1), y[15], "rx-", label=r'$AN=16, n=50$')
    plt.plot(np.arange(1, 100+1), y[16], "r--", label=r'$AN=16, n=80$')
    plt.plot(np.arange(1, 125+1), y[17], "r-.", label=r'$AN=16, n=100$')
    plt.plot(np.arange(1, 150+1), y[18], "r-", label=r'$AN=16, n=120$')
    plt.plot(np.arange(1, 187+1), y[19], "r--.", label=r'$AN=16, n=150$')
    plt.plot(np.arange(1, 250+1), y[20], "r.-", label=r'$AN=16, n=200$')

    plt.xlabel("Parallel UE number")
    plt.ylabel("System average delay (TTI)")

    # 设置坐标轴属性
    plt.xlim(0, 200+1)
    plt.xticks(np.arange(0, 200+1, 10))
    plt.ylim(0.8, 3.2)
    plt.yticks(np.arange(1, 3.2, 0.2))

    plt.grid(b=True, linestyle=':')     # 显示坐标网格
    plt.legend(loc=1)
    plt.show()
    # plt.savefig('./pictures/ds.eps', dpi=600, format='eps')


def show_ds_m_four():
    """
    一张图展示4张配置的天线数不同的子图，分别表示了在满足ds ≤ 1.xx 时，最高能够支持的用户数
    Equipped antennae n = 20, 50, 100, 200
    :return:
    """
    print("runing show_ds_m_four() ...")
    filepaths = ["./data/ds_avg_maxcopies=8_n=20_mmax=25.pkl",
                 "./data/ds_avg_maxcopies=8_n=50_mmax=62.pkl",
                 "./data/ds_avg_maxcopies=8_n=100_mmax=125.pkl",
                 "./data/ds_avg_maxcopies=8_n=200_mmax=250.pkl"]
    m_max = configuration.m_max
    # x = np.arange(1, m_max + 1)
    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ds_avg = pickle.load(f)  # ds_avg是一个m_max*1维数组，保存了m_max个用户下的系统延时
            y0 = ds_avg.reshape(1, len(ds_avg))[0]
            y.append(y0)
    # 计算的最多的用户数为 n + n/4
    # 显示所有计算的用户的延时
    # plt.plot(np.arange(1, len(y[0])+1), y[0], "r")
    # plt.plot(np.arange(1, len(y[1])+1), y[1], "b.")
    # plt.plot(np.arange(1, len(y[2])+1), y[2], "k.")

    # 显示部分计算的用户的延时
    plt.subplot(221)    # -------------------------------
    plt.plot(np.arange(1, 25 + 1), y[0], "k:")

    plt.title("Equipped antennae is 20")
    plt.xlabel("m")
    plt.ylabel("System average delay (TTI)")
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20 + 1, 2))
    plt.ylim(1 - 0.01, 1.1)
    plt.yticks(np.arange(1, 1.1, 0.01))
    plt.grid(b=True, linestyle=':')

    plt.subplot(222)    # -------------------------------
    plt.plot(np.arange(1, 62 + 1), y[1], "k:")

    plt.title("Equipped antennae is 50")
    plt.xlabel("m")
    plt.ylabel("System average delay (TTI)")
    plt.xlim(0, 50)
    plt.xticks(np.arange(0, 50 + 1, 5))
    plt.ylim(1 - 0.01, 1.1)
    plt.yticks(np.arange(1, 1.1, 0.01))
    plt.grid(b=True, linestyle=':')

    plt.subplot(223)    # -------------------------------
    plt.plot(np.arange(1, 125 + 1), y[2], "k:")

    plt.title("Equipped antennae is 100")
    plt.xlabel("m")
    plt.ylabel("System average delay (TTI)")
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100 + 1, 10))
    plt.ylim(1 - 0.01, 1.1)
    plt.yticks(np.arange(1, 1.1, 0.01))
    plt.grid(b=True, linestyle=':')

    plt.subplot(224)    # -------------------------------
    plt.plot(np.arange(1, 250 + 1), y[3], "k:")

    plt.title("Equipped antennae is 200")
    plt.xlabel("m")
    plt.ylabel("System average delay (TTI)")
    plt.xlim(0, 200)
    plt.xticks(np.arange(0, 200 + 1, 25))
    plt.ylim(1 - 0.01, 1.1)
    plt.yticks(np.arange(1, 1.1, 0.01))
    plt.grid(b=True, linestyle=':')  # 显示坐标网格

    plt.show()


def show_ds_m(filepath):
    """ 仿真结果
    在天线数n一定的情况下，展示系统平均延时和同时接入的用户数之间的关系
    :return:
    """
    print("runing show_ds_m() ...")
    # filepath = "./data/ds_avg_maxcopies=5_n=20.pkl"
    ds_avg = []
    with open(filepath, "rb") as f:
        ds_avg = pickle.load(f)

    m = len(ds_avg)
    x = np.arange(1, m + 1)
    y = ds_avg.reshape(1, m)[0]

    plt.plot(x, y, ".")

    plt.title("System average delay for different m \n "
              "(n = {},iternum = {})\n "
              "(ax = {})".format(n, configuration.iternum_ds, DS.get_fa(ax)))
    plt.xlabel("m")
    plt.ylabel("ds_avg (TTI)")

    # 设置坐标轴属性
    # plt.xlim(0, m+0.3)
    plt.xticks(np.arange(0, m+1, 10))
    # plt.ylim(0.9, 2.2)
    # plt.yticks(np.arange(1, 2.2, 0.1))
    plt.ylim(1, 1.2)
    plt.yticks(np.arange(1, 1.2, 0.01))

    plt.grid(b=True, linestyle=':')     # 显示坐标网格
    plt.show()


def show_ps_m_all():
    """
    确定一个最有分布configuration.ax，在配备不同的天线下，展示仿真情况下一次成功概率ps随G=m/n的变化
    数据来源于文件ps_n=xxx_iternum=xxx.pkl，由ps_m_parallel.py并行计算得到
    :return:
    """
    print("runing show_ps_m_all() ...")
    filepaths = ["./data/ps_maxcopies=8_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=200_iternum=10000.pkl"]
    x = np.arange(0, 1)     # G=[0, 1]
    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ps = pickle.load(f)     # ps是一个n*1维数组，保存了n=20,50,100,200个用户下的系统延时
            y0 = ps.reshape(1, len(ps))[0]
            y.append(y0)

    # DS.store2file(y, "test", 0, 0)    # pkl文件数据转换成txt
    # 显示所有计算的用户的延时
    plt.plot(np.arange(0, 1, 1/20), y[0], "r:", label="Equipped antennae is 20")
    plt.plot(np.arange(0, 1, 1/50), y[1], "b-.", label="Equipped antennae is 50")
    plt.plot(np.arange(0, 1, 1/100), y[2], "g--", label="Equipped antennae is 100")
    plt.plot(np.arange(0, 1, 1/200), y[3], "k-", label="Equipped antennae is 200")

    # plt.title("Transmission Successful Rate for different equiped n")
    plt.xlabel("Offered traffic G")
    plt.ylabel("Transmission Success Rate (Rs)")

    # 设置坐标轴属性
    plt.xlim(0, 1+0.03)
    plt.xticks(np.arange(0, 1+0.1, 0.1))
    plt.ylim(0, 1+0.03)
    plt.yticks(np.arange(0, 1+0.1, 0.1))

    plt.grid(b=True, linestyle=':')     # 显示坐标网格
    # plt.legend()        # 给图像加上图例
    plt.show()


def show_ps_m_all_2():
    """

    :return:
    """
    print("runing show_ps_m_all() ...")
    filepaths = ["./data/ps_maxcopies=4_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=512_iternum=10000.pkl"]

    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ps = pickle.load(f)     # ps是一个n*1维数组，保存了n=20,50,100,200个用户下的系统延时
            y0 = ps.reshape(1, len(ps))[0]
            y.append(y0)

    DS.store2file(y, "test", 0, 0)  # pkl文件数据转换成txt
    # 显示所有计算的用户的延时
    fig = plt.figure(figsize=(16, 8))      # 2:1  20:10  16:8
    # fig = plt.figure(figsize=(3.2, 8))      # 4:10  3.2:8
    # fig = plt.figure(figsize=(4.8, 8))      # 4:10  3.2:8  -> 6:10  4.8:8  加上标注

    # # 分布：maxcopies = 4
    # plt.plot(np.arange(0, 20, 1), y[0], "b+-")
    # plt.plot(np.arange(0, 50, 1), y[1], "bx-")
    # plt.plot(np.arange(0, 80, 1), y[2], "b--")
    # plt.plot(np.arange(0, 100, 1), y[3], "b-.")
    # plt.plot(np.arange(0, 120, 1), y[4], "b-")
    # plt.plot(np.arange(0, 150, 1), y[5], "b--.")
    # plt.plot(np.arange(0, 200, 1), y[6], "b.-")
    # plt.plot(np.arange(0, 512, 1), y[7], "b:")
    #
    # # 分布：maxcopies = 8
    # plt.plot(np.arange(0, 20, 1), y[8], "k+-")
    # plt.plot(np.arange(0, 50, 1), y[9], "kx-")
    # plt.plot(np.arange(0, 80, 1), y[10], "k--")
    # plt.plot(np.arange(0, 100, 1), y[11], "k-.")
    # plt.plot(np.arange(0, 120, 1), y[12], "k-")
    # plt.plot(np.arange(0, 150, 1), y[13], "k--.")
    # plt.plot(np.arange(0, 200, 1), y[14], "k.-")
    # plt.plot(np.arange(0, 512, 1), y[15], "k:")
    #
    # # 分布：maxcopies = 16
    # plt.plot(np.arange(0, 20, 1), y[16], "r+-")
    # plt.plot(np.arange(0, 50, 1), y[17], "rx-")
    # plt.plot(np.arange(0, 80, 1), y[18], "r--")
    # plt.plot(np.arange(0, 100, 1), y[19], "r-.")
    # plt.plot(np.arange(0, 120, 1), y[20], "r-")
    # plt.plot(np.arange(0, 150, 1), y[21], "r--.")
    # plt.plot(np.arange(0, 200, 1), y[22], "r.-")
    # plt.plot(np.arange(0, 512, 1), y[23], "r:")

    # 分布：maxcopies = 4
    plt.plot(np.arange(0, 20, 1), y[0], "b+-", label=r'$AN=4, n=20$')
    plt.plot(np.arange(0, 50, 1), y[1], "bx-", label=r'$AN=4, n=50$')
    plt.plot(np.arange(0, 80, 1), y[2], "b--", label=r'$AN=4, n=80$')
    plt.plot(np.arange(0, 100, 1), y[3], "b-.", label=r'$AN=4, n=100$')
    plt.plot(np.arange(0, 120, 1), y[4], "b-", label=r'$AN=4, n=120$')
    plt.plot(np.arange(0, 150, 1), y[5], "b--.", label=r'$AN=4, n=150$')
    plt.plot(np.arange(0, 200, 1), y[6], "b.-", label=r'$AN=4, n=200$')
    plt.plot(np.arange(0, 512, 1), y[7], "b:", label=r'$AN=4, n=512$')

    # 分布：maxcopies = 8
    plt.plot(np.arange(0, 20, 1), y[8], "k+-", label=r'$AN=8, n=20$')
    plt.plot(np.arange(0, 50, 1), y[9], "kx-", label=r'$AN=8, n=50$')
    plt.plot(np.arange(0, 80, 1), y[10], "k--", label=r'$AN=8, n=80$')
    plt.plot(np.arange(0, 100, 1), y[11], "k-.", label=r'$AN=8, n=100$')
    plt.plot(np.arange(0, 120, 1), y[12], "k-", label=r'$AN=8, n=120$')
    plt.plot(np.arange(0, 150, 1), y[13], "k--.", label=r'$AN=8, n=150$')
    plt.plot(np.arange(0, 200, 1), y[14], "k.-", label=r'$AN=8, n=200$')
    plt.plot(np.arange(0, 512, 1), y[15], "k:", label=r'$AN=8, n=512$')

    # 分布：maxcopies = 16
    plt.plot(np.arange(0, 20, 1), y[16], "r+-", label=r'$AN=16, n=20$')
    plt.plot(np.arange(0, 50, 1), y[17], "rx-", label=r'$AN=16, n=50$')
    plt.plot(np.arange(0, 80, 1), y[18], "r--", label=r'$AN=16, n=80$')
    plt.plot(np.arange(0, 100, 1), y[19], "r-.", label=r'$AN=16, n=100$')
    plt.plot(np.arange(0, 120, 1), y[20], "r-", label=r'$AN=16, n=120$')
    plt.plot(np.arange(0, 150, 1), y[21], "r--.", label=r'$AN=16, n=150$')
    plt.plot(np.arange(0, 200, 1), y[22], "r.-", label=r'$AN=16, n=200$')
    plt.plot(np.arange(0, 512, 1), y[23], "r:", label=r'$AN=16, n=512$')

    # plt.title("Transmission Successful Rate for different equiped n")
    plt.xlabel("Parallel UE number")
    plt.ylabel("Transmission Success Rate")

    # 设置坐标轴属性
    plt.xlim(0, 200+5)
    # plt.xlim(360, 520+20+80)
    # plt.xticks(np.append(np.arange(0, 210, 10), np.arange(400, 530, 10)))
    plt.xticks(np.arange(0, 210, 10))
    # plt.xticks(np.arange(400, 530+80, 40))
    plt.ylim(0, 1+0.05)
    plt.yticks(np.arange(0, 1+0.1, 0.1))

    plt.grid(b=True, linestyle=':')     # 显示坐标网格
    plt.legend(loc=1)
    plt.show()
    # plt.savefig('ps2.eps', dpi=600, format='eps')


def show_ps_m_all_3():
    """
    概率分布中的n视为天线数，而不是最大副本数，得出此时的最优分布和相应的G*值
    分析此最优分布的仿真下的成功率和并行用户数的关系
    :return:
    """
    print("runing show_ps_m_all() ...")
    filepaths = ["./data_0/ps_maxcopies=4_n=4_iternum=10000.pkl",
                 "./data_0/ps_maxcopies=8_n=8_iternum=10000.pkl",
                 "./data_0/ps_maxcopies=16_n=16_iternum=10000.pkl"]

    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ps = pickle.load(f)     # ps是一个n*1维数组，保存了n=20,50,100,200个用户下的系统延时
            y0 = ps.reshape(1, len(ps))[0]
            y.append(y0)

    # DS.store2file(y, "test", 0, 0)  # pkl文件数据转换成txt
    # 显示所有计算的用户的延时
    fig = plt.figure(figsize=(16, 8))      # 2:1  20:10  16:8
    # fig = plt.figure(figsize=(3.2, 8))      # 4:10  3.2:8

    plt.plot(np.arange(0, 4, 1), y[0], "b+-")
    plt.plot(np.arange(0, 8, 1), y[1], "bx-")
    plt.plot(np.arange(0, 16, 1), y[2], "b--")

    # plt.title("Transmission Successful Rate for different equiped n")
    plt.xlabel("Parallel UE number m")
    plt.ylabel("Transmission Success Rate Rs")

    # 设置坐标轴属性
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.ylim(0, 1+0.05)
    plt.yticks(np.arange(0, 1+0.1, 0.1))

    plt.grid(b=True, linestyle=':')     # 显示坐标网格
    # plt.legend()
    plt.show()


def show_ps_all_brokenaxes():
    print("runing show_ps_m_all() ...")
    filepaths = ["./data/ps_maxcopies=4_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=4_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=8_n=512_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=20_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=50_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=80_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=100_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=120_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=150_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=200_iternum=10000.pkl",
                 "./data/ps_maxcopies=16_n=512_iternum=10000.pkl"]

    y = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            ps = pickle.load(f)  # ps是一个n*1维数组，保存了n=20,50,100,200个用户下的系统延时
            y0 = ps.reshape(1, len(ps))[0]
            y.append(y0)

    fig = plt.figure(figsize=(15, 8))
    bax = brokenaxes(xlims=((0, 200 + 5), (0 + 320, 200 + 5 + 320)), ylims=((0, 1 + 0.03),), hspace=10)

    bax.plot(np.arange(0, 20, 1), y[0], "b+-")
    bax.plot(np.arange(0, 50, 1), y[1], "bx-")
    bax.plot(np.arange(0, 80, 1), y[2], "b--")
    bax.plot(np.arange(0, 100, 1), y[3], "b-.")
    bax.plot(np.arange(0, 120, 1), y[4], "b-")
    bax.plot(np.arange(0, 150, 1), y[5], "b--.")
    bax.plot(np.arange(0, 200, 1), y[6], "b.-")
    bax.plot(np.arange(0, 512, 1), y[7], "b:")

    # 分布：maxcopies = 8
    bax.plot(np.arange(0, 20, 1), y[8], "k+-")
    bax.plot(np.arange(0, 50, 1), y[9], "kx-")
    bax.plot(np.arange(0, 80, 1), y[10], "k--")
    bax.plot(np.arange(0, 100, 1), y[11], "k-.")
    bax.plot(np.arange(0, 120, 1), y[12], "k-")
    bax.plot(np.arange(0, 150, 1), y[13], "k--.")
    bax.plot(np.arange(0, 200, 1), y[14], "k.-")
    bax.plot(np.arange(0, 512, 1), y[15], "k:")

    # 分布：maxcopies = 16
    bax.plot(np.arange(0, 20, 1), y[16], "r+-")
    bax.plot(np.arange(0, 50, 1), y[17], "rx-")
    bax.plot(np.arange(0, 80, 1), y[18], "r--")
    bax.plot(np.arange(0, 100, 1), y[19], "r-.")
    bax.plot(np.arange(0, 120, 1), y[20], "r-")
    bax.plot(np.arange(0, 150, 1), y[21], "r--.")
    bax.plot(np.arange(0, 200, 1), y[22], "r.-")
    bax.plot(np.arange(0, 512, 1), y[23], "r:")

    bax.grid(axis='both', ls=':')
    bax.set_xlabel('Parallel UE number m')
    bax.set_ylabel('Transmission Success Rate Rs')

    plt.show()


def show_ps_m(filename = None):
    """ Transmission success rate
    天线数n一定的情况下，一次传输就成功的概率(全部成功才算成功)
    :param ax:
    :param n:

    :return:
    """
    print("runing show_ps_m() ...")
    ps = []

    with open(filename, "rb") as f:
        ps = pickle.load(f)

    n = len(ps)
    x = np.arange(1, n + 1)
    y = ps.reshape([1, n])[0]

    plt.plot(x, y, ".")

    plt.title("Transmission Success Rate for different m \n "
              "(n = {}, iternum = {}) \n"
              "(ax = {})".format(n, configuration.iternum_ps, DS.get_fa(ax)))
    plt.xlabel("m")
    plt.ylabel("Ps")
    # 设置x,y坐标轴属性
    plt.xlim(0, n)  # 设置横坐标范围
    plt.xticks(np.linspace(0, n, 11))  # 设置x轴刻度范围[0,n]，共11个刻度
    plt.ylim(0, 1.02)  # 设置纵坐标范围,1.02是为了使得上边缘好看
    plt.yticks(np.linspace(0, 1, 11))

    plt.grid(b=True, linestyle=':')  # 显示坐标网格
    plt.show()


def show_ps_gg_theo():
    """
    展示所有天线数对应的迭代收敛概率q
    :return:
    """
    filepath = "./data/q_all.pkl"

    with open(filepath, "rb") as f:
        q_all = pickle.load(f)

    x = gg = np.arange(0, 1, 0.001)
    for key in q_all:
        if key == 4:
            q4 = q_all[4]
            plt.plot(x, q4, "b-", label="maximum copies = " + str(key))
        if key == 8:
            q8 = q_all[8]
            plt.plot(x, q8, "g-", label="maximum copies = " + str(key))
        if key == 16:
            q16 = q_all[16]
            plt.plot(x, q16, "r-", label="maximum copies = " + str(key))
        if key == 20:
            q20 = q_all[20]
            plt.plot(x, q20, "k-", label="maximum copies = " + str(key))

    # plt.title("Ps vs offered traffic G")
    plt.xlabel("Offered traffic G")
    plt.ylabel("Transmission Success Rate (Rs)")

    # 设置坐标轴属性
    plt.xlim(0, 1)
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylim(-0.03, 1.03)
    plt.yticks(np.linspace(0, 1, 11))

    plt.grid(b=True, linestyle=':')  # 显示坐标网格
    plt.legend()    # 显示图例
    plt.show()


def _test_brokenaxes():
    fig = plt.figure(figsize=(5, 2))
    bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
    x = np.linspace(0, 1, 100)
    bax.plot(x, np.sin(10 * x), label='sin')
    bax.plot(x, np.cos(10 * x), label='cos')
    bax.legend(loc=3)
    bax.set_xlabel('time')
    bax.set_ylabel('value')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    ax = configuration.a_dict[8]

    # ==============================
    # 结果图表展示
    # ==============================
    # 理论分析得出的分布ax的成功概率q随G的变化
    # show_ps_gg_theo()

    # 确定一个最优分布configuration.ax，在配备不同的天线下，展示ds随m的变化
    # show_ds_m_all()
    # show_ds_m_four()
    # 在天线数n一定的情况下，展示系统平均延时和同时接入的用户数之间的关系
    # show_ds_m("./data/ds_avg_maxcopies=8_n=100_mmax=125.pkl")

    # 确定一个最优分布configuration.ax，在配备不同的天线下，展示ps随G的变化
    # show_ps_m_all()       # 废弃
    show_ps_m_all_2()     #
    # show_ps_m_all_3()       #
    # show_ps_all_brokenaxes()
    # 天线数n一定的情况下，一次传输就成功的概率(全部成功才算成功)
    # show_ps_m('./data/ps_n=200_iternum=1000.pkl')
    # _test_brokenaxes()

    end_time = time.time()
    total_time = end_time - start_time
    print("耗时 {:.3f}秒, {:.3f}分钟, {:.3f}小时".format(total_time, total_time / 60, total_time / (60 * 60)))


