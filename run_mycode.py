# -*-coding:utf-8-*-
import DE
from DE import gmax_func
from DS import get_fa
import numpy as np


if __name__ == "__main__":
    # a = []

    # a = a4 = [0, 0, 0.517319660, 0, 0.48268034]
    # gmax = 0.8681640625
    a = a4 = [0, 0, 0.5173, 0, 0.4827]
    # gmax = 0.8681640625

    # a = a5 = [0, 0, 0.55624247, 0.0244032, 0.02445248, 0.39490186]
    # gmax = 0.8955078125
    # a = a5 = [0, 0, 0.5562, 0.0244, 0.0245, 0.3949]
    # gmax = 0.8955078125 = 0.895

    # a = a6 = [0, 0, 0.54695245, 0.13645276, 0.03752448, 0, 0.27907031]
    # gmax = 0.912109375
    # a = a6 = [0, 0, 0.5469, 0.1365, 0.0375, 0, 0.2791]
    # gmax = 0.912109375

    # a = a7 = [0, 0, 0.53040685, 0.227741860, 0, 0, 0, 0.24185129]
    # gmax = 0.9296875
    # a = a7 = [0, 0, 0.5304, 0.2277, 0, 0, 0, 0.2419]
    # gmax = 0.9296875

    # a = a8 = [0, 0, 0.512626115, 0.255117126, 0.0142260539,
    #           0.0000936564970, 0.00363484128, 0.000541534886, 0.213760672]
    # gmax = 0.939453125
    # a = a8 = [0, 0, 0.5126, 0.2551, 0.0142, 0, 0.0036, 0.0005, 0.2140]
    # a = a8 = [0, 0, 0.5126, 0.2556, 0.0142, 0, 0, 0, 0.2176]
    # gmax = 0.939453125

    # a = a9 = [0, 0, 0.50360498, 0.21454012, 0.07915919, 0, 0, 0.03208494, 0, 0.16063859]
    # gmax = 0.9501953125
    # a = a9 = [0, 0, 0.5136, 0.2145, 0.0792, 0, 0, 0.0321, 0, 0.1606]
    # gmax = 0.9423828125

    # a = a10 = [0, 0, 0.55624247, 0.0244032, 0.02445248, 0.39490186]

    # a = a11 = [0, 0, 0.55624247, 0.0244032, 0.02445248, 0.39490186]

    # a = a12 = [0, 0, 0.55624247, 0.0244032, 0.02445248, 0.39490186]

    # a = a13 = [0, 0, 0.55624247, 0.0244032, 0.02445248, 0.39490186]

    # a = a14 = [0, 0, 0.49223889, 0.2271849, 0.06464964, 0.06641725,
    #            0.01261146, 0.0046955, 0.00257912, 0.0027097, 0.00049738,
    #            0.00087016, 0.01611439, 0.04770498, 0.06172661]
    # gmax = 0.962890625
    # a = a14 = [0, 0, 0.4923, 0.2272, 0.0646, 0.0664,
    #            0.0126, 0.0047, 0.0026, 0.0027, 0.0005,
    #            0.0009, 0.0161, 0.0477, 0.0617]
    # gmax = 0.962890625

    # a = a15 = [0, 0, 0.468564417, 0.295296511, 0.00577520051, 0.0509034308,
    #            0.0244822904, 0.0354151826, 0.00833207043, 0.00285002263, 0.00141234221,
    #            0.00847123592, 0.000444787283, 0.00293390306, 0.000471596771, 0.0946470092]
    # gmax = 0.96484375
    # a = a15 = [0, 0, 0.4687, 0.2953, 0.0058, 0.0509,
    #            0.0245, 0.0354, 0.0083, 0.0029, 0.0014,
    #            0.0085, 0.0004, 0.0029, 0.0004, 0.0946]
    # gmax = 0.96484375

    # a = a16 = [0, 0, 0.497362624, 0.181892543,
    #            0.140521082, 0.0242413024, 0.00396685055, 0.0246402064,
    #            0.00568267128, 0.0102770559, 0.000132374788, 0.00218728026,
    #            0.00000165, 0.00216683475, 0.00148100365, 0.0609316823, 0.04451483810]
    # gmax = 0.966796875
    # a = a16 = [0, 0, 0.4975, 0.1819, 0.1420, 0.0242, 0.0040, 0.0246, 0,
    #            0.0103, 0, 0.0022, 0, 0.0022, 0, 0.0609, 0.0502]
    # gmax = 0.966796875

    # a = a20 = [0, 0, 0.504966853, 0.174562648, 0.107155159, 0.0479035618,
    #            0.0411004656, 0.00173082650, 0.0110194982, 0.00227610293, 0.00516351781,
    #            0.00249754533, 0.00746292438, 0.000348394898, 0.0118550384, 0.0350143018,
    #            0.00105006136, 0.00147717922, 0.0128682321, 0.0207812156, 0.01076647310]
    # gmax = 0.970703125
    # a = a20 = [0, 0, 0.5049, 0.1745, 0.1071, 0.0479,
    #            0.0411, 0.0017, 0.0110, 0.0023, 0.0052,
    #            0.0025, 0.0075, 0.0003, 0.0119, 0.0350,
    #            0.0011, 0.0015, 0.0129, 0.0208, 0.0108]
    # gmax = 0.9697265625

    # 判断概率和是否为1
    # temp = np.ones([len(a), 1])
    # result = np.dot(a, temp)
    # print("sum p = {}".format(result))
    #
    # # 将矩阵转换成函数表达式
    # print("f(x) = {}".format(get_fa(a)))
    #
    # # 计算平均使用的天线个数
    # antennae_num_avg = np.dot(a, np.arange(len(a)))
    # print("max copies = {}, antennae_num_avg = {}".format(len(a)-1, antennae_num_avg))
    #
    # gmax = gmax_func(a)
    # print("gmax = {}".format(gmax))

    # ==========================
    # 获取单个分布a的理论成功率q的迭代情况
    # q = DE.tsp_gg_theo(a)
    # print("q = {}".format(q))

    # 获取所有分布a_dict的理论成功率q的迭代情况
    DE.get_q_all()