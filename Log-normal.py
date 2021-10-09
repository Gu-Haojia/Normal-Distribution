import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# 定义标准正态
def cdf(x):
    return norm.cdf(x, loc=0, scale=1)


def pdf(x):
    return norm.pdf(x, loc=0, scale=1)


# 绘图函数（μ*float, σ*float)
def lognormal(mu, sig):
    if sig <= 0:
        print("Scale Error")
        return

    if sig < 1.5:
        x = np.arange(0.0000001, np.exp(mu + 2 * sig), np.exp(mu + 2 * sig) / 1000)
    else:
        x = np.arange(0.0000001, np.exp(mu) + 3, (np.exp(mu) + 3) / 1000)  # 范围，默认1000点,根据sig范围划分不同作图区域

    y1 = norm.pdf(np.log(x), loc=mu, scale=sig) / x  # 作出4个值的序列

    y2 = []
    for t in x:
        y_2 = cdf((np.log(t) - mu) / sig)
        y2.append(y_2)

    y3 = []
    for t in x:
        y_3 = (1 - cdf((np.log(t) - mu) / sig))
        y3.append(y_3)

    y4 = []
    for t in x:
        y_4 = pdf((np.log(t) - mu) / sig) / t / (sig * (1 - cdf((np.log(t) - mu) / sig)))
        y4.append(y_4)

    plt.figure(figsize=(10, 8))  # 绘图
    plt.subplot(2, 2, 1)
    plt.plot(x, y1, label="PDF", color='b')
    plt.title("PDF")

    plt.subplot(2, 2, 2)
    plt.plot(x, y2, label="CDF", color='g')
    plt.title("CDF")

    plt.subplot(2, 2, 3)
    plt.plot(x, y3, label="R", color='y')
    plt.title("R")

    plt.subplot(2, 2, 4)
    plt.plot(x, y4, label="λ", color='r')
    plt.title("λ")

    plt.suptitle("Log-normal Distribution")

    plt.show()
    return


# 实例
sig = float(input("sigma:"))
mu = float(input("mu:"))

lognormal(mu, sig)
