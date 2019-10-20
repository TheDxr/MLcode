# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = np.array([0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
    y = np.array([1,1.7,2.5,4.1,5.3,6.1,7.2,8.1,9.1,9.8])
    z = np.zeros((len(x),len(y)))
    iteration = 100
    rate = 0.001
    b = 0
    k = 0
    for i in range(iteration):
        b_grad = 0.0
        k_grad = 0.0
        for n in range(len(x)):
            b_grad = b_grad - 2.0*(y[n] - b - k*x[n])*1.0
            k_grad = k_grad - 2.0*(y[n] - b - k*x[n])*x[n]
        b = b - rate*b_grad
        k = k - rate*k_grad
        if i <= 10:
            print(k," ",b)
    plt.title("Test")
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.plot(x,y,"o")
    x2 = np.arange(10)
    y2 = k*x2 + b
    plt.plot(y2)
    print(k," ",b)
    plt.show()


