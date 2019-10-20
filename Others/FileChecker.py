# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn import datasets
import os


def check(filePath):
    flag = 0
    f = open(filePath, 'r')
    lines = f.readlines()
    for i in lines:
        if i.find("check") != -1:
            flag = True
    return flag


if __name__ == '__main__':
    path = 'C:\\Users\\Dxr\\Desktop\\Minecraft\\.minecraft\\versions\\1.7.10-Forge10.13.4.1614-1.7.10\\config'
    files = os.listdir(path)
    for file in files:
        if not (file.find('.cfg') != -1):
            continue
        flag = False
        try:
            flag = check(path + '\\' + file)
        except BaseException:
            print('[error]:' + file)

        if flag:
            print(file)

