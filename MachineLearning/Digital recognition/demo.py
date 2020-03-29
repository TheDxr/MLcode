# -*- coding: utf-8 -*-
import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt
import network

def read_data(train_path,test_path):
    """
    读取MNIST数据集
    `return` tuple(数据集,标签)
    """
    data = open(train_path,'rb').read()
    data2 = open(test_path,'rb').read()
    index = 0
    index2 = 8
    ret_data = []
    magic, num, width, height = struct.unpack_from('>IIII',data,index)
    index += struct.calcsize('>IIII')
    for i in range(num):
        img = struct.unpack_from('%dB'%(height*width),data,index)
        lable = struct.unpack_from('B',data2,index2)
        index += struct.calcsize('%dB'%(height*width))
        index2 += 1
        ret_data.append( [np.array(img).reshape(width*height,1),int(lable[0])] )
    return ret_data

if __name__ == "__main__":
    training_data = read_data("MachineLearning/Digital recognition/DataSet/train-images.idx3-ubyte", \
    "MachineLearning/Digital recognition/DataSet/train-labels.idx1-ubyte")
    test_data = read_data('MachineLearning/Digital recognition/DataSet/t10k-images.idx3-ubyte',\
        "MachineLearning/Digital recognition/DataSet/t10k-labels.idx1-ubyte")
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 20, 10, 3.0, test_data=test_data)
    # plt.figure()
    # for i in range(0,1):
    #     plt.subplot((251 + i))
    #     plt.title('%d'%test_data[i][1])
    #     plt.imshow(test_data[i][0].reshape(28,28),cmap='gray_r')
    # plt.show()