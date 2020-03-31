# -*- coding: utf-8 -*-
import struct
import numpy as np
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, sizes):
        """``sizes``每一层的神经元数量(For example, if the listwas [2, 3, 1] 
        then it would be a three-layer network, with thefirst layer containing 
        2 neurons, the second layer 3 neurons,and the third layer 1 neuron) 
        ``biases``weights: (0,1)区间的随机数 """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ ``a``输入``Return``神经网络的输出  """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """用mini-batch stochasticgradient descent训练神经网络.
        `training_data为元祖`(x, y)表示输入和输出.  
        `epoch表示迭代次数,.
        `mini_batch_size`表示取样块的大小"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("epoch {0}: {1} / {2}".format(j + 1,
                                                   self.evaluate(test_data), n_test))
            else:
                print("epoch {0} complete".format(j + 1))

    def update_mini_batch(self, mini_batch, eta):
        """应用梯度下降更新神经网络,使用反向传播对单个mini batch.
        ``mini_batch`` 为元祖``(x, y)``,和``eta``是learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        反向传播计算梯度
        :param x: 单个输入
        :param y: 期望输出
        :return: 梯度
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # list to store all the activations, layer by layer
        activations = [np.array(x)]
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(
                self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # @staticmethod
    # def cost_derivative(output_activations, y):
    #     """Return the vector of partial derivatives \partial C_x /
    #     \partial a for the output activations."""
    #     return (output_activations-y)
    
    def cost_derivative(self,output_activation, y):
        """
        :param output_activation: 网络的实际输出，是一个十维的向量，0-9 的激活值
        :param y: 预期输出：在这里是一个整数值：0-9
        :return: 返回一个代表差异的向量，分别表示输出层各个节点的误差，正值表示比期望大，负值表示比期望小，
                 绝对值表示偏离期望的程度（修改的优先级）
        此处采用二次代价：
        对于单个样本：
        二次代价 C_x = \frac{(y-a)^2}{2}
                a = \sigma(z)
                \delta^L= a - y
        """
        # 由于期望输出向量是[0,0,0,0,0,0,0,0,0,0][y]=1,
        # 这里避免额外的向量计算，直接计算 output_activation - 期望输出向量
        #output_activation = output_activation[:]
        output_activation[y] -= 1
        return output_activation

#### Miscellaneous functions
def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """sigmoid函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))
