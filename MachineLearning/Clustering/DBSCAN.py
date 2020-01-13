# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from sklearn import datasets


#读取文件
datafile = 'MachineLearning\\DataSet\\go_track_trackspoints.csv'
outfile = 'out.csv'
data = pd.read_csv(datafile,usecols=["latitude","longitude"],nrows = 500)
d = DataFrame(data)
X = np.array(d)

# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                       noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
#X = np.concatenate((X1, X2))

if __name__ == '__main__':
    from sklearn.cluster import DBSCAN
    y_pred = DBSCAN(eps = 0.005, min_samples = 5).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()