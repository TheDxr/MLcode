# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
 
#读取文件
datafile = 'MachineLearning\\DataSet\\go_track_trackspoints.csv'
outfile = 'out.csv'
data = pd.read_csv(datafile,usecols=["latitude","longitude"])
d = DataFrame(data)
d.head()
# ----------------------------------聚类-------------------------------------------

mod = KMeans(n_clusters=3, n_jobs = 4, max_iter = 500)#聚成3类数据,并发数为4，最大循环次数为500
mod.fit_predict(d)#y_pred表示聚类的结果
 
#聚成3类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis = 1)
r.columns = list(d.columns) + ["Clustering"]
print(r)
 
#给每一条数据标注上被分为哪一类
r = pd.concat([d, pd.Series(mod.labels_, index = d.index)], axis = 1)
r.columns = list(d.columns) + ["Clustering"]
print(r.head())
r.to_csv(outfile)#如果需要保存到本地，就写上这一列

# ------------------------------------可视化过程------------------------------------------

from sklearn.manifold import TSNE
 
ts = TSNE()
ts.fit_transform(r)
ts = pd.DataFrame(ts.embedding_, index = r.index)
 
import matplotlib.pyplot as plt
 
a = ts[r["Clustering"] == 0]
plt.plot(a[0], a[1], 'r.')
a = ts[r["Clustering"] == 1]
plt.plot(a[0], a[1], 'go')
a = ts[r["Clustering"] == 2]
plt.plot(a[0], a[1], 'b*')
plt.show()