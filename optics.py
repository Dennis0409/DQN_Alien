from sklearn.cluster import OPTICS
import numpy as np
class optic():
    def __init__(self) :
        self.opt= OPTICS(min_samples=2, metric="precomputed")
    def clustering(self,graph):
        arr = np.array(graph)
        cluster = self.opt.fit(arr)
        print(max(cluster.labels_))
        print (cluster.labels_)

dis_graph=[]
path = 'test'
f = open(path, 'r')
lis = f.readlines()
for i in lis :
    dis_graph.append(i.split())
f.close()
optic().clustering(dis_graph)
