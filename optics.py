from sklearn.cluster import OPTICS
import numpy as np
class optic():
    def __init__(self) :
        self.opt= OPTICS(min_samples=3, metric="precomputed")
    def clustering(self,graph):
        arr = np.array(graph)
        cluster = self.opt.fit(arr)
        print(max(cluster.labels_))
        print (cluster.labels_)
        
        