from sklearn.cluster import OPTICS
import numpy as np
class optics():
    def __init__(self) :
        self.opt= OPTICS(min_samples=5, metric="precomputed")
    def clustering(self,graph):
        arr = np.array(graph)
        cluster = self.opt.fit(arr)
        print(min(cluster.labels_))
        print (cluster.labels_)
        
        