from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
class optic():
    def __init__(self) :
        self.opt= OPTICS(min_samples=4, metric="precomputed")
    def clustering(self,graph):
        arr = np.array(graph)
        cluster = self.opt.fit(arr)
        print(max(cluster.labels_))
        print (cluster.labels_)
        return cluster
    def test(self):
        dis_graph=[]
            
        path = 'test'
        f = open(path, 'r')
        lis = f.readlines()
        for i in lis :
            dis_graph.append(list(map(int,i.split())))
        f.close()
        for i in range(len(dis_graph)) :
            for j in range (len(dis_graph)):
                dis_graph[i][j]/=10
                dis_graph[i][j]**=2
        clust = optic().clustering(dis_graph)
        space = np.arange(len(dis_graph))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]
        plt.figure(figsize=(10, 7))
        G = gridspec.GridSpec(1,1)
        ax1 = plt.subplot(G[0, :])
        colors = ['g.', 'r.', 'b.', 'y.', 'c.']
        for klass, color in zip(range(0, 5), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            ax1.plot(Xk, Rk, color, alpha=0.3)
        ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
        #ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
        #ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
        ax1.set_ylabel('Reachability')
        ax1.set_title('Reachability Plot')
        plt.tight_layout()
        plt.show()
