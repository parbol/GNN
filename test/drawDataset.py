import numpy as np
import torch 
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import optparse



def drawGraph(dataset, ax):
    
    x = dataset['source'].x
    ax.plot(x[:,0].numpy(), x[:,1].numpy(), 'g*')
    edge_index = dataset['source', 'weight', 'target'].edge_index
    edge_label = dataset['source', 'weight', 'target'].edge_label

    for i, edge in enumerate(torch.t(edge_index)):
        print(x)
        edge1 = edge[0].numpy()
        edge2 = edge[1].numpy()
        if edge_label[i] > 0.75:
            x1, y1 = x[edge1,0], x[edge1,1]
            x2, y2 = x[edge2,0], x[edge2,1]
            xg = [] 
            xg.append(x1)
            xg.append(x2)
            yg = []
            yg.append(y1)
            yg.append(y2)
            ax.plot(xg, yg, 'r-')    

    



if __name__=='__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='input.pt', help='Input Reference Dataset')
    parser.add_option('-r', '--real', action='store', type='string', dest='realFile', default='inputReal.pt', help='Input Real Dataset')
    (opts, args) = parser.parse_args()
    
    dataset = torch.load(opts.inputFile)
    
    fig = plt.figure(figsize = (16, 8), layout="constrained")
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    drawGraph(dataset, ax1)
    drawGraph(dataset, ax2)
    plt.show()

