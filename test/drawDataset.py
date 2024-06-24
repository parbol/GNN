import numpy as np
import torch 
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import optparse
import math




def cleanSolution(dataset):

    x = dataset['source'].x
    edge_index = torch.t(dataset['source', 'weight', 'target'].edge_index)
    label_index = dataset['source', 'weight', 'target'].label_index
    
    lista = dict()

    for i, edge in enumerate(edge_index):      
        ref = edge[0].numpy()
        target = edge[1].numpy()
        r1 = math.sqrt(x[ref, 0]**2 + x[ref, 1]**2)
        r2 = math.sqrt(x[target, 0]**2 + x[target, 1]**2)
        if r2 > r1:
            if ref not in lista and r2 > r1:
                lista[ref] = []
            lista[ref].append(i)
 
    for node, edges in lista:

        if len(edges) == 1:
            continue
        maxScore = -1.0
        maxIndex = -1
        for j in edges:
            if label_index[j] > maxScore:
                maxScore = label_index[j]
                maxIndex = j
        for j in edges:
            if j != maxIndex:
                dataset['source', 'weight', 'target'].label_index[j] = 0.0

    







       
def drawGraph(dataset, ax):
    
    x = dataset['source'].x
    #ax.plot(x[:,0].numpy(), x[:,1].numpy(), 'g*')
    ax.plot(x[:,0].numpy(), x[:,1].numpy(), x[:,2].numpy(), 'g*')

    edge_index = dataset['source', 'weight', 'target'].edge_index
    edge_label = dataset['source', 'weight', 'target'].edge_label

    print(edge_index)

    for i, edge in enumerate(torch.t(edge_index)):
       
        edge1 = edge[0].numpy()
        edge2 = edge[1].numpy()
        if edge_label[i] > 0.5:
            #x1, y1 = x[edge1,0], x[edge1,1]
            #x2, y2 = x[edge2,0], x[edge2,1]
            x1, y1, z1 = x[edge1,0], x[edge1,1], x[edge1, 2]
            x2, y2, z2 = x[edge2,0], x[edge2,1], x[edge2, 2]
            xg = [] 
            xg.append(x1)
            xg.append(x2)
            yg = []
            yg.append(y1)
            yg.append(y2)
            zg = []
            zg.append(z1)
            zg.append(z2)
            #ax.plot(xg, yg, 'r-')    
            ax.plot(xg, yg, zg, 'r-')    

    


if __name__=='__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='input.pt', help='Input Reference Dataset')
    parser.add_option('-r', '--real', action='store', type='string', dest='realFile', default='inputReal.pt', help='Input Real Dataset')
    (opts, args) = parser.parse_args()
    
    dataset = torch.load(opts.inputFile)
    
    fig = plt.figure(figsize = (16, 8), layout="constrained")
    ax1 = fig.add_subplot(1,2,1, projection = '3d')
    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    drawGraph(dataset, ax1)
    drawGraph(dataset, ax2)
    plt.show()

