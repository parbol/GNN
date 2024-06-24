import numpy as np
import torch 
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def getIntersection(r, phi):

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x,y


def getTracks(r, N, trainc, testc):

    phis = np.random.uniform(0, 2.0*np.pi, N)
    xp = np.asarray([])
    yp = np.asarray([])
    counter  = 0
    edges = []
    yc = []
    typeCounter = 0
    for phi in phis:
        x, y = getIntersection(r, phi)
        xp = np.concatenate((xp, x), axis=0)
        yp = np.concatenate((yp, y), axis=0)
        for i in range(0, len(x)-1):
            edges.append([counter, counter+1])
            edges.append([counter+1, counter])
            counter = counter + 1
        for i in range(0, len(x)):
            yc.append(typeCounter)
        typeCounter = typeCounter + 1
    
    xnm = np.column_stack((xp, yp))
    x = torch.from_numpy(xnm)
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.t(torch.tensor(edges, dtype=torch.long))
    y = torch.tensor(yc, dtype=torch.long)
    toTrain = []
    toTest = []
    toVal = []
    
    for i in range(0, len(xp)):
        if i < N*trainc:
            toTrain.append(1)
            toTest.append(0)
            toVal.append(0)
        elif i >= N*trainc and i < N*(trainc+testc):
            toTrain.append(0)
            toTest.append(1) 
            toVal.append(0)
        else:
            toTrain.append(0)
            toTest.append(1)
            toVal.append(1)

    #return Data(x = x, edge_index=edge_index, y = y, train_mask = toTrain, test_mask = toTest, val_mask = toVal)    
    return Data(x = x, edge_index=edge_index, y = y)    



if __name__=='__main__':
    Ntot = 200
    ptrain = 0.4
    ptest = 0.3
    pval = 0.3
    Ntrain = int(Ntot*ptrain)
    Ntest = int(Ntot*ptest)
    Nval = int(Ntot*pval)
    #Radious of the tracker layers
    r = np.asarray([1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0])
    dataset = getTracks(r, Ntrain, ptrain, ptest)
    torch.save(dataset, 'dataset.pt')
    #dataset = getTracks(r, Ntest, 'test')
    #torch.save(dataset, 'test.pt')
    #dataset = getTracks(r, Nval, 'val')
    #torch.save(dataset, 'val.pt')
    plt.plot(dataset.x[:,0].numpy(), dataset.x[:,1].numpy(), 'g*')
    plt.show()
