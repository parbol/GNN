import numpy as np
import torch 
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import optparse
import math


##############################################################################
#####These functions provide information about the metrics for the tracks#####
##############################################################################
def giveMeNext(edge_index, edge_label, target):
    
    for i in range(0, len(edge_index)):
        if edge_label[i] < 0.7:
            continue
        sou = int(edge_index[i,0].numpy())
        tar = int(edge_index[i,1].numpy())
        if sou == target:
            return True, tar
    return False, -1


def trackMetric(reference, prediction):

    #We identify tracks by the first node
    x = reference['source'].x
    edge_index = torch.t(reference['source', 'weight', 'target'].edge_index)
    edge_label = reference['source', 'weight', 'target'].edge_label

    lista = dict()
    #I fill the seeds for the tracks
    for i, ix in enumerate(x):
        r = math.sqrt(ix[0]**2 + ix[1]**2)
        if r < 11.0:
            lista[i] = []
            

    #I create the dictionary for the ground truth
    for i, edge in enumerate(edge_index):

        if edge_label[i] < 0.7:
            continue
        source = int(edge[0].numpy())
        target = int(edge[1].numpy())
        
        if source in lista:
            lista[source].append(source)        
            lista[source].append(target)
            continueSearching = True
            while continueSearching:
                valid, next = giveMeNext(edge_index, edge_label, target)
                if valid:
                    lista[source].append(next)
                    target = next
                else:
                    continueSearching = False
    
    #Now getting the information for the target dataset
    x_pred = prediction['source'].x
    edge_index_pred = torch.t(prediction['source', 'weight', 'target'].edge_index)
    edge_label_pred = prediction['source', 'weight', 'target'].edge_label
    
    #Creating track results
    tracks = dict()

    for source, tr in lista:
        goodLink = 0
        missingLink = 0
        fakeLink = 0
        tracks[source] = []
        for i, node in enumerate(tr):
            source = tr[i]
            if i < len(tr)-1:
                target = tr[i+1]
            else:
                target = -1
            sourcePartner = getPartners(edge_index_pred, edge_label_pred, source)
            if target == -1:
                if sourcePartner != -1:
                    fakeLink = fakeLink + 1
            else:
                if target == sourcePartner:
                    goodLink = goodLink + 1
                elif sourcePartner == -1:
                    missingLink = missingLink + 1
                else:
                    fakeLink = fakeLink + 1

            


    return lista


def getPartners(index, label, source):

    sourcePartner = -1
    for i, edge in enumerate(index):
        sourceP = int(edge[0].numpy())
        targetP = int(edge[1].numpy())
        
        if sourceP == source and label[i] > 0.5:
            sourcePartner = targetP
            
    return sourcePartner


##############################################################################
##############################################################################
##############################################################################



def cleanSolution(dataset):

    x = dataset['source'].x
    edge_index = torch.t(dataset['source', 'weight', 'target'].edge_index)
    label_index = dataset['source', 'weight', 'target'].edge_label
    
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

    
def printTrack(dataset, lista, number):

    x = dataset['source'].x
    print('------------------------------------')
    r = math.sqrt(x[number,0]**2 + x[number,1]**2)
    phi = math.atan2(x[number, 1], x[number, 0])
    print('Node:', r, phi)
    for i in lista[number]:
        r = math.sqrt(x[i,0]**2 + x[i,1]**2)
        phi = math.atan2(x[i, 1], x[i, 0])
        print('Value:', r, phi)




if __name__=='__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='input.pt', help='Input Reference Dataset')
    parser.add_option('-r', '--real', action='store', type='string', dest='realFile', default='inputReal.pt', help='Input Real Dataset')
    (opts, args) = parser.parse_args()
    
    dataset = torch.load(opts.inputFile)
    

    

    lista = trackMetric(dataset, dataset)
   
   
    printTrack(dataset, lista, 10)
    printTrack(dataset, lista, 490)


    fig = plt.figure(figsize = (16, 8), layout="constrained")
    ax1 = fig.add_subplot(1,2,1, projection = '3d')
    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    #drawGraph(dataset, ax1)
    #drawGraph(dataset, ax2)
    #plt.show()

