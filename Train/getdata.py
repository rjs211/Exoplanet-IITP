import numpy as np
import sklearn as sk
import sklearn
from sklearn.utils import shuffle


'''
returns data,label,indices after oversampling by repeating

args: 

data: the datapoints
label: the labels correspnding to datapoints
indpos : the indices of positive samples in the datapoints
indneg:  the indices of positive samples in the datapoints
posTimes:  the number of times the positive samples has to be added to the data
if_shuffle:  if radom shuffling is required or not 

'''

def getxy(data , label , indpos,indneg,posTimes = 1 , if_shuffle = True ):    
    posdata = data[indpos]
    poslabel = label[indpos]
    posind = indpos
    fposdata = data[indpos]
    fposlabel = label[indpos]
    fposind = indpos
    for i in range(posTimes -1) :
        fposdata = np.concatenate( (fposdata,posdata),axis = 0)
        fposlabel = np.concatenate((fposlabel,poslabel),axis = 0)
        fposind = fposind + posind
    
    allx = data[indneg]
    ally = label[indneg]
    allind = indneg
    #print(type(fposdata))
    #print(fposdata.shape )
    #print( fposlabel.shape)
    #print(allx.shape)
    #print(ally.shape)
    
    allx = np.concatenate((allx,fposdata), axis = 0)
    ally = np.concatenate((ally,fposlabel), axis = 0)
    allind = allind + fposind
    
    if if_shuffle == True:
        allx,ally,allind = shuffle(allx,ally,allind ,random_state = 0)
    
    return allx,ally,allind
